# Comparison: Uniform i1 Extension Approaches for AMDGPU

## The Problem

When extending a uniform scalar i1 (from SCC) to i32, the current lowering produces an inefficient VGPR roundtrip:

```asm
; Current (wasteful)
s_cmp_eq_u32 s0, s1
s_cselect_b32 s1, -1, 0    ; SCC → SGPR as -1/0
v_cndmask_b32 v0, 0, 1, s1 ; SGPR → VGPR
v_readfirstlane_b32 s2, v0 ; VGPR → SGPR (expensive!)
```

What we want:
```asm
; Ideal
s_cmp_eq_u32 s0, s1
s_cselect_b32 s1, 1, 0     ; Direct selection of 0/1 into SGPR
```

---

## Background: Why i1 is Ambiguous on AMDGPU

On AMDGPU, a boolean (i1) can be represented in different ways:

| Location | Size | Representation | Use Case |
|----------|------|----------------|----------|
| SCC | 1-bit | Scalar condition code | Scalar comparisons |
| VCC/EXEC | 64-bit (Wave64) or 32-bit (Wave32) | Lane mask | Vector comparisons |
| SGPR | 32-bit | 0 or -1 | Materialized scalar boolean |
| VGPR | 32-bit per lane | 0 or 1 per lane | Materialized vector boolean |

The problem is that SelectionDAG only sees the **type** (i1), not **where** the value lives. This ambiguity causes issues when InstrEmitter assigns register classes.

---

## Approach 1: TableGen Uniform/Divergent Patterns (tdm_isel branch)

**Commit:** `cc1f6f13c792`

**Implementation:**
```tablegen
// For uniform i1 (Wave32)
def : GCNPat <
  (i32 (UniformUnaryFrag<zext> i1:$src0)),
  (S_AND_B32 (COPY_TO_REGCLASS i1:$src0, SReg_32), 1)
>;

// For uniform i1 (Wave64)
def : GCNPat <
  (i32 (UniformUnaryFrag<zext> i1:$src0)),
  (S_AND_B32 (EXTRACT_SUBREG i1:$src0, sub0), 1)
>;
```

**Issues:**
1. **Two instructions instead of one:** Still generates `s_cselect -1,0` + `s_and 1` instead of `s_cselect 1,0`
2. **Wave64 semantic issue:** Using `EXTRACT_SUBREG sub0` from a 64-bit wave mask is conceptually incorrect (even if it works mathematically)
3. **VGPR destination regression:** When result needs VGPR, generates `s_and + v_mov` (2 instructions) instead of `v_cndmask` (1 instruction)

---

## Approach 2: PR #174539 (InstrEmitter + S_AND_B32 pattern)

**PR:** https://github.com/llvm/llvm-project/pull/174539

### TableGen Pattern
```tablegen
def : GCNPat <
  (i32 (UniformUnaryFrag<zext> i1:$src)),
  (S_AND_B32 $src, (i32 1))
>;

def : GCNPat <
  (i64 (UniformUnaryFrag<zext> i1:$src)),
  (S_AND_B64 $src, (i64 1))
>;
```

### InstrEmitter.cpp Fix (the key contribution)

**The Problem:** InstrEmitter determines virtual register classes based on DAG type:
```cpp
// Before: Just look at DAG type and divergence
UseRC = TLI->getRegClassFor(VT, Op->isDivergent());
// For i1 on Wave64, this returns SReg_64 - WRONG for SCC!
```

**The Fix:** When copying from a physical register (like SCC), get the register class from the physical register itself:
```cpp
// After: Check if source is an implicit physical register def
if (SrcRegIsImplicitDef) {
  // Get register class FROM the physical register itself
  const TargetRegisterClass *ImplicitRC = TRI->getMinimalPhysRegClass(SrcReg);
  UseRC = TRI->getCrossCopyRegClass(ImplicitRC);
  // For SCC: getMinimalPhysRegClass(SCC) → SCC's reg class
  //          getCrossCopyRegClass(...) → SReg_32
} else {
  UseRC = TLI->getRegClassFor(VT, Op->isDivergent());
}
```

**How it detects implicit defs:**
```cpp
auto Node = Op.getNode();
if (Node->isMachineOpcode() && SrcReg.isPhysical()) {
  const MCInstrDesc &II = TII->get(Op.getMachineOpcode());
  auto ImplicitDefs = II.implicit_defs();
  auto MCSrcReg = SrcReg.asMCReg();

  SrcRegIsImplicitDef = II.NumDefs == 0 &&
    llvm::any_of(ImplicitDefs, [MCSrcReg](MCPhysReg Reg) {
      return Reg == MCSrcReg;
    });
}
```

This checks if:
1. The source is a physical register
2. It's an implicit def of the instruction (like SCC from S_CMP_*)
3. The instruction has no explicit defs (NumDefs == 0)

### What This Fixes

| Scenario | Before | After |
|----------|--------|-------|
| Copy from SCC on Wave64 | Uses SReg_64 ❌ | Uses SReg_32 ✓ |
| Copy from SCC on Wave32 | Uses SReg_32 ✓ | Uses SReg_32 ✓ |
| Copy from VCC on Wave64 | Uses SReg_64 ✓ | Uses SReg_64 ✓ |

### Empirical Test Results (2025-01-29)

We tested PR #174539 directly by applying its changes and running llc.

#### Test Case: Uniform zext with SGPR destination
```llvm
define amdgpu_ps i32 @test_uniform_zext(i32 inreg %a, i32 inreg %b) {
  %cmp = icmp slt i32 %a, %b
  %ext = zext i1 %cmp to i32
  ret i32 %ext
}
```

| Version | gfx1100 (Wave32) Output | Instruction Count |
|---------|------------------------|-------------------|
| **Baseline** | `s_cmp` + `s_cselect -1,0` + `v_cndmask 0,1` + `v_readfirstlane` | 4 |
| **PR #174539** | `s_cmp` + `s_cselect -1,0` + `s_and 1` | 3 |
| **Ideal** | `s_cmp` + `s_cselect 1,0` | 2 |

**Key finding:** PR #174539 **does NOT eliminate s_cselect** (contrary to author's claim). The output is:
```asm
s_cmp_lt_i32 s0, s1
s_cselect_b32 s0, -1, 0   ; <-- s_cselect is still here!
s_delay_alu instid0(SALU_CYCLE_1)
s_and_b32 s0, s0, 1
```

#### Test Case: Uniform zext with VGPR destination (Issue 3)

From `llvm/test/CodeGen/AMDGPU/anyext.ll`:
```llvm
; Result stored to global memory (requires VGPR)
%cmp = icmp eq i32 %cond, 0
%ext = zext i1 %cmp to i32
store i32 %ext, ptr addrspace(1) %out
```

**Wave32 (gfx1100):**

| Version | Output | Instruction Count |
|---------|--------|-------------------|
| **Baseline** | `s_cselect -1,0` + `v_cndmask 0,1` | 2 |
| **PR #174539** | `s_cselect -1,0` + `s_and 1` + `v_mov` | 3 |

**Wave64 (verde):**

| Version | Output | Instruction Count |
|---------|--------|-------------------|
| **Baseline** | `s_cselect_b64 -1,0` + `v_cndmask 0,1` | 2 |
| **PR #174539** | `s_cselect_b64 -1,0` + **illegal copy** + `s_and 1` + `v_mov` | 4 + BUG! |

**Critical finding:** PR #174539 has a **Wave64 bug** that generates an "illegal copy" warning:
```asm
s_cselect_b64 s[4:5], -1, 0
; illegal copy s[4:5] to s4     ; <-- BUG!
s_and_b32 s4, s4, 1
v_mov_b32_e32 v0, s4
```

**Root cause of Wave64 bug:** The InstrEmitter fix and the TableGen pattern are working at cross-purposes:

1. **InstrEmitter fix:** Correctly identifies that i1 values should use the wave-size register class. On Wave64, this means `SReg_64`, so it generates `s_cselect_b64` to materialize the boolean (this is correct for general i1 handling).

2. **The pattern in SIInstructions.td:** Uses `S_AND_B32` which is a **fixed 32-bit** operation:
   ```tablegen
   def : GCNPat<
     (i32 (UniformUnaryFrag<zext> i1:$src)),
     (S_AND_B32 $src, (i32 1))
   >;
   ```
   When this pattern tries to use the i1 value in `S_AND_B32`, it needs a 32-bit input. But the InstrEmitter has already chosen `SReg_64` for the i1, resulting in an invalid 64-bit to 32-bit copy.

**To fix this properly**, the pattern would need to be wave-size aware:
- On Wave32: `S_AND_B32`
- On Wave64: `S_AND_B64` (or properly extract the low 32 bits)

### What PR #174539 Does NOT Fix

1. **Issue 1 NOT FIXED:** Still generates `s_cselect -1/0` + `s_and_b32` (3 instructions vs ideal 2)
2. **Issue 3 NOT FIXED:** VGPR destination regression - generates `s_and + v_mov` instead of `v_cndmask`
3. **Wave64 bug:** Generates illegal copy from 64-bit to 32-bit register
4. **sext not handled:** PR only adds zext pattern, sext still goes through VGPR roundtrip

---

## Approach 3: Custom C++ ISel (tdm_isel_scc branch, stashed)

**Implementation:** Custom instruction selection in `AMDGPUISelDAGToDAG::Select()` that matches `(zext/sext (setcc a, b, cc))` for uniform scalar comparisons.

**Key insight:** Must emit both `S_CMP` and `S_CSELECT_B32` together with glue, because if we only replace the extension, the setcc becomes dead (no users) and gets eliminated.

**Why glue is needed:**
```
Without glue:
  setcc → S_CMP (defines SCC, but no users!)  → ELIMINATED
  zext  → S_CSELECT_B32 (reads SCC)           → SCC undefined!

With glue:
  setcc → S_CMP ─┬─glue─→ S_CSELECT_B32
                 └─SCC──→
  Both instructions stay together
```

**Result:**
```asm
; Before
s_cmpk_lt_i32 s0, 0x3e9
s_cselect_b32 s0, -1, 0
v_cndmask_b32_e64 v2, 0, 1, s0

; After (with this approach)
s_cmpk_lt_i32 s0, 0x3e9
s_cselect_b32 s0, 1, 0
v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, s0
```

**Fixes:**
- Issue 1: YES - Directly generates `s_cselect 1, 0`
- Issue 2: YES - No Wave64 semantic issues (works at setcc level)
- Issue 3: PARTIAL - Still needs v_mov when result goes to VGPR

---

## Reviewer's Comment (Explained)

> "The problem is SelectionDAG thinks the type can be informed by the type, but i1 is ambiguous for AMDGPU. Currently InstrEmitter chooses the wave size register for the vreg for the boolean value. But for copies from SCC, it should always use i32 as the virtual register class to copy to. InstrEmitter needs to take the type directly from the physical register source. i.e., #174539 is solving the same problem but is closer."

**Translation:**

### The Compilation Flow

```
Stage 1: SelectionDAG (types only)
  (setcc a, b, eq)  → i1
  (zext i1 → i32)   → i32

Stage 2: Instruction Selection
  setcc → S_CMP_EQ_U32 (implicitly defines SCC)
  zext  → ???

Stage 3: InstrEmitter (assigns register classes)
  Problem: Sees i1, picks wave-size register class
  Fix: Look at physical register source (SCC) instead
```

### Why PR #174539 is "Closer"

The reviewer is saying:
1. Your approach (glue-based setcc+zext fusion) is a **workaround** at the ISel level
2. PR #174539 fixes the **root cause** in InstrEmitter's register class assignment
3. With the InstrEmitter fix, other patterns can work correctly without needing special handling

However, our testing shows PR #174539:
- Still doesn't generate optimal code (`s_cselect 1,0` directly)
- Has VGPR destination regression (same as Approach 1)
- Has Wave64 bugs

---

## Summary Comparison

| Aspect | Baseline | TableGen Patterns (tdm_isel) | PR #174539 | Custom C++ ISel (stashed) |
|--------|----------|------------------------------|------------|---------------------------|
| **SGPR dest: instruction count** | 4 | 3 | 3 | **2** |
| **SGPR dest: output** | s_cselect + v_cndmask + readfirstlane | s_cselect + s_and | s_cselect + s_and | **s_cselect 1,0** |
| **VGPR dest: instruction count** | 2 | 3 (regression) | 3 (regression) | 3 |
| **VGPR dest: output** | s_cselect + v_cndmask | s_cselect + s_and + v_mov | s_cselect + s_and + v_mov | s_cselect + v_mov |
| **Wave64 correctness** | ✓ | Questionable (EXTRACT_SUBREG) | ❌ (illegal copy bug) | ✓ |
| **Fixes root cause** | N/A | No | **Yes** (InstrEmitter) | No |
| **Scope** | N/A | zext, sext, anyext | zext only | zext, sext |
| **Requires glue** | N/A | No | No | Yes |

---

## Ideal Solution

The ideal solution would combine:

1. **InstrEmitter fix (from PR #174539):** Properly handle copies from SCC to always use SReg_32
   - Fixes the infrastructure so i1 from SCC is handled correctly everywhere
   - Enables other patterns to work without special-casing
   - **But needs Wave64 bug fix**

2. **Custom ISel or optimized pattern:** Match `(zext/sext (setcc ...))` to emit `S_CMP + S_CSELECT 1/0/-1` directly
   - Generates optimal single-instruction materialization
   - Works at the right level (before i1 is materialized)

3. **VGPR-aware pattern:** When the result is used by VGPR instructions, use `V_CNDMASK_B32` directly instead of forcing SGPR path
   - Avoids the s_and + v_mov regression
   - The current `DivergentUnaryFrag<zext>` pattern should take priority for VGPR destinations
   - Requires careful pattern ordering or destination-aware matching

---

## Open Questions

1. **Can both approaches coexist?** The InstrEmitter fix is orthogonal to the ISel pattern matching. They could potentially be combined.

2. **Is the glue approach acceptable?** Using glue to tie S_CMP and S_CSELECT is a workaround. Is there a cleaner way? (See alternatives explored: pseudo instructions, CopyToReg chaining, custom DAG nodes)

3. **How to handle VGPR destinations?** When the result goes to VGPR anyway, using V_CNDMASK_B32 directly is more efficient. Options:
   - Don't add uniform pattern, let divergent pattern handle VGPR cases
   - Add complexity cost to uniform pattern so v_cndmask is preferred when result needs VGPR
   - Detect destination register class during ISel

4. **Should we report the Wave64 bug in PR #174539?** The "illegal copy" from s[4:5] to s4 is a correctness issue.

---

## Alternatives to Glue (Explored)

Based on codebase exploration, these patterns avoid glue while keeping instructions alive:

### Option 1: Pseudo with Explicit SCC Output
Following `S_ADD_CO_PSEUDO` pattern:
```tablegen
def S_SETCC_ZEXT_PSEUDO : SPseudoInstSI <
  (outs SReg_32:$sdst, SSrc_i1:$scc_out),  // SCC is explicit output
  (ins SSrc_b32:$src0, SSrc_b32:$src1, i32imm:$cond_code)
> {
  let usesCustomInserter = 1;  // Expand to S_CMP + S_CSELECT post-ISel
}
```

### Option 2: CopyToReg Chain
From `SelectBR_CC_SCC` pattern:
```cpp
SDValue Cmp = CurDAG->getMachineNode(AMDGPU::S_CMP_EQ_U32, ...);
SDValue SCCCopy = CurDAG->getCopyToReg(Chain, SL, AMDGPU::SCC, Cmp);
SDValue Sel = CurDAG->getMachineNode(AMDGPU::S_CSELECT_B32, ...,
                                      SCCCopy.getValue(0));  // chain dependency
```

### Option 3: Custom DAG Node
Create target-specific node during DAG combine:
```cpp
// AMDGPUISD::SCALAR_SETCC_ZEXT combines setcc + zext
return DAG.getNode(AMDGPUISD::SCALAR_SETCC_ZEXT, DL, MVT::i32,
                   LHS, RHS, CondCode);
```

---

## Real-World Performance: TDM Attention Kernel (gfx1250)

We tested our TableGen pattern fix (Approach 1) on a real attention forward kernel that uses Tensor DMA (TDM) descriptors on gfx1250.

### The Impact on TDM Descriptors

TDM instructions require their descriptors to be in SGPRs. When a descriptor field is computed from a uniform boolean (i1), the SGPR→VGPR→SGPR roundtrip forces the entire descriptor into VGPRs, requiring `V_READFIRSTLANE` to get it back.

**Original MIR (tdm.mir):**
```mir
; 1. Compare and get boolean in SGPR (as -1/0)
S_CMP_LG_U32 %370, killed %5402, implicit-def $scc
%14138:sreg_32_xm0_xexec = S_CSELECT_B32 -1, 0, implicit $scc

; 2. V_CNDMASK forces result to VGPR (the bug!)
%5415:vgpr_32 = V_CNDMASK_B32_e64 0, 0, 0, 1, killed %5403, implicit $exec

; 3. Build TDM descriptor in VGPR (contaminated by step 2)
%14126:vreg_128_align2 = INSERT_SUBREG ... %5415 ...

; 4. Must use V_READFIRSTLANE to get descriptor back to SGPR for TDM
%14129:sgpr_32 = V_READFIRSTLANE_B32 %14126.sub0
%14130:sgpr_32 = V_READFIRSTLANE_B32 %14126.sub1
%14131:sgpr_32 = V_READFIRSTLANE_B32 %14126.sub2
%14132:sgpr_32 = V_READFIRSTLANE_B32 %14126.sub3
```

**Fixed MIR (tdm_post_misched.mir):**
```mir
; 1. Compare and get boolean in SGPR (as -1/0)
S_CMP_LG_U32 renamable $sgpr93, -7808, implicit-def $scc
renamable $sgpr14 = S_CSELECT_B32 -1, 0, implicit $scc

; 2. S_AND_B32 keeps result in SGPR (our fix!)
renamable $sgpr52 = S_AND_B32 killed renamable $sgpr14, 1, implicit-def dead $scc

; 3. Descriptor built directly in SGPR registers - NO V_READFIRSTLANE needed!
$sgpr56_sgpr57 = S_MOV_B64 $sgpr52_sgpr53 ...

; 4. Used directly by TDM instruction
TENSOR_LOAD_TO_LDS_D2 $sgpr52_sgpr53_sgpr54_sgpr55, ...
```

### Instruction Count Comparison

| Instruction | Original | Fixed | Change |
|-------------|----------|-------|--------|
| `V_READFIRSTLANE` | 8 | **0** | **-8 eliminated** |
| `V_CNDMASK` | 1 | **0** | **-1 eliminated** |
| `S_AND_B32` | 11 | 12 | +1 added |
| **Net VALU removed** | - | - | **9** |
| **Net SALU added** | - | - | **1** |

### Static Performance Simulation

The assembly files include annotations from a static performance simulator. Here's how to read them:

#### Block Header Format
```asm
;=== Block: 21283 cycles ===
;  VALU:3870(VOPD:33+PK:700) SALU:137 TRANS:648 WMMA:96 DS:144 VMEM:64 SMEM:1 TDM:2 Ctrl:3854
;  Stall: 15131 cycles (71%)
;    FU:755 | WMMACoExec:391(...) | DelayAlu:12245 | MemFIFO:1372 | Wait:43 | RegBank:192 | ...
;      FU: XDL:164 VALU:455 TRANS:136
```

| Line | Field | Description |
|------|-------|-------------|
| 1 | `Block: N cycles` | Total estimated cycles for this basic block |
| 2 | `VALU:N` | Vector ALU cycles (includes v_cndmask, v_readfirstlane) |
| 2 | `SALU:N` | Scalar ALU cycles (s_add, s_and, s_cmp, etc.) |
| 2 | `TRANS:N` | Transcendental unit (v_exp, v_log, v_sqrt) |
| 2 | `WMMA:N` | Wave matrix multiply-accumulate |
| 2 | `DS:N` | Data share / LDS operations |
| 2 | `VMEM:N` | Vector memory (global loads/stores) |
| 2 | `SMEM:N` | Scalar memory (constant loads) |
| 2 | `TDM:N` | Tensor DMA operations |
| 2 | `Ctrl:N` | Control flow (branches, barriers) |
| 3 | `Stall: N cycles (P%)` | Total stall cycles and percentage |
| 4 | `FU:N` | Functional unit busy stalls |
| 4 | `DelayAlu:N` | Delay ALU hazard stalls |
| 4 | `MemFIFO:N` | Memory queue full stalls |
| 4 | `RegBank:N` | Register bank conflict stalls |
| 5 | `FU: XDL:N VALU:N ...` | FU stall breakdown by unit |

#### Main Loop Block Comparison

| Metric | Original | Fixed | Change | Interpretation |
|--------|----------|-------|--------|----------------|
| **Total cycles** | 21283 | 21276 | **-7** | Slightly faster |
| **VALU** | 3870 | 3869 | **-1** | 1 fewer VALU cycle |
| **SALU** | 137 | 137 | 0 | Unchanged |
| **Stall cycles** | 15131 | 15126 | **-5** | Fewer stalls |
| **FU stalls** | 755 | 754 | **-1** | Less FU contention |
| **RegBank** | 192 | 183 | **-9** | Fewer bank conflicts |
| **FU:VALU** | 455 | 453 | **-2** | Less VALU FU pressure |

### Why the Cycle Improvement is Modest

1. **The block is 71% stall-bound** - mostly from `DelayAlu` (12245 cycles) and `MemFIFO` (1372 cycles). The i1 extension fix helps, but other bottlenecks dominate.

2. **`v_readfirstlane` is fast** (1 cycle each) - eliminating 8 of them saves only ~8 cycles in a 21283-cycle block.

3. **Key improvement is RegBank conflicts** - dropped from 192 to 183 (-9). The VGPR roundtrip was causing register bank conflicts that are now eliminated.

### Conclusion

Despite the modest cycle improvement (7 cycles per iteration), the fix provides significant code quality improvements:

- **Eliminates unnecessary VGPR usage** for uniform values
- **Removes 9 VALU instructions** (8 `v_readfirstlane` + 1 `v_cndmask`)
- **Reduces register bank conflicts** by keeping values in SGPRs
- **Enables cleaner TDM descriptor construction** without cross-lane operations

For TDM-heavy kernels, this fix prevents the "contamination" of descriptor values by the i1 extension bug, allowing the entire descriptor to stay in SGPRs as intended.

### Final Assembly Analysis

After all compiler optimizations, the final assembly shows the fix is working correctly. Here's what the code looks like in the main loop:

**Original (tdm_orig.s):**
```asm
; Line 1999-2000: Compare and select
s_cmp_lg_u32 s93, 0xffffe180
s_cselect_b32 s6, -1, 0

; Line 2833: Force to VGPR (the bug!)
v_cndmask_b32_e64 v122, 0, 1, s6

; Lines 3550, 3557: Read back to SGPR for descriptor
v_readfirstlane_b32 s56, v122
v_readfirstlane_b32 s52, v122
```

**Fixed (tdm_fixed_isel.s):**
```asm
; Line 2028-2029: Compare and select
s_cmp_lg_u32 s93, 0xffffe180
s_cselect_b32 s14, -1, 0

; Line 2329: Stay in SGPR (our fix!)
s_and_b32 s52, s14, 1

; Lines 2339-2340: Direct SGPR descriptor copy
s_mov_b64 s[58:59], s[54:55]
s_mov_b64 s[56:57], s[52:53]
```

### Is the Boolean Computation Loop-Invariant?

We checked whether the comparison could be hoisted out of the loop:

```asm
; s93 is the loop induction variable - updated each iteration
s_addk_co_i32 s93, 0xff80          ; Line 2673: s93 changes every iteration

; The comparison depends on s93
s_cmp_lg_u32 s93, 0xffffe180       ; Line 2028: loop-varying comparison
```

**Result:** The boolean computation is **loop-varying** (depends on the loop counter), so it cannot be hoisted. The comparison must be re-evaluated each iteration.

### Remaining Optimization Opportunities

| Optimization | Potential Savings | Feasibility |
|--------------|-------------------|-------------|
| **Fold `s_cselect -1,0 + s_and 1` → `s_cselect 1,0`** | 1 SALU/iteration | Requires DAG combine or glue-based ISel |
| **Reduce `s_mov_b64` descriptor copies** | 1-2 SALU/iteration | Register allocation issue, not ISel |
| **Hoist loop-invariant parts** | N/A | Already optimized - the comparison is loop-varying |

### Why Further Optimization Has Diminishing Returns

The main loop statistics show the bottleneck is elsewhere:

```
;=== Block: 21276 cycles ===
;  Stall: 15126 cycles (71%)
;    DelayAlu:12250 | MemFIFO:1372 | RegBank:183 | ...
```

- **71% of cycles are stalls** - the loop is heavily stall-bound
- **DelayAlu dominates** (12250 cycles) - instruction dependency stalls from WMMA and transcendental operations
- **TDM overhead is now negligible** - descriptor construction uses a few SALU instructions

Saving 1-2 SALU cycles per iteration (from folding `s_cselect + s_and`) in a 21276-cycle loop would yield <0.01% improvement. The fix has already addressed the main issue (eliminating the VGPR roundtrip).

### Summary

The i1 extension fix is **working correctly in the final assembly**:

1. **No VGPR contamination** - descriptors stay entirely in SGPRs
2. **No `v_readfirstlane`** - eliminated all 8 instances
3. **Minimal overhead** - just `s_cselect + s_and` (2 SALU instructions)
4. **Loop-varying computation handled correctly** - can't be hoisted, but that's expected

The remaining optimization (`s_cselect 1,0` directly) would require either:
- A DAG combine to fold `(and (select -1, 0), 1)` → `(select 1, 0)`
- Or the glue-based custom ISel approach from Approach 3

But the benefit is minimal given the loop is dominated by WMMA/transcendental stalls.

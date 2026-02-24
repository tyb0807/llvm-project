//===-- AMDGPUInsertCondBarriers.cpp - Insert conditional barriers ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass automatically inserts conditional barrier sequences before and
// after loops using wavefront-level scalar comparisons.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/TargetParser/TargetParser.h"

using namespace llvm;
using namespace llvm::AMDGPU;

#define DEBUG_TYPE "amdgpu-insert-cond-barriers"

static cl::opt<unsigned> CondBarrierThreshold(
    "amdgpu-cond-barrier-threshold",
    cl::desc("Thread ID threshold for conditional barriers"),
    cl::init(256), cl::Hidden);

class AMDGPUInsertCondBarriers : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUInsertCondBarriers() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU Insert Conditional Barriers";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineLoopInfoWrapperPass>();
    // We modify the CFG
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  struct CondBarrierConfig {
    unsigned ThreadIdReg;
    unsigned CompareValue;
  };

  // Validation functions - check if barriers can be inserted
  bool canInsertCondBarrierPrologue(MachineLoop *ML);
  bool canInsertCondBarrierEpilogue(MachineLoop *ML);

  // Insertion functions - actually insert the barriers
  bool insertCondBarrierPrologue(MachineLoop *ML, const CondBarrierConfig &Config);
  bool insertCondBarrierEpilogue(MachineLoop *ML, const CondBarrierConfig &Config);

  // Enhanced barrier duplication in loop body
  bool duplicateBarriersInLoopBody(MachineLoop *ML);

  // Helper to find workItemIDX register using SIMachineFunctionInfo
  unsigned findWorkItemIDXRegister(MachineFunction &MF);

  // Helper to build conditional barrier sequence in BarrierBB
  // Creates DoBarrierBB with S_WAITCNT + S_BARRIER, and conditional branch logic
  // CmpOpcode: S_CMP_GE_U32 for prologue, S_CMP_LT_U32 for epilogue
  // SkipBB: target when condition is false (skip barrier)
  // FallThroughBB: target after executing barrier
  MachineBasicBlock *buildCondBarrierSequence(
      MachineBasicBlock *BarrierBB, MachineBasicBlock *SkipBB,
      MachineBasicBlock *FallThroughBB, unsigned CmpOpcode,
      const CondBarrierConfig &Config, const DebugLoc &DL);
};

char AMDGPUInsertCondBarriers::ID = 0;

char &llvm::AMDGPUInsertCondBarriersID = AMDGPUInsertCondBarriers::ID;

INITIALIZE_PASS_BEGIN(AMDGPUInsertCondBarriers, DEBUG_TYPE,
                      "Insert conditional barrier sequences", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_END(AMDGPUInsertCondBarriers, DEBUG_TYPE,
                    "Insert conditional barrier sequences", false, false)

bool AMDGPUInsertCondBarriers::runOnMachineFunction(MachineFunction &MF) {
  bool Changed = false;

  // Get loop info
  MachineLoopInfo &MLI = getAnalysis<MachineLoopInfoWrapperPass>().getLI();

  // Try to find workItemIDX register using metadata
  unsigned ThreadIdReg = findWorkItemIDXRegister(MF);
  if (!ThreadIdReg) {
    LLVM_DEBUG(dbgs() << "Could not find workItemIDX register, skipping\n");
    return false;
  }

  LLVM_DEBUG(dbgs() << "Found workItemIDX register: %" << ThreadIdReg << "\n");

  // Configuration for conditional barriers
  CondBarrierConfig Config;
  Config.ThreadIdReg = ThreadIdReg;
  Config.CompareValue = CondBarrierThreshold;

  // Process each top-level loop
  for (MachineLoop *ML : MLI) {
    LLVM_DEBUG(dbgs() << "Processing loop with header: "
                      << ML->getHeader()->getFullName() << "\n");

    // VALIDATION PHASE - check if we can insert both barriers
    if (!canInsertCondBarrierPrologue(ML)) {
      LLVM_DEBUG(dbgs() << "Cannot insert prologue, skipping loop\n");
      continue;
    }

    if (!canInsertCondBarrierEpilogue(ML)) {
      LLVM_DEBUG(dbgs() << "Cannot insert epilogue, skipping loop\n");
      continue;
    }

    LLVM_DEBUG(dbgs() << "Validation passed, inserting both barriers\n");

    // INSERTION PHASE - both barriers can be inserted
    bool PrologueInserted = insertCondBarrierPrologue(ML, Config);
    bool EpilogueInserted = insertCondBarrierEpilogue(ML, Config);
    bool BarriersDuplicated = duplicateBarriersInLoopBody(ML);

    // Both should succeed since we validated
    if (PrologueInserted && EpilogueInserted) {
      LLVM_DEBUG(dbgs() << "Successfully inserted both conditional barriers\n");
      if (BarriersDuplicated) {
        LLVM_DEBUG(dbgs() << "Successfully duplicated barriers in loop body\n");
      }
      Changed = true;
    } else {
      LLVM_DEBUG(dbgs() << "ERROR: Insertion failed after validation!\n");
      // This should not happen if validation was correct
    }
  }

  return Changed;
}

bool AMDGPUInsertCondBarriers::canInsertCondBarrierPrologue(MachineLoop *ML) {
  // We can always insert a prologue conditional barrier
  // Either in the existing preheader, or by creating a new barrier block
  return true;
}

bool AMDGPUInsertCondBarriers::canInsertCondBarrierEpilogue(MachineLoop *ML) {
  // Epilogue needs exit blocks to insert after the loop
  SmallVector<MachineBasicBlock *, 4> ExitBlocks;
  ML->getExitBlocks(ExitBlocks);

  if (ExitBlocks.empty()) {
    LLVM_DEBUG(dbgs() << "Loop has no exit blocks\n");
    return false;
  }

  return true;
}

unsigned AMDGPUInsertCondBarriers::findWorkItemIDXRegister(MachineFunction &MF) {
  const SIMachineFunctionInfo *Info = MF.getInfo<SIMachineFunctionInfo>();
  if (!Info) {
    LLVM_DEBUG(dbgs() << "No SIMachineFunctionInfo available\n");
    return 0;
  }

  // Get the ArgDescriptor for WorkItemIDX
  const ArgDescriptor &WorkItemIDXDesc = Info->getArgInfo().WorkItemIDX;

  // Check if WorkItemIDX is set (not default/empty)
  if (!WorkItemIDXDesc.isSet()) {
    LLVM_DEBUG(dbgs() << "WorkItemIDX not set in ArgInfo\n");
    return 0;
  }

  // Check if WorkItemIDX is in a register (not on stack)
  if (!WorkItemIDXDesc.isRegister()) {
    LLVM_DEBUG(dbgs() << "WorkItemIDX is not in a register\n");
    return 0;
  }

  Register PhysReg = WorkItemIDXDesc.getRegister();
  LLVM_DEBUG(dbgs() << "WorkItemIDX physical register: " << PhysReg << "\n");

  // Find the virtual register that receives the workItemIDX value
  // Look for COPY instruction: %virt_reg = COPY $vgpr0 (or whatever PhysReg is)
  for (auto &MBB : MF) {
    for (auto &MI : MBB) {
      if (MI.getOpcode() == AMDGPU::COPY &&
          MI.getNumOperands() >= 2 &&
          MI.getOperand(1).isReg() &&
          MI.getOperand(1).getReg() == PhysReg) {

        if (MI.getOperand(0).isReg()) {
          Register VirtReg = MI.getOperand(0).getReg();
          LLVM_DEBUG(dbgs() << "Found COPY from WorkItemIDX: %"
                           << VirtReg << " = COPY " << PhysReg << "\n");
          return VirtReg;
        }
      }
    }
  }

  LLVM_DEBUG(dbgs() << "Could not find COPY instruction for WorkItemIDX\n");
  return 0;
}

MachineBasicBlock *AMDGPUInsertCondBarriers::buildCondBarrierSequence(
    MachineBasicBlock *BarrierBB, MachineBasicBlock *SkipBB,
    MachineBasicBlock *FallThroughBB, unsigned CmpOpcode,
    const CondBarrierConfig &Config, const DebugLoc &DL) {

  MachineFunction *MF = BarrierBB->getParent();
  const GCNSubtarget &ST = MF->getSubtarget<GCNSubtarget>();
  const SIInstrInfo *TII = ST.getInstrInfo();
  MachineRegisterInfo &MRI = MF->getRegInfo();

  // Create separate basic block for executing the barrier
  MachineBasicBlock *DoBarrierBB = MF->CreateMachineBasicBlock();
  MF->insert(std::next(BarrierBB->getIterator()), DoBarrierBB);

  unsigned CompareValReg = MRI.createVirtualRegister(&AMDGPU::SReg_32RegClass);

  // Create SGPR to hold workItemIDX value converted from VGPR
  // V_READFIRSTLANE_B32 requires its destination to be SReg_32_XM0
  const MCInstrDesc &ReadFirstLaneDesc = TII->get(AMDGPU::V_READFIRSTLANE_B32);
  const TargetRegisterClass *ThreadIdRC = TII->getRegClass(ReadFirstLaneDesc, 0);
  unsigned ThreadIdSgpr = MRI.createVirtualRegister(ThreadIdRC);

  // Copy VGPR workItemIDX to SGPR using V_READFIRSTLANE_B32
  BuildMI(*BarrierBB, BarrierBB->end(), DL, TII->get(AMDGPU::V_READFIRSTLANE_B32), ThreadIdSgpr)
      .addReg(Config.ThreadIdReg);

  // %compare_val = S_MOV_B32 <threshold>
  BuildMI(*BarrierBB, BarrierBB->end(), DL, TII->get(AMDGPU::S_MOV_B32), CompareValReg)
      .addImm(Config.CompareValue);

  // S_CMP_*_U32 %thread_id_sgpr, %compare_val  (sets SCC)
  BuildMI(*BarrierBB, BarrierBB->end(), DL, TII->get(CmpOpcode))
      .addReg(ThreadIdSgpr)
      .addReg(CompareValReg);

  // S_CBRANCH_SCC0 %skip_bb  (branch if condition false - skip barrier)
  BuildMI(*BarrierBB, BarrierBB->end(), DL, TII->get(AMDGPU::S_CBRANCH_SCC0))
      .addMBB(SkipBB);

  // S_BRANCH %DoBarrierBB  (branch to barrier block if condition true)
  BuildMI(*BarrierBB, BarrierBB->end(), DL, TII->get(AMDGPU::S_BRANCH))
      .addMBB(DoBarrierBB);

  // DoBarrierBB: Execute synchronization barrier.
  // For MI450 (gfx1250): use S_WAIT_DSCNT 0 + split barrier signal/wait pattern.
  // For MI350 (gfx950): use vmcnt(4) to wait until first buffer is filled,
  //   with lgkmcnt(max) - don't wait for LDS/GDS.
  // For MI300 (gfx942) and others: use max vmcnt (don't wait for VM
  //   operations), with lgkmcnt(0) - wait for all LDS/GDS.
  if (ST.hasGFX1250Insts()) {
    BuildMI(*DoBarrierBB, DoBarrierBB->end(), DL,
            TII->get(AMDGPU::S_WAIT_DSCNT))
        .addImm(0);
    BuildMI(*DoBarrierBB, DoBarrierBB->end(), DL,
            TII->get(AMDGPU::S_BARRIER_SIGNAL_IMM))
        .addImm(-1);
    BuildMI(*DoBarrierBB, DoBarrierBB->end(), DL,
            TII->get(AMDGPU::S_BARRIER_WAIT))
        .addImm(-1);
  } else {
    AMDGPU::IsaVersion IV = AMDGPU::getIsaVersion(ST.getCPU());
    unsigned Vmcnt = ST.hasGFX950Insts() ? 4 : AMDGPU::getVmcntBitMask(IV);
    unsigned ExpcntMax = AMDGPU::getExpcntBitMask(IV);
    unsigned Lgkmcnt =
        ST.hasGFX950Insts() ? AMDGPU::getLgkmcntBitMask(IV) : 0;
    unsigned WaitcntImm =
        AMDGPU::encodeWaitcnt(IV, Vmcnt, ExpcntMax, Lgkmcnt);
    // For MI350 (gfx950), set unused bits [13:12] and [7] to 1 for
    // consistency.
    if (ST.hasGFX950Insts())
      WaitcntImm |= 0x3080;
    BuildMI(*DoBarrierBB, DoBarrierBB->end(), DL,
            TII->get(AMDGPU::S_WAITCNT))
        .addImm(WaitcntImm);
    BuildMI(*DoBarrierBB, DoBarrierBB->end(), DL,
            TII->get(AMDGPU::S_BARRIER));
  }
  BuildMI(*DoBarrierBB, DoBarrierBB->end(), DL, TII->get(AMDGPU::S_BRANCH))
      .addMBB(FallThroughBB);

  // Set up CFG edges (SkipBB might already be a successor if reusing preheader)
  if (!BarrierBB->isSuccessor(SkipBB))
    BarrierBB->addSuccessor(SkipBB);
  BarrierBB->addSuccessor(DoBarrierBB);
  DoBarrierBB->addSuccessor(FallThroughBB);

  return DoBarrierBB;
}

bool AMDGPUInsertCondBarriers::insertCondBarrierPrologue(
    MachineLoop *ML, const CondBarrierConfig &Config) {

  MachineFunction *MF = ML->getHeader()->getParent();
  MachineBasicBlock *Header = ML->getHeader();
  MachineBasicBlock *Preheader = ML->getLoopPreheader();
  DebugLoc DL = Header->begin() != Header->end() ? Header->begin()->getDebugLoc() : DebugLoc();

  MachineBasicBlock *BarrierBB = nullptr;

  if (Preheader) {
    // Case 1: Use existing preheader
    BarrierBB = Preheader;

    // Remove the original terminator instruction from preheader
    // The existing CFG edge to Header is preserved and will be used by our conditional branch
    MachineBasicBlock::iterator TerminatorPos = Preheader->getFirstTerminator();
    if (TerminatorPos != Preheader->end()) {
      Preheader->erase(TerminatorPos);
    }
  } else {
    // Case 2: Create new barrier block for loops without preheader
    BarrierBB = MF->CreateMachineBasicBlock();

    // Insert barrier block before header in function
    MachineFunction::iterator InsertIt = Header->getIterator();
    MF->insert(InsertIt, BarrierBB);

    // Redirect all predecessors of Header (that are outside the loop) to BarrierBB
    SmallVector<MachineBasicBlock *, 4> HeaderPreds(Header->pred_begin(), Header->pred_end());
    for (MachineBasicBlock *PredBB : HeaderPreds) {
      if (!ML->contains(PredBB)) {
        PredBB->replaceSuccessor(Header, BarrierBB);
      }
    }
  }

  // Build conditional barrier sequence: S_CMP_GE_U32 for prologue
  // SkipBB = Header (skip barrier and go to loop), FallThroughBB = Header (after barrier)
  buildCondBarrierSequence(BarrierBB, Header, Header, AMDGPU::S_CMP_GE_U32, Config, DL);

  LLVM_DEBUG(dbgs() << "Inserted conditional barrier prologue\n");
  return true;
}

bool AMDGPUInsertCondBarriers::insertCondBarrierEpilogue(
    MachineLoop *ML, const CondBarrierConfig &Config) {

  MachineFunction *MF = ML->getHeader()->getParent();

  // Find loop exit blocks
  SmallVector<MachineBasicBlock *, 4> ExitBlocks;
  ML->getExitBlocks(ExitBlocks);

  // We already validated this is non-empty
  assert(!ExitBlocks.empty() && "Exit blocks should exist after validation");

  bool Changed = false;

  for (MachineBasicBlock *ExitBB : ExitBlocks) {
    // Collect predecessors that are in the loop BEFORE we modify the CFG
    SmallVector<MachineBasicBlock *, 4> ExitPreds(ExitBB->pred_begin(), ExitBB->pred_end());

    // Create new barrier block to insert between loop and exit
    MachineBasicBlock *BarrierBB = MF->CreateMachineBasicBlock();

    // Insert barrier block before exit block in function
    MachineFunction::iterator InsertIt = ExitBB->getIterator();
    MF->insert(InsertIt, BarrierBB);

    DebugLoc DL = ExitBB->begin() != ExitBB->end() ? ExitBB->begin()->getDebugLoc() : DebugLoc();

    // Build conditional barrier sequence: S_CMP_LT_U32 for epilogue
    // SkipBB = ExitBB (skip barrier and exit), FallThroughBB = ExitBB (after barrier)
    buildCondBarrierSequence(BarrierBB, ExitBB, ExitBB, AMDGPU::S_CMP_LT_U32, Config, DL);

    // Redirect all predecessors of ExitBB (that are in the loop) to BarrierBB
    // Use the predecessors we collected before modifying the CFG
    for (MachineBasicBlock *PredBB : ExitPreds) {
      if (ML->contains(PredBB)) {
        // Update CFG edge
        PredBB->replaceSuccessor(ExitBB, BarrierBB);

        // Update actual branch instruction operands
        for (MachineInstr &MI : PredBB->terminators()) {
          for (MachineOperand &MO : MI.operands()) {
            if (MO.isMBB() && MO.getMBB() == ExitBB) {
              MO.setMBB(BarrierBB);
            }
          }
        }
      }
    }

    LLVM_DEBUG(dbgs() << "Inserted conditional barrier epilogue\n");
    Changed = true;
  }

  return Changed;
}

bool AMDGPUInsertCondBarriers::duplicateBarriersInLoopBody(MachineLoop *ML) {
  MachineFunction *MF = ML->getHeader()->getParent();
  const GCNSubtarget &ST = MF->getSubtarget<GCNSubtarget>();
  const SIInstrInfo *TII = ST.getInstrInfo();

  bool Changed = false;

  if (ST.hasGFX1250Insts()) {
    // MI450 (gfx1250): Find S_WAIT_DSCNT 0 + S_BARRIER_SIGNAL_IMM -1 +
    // S_BARRIER_WAIT -1 pattern and duplicate all 3 instructions.
    SmallVector<MachineInstr *, 4> WaitDscntInstrs;
    for (MachineBasicBlock *MBB : ML->blocks())
      for (MachineInstr &MI : *MBB)
        if (MI.getOpcode() == AMDGPU::S_WAIT_DSCNT &&
            MI.getOperand(0).getImm() == 0)
          WaitDscntInstrs.push_back(&MI);

    for (MachineInstr *WaitDscntMI : WaitDscntInstrs) {
      MachineBasicBlock *MBB = WaitDscntMI->getParent();
      DebugLoc DL = WaitDscntMI->getDebugLoc();
      auto SignalIt = std::next(WaitDscntMI->getIterator());

      // Expect S_BARRIER_SIGNAL_IMM -1 immediately after S_WAIT_DSCNT 0.
      if (SignalIt == MBB->end() ||
          SignalIt->getOpcode() != AMDGPU::S_BARRIER_SIGNAL_IMM ||
          SignalIt->getOperand(0).getImm() != -1)
        continue;

      auto WaitIt = std::next(SignalIt);

      // Expect S_BARRIER_WAIT -1 immediately after S_BARRIER_SIGNAL_IMM -1.
      if (WaitIt == MBB->end() ||
          WaitIt->getOpcode() != AMDGPU::S_BARRIER_WAIT ||
          WaitIt->getOperand(0).getImm() != -1)
        continue;

      MachineBasicBlock::iterator InsertPos = std::next(WaitIt);

      // Emit duplicate: S_WAIT_DSCNT 0 + S_BARRIER_SIGNAL_IMM -1 +
      // S_BARRIER_WAIT -1.
      BuildMI(*MBB, InsertPos, DL, TII->get(AMDGPU::S_WAIT_DSCNT))
          .addImm(0);
      BuildMI(*MBB, InsertPos, DL,
              TII->get(AMDGPU::S_BARRIER_SIGNAL_IMM))
          .addImm(-1);
      BuildMI(*MBB, InsertPos, DL, TII->get(AMDGPU::S_BARRIER_WAIT))
          .addImm(-1);

      Changed = true;
      LLVM_DEBUG(dbgs() << "Duplicated S_WAIT_DSCNT + barrier pattern "
                        << "in loop body for gfx1250\n");
    }
  } else {
    // MI300/MI350 path: collect and duplicate S_BARRIER instructions.

    // For MI350 (gfx950), compute waitcnt values.
    // vmcnt(4) is what we look for, vmcnt(0) is what we change to.
    AMDGPU::IsaVersion IV = AMDGPU::getIsaVersion(ST.getCPU());
    unsigned Vmcnt4WaitcntImm = 0;
    unsigned Vmcnt0WaitcntImm = 0;
    if (ST.hasGFX950Insts()) {
      unsigned ExpcntMax = AMDGPU::getExpcntBitMask(IV);
      unsigned LgkmcntMax = AMDGPU::getLgkmcntBitMask(IV);
      Vmcnt4WaitcntImm =
          AMDGPU::encodeWaitcnt(IV, 4, ExpcntMax, LgkmcntMax);
      Vmcnt4WaitcntImm |= 0x3080;
      Vmcnt0WaitcntImm =
          AMDGPU::encodeWaitcnt(IV, 0, ExpcntMax, LgkmcntMax);
      Vmcnt0WaitcntImm |= 0x3080;
    }

    SmallVector<MachineInstr *, 4> Barriers;

    for (MachineBasicBlock *MBB : ML->blocks())
      for (MachineInstr &MI : *MBB)
        if (MI.getOpcode() == AMDGPU::S_BARRIER)
          Barriers.push_back(&MI);

    for (MachineInstr *BarrierMI : Barriers) {
      MachineBasicBlock *MBB = BarrierMI->getParent();
      DebugLoc DL = BarrierMI->getDebugLoc();
      MachineBasicBlock::iterator BarrierIt = BarrierMI->getIterator();

      // Insert duplicate after the current S_BARRIER.
      MachineBasicBlock::iterator InsertPos = std::next(BarrierIt);

      if (ST.hasGFX950Insts()) {
        // MI350: Find S_WAITCNT with vmcnt(4) before the barrier and change
        // to vmcnt(0). Search backwards from the barrier.
        MachineInstr *Vmcnt4WaitcntMI = nullptr;
        for (auto It = BarrierIt; It != MBB->begin();) {
          --It;
          if (It->getOpcode() == AMDGPU::S_WAITCNT) {
            int64_t WaitcntVal = It->getOperand(0).getImm();
            // Check if this is vmcnt(4) - compare lower 16 bits.
            if ((WaitcntVal & 0xFFFF) == (Vmcnt4WaitcntImm & 0xFFFF)) {
              Vmcnt4WaitcntMI = &(*It);
              break;
            }
          }
          // Stop at non-waitcnt instructions.
          if (It->getOpcode() != AMDGPU::S_WAITCNT)
            break;
        }

        if (Vmcnt4WaitcntMI) {
          Vmcnt4WaitcntMI->getOperand(0).setImm(Vmcnt0WaitcntImm);
          BuildMI(*MBB, InsertPos, DL, TII->get(AMDGPU::S_BARRIER));
          Changed = true;
          LLVM_DEBUG(
              dbgs()
              << "Updated S_WAITCNT from vmcnt(4) to vmcnt(0) and "
              << "duplicated S_BARRIER in loop body\n");
        }
      } else {
        // Other targets: duplicate S_WAITCNT + S_BARRIER if S_WAITCNT
        // exists.
        if (BarrierIt != MBB->begin()) {
          MachineBasicBlock::iterator PrevIt = std::prev(BarrierIt);
          if (PrevIt->getOpcode() == AMDGPU::S_WAITCNT) {
            unsigned WaitcntImm = PrevIt->getOperand(0).getImm();
            BuildMI(*MBB, InsertPos, DL, TII->get(AMDGPU::S_WAITCNT))
                .addImm(WaitcntImm);
            BuildMI(*MBB, InsertPos, DL, TII->get(AMDGPU::S_BARRIER));
            Changed = true;
            LLVM_DEBUG(dbgs() << "Duplicated S_WAITCNT + S_BARRIER "
                              << "pattern in loop body\n");
          }
        }
      }
    }
  }

  return Changed;
}

FunctionPass *llvm::createAMDGPUInsertCondBarriersPass() {
  return new AMDGPUInsertCondBarriers();
}
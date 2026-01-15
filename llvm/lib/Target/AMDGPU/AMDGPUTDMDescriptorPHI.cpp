//===- AMDGPUTDMDescriptorPHI.cpp - TDM Descriptor PHI Optimization -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass optimizes TDM (Tensor Data Movement) descriptor construction
/// in loops by using PHI nodes to carry descriptors across iterations and
/// updating only the fields that change with affine strides.
///
/// Instead of rebuilding descriptors from scratch each iteration:
///   loop:
///     %field2 = <complex computation involving IV>
///     %desc = insertelement <4 x i32> ..., i32 %field2, i64 2
///     call @llvm.amdgcn.tensor.load.to.lds.d2(<4 x i32> %desc, ...)
///
/// We transform to:
///   preheader:
///     %init.desc = <initial descriptor>
///   loop:
///     %desc.phi = phi <4 x i32> [ %init.desc, %preheader ], [ %next.desc, %loop ]
///     call @llvm.amdgcn.tensor.load.to.lds.d2(<4 x i32> %desc.phi, ...)
///     %cur = extractelement <4 x i32> %desc.phi, i64 2
///     %next = add i32 %cur, <stride>
///     %next.desc = insertelement <4 x i32> %desc.phi, i32 %next, i64 2
///
/// This lowers to INSERT_SUBREG operations that reuse the register tuple.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "amdgpu-tdm-descriptor-phi"

STATISTIC(NumDescriptorsOptimized, "Number of TDM descriptors optimized");
STATISTIC(NumFieldsOptimized, "Number of descriptor fields converted to PHI+stride");

static cl::opt<bool> EnableTDMDescriptorPHI(
    "amdgpu-tdm-descriptor-phi",
    cl::desc("Enable TDM descriptor PHI optimization"),
    cl::init(true), cl::Hidden);

namespace {

/// Classification of a descriptor field for optimization purposes.
enum class FieldKind {
  Constant,      // Field is a compile-time constant
  LoopInvariant, // Field doesn't change across loop iterations
  Affine,        // Field has affine recurrence: {start, +, stride}
  NonAffine      // Field has non-affine computation (not optimizable)
};

/// Information about a single field in a TDM descriptor.
struct FieldInfo {
  FieldKind Kind = FieldKind::NonAffine;
  Value *CurrentValue = nullptr;     // The value used for this field
  const SCEV *StartSCEV = nullptr;   // For affine: the start value
  const SCEV *StrideSCEV = nullptr;  // For affine: the stride
  int64_t StrideValue = 0;           // Constant stride value (if computable)
  bool HasConstantStride = false;    // Whether stride is a constant
};

/// Information about a TDM intrinsic call and its descriptor.
struct TDMCallInfo {
  IntrinsicInst *Call = nullptr;
  Value *AddrDescriptor = nullptr;   // The <4 x i32> address descriptor
  FieldInfo Fields[4];               // Info for each field
  bool HasOptimizableFields = false; // Whether any field can be optimized
};

class AMDGPUTDMDescriptorPHI : public FunctionPass {
public:
  static char ID;

  AMDGPUTDMDescriptorPHI() : FunctionPass(ID) {
    initializeAMDGPUTDMDescriptorPHIPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override {
    return "AMDGPU TDM Descriptor PHI Optimization";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addPreserved<LoopInfoWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
  }

  bool runOnFunction(Function &F) override;

private:
  LoopInfo *LI = nullptr;
  ScalarEvolution *SE = nullptr;
  DominatorTree *DT = nullptr;

  /// Find all TDM intrinsic calls in the given loop.
  void findTDMCalls(Loop *L, SmallVectorImpl<TDMCallInfo> &Calls);

  /// Trace back the value of a specific field in a descriptor.
  Value *traceFieldValue(Value *Desc, unsigned FieldIdx);

  /// Analyze a descriptor field using SCEV.
  void analyzeField(FieldInfo &FI, Value *FieldVal, Loop *L);

  /// Check if a TDM call has any fields that can be optimized.
  bool hasOptimizableFields(TDMCallInfo &Info);

  /// Transform a single TDM call to use PHI-based field updates.
  bool transformCall(TDMCallInfo &Info, Loop *L);

  /// Create the initial descriptor value in the preheader.
  Value *createPreheaderInit(TDMCallInfo &Info, Loop *L, IRBuilder<> &Builder);

  /// Create the stride update for affine fields in the loop.
  Value *createStrideUpdate(Value *DescPHI, TDMCallInfo &Info,
                            IRBuilder<> &Builder);
};

} // end anonymous namespace

char AMDGPUTDMDescriptorPHI::ID = 0;
char &llvm::AMDGPUTDMDescriptorPHIID = AMDGPUTDMDescriptorPHI::ID;

INITIALIZE_PASS_BEGIN(AMDGPUTDMDescriptorPHI, DEBUG_TYPE,
                      "AMDGPU TDM Descriptor PHI Optimization", false, false)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(AMDGPUTDMDescriptorPHI, DEBUG_TYPE,
                    "AMDGPU TDM Descriptor PHI Optimization", false, false)

/// Trace back through insertelement chain to find the value at a specific index.
Value *AMDGPUTDMDescriptorPHI::traceFieldValue(Value *Desc, unsigned FieldIdx) {
  // Follow insertelement chain backwards
  while (auto *IE = dyn_cast<InsertElementInst>(Desc)) {
    auto *IdxCI = dyn_cast<ConstantInt>(IE->getOperand(2));
    if (!IdxCI)
      return nullptr;

    if (IdxCI->getZExtValue() == FieldIdx) {
      // Found the insertelement that sets this field
      return IE->getOperand(1);
    }

    // Continue up the chain
    Desc = IE->getOperand(0);
  }

  // Check if it's from a constant vector
  if (auto *CV = dyn_cast<Constant>(Desc)) {
    return CV->getAggregateElement(FieldIdx);
  }

  // Could be a shufflevector or other complex construction
  // For now, we can't trace these
  return nullptr;
}

void AMDGPUTDMDescriptorPHI::analyzeField(FieldInfo &FI, Value *FieldVal,
                                          Loop *L) {
  FI.CurrentValue = FieldVal;

  if (!FieldVal) {
    FI.Kind = FieldKind::NonAffine;
    return;
  }

  // Check for constant
  if (isa<Constant>(FieldVal)) {
    FI.Kind = FieldKind::Constant;
    return;
  }

  // Use SCEV to analyze the value
  const SCEV *S = SE->getSCEV(FieldVal);

  // Check if loop-invariant
  if (SE->isLoopInvariant(S, L)) {
    FI.Kind = FieldKind::LoopInvariant;
    return;
  }

  // Check for affine recurrence
  if (auto *AR = dyn_cast<SCEVAddRecExpr>(S)) {
    if (AR->isAffine() && AR->getLoop() == L) {
      FI.Kind = FieldKind::Affine;
      FI.StartSCEV = AR->getStart();
      FI.StrideSCEV = AR->getStepRecurrence(*SE);

      // Try to get constant stride
      if (auto *StrideC = dyn_cast<SCEVConstant>(FI.StrideSCEV)) {
        FI.StrideValue = StrideC->getAPInt().getSExtValue();
        FI.HasConstantStride = true;
      }

      LLVM_DEBUG(dbgs() << "  Found affine field with SCEV: " << *S << "\n");
      return;
    }
  }

  // Not optimizable
  FI.Kind = FieldKind::NonAffine;
}

void AMDGPUTDMDescriptorPHI::findTDMCalls(Loop *L,
                                          SmallVectorImpl<TDMCallInfo> &Calls) {
  for (BasicBlock *BB : L->blocks()) {
    for (Instruction &I : *BB) {
      auto *II = dyn_cast<IntrinsicInst>(&I);
      if (!II)
        continue;

      // Check for TDM intrinsics by name since they may be target-specific
      // Common patterns: amdgcn.tensor.load.to.lds.d2, etc.
      StringRef Name = II->getCalledFunction()->getName();
      if (!Name.contains("tensor") || !Name.contains("lds"))
        continue;

      LLVM_DEBUG(dbgs() << "Found TDM call: " << *II << "\n");

      TDMCallInfo Info;
      Info.Call = II;

      // First argument is the address descriptor <4 x i32>
      Value *AddrDesc = II->getArgOperand(0);
      auto *VecTy = dyn_cast<FixedVectorType>(AddrDesc->getType());
      if (!VecTy || VecTy->getNumElements() != 4 ||
          !VecTy->getElementType()->isIntegerTy(32))
        continue;

      Info.AddrDescriptor = AddrDesc;

      // Analyze each field
      for (unsigned i = 0; i < 4; ++i) {
        Value *FieldVal = traceFieldValue(AddrDesc, i);
        analyzeField(Info.Fields[i], FieldVal, L);

        LLVM_DEBUG({
          dbgs() << "  Field " << i << ": ";
          switch (Info.Fields[i].Kind) {
          case FieldKind::Constant: dbgs() << "Constant"; break;
          case FieldKind::LoopInvariant: dbgs() << "LoopInvariant"; break;
          case FieldKind::Affine: dbgs() << "Affine"; break;
          case FieldKind::NonAffine: dbgs() << "NonAffine"; break;
          }
          if (FieldVal)
            dbgs() << " = " << *FieldVal;
          dbgs() << "\n";
        });
      }

      Calls.push_back(Info);
    }
  }
}

bool AMDGPUTDMDescriptorPHI::hasOptimizableFields(TDMCallInfo &Info) {
  unsigned NumAffine = 0;

  for (unsigned i = 0; i < 4; ++i) {
    if (Info.Fields[i].Kind == FieldKind::Affine &&
        Info.Fields[i].HasConstantStride) {
      NumAffine++;
    }
  }

  // Only optimize if we have at least 2 affine fields.
  // With only 1 affine field, the PHI overhead outweighs the benefit
  // since LLVM's existing passes already handle scalar stride updates well.
  Info.HasOptimizableFields = (NumAffine >= 2);
  return Info.HasOptimizableFields;
}

Value *AMDGPUTDMDescriptorPHI::createPreheaderInit(TDMCallInfo &Info, Loop *L,
                                                    IRBuilder<> &Builder) {
  LLVMContext &Ctx = Builder.getContext();
  Type *I32Ty = Type::getInt32Ty(Ctx);
  auto *VecTy = FixedVectorType::get(I32Ty, 4);

  // Start with undef vector
  Value *InitDesc = UndefValue::get(VecTy);

  for (unsigned i = 0; i < 4; ++i) {
    FieldInfo &FI = Info.Fields[i];
    Value *InitVal = nullptr;

    switch (FI.Kind) {
    case FieldKind::Constant:
      // Use the constant directly
      InitVal = FI.CurrentValue;
      break;

    case FieldKind::LoopInvariant:
      // Value is already available in preheader
      InitVal = FI.CurrentValue;
      break;

    case FieldKind::Affine: {
      // Compute the initial value using SCEV expander
      // For simplicity, we compute start value by evaluating at IV=0
      SCEVExpander Expander(*SE, Builder.GetInsertBlock()->getModule()->getDataLayout(),
                            "tdm.init");
      Instruction *InsertPt = &*Builder.GetInsertPoint();
      InitVal = Expander.expandCodeFor(FI.StartSCEV, I32Ty, InsertPt);
      break;
    }

    case FieldKind::NonAffine:
      // Leave as undef - will be set in loop body
      InitVal = UndefValue::get(I32Ty);
      break;
    }

    if (InitVal && !isa<UndefValue>(InitVal)) {
      InitDesc = Builder.CreateInsertElement(InitDesc, InitVal, i,
                                              "tdm.init." + Twine(i));
    }
  }

  return InitDesc;
}

Value *AMDGPUTDMDescriptorPHI::createStrideUpdate(Value *DescPHI,
                                                   TDMCallInfo &Info,
                                                   IRBuilder<> &Builder) {
  Value *NextDesc = DescPHI;

  for (unsigned i = 0; i < 4; ++i) {
    FieldInfo &FI = Info.Fields[i];

    if (FI.Kind != FieldKind::Affine || !FI.HasConstantStride)
      continue;

    // Extract current value
    Value *CurVal = Builder.CreateExtractElement(DescPHI, i,
                                                  "tdm.cur." + Twine(i));

    // Add stride
    Value *NextVal = Builder.CreateAdd(CurVal,
                                        Builder.getInt32(FI.StrideValue),
                                        "tdm.next." + Twine(i));

    // Insert back
    NextDesc = Builder.CreateInsertElement(NextDesc, NextVal, i,
                                            "tdm.upd." + Twine(i));

    ++NumFieldsOptimized;
  }

  return NextDesc;
}

bool AMDGPUTDMDescriptorPHI::transformCall(TDMCallInfo &Info, Loop *L) {
  BasicBlock *Preheader = L->getLoopPreheader();
  BasicBlock *Header = L->getHeader();
  BasicBlock *Latch = L->getLoopLatch();

  if (!Preheader || !Latch) {
    LLVM_DEBUG(dbgs() << "  Skip: no preheader or latch\n");
    return false;
  }

  // For simplicity, only handle single-block loops for now
  // Multi-block loops would need more careful handling of the update placement
  if (L->getNumBlocks() > 1) {
    LLVM_DEBUG(dbgs() << "  Skip: multi-block loop (not yet supported)\n");
    return false;
  }

  // Create initial descriptor in preheader
  IRBuilder<> PreheaderBuilder(Preheader->getTerminator());
  Value *InitDesc = createPreheaderInit(Info, L, PreheaderBuilder);

  // Create PHI node at loop header
  LLVMContext &Ctx = Header->getContext();
  auto *VecTy = FixedVectorType::get(Type::getInt32Ty(Ctx), 4);

  // Insert PHI at the beginning of the header
  PHINode *DescPHI = PHINode::Create(VecTy, 2, "tdm.desc.phi");
  DescPHI->insertBefore(Header->begin());
  DescPHI->addIncoming(InitDesc, Preheader);

  // Build the descriptor to use: PHI + non-affine fields
  Value *UseDesc = DescPHI;
  IRBuilder<> UseBuilder(Info.Call);

  for (unsigned i = 0; i < 4; ++i) {
    FieldInfo &FI = Info.Fields[i];

    // Only insert non-affine fields - affine fields are already in PHI
    if (FI.Kind == FieldKind::NonAffine && FI.CurrentValue) {
      UseDesc = UseBuilder.CreateInsertElement(UseDesc, FI.CurrentValue, i,
                                                "tdm.use." + Twine(i));
    }
  }

  // Replace the descriptor operand in the TDM call
  Info.Call->setArgOperand(0, UseDesc);

  // Create stride update before the latch terminator
  // In a single-block loop, the latch is the header, so insert before terminator
  Instruction *LatchTerm = Latch->getTerminator();
  IRBuilder<> UpdateBuilder(LatchTerm);
  Value *NextDesc = createStrideUpdate(DescPHI, Info, UpdateBuilder);

  // Complete the PHI with the updated descriptor from the latch
  DescPHI->addIncoming(NextDesc, Latch);

  ++NumDescriptorsOptimized;
  LLVM_DEBUG(dbgs() << "  Transformed TDM call\n");

  return true;
}

bool AMDGPUTDMDescriptorPHI::runOnFunction(Function &F) {
  if (!EnableTDMDescriptorPHI)
    return false;

  LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  SE = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();

  bool Changed = false;

  // Process loops in post-order (innermost first)
  for (Loop *L : LI->getLoopsInPreorder()) {
    // Only process innermost loops for now
    if (!L->getSubLoops().empty())
      continue;

    LLVM_DEBUG(dbgs() << "Processing loop: " << L->getName() << "\n");

    SmallVector<TDMCallInfo, 4> TDMCalls;
    findTDMCalls(L, TDMCalls);

    for (TDMCallInfo &Info : TDMCalls) {
      if (hasOptimizableFields(Info)) {
        Changed |= transformCall(Info, L);
      } else {
        LLVM_DEBUG(dbgs() << "  Skip: no optimizable fields\n");
      }
    }
  }

  return Changed;
}

FunctionPass *llvm::createAMDGPUTDMDescriptorPHIPass() {
  return new AMDGPUTDMDescriptorPHI();
}

PreservedAnalyses AMDGPUTDMDescriptorPHIPass::run(Function &F,
                                                   FunctionAnalysisManager &AM) {
  if (!EnableTDMDescriptorPHI)
    return PreservedAnalyses::all();

  auto &LI = AM.getResult<LoopAnalysis>(F);
  auto &SE = AM.getResult<ScalarEvolutionAnalysis>(F);
  (void)AM.getResult<DominatorTreeAnalysis>(F); // Required for future enhancements

  bool Changed = false;

  // Helper lambdas that capture the analyses

  // Trace back through insertelement chain to find the value at a specific index.
  auto traceFieldValue = [](Value *Desc, unsigned FieldIdx) -> Value * {
    while (auto *IE = dyn_cast<InsertElementInst>(Desc)) {
      auto *IdxCI = dyn_cast<ConstantInt>(IE->getOperand(2));
      if (!IdxCI)
        return nullptr;
      if (IdxCI->getZExtValue() == FieldIdx)
        return IE->getOperand(1);
      Desc = IE->getOperand(0);
    }
    if (auto *CV = dyn_cast<Constant>(Desc))
      return CV->getAggregateElement(FieldIdx);
    return nullptr;
  };

  // Analyze a descriptor field using SCEV.
  auto analyzeField = [&SE](FieldInfo &FI, Value *FieldVal, Loop *L) {
    FI.CurrentValue = FieldVal;
    if (!FieldVal) {
      FI.Kind = FieldKind::NonAffine;
      return;
    }
    if (isa<Constant>(FieldVal)) {
      FI.Kind = FieldKind::Constant;
      return;
    }
    const SCEV *S = SE.getSCEV(FieldVal);
    if (SE.isLoopInvariant(S, L)) {
      FI.Kind = FieldKind::LoopInvariant;
      return;
    }
    if (auto *AR = dyn_cast<SCEVAddRecExpr>(S)) {
      if (AR->isAffine() && AR->getLoop() == L) {
        FI.Kind = FieldKind::Affine;
        FI.StartSCEV = AR->getStart();
        FI.StrideSCEV = AR->getStepRecurrence(SE);
        if (auto *StrideC = dyn_cast<SCEVConstant>(FI.StrideSCEV)) {
          FI.StrideValue = StrideC->getAPInt().getSExtValue();
          FI.HasConstantStride = true;
        }
        return;
      }
    }
    FI.Kind = FieldKind::NonAffine;
  };

  // Process loops in post-order (innermost first)
  for (Loop *L : LI.getLoopsInPreorder()) {
    // Only process innermost loops for now
    if (!L->getSubLoops().empty())
      continue;

    // Find TDM calls
    SmallVector<TDMCallInfo, 4> TDMCalls;
    for (BasicBlock *BB : L->blocks()) {
      for (Instruction &I : *BB) {
        auto *II = dyn_cast<IntrinsicInst>(&I);
        if (!II)
          continue;
        StringRef Name = II->getCalledFunction()->getName();
        if (!Name.contains("tensor") || !Name.contains("lds"))
          continue;

        TDMCallInfo Info;
        Info.Call = II;
        Value *AddrDesc = II->getArgOperand(0);
        auto *VecTy = dyn_cast<FixedVectorType>(AddrDesc->getType());
        if (!VecTy || VecTy->getNumElements() != 4 ||
            !VecTy->getElementType()->isIntegerTy(32))
          continue;

        Info.AddrDescriptor = AddrDesc;
        for (unsigned i = 0; i < 4; ++i) {
          Value *FieldVal = traceFieldValue(AddrDesc, i);
          analyzeField(Info.Fields[i], FieldVal, L);
        }
        TDMCalls.push_back(Info);
      }
    }

    // Process each TDM call
    for (TDMCallInfo &Info : TDMCalls) {
      // Check if optimizable - need at least 2 affine fields
      unsigned NumAffine = 0;
      for (unsigned i = 0; i < 4; ++i) {
        if (Info.Fields[i].Kind == FieldKind::Affine &&
            Info.Fields[i].HasConstantStride) {
          NumAffine++;
        }
      }
      // Only optimize if we have at least 2 affine fields.
      // With only 1 affine field, the PHI overhead outweighs the benefit.
      if (NumAffine < 2)
        continue;

      BasicBlock *Preheader = L->getLoopPreheader();
      BasicBlock *Header = L->getHeader();
      BasicBlock *Latch = L->getLoopLatch();

      if (!Preheader || !Latch || L->getNumBlocks() > 1)
        continue;

      // Create initial descriptor in preheader
      IRBuilder<> PreheaderBuilder(Preheader->getTerminator());
      LLVMContext &Ctx = PreheaderBuilder.getContext();
      Type *I32Ty = Type::getInt32Ty(Ctx);
      auto *VecTy = FixedVectorType::get(I32Ty, 4);
      Value *InitDesc = UndefValue::get(VecTy);

      for (unsigned i = 0; i < 4; ++i) {
        FieldInfo &FI = Info.Fields[i];
        Value *InitVal = nullptr;
        switch (FI.Kind) {
        case FieldKind::Constant:
        case FieldKind::LoopInvariant:
          InitVal = FI.CurrentValue;
          break;
        case FieldKind::Affine: {
          SCEVExpander Expander(SE, PreheaderBuilder.GetInsertBlock()->getModule()->getDataLayout(),
                                "tdm.init");
          Instruction *InsertPt = &*PreheaderBuilder.GetInsertPoint();
          InitVal = Expander.expandCodeFor(FI.StartSCEV, I32Ty, InsertPt);
          break;
        }
        case FieldKind::NonAffine:
          InitVal = UndefValue::get(I32Ty);
          break;
        }
        if (InitVal && !isa<UndefValue>(InitVal))
          InitDesc = PreheaderBuilder.CreateInsertElement(InitDesc, InitVal, i,
                                                          "tdm.init." + Twine(i));
      }

      // Create PHI node at loop header
      PHINode *DescPHI = PHINode::Create(VecTy, 2, "tdm.desc.phi");
      DescPHI->insertBefore(Header->begin());
      DescPHI->addIncoming(InitDesc, Preheader);

      // Build the descriptor to use: PHI + non-affine fields
      Value *UseDesc = DescPHI;
      IRBuilder<> UseBuilder(Info.Call);
      for (unsigned i = 0; i < 4; ++i) {
        FieldInfo &FI = Info.Fields[i];
        if (FI.Kind == FieldKind::NonAffine && FI.CurrentValue)
          UseDesc = UseBuilder.CreateInsertElement(UseDesc, FI.CurrentValue, i,
                                                    "tdm.use." + Twine(i));
      }
      Info.Call->setArgOperand(0, UseDesc);

      // Create stride update before the latch terminator
      IRBuilder<> UpdateBuilder(Latch->getTerminator());
      Value *NextDesc = DescPHI;
      for (unsigned i = 0; i < 4; ++i) {
        FieldInfo &FI = Info.Fields[i];
        if (FI.Kind != FieldKind::Affine || !FI.HasConstantStride)
          continue;
        Value *CurVal = UpdateBuilder.CreateExtractElement(DescPHI, i,
                                                            "tdm.cur." + Twine(i));
        Value *NextVal = UpdateBuilder.CreateAdd(CurVal,
                                                  UpdateBuilder.getInt32(FI.StrideValue),
                                                  "tdm.next." + Twine(i));
        NextDesc = UpdateBuilder.CreateInsertElement(NextDesc, NextVal, i,
                                                      "tdm.upd." + Twine(i));
      }
      DescPHI->addIncoming(NextDesc, Latch);

      Changed = true;
    }
  }

  if (Changed) {
    PreservedAnalyses PA;
    PA.preserveSet<CFGAnalyses>();
    return PA;
  }

  return PreservedAnalyses::all();
}

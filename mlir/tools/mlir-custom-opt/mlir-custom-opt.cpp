//===- mlir-custom-opt.cpp - MLIR Optimizer Driver ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for mlir-custom-opt for when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

// Dialect includes for registration
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

using namespace llvm;
using namespace mlir;

namespace {
struct AssertInserterPipelineOptions
    : public PassPipelineOptions<AssertInserterPipelineOptions> {
  PassOptions::Option<bool> warnOnUnknown{
      *this, "warn-on-unknown",
      llvm::cl::desc("Warn on unknown side-effecting operations"),
      llvm::cl::init(true)};
  PassOptions::Option<bool> includeVectorLoadStore{
      *this, "include-vector-load-store",
      llvm::cl::desc(
          "Include vector.load/store operations despite them allowing "
          "out-of-bounds"),
      llvm::cl::init(false)};
  PassOptions::Option<bool> checkEachDim{
      *this, "check-each-dim",
      llvm::cl::desc("Check each dimension individually"),
      llvm::cl::init(true)};
  PassOptions::Option<bool> createSpeculativeFuncs{
      *this, "create-speculative-funcs",
      llvm::cl::desc("Create a function that performs assertions speculatively "
                     "instead of in-place checks"),
      llvm::cl::init(false)};
};

void buildAssertInserterPipeline(mlir::OpPassManager &pm,
                                 const AssertInserterPipelineOptions &options) {
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::memref::createAssertInBoundsPass(
      mlir::memref::AssertInBoundsPassOptions{
          options.warnOnUnknown, options.includeVectorLoadStore,
          options.checkEachDim, options.createSpeculativeFuncs}));
  pm.addPass(mlir::arith::createIntRangeOptimizationsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::memref::createCheckStaticAssertionsPass());
}

void registerAssertInserterPipeline() {
  PassPipelineRegistration<AssertInserterPipelineOptions>(
      "assert-inserter-pipeline",
      "The pipeline inserts, simplifies and canonicalizes assertions to make "
      "sure memory accesses are in-bounds.",
      buildAssertInserterPipeline);
}
} // namespace

int main(int argc, char **argv) {
  registerAssertInserterPipeline();
  DialectRegistry registry;
  registry.insert<mlir::affine::AffineDialect, mlir::arith::ArithDialect,
                  mlir::func::FuncDialect, mlir::amdgpu::AMDGPUDialect,
                  mlir::gpu::GPUDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::vector::VectorDialect>();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "MLIR custom modular optimizer driver\n", registry));
}

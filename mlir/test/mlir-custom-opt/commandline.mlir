// RUN: echo "" | mlir-custom-opt --show-dialects | FileCheck %s
// CHECK: Available Dialects:
// CHECK-SAME: affine
// CHECK-SAME: amdgpu
// CHECK-SAME: arith
// CHECK-SAME: builtin
// CHECK-SAME: func
// CHECK-SAME: gpu
// CHECK-SAME: memref
// CHECK-SAME: scf
// CHECK-SAME: vector

// RUN: mlir-custom-opt --help | FileCheck %s -check-prefix=CHECK-HELP
// CHECK-HELP:      Pass Pipelines:
// CHECK-HELP-NEXT:   assert-inserter-pipeline
// CHECK-HELP-SAME:   The pipeline inserts, simplifies and canonicalizes assertions to make sure memory accesses are in-bounds.
// CHECK-HELP-NEXT:     check-each-dim
// CHECK-HELP-SAME:     Check each dimension individually
// CHECK-HELP-NEXT:     create-speculative-funcs
// CHECK-HELP-SAME:     Create a function that performs assertions speculatively instead of in-place checks
// CHECK-HELP-NEXT:     include-vector-load-store
// CHECK-HELP-SAME:     Include vector.load/store operations despite them allowing out-of-bounds
// CHECK-HELP-NEXT:     warn-on-unknown
// CHECK-HELP-SAME:     Warn on unknown side-effecting operations

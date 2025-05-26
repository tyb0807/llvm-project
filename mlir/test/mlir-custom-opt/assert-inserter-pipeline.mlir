// RUN:   mlir-custom-opt %s -pass-pipeline="builtin.module(assert-inserter-pipeline{include-vector-load-store=true})" \
// RUN: | FileCheck %s

// CHECK: cf.assert
// CHECK-NOT: affine

#map = affine_map<()[s0, s1] -> (s0 + s1 * 32 - (s0 floordiv 16) * 16 + (s0 floordiv 64) * 16)>
#map1 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 16)>
#map2 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4)>
#map3 = affine_map<()[s0, s1, s2] -> (s0 + s1 * 32 + s2 * 16 - (s0 floordiv 16) * 16)>
#map4 = affine_map<()[s0, s1] -> (s0 + s1 * 16 - (s0 floordiv 16) * 16)>
#map5 = affine_map<()[s0, s1] -> (s0 * 16 + ((s1 mod 64) floordiv 16) * 4)>
#map6 = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4)>
#map7 = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 1)>
#map8 = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 2)>
#map9 = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 3)>
module {
  func.func @gemm(%arg0: memref<128x64xf16, strided<[64, 1], offset: ?>>,
                  %arg1: memref<64x64xf16, strided<[64, 1], offset: ?>>,
                  %arg2: memref<64x128xf32, strided<[128, 1], offset: ?>>) {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : vector<4xf32>

    // Define workgroup and thread dimensions
    %workgroup_dim_x = arith.constant 4 : index  // Assuming 4 workgroups in x-dim
    %workgroup_dim_y = arith.constant 4 : index  // Assuming 4 workgroups in y-dim
    %thread_dim_x = arith.constant 64 : index    // Assuming 64 threads in x-dim
    %thread_dim_y = arith.constant 16 : index    // Assuming 16 threads in y-dim

    // Workgroup loops
    scf.forall (%workgroup_id_0, %workgroup_id_1) = (%c0, %c0)
        to (%workgroup_dim_x, %workgroup_dim_y) step (%c1, %c1) {
      // Thread loops
      scf.forall (%thread_id_x, %thread_id_y) = (%c0, %c0)
          to (%thread_dim_x, %thread_dim_y) step (%c1, %c1) {
        %alloc = memref.alloc() : memref<32x20xf16, #gpu.address_space<workgroup>>
        %alloc_0 = memref.alloc() : memref<32x20xf16, #gpu.address_space<workgroup>>

        %2 = affine.apply #map()[%thread_id_x, %workgroup_id_0]
        %3 = affine.apply #map1()[%thread_id_x]
        %4 = affine.apply #map2()[%thread_id_x]
        %5 = affine.apply #map3()[%thread_id_x, %workgroup_id_1, %thread_id_y]
        %6 = affine.apply #map4()[%thread_id_x, %thread_id_y]

        %7 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %cst) -> (vector<4xf32>) {
          amdgpu.lds_barrier

          %18 = affine.apply #map5()[%arg3, %thread_id_x]
          %19 = vector.load %arg1[%2, %18] : memref<64x64xf16, strided<[64, 1], offset: ?>>, vector<4xf16>
          vector.store %19, %alloc_0[%3, %4] : memref<32x20xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %20 = vector.load %arg0[%5, %18] : memref<128x64xf16, strided<[64, 1], offset: ?>>, vector<4xf16>
          vector.store %20, %alloc[%6, %4] : memref<32x20xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          amdgpu.lds_barrier

          %21 = vector.load %alloc[%6, %4] : memref<32x20xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %22 = vector.load %alloc_0[%3, %4] : memref<32x20xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %23 = amdgpu.mfma %22 * %21 + %arg4 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          scf.yield %23 : vector<4xf32>
        }

        %8 = vector.extract_strided_slice %7 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %10 = affine.apply #map6()[%workgroup_id_0, %thread_id_x]
        %11 = affine.apply #map3()[%thread_id_x, %workgroup_id_1, %thread_id_y]
        vector.store %8, %arg2[%10, %11] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>

        %12 = vector.extract_strided_slice %7 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %13 = affine.apply #map7()[%workgroup_id_0, %thread_id_x]
        vector.store %12, %arg2[%13, %11] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>

        %14 = vector.extract_strided_slice %7 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %15 = affine.apply #map8()[%workgroup_id_0, %thread_id_x]
        vector.store %14, %arg2[%15, %11] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>

        %16 = vector.extract_strided_slice %7 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %17 = affine.apply #map9()[%workgroup_id_0, %thread_id_x]
        vector.store %16, %arg2[%17, %11] : memref<64x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
      }
    }
    return
  }
}

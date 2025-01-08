/// tiling strategy
// for n_o = 0:N_G:N
//   for m_o = 0:M_G:M
//     for k_o = 0:K_G:K
//       A(M_G*K_G) -> Ag
//       K_A = K_G
//       for n_i = 0:N_A:N_G
//         B(K_A*N_A) -> Ba
//         for m_i = 0:M_A:M_G
//           C(M_A*N_A) -> Ca
//           for m_s = 0:M_S:M_A
//             Ag(M_S*K_A) -> As
//             call micro_kernel(As, Ba, Ca)
//           end for
//           Ca -> C(M_A*N_A)
//         end for
//       end for
//     end for
//   end for
// end for

////////////////////////////////// Original code //////////////////////////////////

module {
  func.func @matmul_elemwise(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>, %arg3: tensor<?x?xf32>) -> tensor<?x?xf32> 
  attributes {mtfusion.function_kind = #mtfusion.function_kind<Host>}
  {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim_m = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %dim_n = tensor.dim %arg1, %c1 : tensor<?x?xf32>
    %0 = tensor.empty(%dim_m, %dim_n) : tensor<?x?xf32>
    %1 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %2 = tensor.empty(%dim_m, %dim_n) : tensor<?x?xf32>
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%2 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %5 = tensor.empty(%dim_m, %dim_n) : tensor<?x?xf32>
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%3, %arg3 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%5 : tensor<?x?xf32>) -> tensor<?x?xf32>
    return %4 : tensor<?x?xf32>
  }
}

////////////////////////////////// Before tiling //////////////////////////////////
// -mtfusion-infer-func-fusion-kind
// -mtfusion-fuse-ops

module {
  func.func @matmul_elemwise_0(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>, %arg3: tensor<?x?xf32>, %arg4: tensor<?x?xf32>) -> tensor<?x?xf32> attributes {mtfusion.entry, mtfusion.function_kind = #mtfusion.function_kind<Device>, mtfusion.fusion_kind = #mtfusion.fusion_kind<MIX_CV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %dim_0 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
    %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
    %1 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%2, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%1 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%3, %arg3 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg4 : tensor<?x?xf32>) -> tensor<?x?xf32>
    return %4 : tensor<?x?xf32>
  }
  func.func @matmul_elemwise(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>, %arg3: tensor<?x?xf32>) -> tensor<?x?xf32> attributes {mtfusion.function_kind = #mtfusion.function_kind<Host>, mtfusion.fusion_kind = #mtfusion.fusion_kind<MIX_CV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %dim_0 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
    %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
    %1 = call @matmul_elemwise_0(%arg0, %arg1, %arg2, %arg3, %0) : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    return %1 : tensor<?x?xf32>
  }
}

////////////////////////////////// After predefined-tiling //////////////////////////////////
// -mtfusion-predefined-tiling=
//   "fusion-mode=MIX_CV tiling-seq=
//   {-axis=k},
//   {-axis=m -copy-mat=A -copy-dst=GSM},
//   {-axis=n -nthreads=8 -copy-mat=B copy-dst=AM},
//   {-axis=m -copy-mat=C -copy-dst=AM},
//   {-axis=m -copy-mat=A -copy-dst=SM}"

#map = affine_map<(d0)[s0, s1] -> (-d0 + s1, s0)>
module {
  func.func @matmul_elemwise_0_tiling(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>, %arg3: tensor<?x?xf32>, %arg4: tensor<?x?xf32>, %arg5: i64 {mtfusion.tiling_axis = #mtfusion.tiling_axis<K>, mtfusion.tiling_data}, %arg6: i64 {mtfusion.copy_dst = #mtfusion.copy_dst<GSM>, mtfusion.copy_mat = #mtfusion.copy_mat<MatA>, mtfusion.tiling_axis = #mtfusion.tiling_axis<M>, mtfusion.tiling_data}, %arg7: i64 {mtfusion.copy_dst = #mtfusion.copy_dst<AM>, mtfusion.copy_mat = #mtfusion.copy_mat<MatB>, mtfusion.nthreads = #mtfusion.nthreads<8>, mtfusion.tiling_axis = #mtfusion.tiling_axis<N>, mtfusion.tiling_data}, %arg8: i64 {mtfusion.copy_dst = #mtfusion.copy_dst<AM>, mtfusion.copy_mat = #mtfusion.copy_mat<MatC>, mtfusion.tiling_axis = #mtfusion.tiling_axis<M>, mtfusion.tiling_data}, %arg9: i64 {mtfusion.copy_dst = #mtfusion.copy_dst<SM>, mtfusion.copy_mat = #mtfusion.copy_mat<MatA>, mtfusion.tiling_axis = #mtfusion.tiling_axis<M>, mtfusion.tiling_data}) -> tensor<?x?xf32> attributes {mtfusion.entry, mtfusion.function_kind = #mtfusion.function_kind<Device>, mtfusion.fusion_kind = #mtfusion.fusion_kind<MIX_CV>} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = index.casts %arg5 {mtfusion.tiling_data} : i64 to index
    %1 = index.casts %arg6 {mtfusion.tiling_data} : i64 to index
    %2 = index.casts %arg7 {mtfusion.tiling_data} : i64 to index
    %3 = index.casts %arg8 {mtfusion.tiling_data} : i64 to index
    %4 = index.casts %arg9 {mtfusion.tiling_data} : i64 to index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %dim_0 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
    %5 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
    %dim_1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %6 = scf.for %arg10 = %c0 to %dim_1 step %0 iter_args(%arg11 = %5) -> (tensor<?x?xf32>) {
      %9 = affine.min #map(%arg10)[%0, %dim_1]
      %extracted_slice = tensor.extract_slice %arg0[0, %arg10] [%dim, %9] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
      %extracted_slice_2 = tensor.extract_slice %arg1[%arg10, 0] [%9, %dim_0] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
      %extracted_slice_3 = tensor.extract_slice %arg11[0, 0] [%dim, %dim_0] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
      %10 = scf.for %arg12 = %c0 to %dim step %1 iter_args(%arg13 = %extracted_slice_3) -> (tensor<?x?xf32>) {
        %11 = affine.min #map(%arg12)[%1, %dim]
        %extracted_slice_4 = tensor.extract_slice %extracted_slice[%arg12, 0] [%11, %9] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %12 = tensor.empty(%11, %9) : tensor<?x?xf32>
        %13 = linalg.copy {"DDR : GSM"} ins(%extracted_slice_4 : tensor<?x?xf32>) outs(%12 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %extracted_slice_5 = tensor.extract_slice %extracted_slice_2[0, 0] [%9, %dim_0] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %extracted_slice_6 = tensor.extract_slice %arg13[%arg12, 0] [%11, %dim_0] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %14 = scf.for %arg14 = %c0 to %dim_0 step %2 iter_args(%arg15 = %extracted_slice_6) -> (tensor<?x?xf32>) {
          %15 = affine.min #map(%arg14)[%2, %dim_0]
          %extracted_slice_8 = tensor.extract_slice %13[0, 0] [%11, %9] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
          %extracted_slice_9 = tensor.extract_slice %extracted_slice_5[0, %arg14] [%9, %15] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
          %16 = tensor.empty(%9, %15) : tensor<?x?xf32>
          %17 = linalg.copy {"DDR : AM"} ins(%extracted_slice_9 : tensor<?x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
          %extracted_slice_10 = tensor.extract_slice %arg15[0, %arg14] [%11, %15] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
          %18 = scf.for %arg16 = %c0 to %11 step %3 iter_args(%arg17 = %extracted_slice_10) -> (tensor<?x?xf32>) {
            %19 = affine.min #map(%arg16)[%3, %11]
            %extracted_slice_12 = tensor.extract_slice %extracted_slice_8[%arg16, 0] [%19, %9] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
            %extracted_slice_13 = tensor.extract_slice %17[0, 0] [%9, %15] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
            %extracted_slice_14 = tensor.extract_slice %arg17[%arg16, 0] [%19, %15] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
            %20 = tensor.empty(%19, %15) : tensor<?x?xf32>
            %21 = linalg.copy {"DDR : AM"} ins(%extracted_slice_14 : tensor<?x?xf32>) outs(%20 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %22 = scf.for %arg18 = %c0 to %19 step %4 iter_args(%arg19 = %21) -> (tensor<?x?xf32>) {
              %25 = affine.min #map(%arg18)[%4, %19]
              %extracted_slice_18 = tensor.extract_slice %extracted_slice_12[%arg18, 0] [%25, %9] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
              %26 = tensor.empty(%25, %9) : tensor<?x?xf32>
              %27 = linalg.copy {"GSM : SM"} ins(%extracted_slice_18 : tensor<?x?xf32>) outs(%26 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %extracted_slice_19 = tensor.extract_slice %extracted_slice_13[0, 0] [%9, %15] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
              %extracted_slice_20 = tensor.extract_slice %arg19[%arg18, 0] [%25, %15] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
              %28 = linalg.matmul ins(%27, %extracted_slice_19 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%extracted_slice_20 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %inserted_slice_21 = tensor.insert_slice %28 into %arg19[%arg18, 0] [%25, %15] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
              scf.yield %inserted_slice_21 : tensor<?x?xf32>
            } {__tiled_for___4}
            %dim_15 = tensor.dim %22, %c0 : tensor<?x?xf32>
            %dim_16 = tensor.dim %22, %c1 : tensor<?x?xf32>
            %23 = tensor.empty(%dim_15, %dim_16) : tensor<?x?xf32>
            %24 = linalg.copy {"AM : DDR"} ins(%22 : tensor<?x?xf32>) outs(%23 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %inserted_slice_17 = tensor.insert_slice %24 into %arg17[%arg16, 0] [%19, %15] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
            scf.yield %inserted_slice_17 : tensor<?x?xf32>
          } {__tiled_for___3}
          %inserted_slice_11 = tensor.insert_slice %18 into %arg15[0, %arg14] [%11, %15] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
          scf.yield %inserted_slice_11 : tensor<?x?xf32>
        } {__tiled_for___2}
        %inserted_slice_7 = tensor.insert_slice %14 into %arg13[%arg12, 0] [%11, %dim_0] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
        scf.yield %inserted_slice_7 : tensor<?x?xf32>
      } {__tiled_for___1}
      %inserted_slice = tensor.insert_slice %10 into %arg11[0, 0] [%dim, %dim_0] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
      scf.yield %inserted_slice : tensor<?x?xf32>
    } {__tiled_for__}
    %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%6, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%5 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %8 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%7, %arg3 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg4 : tensor<?x?xf32>) -> tensor<?x?xf32>
    return %8 : tensor<?x?xf32>
  }
  transform.sequence  failures(propagate) attributes {transform.target_tag = "matmul_elemwise_0_tiling_transform"} {
  ^bb0(%arg0: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = split_handle %0 : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.match attributes {mtfusion.tiling_data} in %arg0 : (!transform.any_op) -> !transform.any_op
    %3:5 = split_handle %2 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %tiled_linalg_op, %loops = transform.structured.tile_using_for %1[0, 0, %3#0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    annotate %loops "__tiled_for__" : !transform.any_op
    %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op[%3#1, 0, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    annotate %loops_1 "__tiled_for___1" : !transform.any_op
    %4 = get_operand %tiled_linalg_op_0[0] : (!transform.any_op) -> !transform.any_value
    %5 = transform.structured.cache_read %4 : (!transform.any_value) -> !transform.any_op
    annotate %5 "DDR : GSM" : !transform.any_op
    %6 = transform.structured.match attributes {"DDR : GSM"} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %tiled_linalg_op_0[0, %3#2, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    annotate %loops_3 "__tiled_for___2" : !transform.any_op
    %7 = get_operand %tiled_linalg_op_2[1] : (!transform.any_op) -> !transform.any_value
    %8 = transform.structured.cache_read %7 : (!transform.any_value) -> !transform.any_op
    annotate %8 "DDR : AM" : !transform.any_op
    %9 = transform.structured.match attributes {"DDR : AM"} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2[%3#3, 0, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    annotate %loops_5 "__tiled_for___3" : !transform.any_op
    %10 = get_result %tiled_linalg_op_4[0] : (!transform.any_op) -> !transform.any_value
    %11 = transform.structured.cache_write %10 {output_only = true} : (!transform.any_value) -> !transform.any_op
    annotate %11 "AM : DDR" : !transform.any_op
    %12 = transform.structured.match attributes {"AM : DDR"} in %arg0 : (!transform.any_op) -> !transform.any_op
    %13 = reverse %12 : (!transform.any_op) -> !transform.any_op
    %14 = get_operand %tiled_linalg_op_4[2] : (!transform.any_op) -> !transform.any_value
    %15 = transform.structured.cache_read %14 : (!transform.any_value) -> !transform.any_op
    annotate %15 "DDR : AM" : !transform.any_op
    %16 = transform.structured.match attributes {"DDR : AM"} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %tiled_linalg_op_4[%3#4, 0, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    annotate %loops_7 "__tiled_for___4" : !transform.any_op
    %17 = get_operand %tiled_linalg_op_6[0] : (!transform.any_op) -> !transform.any_value
    %18 = transform.structured.cache_read %17 : (!transform.any_value) -> !transform.any_op
    annotate %18 "GSM : SM" : !transform.any_op
    %19 = transform.structured.match attributes {"GSM : SM"} in %arg0 : (!transform.any_op) -> !transform.any_op
    %20 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    apply_patterns to %20 {
      transform.apply_patterns.canonicalization
    } {apply_cse} : !transform.any_op
  }
  func.func @matmul_elemwise(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>, %arg3: tensor<?x?xf32>, %arg4: i64 {mtfusion.tiling_data}, %arg5: i64 {mtfusion.tiling_data}, %arg6: i64 {mtfusion.tiling_data}, %arg7: i64 {mtfusion.tiling_data}, %arg8: i64 {mtfusion.tiling_data}) -> tensor<?x?xf32> attributes {mtfusion.function_kind = #mtfusion.function_kind<Host>, mtfusion.fusion_kind = #mtfusion.fusion_kind<MIX_CV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %dim_0 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
    %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
    %1 = call @matmul_elemwise_0_tiling(%arg0, %arg1, %arg2, %arg3, %0, %arg4, %arg5, %arg6, %arg7, %arg8) : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, i64, i64, i64, i64, i64) -> tensor<?x?xf32>
    return %1 : tensor<?x?xf32>
  }
}

////////////////////////////////// After bufferization //////////////////////////////////
// -one-shot-bufferize="bufferize-function-boundaries=true"
// -mtfusion-buffer-normalize

#map = affine_map<(d0)[s0, s1] -> (-d0 + s1, s0)>
module {
  func.func @matmul_elemwise_0_tiling(
      %arg0: memref<?x?xf32, strided<[?, ?], offset: ?>>, 
      %arg1: memref<?x?xf32, strided<[?, ?], offset: ?>>, 
      %arg2: memref<?x?xf32, strided<[?, ?], offset: ?>>, 
      %arg3: memref<?x?xf32, strided<[?, ?], offset: ?>>, 
      %arg4: memref<?x?xf32, strided<[?, ?], offset: ?>>, 
      %arg5: i64 {mtfusion.tiling_axis = #mtfusion.tiling_axis<K>, mtfusion.tiling_data}, 
      %arg6: i64 {mtfusion.copy_dst = #mtfusion.copy_dst<GSM>, mtfusion.copy_mat = #mtfusion.copy_mat<MatA>, mtfusion.tiling_axis = #mtfusion.tiling_axis<M>, mtfusion.tiling_data}, 
      %arg7: i64 {mtfusion.copy_dst = #mtfusion.copy_dst<AM>, mtfusion.copy_mat = #mtfusion.copy_mat<MatB>, mtfusion.nthreads = #mtfusion.nthreads<8>, mtfusion.tiling_axis = #mtfusion.tiling_axis<N>, mtfusion.tiling_data}, 
      %arg8: i64 {mtfusion.copy_dst = #mtfusion.copy_dst<AM>, mtfusion.copy_mat = #mtfusion.copy_mat<MatC>, mtfusion.tiling_axis = #mtfusion.tiling_axis<M>, mtfusion.tiling_data}, 
      %arg9: i64 {mtfusion.copy_dst = #mtfusion.copy_dst<SM>, mtfusion.copy_mat = #mtfusion.copy_mat<MatA>, mtfusion.tiling_axis = #mtfusion.tiling_axis<M>, mtfusion.tiling_data})
       -> memref<?x?xf32, strided<[?, ?], offset: ?>> 
      attributes {mtfusion.entry, mtfusion.function_kind = #mtfusion.function_kind<Device>, mtfusion.fusion_kind = #mtfusion.fusion_kind<MIX_CV>} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = index.casts %arg5 {mtfusion.tiling_data} : i64 to index
    %1 = index.casts %arg6 {mtfusion.tiling_data} : i64 to index
    %2 = index.casts %arg7 {mtfusion.tiling_data} : i64 to index
    %3 = index.casts %arg8 {mtfusion.tiling_data} : i64 to index
    %4 = index.casts %arg9 {mtfusion.tiling_data} : i64 to index
    %dim = memref.dim %arg0, %c0 : memref<?x?xf32, strided<[?, ?], offset: ?>>
    %dim_0 = memref.dim %arg1, %c1 : memref<?x?xf32, strided<[?, ?], offset: ?>>
    %alloc = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf32>
    %dim_1 = memref.dim %arg0, %c1 : memref<?x?xf32, strided<[?, ?], offset: ?>>
    %alloc_2 = memref.alloc(%dim, %dim_0) {alignment = 64 : i64, mtfusion.memory_level = #mtfusion.memory_level<DDR>} : memref<?x?xf32>
    scf.for %arg10 = %c0 to %dim_1 step %0 {
      %5 = affine.min #map(%arg10)[%0, %dim_1]
      %subview = memref.subview %arg0[0, %arg10] [%dim, %5] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %subview_3 = memref.subview %arg1[%arg10, 0] [%5, %dim_0] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      scf.for %arg11 = %c0 to %dim step %1 {
        %6 = affine.min #map(%arg11)[%1, %dim]
        %subview_4 = memref.subview %subview[%arg11, 0] [%6, %5] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %alloc_5 = memref.alloc(%1, %0) {alignment = 64 : i64, mtfusion.memory_level = #mtfusion.memory_level<GSM>} : memref<?x?xf32>
        linalg.copy {"DDR : GSM"} ins(%subview_4 : memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%alloc_5 : memref<?x?xf32>)
        %subview_6 = memref.subview %alloc_2[%arg11, 0] [%6, %dim_0] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        scf.for %arg12 = %c0 to %dim_0 step %2 {
          %7 = affine.min #map(%arg12)[%2, %dim_0]
          %subview_7 = memref.subview %subview_3[0, %arg12] [%5, %7] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
          %alloc_8 = memref.alloc(%0, %2) {alignment = 64 : i64, mtfusion.memory_level = #mtfusion.memory_level<AM>} : memref<?x?xf32>
          linalg.copy {"DDR : AM"} ins(%subview_7 : memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%alloc_8 : memref<?x?xf32>)
          %subview_9 = memref.subview %subview_6[0, %arg12] [%6, %7] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
          scf.for %arg13 = %c0 to %6 step %3 {
            %8 = affine.min #map(%arg13)[%3, %6]
            %subview_10 = memref.subview %alloc_5[%arg13, 0] [%8, %5] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
            %subview_11 = memref.subview %subview_9[%arg13, 0] [%8, %7] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
            %alloc_12 = memref.alloc(%3, %2) {alignment = 64 : i64, mtfusion.memory_level = #mtfusion.memory_level<AM>} : memref<?x?xf32>
            linalg.copy {"DDR : AM"} ins(%subview_11 : memref<?x?xf32, strided<[?, 1], offset: ?>>) outs(%alloc_12 : memref<?x?xf32>)
            scf.for %arg14 = %c0 to %8 step %4 {
              %9 = affine.min #map(%arg14)[%4, %8]
              %subview_13 = memref.subview %subview_10[%arg14, 0] [%9, %5] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              %alloc_14 = memref.alloc(%4, %0) {alignment = 64 : i64, mtfusion.memory_level = #mtfusion.memory_level<SM>} : memref<?x?xf32>
              linalg.copy {"GSM : SM"} ins(%subview_13 : memref<?x?xf32, strided<[?, 1], offset: ?>>) outs(%alloc_14 : memref<?x?xf32>)
              %subview_15 = memref.subview %alloc_12[%arg14, 0] [%9, %7] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              linalg.matmul ins(%alloc_14, %alloc_8 : memref<?x?xf32>, memref<?x?xf32>) outs(%subview_15 : memref<?x?xf32, strided<[?, 1], offset: ?>>)
            } {__tiled_for___4}
            linalg.copy {"AM : DDR"} ins(%alloc_12 : memref<?x?xf32>) outs(%subview_11 : memref<?x?xf32, strided<[?, 1], offset: ?>>)
          } {__tiled_for___3}
        } {__tiled_for___2}
      } {__tiled_for___1}
    } {__tiled_for__}
    linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%alloc_2, %arg2 : memref<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%alloc : memref<?x?xf32>)
    linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%alloc, %arg3 : memref<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%arg4 : memref<?x?xf32, strided<[?, ?], offset: ?>>)
    return %arg4 : memref<?x?xf32, strided<[?, ?], offset: ?>>
  }
  func.func @matmul_elemwise(%arg0: memref<?x?xf32, strided<[?, ?], offset: ?>>, %arg1: memref<?x?xf32, strided<[?, ?], offset: ?>>, %arg2: memref<?x?xf32, strided<[?, ?], offset: ?>>, %arg3: memref<?x?xf32, strided<[?, ?], offset: ?>>, %arg4: i64 {mtfusion.tiling_data}, %arg5: i64 {mtfusion.tiling_data}, %arg6: i64 {mtfusion.tiling_data}, %arg7: i64 {mtfusion.tiling_data}, %arg8: i64 {mtfusion.tiling_data}) -> memref<?x?xf32, strided<[?, ?], offset: ?>> attributes {mtfusion.function_kind = #mtfusion.function_kind<Host>, mtfusion.fusion_kind = #mtfusion.fusion_kind<MIX_CV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = memref.dim %arg0, %c0 : memref<?x?xf32, strided<[?, ?], offset: ?>>
    %dim_0 = memref.dim %arg1, %c1 : memref<?x?xf32, strided<[?, ?], offset: ?>>
    %alloc = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf32>
    %cast = memref.cast %alloc : memref<?x?xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
    %0 = call @matmul_elemwise_0_tiling(%arg0, %arg1, %arg2, %arg3, %cast, %arg4, %arg5, %arg6, %arg7, %arg8) : (memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>, i64, i64, i64, i64, i64) -> memref<?x?xf32, strided<[?, ?], offset: ?>>
    return %0 : memref<?x?xf32, strided<[?, ?], offset: ?>>
  }
}

////////////////////////////////// After multi-buffering //////////////////////////////////
// -mtfusion-multi-buffering

#map = affine_map<(d0)[s0, s1] -> (-d0 + s1, s0)>
module {
  func.func @matmul_elemwise_0_tiling(
      %arg0: memref<?x?xf32, strided<[?, ?], offset: ?>>, 
      %arg1: memref<?x?xf32, strided<[?, ?], offset: ?>>,
      %arg2: memref<?x?xf32, strided<[?, ?], offset: ?>>, 
      %arg3: memref<?x?xf32, strided<[?, ?], offset: ?>>, 
      %arg4: memref<?x?xf32, strided<[?, ?], offset: ?>>, 
      %arg5: i64 {mtfusion.tiling_axis = #mtfusion.tiling_axis<K>, mtfusion.tiling_data}, 
      %arg6: i64 {mtfusion.copy_dst = #mtfusion.copy_dst<GSM>, mtfusion.copy_mat = #mtfusion.copy_mat<MatA>, mtfusion.tiling_axis = #mtfusion.tiling_axis<M>, mtfusion.tiling_data}, 
      %arg7: i64 {mtfusion.copy_dst = #mtfusion.copy_dst<AM>, mtfusion.copy_mat = #mtfusion.copy_mat<MatB>, mtfusion.nthreads = #mtfusion.nthreads<8>, mtfusion.tiling_axis = #mtfusion.tiling_axis<N>, mtfusion.tiling_data}, 
      %arg8: i64 {mtfusion.copy_dst = #mtfusion.copy_dst<AM>, mtfusion.copy_mat = #mtfusion.copy_mat<MatC>, mtfusion.tiling_axis = #mtfusion.tiling_axis<M>, mtfusion.tiling_data}, 
      %arg9: i64 {mtfusion.copy_dst = #mtfusion.copy_dst<SM>, mtfusion.copy_mat = #mtfusion.copy_mat<MatA>, mtfusion.tiling_axis = #mtfusion.tiling_axis<M>, mtfusion.tiling_data}) 
      -> memref<?x?xf32, strided<[?, ?], offset: ?>> 
      attributes {mtfusion.entry, mtfusion.function_kind = #mtfusion.function_kind<Device>, mtfusion.fusion_kind = #mtfusion.fusion_kind<MIX_CV>} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = index.casts %arg5 {mtfusion.tiling_data} : i64 to index
    %1 = index.casts %arg6 {mtfusion.tiling_data} : i64 to index
    %alloc = memref.alloc(%1, %0) {alignment = 64 : i64, mtfusion.memory_level = #mtfusion.memory_level<GSM>} : memref<2x?x?xf32>
    %2 = index.casts %arg7 {mtfusion.tiling_data} : i64 to index
    %alloc_0 = memref.alloc(%0, %2) {alignment = 64 : i64, mtfusion.memory_level = #mtfusion.memory_level<AM>} : memref<2x?x?xf32>
    %3 = index.casts %arg8 {mtfusion.tiling_data} : i64 to index
    %alloc_1 = memref.alloc(%3, %2) {alignment = 64 : i64, mtfusion.memory_level = #mtfusion.memory_level<AM>} : memref<3x?x?xf32>
    %4 = index.casts %arg9 {mtfusion.tiling_data} : i64 to index
    %alloc_2 = memref.alloc(%4, %0) {alignment = 64 : i64, mtfusion.memory_level = #mtfusion.memory_level<SM>} : memref<2x?x?xf32>
    %dim = memref.dim %arg0, %c0 : memref<?x?xf32, strided<[?, ?], offset: ?>>
    %dim_3 = memref.dim %arg1, %c1 : memref<?x?xf32, strided<[?, ?], offset: ?>>
    %alloc_4 = memref.alloc(%dim, %dim_3) {alignment = 64 : i64} : memref<?x?xf32>
    %dim_5 = memref.dim %arg0, %c1 : memref<?x?xf32, strided<[?, ?], offset: ?>>
    %alloc_6 = memref.alloc(%dim, %dim_3) {alignment = 64 : i64, mtfusion.memory_level = #mtfusion.memory_level<DDR>} : memref<?x?xf32>
    %c0_7 = arith.constant 0 : index
    scf.for %arg10 = %c0 to %dim_5 step %0 {
      %5 = affine.min #map(%arg10)[%0, %dim_5]
      %subview = memref.subview %arg0[0, %arg10] [%dim, %5] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %subview_8 = memref.subview %arg1[%arg10, 0] [%5, %dim_3] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %c0_9 = arith.constant 0 : index

      %subview_10 = memref.subview %alloc[0, 0, 0] [1, %1, %0] [1, 1, 1] : memref<2x?x?xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %6 = affine.min #map(%c0)[%1, %dim]
      %subview_11 = memref.subview %subview[%c0, 0] [%6, %5] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      linalg.copy {"DDR : GSM"} ins(%subview_11 : memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%subview_10 : memref<?x?xf32, strided<[?, ?], offset: ?>>)
      %subview_12 = memref.subview %alloc_6[%c0, 0] [%6, %dim_3] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %7:3 = scf.for %arg11 = %c0 to %dim step %1 iter_args(%arg12 = %subview_11, %arg13 = %subview_10, %arg14 = %subview_12) -> (memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>) {
        %8 = arith.addi %arg11, %1 : index
        %9 = arith.cmpi slt, %8, %dim : index
        %10 = arith.subi %8, %c0 : index
        %11 = arith.divsi %10, %1 : index
        %c2 = arith.constant 2 : index
        %12 = arith.remsi %11, %c2 : index
        %subview_13 = memref.subview %alloc[%12, 0, 0] [1, %1, %0] [1, 1, 1] : memref<2x?x?xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %13 = affine.min #map(%8)[%1, %dim]
        %14:3 = scf.if %9 -> (memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>) {
          %subview_18 = memref.subview %subview[%8, 0] [%13, %5] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
          linalg.copy {"DDR : GSM"} ins(%subview_18 : memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%subview_13 : memref<?x?xf32, strided<[?, ?], offset: ?>>)
          %subview_19 = memref.subview %alloc_6[%8, 0] [%13, %dim_3] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
          scf.yield %subview_18, %subview_13, %subview_19 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>
        } else {
          scf.yield %arg12, %arg13, %arg14 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>
        }
        %15 = affine.min #map(%arg11)[%1, %dim]
        %c0_14 = arith.constant 0 : index

        %subview_15 = memref.subview %alloc_0[0, 0, 0] [1, %0, %2] [1, 1, 1] : memref<2x?x?xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %16 = affine.min #map(%c0)[%2, %dim_3]
        %subview_16 = memref.subview %subview_8[0, %c0] [%5, %16] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        linalg.copy {"DDR : AM"} ins(%subview_16 : memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%subview_15 : memref<?x?xf32, strided<[?, ?], offset: ?>>)
        %subview_17 = memref.subview %arg14[0, %c0] [%15, %16] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %17:3 = scf.for %arg15 = %c0 to %dim_3 step %2 iter_args(%arg16 = %subview_16, %arg17 = %subview_15, %arg18 = %subview_17) -> (memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>) {
          %18 = arith.addi %arg15, %2 : index
          %19 = arith.cmpi slt, %18, %dim_3 : index
          %20 = arith.subi %18, %c0 : index
          %21 = arith.divsi %20, %2 : index
          %c2_18 = arith.constant 2 : index
          %22 = arith.remsi %21, %c2_18 : index
          %subview_19 = memref.subview %alloc_0[%22, 0, 0] [1, %0, %2] [1, 1, 1] : memref<2x?x?xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
          %23 = affine.min #map(%18)[%2, %dim_3]
          %24:3 = scf.if %19 -> (memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>) {
            %subview_24 = memref.subview %subview_8[0, %18] [%5, %23] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            linalg.copy {"DDR : AM"} ins(%subview_24 : memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%subview_19 : memref<?x?xf32, strided<[?, ?], offset: ?>>)
            %subview_25 = memref.subview %arg14[0, %18] [%15, %23] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
            scf.yield %subview_24, %subview_19, %subview_25 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>
          } else {
            scf.yield %arg16, %arg17, %arg18 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>
          }
          %25 = affine.min #map(%arg15)[%2, %dim_3]
          %c0_20 = arith.constant 0 : index

          %subview_21 = memref.subview %alloc_1[0, 0, 0] [1, %3, %2] [1, 1, 1] : memref<3x?x?xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
          %26 = affine.min #map(%c0)[%3, %15]
          %subview_22 = memref.subview %arg13[%c0, 0] [%26, %5] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
          %subview_23 = memref.subview %arg18[%c0, 0] [%26, %25] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
          linalg.copy {"DDR : AM"} ins(%subview_23 : memref<?x?xf32, strided<[?, 1], offset: ?>>) outs(%subview_21 : memref<?x?xf32, strided<[?, ?], offset: ?>>)
          %27:5 = scf.for %arg19 = %c0 to %15 step %3 iter_args(%arg20 = %subview_22, %arg21 = %subview_23, %arg22 = %subview_21, %arg23 = %subview_21, %arg24 = %subview_23) -> (memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>) {
            %28 = arith.addi %arg19, %3 : index
            %29 = arith.cmpi slt, %28, %15 : index
            %30 = arith.subi %28, %c0 : index
            %31 = arith.divsi %30, %3 : index
            %c3 = arith.constant 3 : index
            %32 = arith.remsi %31, %c3 : index
            %subview_24 = memref.subview %alloc_1[%32, 0, 0] [1, %3, %2] [1, 1, 1] : memref<3x?x?xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            %33 = affine.min #map(%28)[%3, %15]
            %34:3 = scf.if %29 -> (memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) {
              %subview_29 = memref.subview %arg13[%28, 0] [%33, %5] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              %subview_30 = memref.subview %arg18[%28, 0] [%33, %25] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              linalg.copy {"DDR : AM"} ins(%subview_30 : memref<?x?xf32, strided<[?, 1], offset: ?>>) outs(%subview_24 : memref<?x?xf32, strided<[?, ?], offset: ?>>)
              scf.yield %subview_29, %subview_30, %subview_24 : memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>
            } else {
              scf.yield %arg20, %arg21, %arg22 : memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>
            }
            %35 = affine.min #map(%arg19)[%3, %15]
            %c0_25 = arith.constant 0 : index

            %subview_26 = memref.subview %alloc_2[0, 0, 0] [1, %4, %0] [1, 1, 1] : memref<2x?x?xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
            %36 = affine.min #map(%c0)[%4, %35]
            %subview_27 = memref.subview %arg20[%c0, 0] [%36, %5] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
            linalg.copy {"GSM : SM"} ins(%subview_27 : memref<?x?xf32, strided<[?, 1], offset: ?>>) outs(%subview_26 : memref<?x?xf32, strided<[?, ?], offset: ?>>)
            %subview_28 = memref.subview %arg22[%c0, 0] [%36, %25] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
            %37:3 = scf.for %arg25 = %c0 to %35 step %4 iter_args(%arg26 = %subview_27, %arg27 = %subview_26, %arg28 = %subview_28) -> (memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>) {
              %39 = arith.addi %arg25, %4 : index
              %40 = arith.cmpi slt, %39, %35 : index
              %41 = arith.subi %39, %c0 : index
              %42 = arith.divsi %41, %4 : index
              %c2_29 = arith.constant 2 : index
              %43 = arith.remsi %42, %c2_29 : index
              %subview_30 = memref.subview %alloc_2[%43, 0, 0] [1, %4, %0] [1, 1, 1] : memref<2x?x?xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              %44 = affine.min #map(%39)[%4, %35]
              %45:3 = scf.if %40 -> (memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>) {
                %subview_31 = memref.subview %arg20[%39, 0] [%44, %5] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                linalg.copy {"GSM : SM"} ins(%subview_31 : memref<?x?xf32, strided<[?, 1], offset: ?>>) outs(%subview_30 : memref<?x?xf32, strided<[?, ?], offset: ?>>)
                %subview_32 = memref.subview %arg22[%39, 0] [%44, %25] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                scf.yield %subview_31, %subview_30, %subview_32 : memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>
              } else {
                scf.yield %arg26, %arg27, %arg28 : memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>
              }
              %46 = affine.min #map(%arg25)[%4, %35]
              linalg.matmul ins(%arg27, %arg17 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%arg28 : memref<?x?xf32, strided<[?, 1], offset: ?>>)
              scf.yield %45#0, %45#1, %45#2 : memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>
            }

            %38 = arith.cmpi ne, %arg19, %c0 : index
            scf.if %38 {
              linalg.copy {"AM : DDR"} ins(%arg23 : memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%arg24 : memref<?x?xf32, strided<[?, 1], offset: ?>>)
            }

            scf.yield %34#0, %34#1, %34#2, %arg22, %arg21 : memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>
          }
          linalg.copy {"AM : DDR"} ins(%27#3 : memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%27#4 : memref<?x?xf32, strided<[?, 1], offset: ?>>)
          scf.yield %24#0, %24#1, %24#2 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>
        }
        scf.yield %14#0, %14#1, %14#2 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, 1], offset: ?>>
      }
    } {__tiled_for__}
    linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%alloc_6, %arg2 : memref<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%alloc_4 : memref<?x?xf32>)
    linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%alloc_4, %arg3 : memref<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%arg4 : memref<?x?xf32, strided<[?, ?], offset: ?>>)
    return %arg4 : memref<?x?xf32, strided<[?, ?], offset: ?>>
  }
  func.func @matmul_elemwise(%arg0: memref<?x?xf32, strided<[?, ?], offset: ?>>, %arg1: memref<?x?xf32, strided<[?, ?], offset: ?>>, %arg2: memref<?x?xf32, strided<[?, ?], offset: ?>>, %arg3: memref<?x?xf32, strided<[?, ?], offset: ?>>, %arg4: i64 {mtfusion.tiling_data}, %arg5: i64 {mtfusion.tiling_data}, %arg6: i64 {mtfusion.tiling_data}, %arg7: i64 {mtfusion.tiling_data}, %arg8: i64 {mtfusion.tiling_data}) -> memref<?x?xf32, strided<[?, ?], offset: ?>> attributes {mtfusion.function_kind = #mtfusion.function_kind<Host>, mtfusion.fusion_kind = #mtfusion.fusion_kind<MIX_CV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = memref.dim %arg0, %c0 : memref<?x?xf32, strided<[?, ?], offset: ?>>
    %dim_0 = memref.dim %arg1, %c1 : memref<?x?xf32, strided<[?, ?], offset: ?>>
    %alloc = memref.alloc(%dim, %dim_0) {alignment = 64 : i64} : memref<?x?xf32>
    %cast = memref.cast %alloc : memref<?x?xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
    %0 = call @matmul_elemwise_0_tiling(%arg0, %arg1, %arg2, %arg3, %cast, %arg4, %arg5, %arg6, %arg7, %arg8) : (memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>, i64, i64, i64, i64, i64) -> memref<?x?xf32, strided<[?, ?], offset: ?>>
    return %0 : memref<?x?xf32, strided<[?, ?], offset: ?>>
  }
}

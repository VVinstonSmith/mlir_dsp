

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


////////////////////////////////// After predefined-tiling //////////////////////////////////

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
      %10 = scf.for %arg12 = %c0 to %dim step %1 iter_args(%arg13 = %5) -> (tensor<?x?xf32>) {
        %11 = affine.min #map(%arg12)[%1, %dim]
        %extracted_slice_3 = tensor.extract_slice %extracted_slice[%arg12, 0] [%11, %9] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %12 = tensor.empty(%11, %9) : tensor<?x?xf32>
        %13 = linalg.copy {"DDR : GSM"} ins(%extracted_slice_3 : tensor<?x?xf32>) outs(%12 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %extracted_slice_4 = tensor.extract_slice %arg13[%arg12, 0] [%11, %dim_0] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %14 = scf.for %arg14 = %c0 to %dim_0 step %2 iter_args(%arg15 = %extracted_slice_4) -> (tensor<?x?xf32>) {
          %15 = affine.min #map(%arg14)[%2, %dim_0]
          %extracted_slice_6 = tensor.extract_slice %extracted_slice_2[0, %arg14] [%9, %15] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
          %16 = tensor.empty(%9, %15) : tensor<?x?xf32>
          %17 = linalg.copy {"DDR : AM"} ins(%extracted_slice_6 : tensor<?x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
          %extracted_slice_7 = tensor.extract_slice %arg15[0, %arg14] [%11, %15] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
          %18 = scf.for %arg16 = %c0 to %11 step %3 iter_args(%arg17 = %extracted_slice_7) -> (tensor<?x?xf32>) {
            %19 = affine.min #map(%arg16)[%3, %11]
            %extracted_slice_9 = tensor.extract_slice %extracted_slice_3[%arg16, 0] [%19, %9] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
            %extracted_slice_10 = tensor.extract_slice %arg17[%arg16, 0] [%19, %15] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
            %20 = tensor.empty(%19, %15) : tensor<?x?xf32>
            %21 = linalg.copy {"DDR : AM"} ins(%extracted_slice_10 : tensor<?x?xf32>) outs(%20 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %22 = scf.for %arg18 = %c0 to %19 step %4 iter_args(%arg19 = %21) -> (tensor<?x?xf32>) {
              %25 = affine.min #map(%arg18)[%4, %19]
              %extracted_slice_14 = tensor.extract_slice %extracted_slice_9[%arg18, 0] [%25, %9] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
              %26 = tensor.empty(%25, %9) : tensor<?x?xf32>
              %27 = linalg.copy {"GSM : SM"} ins(%extracted_slice_14 : tensor<?x?xf32>) outs(%26 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %extracted_slice_15 = tensor.extract_slice %arg19[%arg18, 0] [%25, %15] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
              %28 = linalg.matmul ins(%27, %extracted_slice_6 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%extracted_slice_15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %inserted_slice_16 = tensor.insert_slice %28 into %arg19[%arg18, 0] [%25, %15] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
              scf.yield %inserted_slice_16 : tensor<?x?xf32>
            } {__tiled_for___4}
            %dim_11 = tensor.dim %22, %c0 : tensor<?x?xf32>
            %dim_12 = tensor.dim %22, %c1 : tensor<?x?xf32>
            %23 = tensor.empty(%dim_11, %dim_12) : tensor<?x?xf32>
            %24 = linalg.copy {"AM : DDR"} ins(%22 : tensor<?x?xf32>) outs(%23 : tensor<?x?xf32>) -> tensor<?x?xf32>
            %inserted_slice_13 = tensor.insert_slice %24 into %arg17[%arg16, 0] [%19, %15] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
            scf.yield %inserted_slice_13 : tensor<?x?xf32>
          } {__tiled_for___3}
          %inserted_slice_8 = tensor.insert_slice %18 into %arg15[0, %arg14] [%11, %15] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
          scf.yield %inserted_slice_8 : tensor<?x?xf32>
        } {__tiled_for___2}
        %inserted_slice_5 = tensor.insert_slice %14 into %arg13[%arg12, 0] [%11, %dim_0] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
        scf.yield %inserted_slice_5 : tensor<?x?xf32>
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

#map = affine_map<(d0)[s0, s1] -> (-d0 + s1, s0)>
module {
  func.func @matmul_elemwise_0_tiling(%arg0: memref<?x?xf32, strided<[?, ?], offset: ?>>, %arg1: memref<?x?xf32, strided<[?, ?], offset: ?>>, %arg2: memref<?x?xf32, strided<[?, ?], offset: ?>>, %arg3: memref<?x?xf32, strided<[?, ?], offset: ?>>, %arg4: memref<?x?xf32, strided<[?, ?], offset: ?>>, %arg5: i64 {mtfusion.tiling_axis = #mtfusion.tiling_axis<K>, mtfusion.tiling_data}, %arg6: i64 {mtfusion.copy_dst = #mtfusion.copy_dst<GSM>, mtfusion.copy_mat = #mtfusion.copy_mat<MatA>, mtfusion.tiling_axis = #mtfusion.tiling_axis<M>, mtfusion.tiling_data}, %arg7: i64 {mtfusion.copy_dst = #mtfusion.copy_dst<AM>, mtfusion.copy_mat = #mtfusion.copy_mat<MatB>, mtfusion.nthreads = #mtfusion.nthreads<8>, mtfusion.tiling_axis = #mtfusion.tiling_axis<N>, mtfusion.tiling_data}, %arg8: i64 {mtfusion.copy_dst = #mtfusion.copy_dst<AM>, mtfusion.copy_mat = #mtfusion.copy_mat<MatC>, mtfusion.tiling_axis = #mtfusion.tiling_axis<M>, mtfusion.tiling_data}, %arg9: i64 {mtfusion.copy_dst = #mtfusion.copy_dst<SM>, mtfusion.copy_mat = #mtfusion.copy_mat<MatA>, mtfusion.tiling_axis = #mtfusion.tiling_axis<M>, mtfusion.tiling_data}) -> memref<?x?xf32, strided<[?, ?], offset: ?>> attributes {mtfusion.entry, mtfusion.function_kind = #mtfusion.function_kind<Device>, mtfusion.fusion_kind = #mtfusion.fusion_kind<MIX_CV>} {
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

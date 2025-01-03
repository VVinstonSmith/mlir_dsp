
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



mtir-opt matmul_elemwise.mlir -tiling-pass= \
   "{-tiling-axis=k}
    {-tiling-axis=m -copy-mat = A ; -copy-dst= GSM}
    {-tiling-axis=n ; -nthread = 8 ; -copy-mat= B, copy-dst = AM}
    {-tiling-axis = m ; -copy-mat = C ; -copy-dst = AM}
    {-tiling-axis = m ; -copy-mat = A ; -copy-dst = SM}"

func.func @matmul_kernel(
        %arg0: tensor<1024x1024xf32>, %arg1: tensor<1024x1024xf32>, %arg2: tensor<1024x1024xf32>,  
        %arg3: i64 {mtfusion.tiling_data, tiling_axis=#mtfusion.axis<N>, mtfusion.parallel_dim},
        %arg4: i64 {mtfusion.tiling_data, tiling_axis=#mtfusion.axis<K>},
        %arg5: i64 {mtfusion.tiling_data, tiling_axis=#mtfusion.axis<M>, copy_data={#mtfusion.matrix<A>}, copy_destinies={#mtfusion.memory<GSM>}}, 
        %arg6: i64 {mtfusion.tiling_data, tiling_axis=#mtfusion.axis<N>, copy_data={#mtfusion.matrix<B>}, copy_destinies={#mtfusion.memory<AM>}}, 
        %arg7: i64 {mtfusion.tiling_data, tiling_axis=#mtfusion.axis<M>, copy_data={#mtfusion.matrix<C>}, copy_destinies={#mtfusion.memory<AM>}}, 
        %arg8: i64 {mtfusion.tiling_data, tiling_axis=#mtfusion.axis<M>, copy_data={#mtfusion.matrix<A>}, copy_destinies={#mtfusion.memory<SM>}}
        ) attributes {
        always_inline = #hfusion.always_inline, 
        mtfusion.function_kind = #hfusion.function_kind<Device>, 
        mtfusion.block_dim = 8 : i64,
        hfusion.fusion_kind = #hfusion.fusion_kind<MixCV>} 
{
    %0 = tensor.empty() : tensor<1024x1024xf32>
    %1 = linalg.copy {__cache_read__} ins(%arg0 : tensor<1024x1024xf32>) outs(%0 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %2 = linalg.copy {__cache_read__} ins(%arg1 : tensor<1024x1024xf32>) outs(%0 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %3 = linalg.matmul ins(%1, %2: tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%0 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %4 = linalg.copy {__cache_write__} ins(%3 : tensor<1024x1024xf32>) outs(%arg2 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    return %4 : tensor<1024x1024xf32>
}


## original version
func @matmul_elemwise(%argA, %argB, %argC, %argD, %argR)
{
    %empty = empty()
    %tmp_1 = matmul ins(%argA, %argB) outs(%empty)
    %tmp_2 = elemwise ins(%tmp_1, %argC) outs(%empty)
    %final = elemwise ins(%tmp_2, %argD) outs(%argR)
    return %final
}

## caching
{
    %empty = empty()
    %A = copy ins(%argA) outs(%empty)
    %B = copy ins(%argB) outs(%empty)
    %C = copy ins(%argC) outs(%empty)
    %D = copy ins(%argD) outs(%empty)
    %tmp_1 = matmul ins(%A, %B) outs(%empty)
    %tmp_2 = elemwise ins(%tmp_1, %C) outs(%empty)
    %tmp_3 = elemwise ins(%tmp_2, %D) outs(%empty)
    %final = copy ins(%tmp_3) outs(%argR)
    return %final
}

## tiling ops in dim N
{
    %empty = empty()
    %A = copy ins(%argA) outs(%empty)
    %B = copy ins(%argB) outs(%empty)
    %C = copy ins(%argC) outs(%empty)
    %D = copy ins(%argD) outs(%empty)
    scf.for {
        %B_n = extract_slice %B
        %empty_n = extract_slice %empty
        %tmp_1_n = matmul ins(%A, %B_n) outs(%empty_n)
        %tmp_1 = insert_slice %tmp_1_n into %empty
        scf.yield %tmp_1
    }
    %tmp_2 = elemwise ins(%tmp_1, %C) outs(%empty)
    %tmp_3 = elemwise ins(%tmp_2, %D) outs(%empty)
    %final = copy ins(%tmp_3) outs(%argR)
    return %final
}

## fusing ops in dim N
{
    %empty = empty()
    %A = copy ins(%argA) outs(%empty)
    %B = copy ins(%argB) outs(%empty)
    %C = copy ins(%argC) outs(%empty)
    %D = copy ins(%argD) outs(%empty)
    scf.for {
        %B_n = extract_slice %B
        %empty_n = extract_slice %empty
        %tmp_1_n = matmul ins(%A, %B_n) outs(%empty_n)
        %C_n = extract_slice %C
        %tmp_2_n = elemwise ins(%tmp_1_n, %C_n) outs(%empty_n)
        %D_n = extract_slice %D
        %tmp_3_n = elemwise ins(%tmp_2_n, %D_n) outs(%empty_n)
        %tmp_3 = insert_slice %tmp_3_n into %empty
        scf.yield %tmp_3
    }
    %final = copy ins(%tmp_3) outs(%argR)
    return %final
}

## fusing copies in dim N
{
    %empty = empty()
    %A = copy ins(%argA) outs(%empty)
    scf.for {
        %empty_n = extract_slice %empty
        %argB_n = extract_slice %argB
        %argC_n = extract_slice %argC
        %argD_n = extract_slice %argD
        %B_n = copy ins(%argB_n) outs(%empty_n)
        %C_n = copy ins(%argC_n) outs(%empty_n)
        %D_n = copy ins(%argD_n) outs(%empty_n)
        %tmp_1_n = matmul ins(%A, %B_n) outs(%empty_n)
        %tmp_2_n = elemwise ins(%tmp_1_n, %C_n) outs(%empty_n)
        %tmp_3_n = elemwise ins(%tmp_2_n, %D_n) outs(%empty_n)
        %argR_n = extract_slice %argR
        %final_n = copy ins(%tmp_3_n) outs(%argR_n)
        %final = insert_slice %final_n into %argR
        scf.yield %final
    }
    return %final
}

## tiling ops in dim M
{
    %empty = empty()
    %A = copy ins(%argA) outs(%empty)
    scf.for {
        %empty_n = extract_slice %empty
        %argB_n = extract_slice %argB
        %argC_n = extract_slice %argC
        %argD_n = extract_slice %argD
        %argR_n = extract_slice %argR
        %B_n = copy ins(%argB_n) outs(%empty_n)
        %C_n = copy ins(%argC_n) outs(%empty_n)
        %D_n = copy ins(%argD_n) outs(%empty_n)
        scf.for {
            %empty_n_m = extract_slice %empty_n
            %A_m = extract_slice %A
            %tmp_1_n_m = matmul ins(%A_m, %B_n) outs(%empty_n_m)
            %tmp_1_n = insert_slice %tmp_1_n_m into %empty_n
            scf.yield %tmp_1_n
        }
        %tmp_2_n = elemwise ins(%tmp_1_n, %C_n) outs(%empty_n)
        %tmp_3_n = elemwise ins(%tmp_2_n, %D_n) outs(%empty_n)
        %final_n = copy ins(%tmp_3_n) outs(%argR_n)
        %final = insert_slice %final_n into %argR
        scf.yield %final
    }
    return %final
}

## fusing ops in dim M
{
    %empty = empty()
    %A = copy ins(%argA) outs(%empty)
    scf.for {
        %empty_n = extract_slice %empty
        %argB_n = extract_slice %argB
        %argC_n = extract_slice %argC
        %argD_n = extract_slice %argD
        %argR_n = extract_slice %argR
        %B_n = copy ins(%argB_n) outs(%empty_n)
        %C_n = copy ins(%argC_n) outs(%empty_n)
        %D_n = copy ins(%argD_n) outs(%empty_n)
        scf.for {
            %empty_n_m = extract_slice %empty_n
            %A_m = extract_slice %A
            %tmp_1_n_m = matmul ins(%A_m, %B_n) outs(%empty_n_m)
            %C_n_m = extract_slice %C_n
            %tmp_2_n_m = elemwise ins(%tmp_1_n_m, %C_n_m) outs(%empty_n_m)
            %D_n_m = extract_slice %D_n
            %tmp_3_n_m = elemwise ins(%tmp_2_n_m, %D_n_m) outs(%empty_n_m)
            %tmp_3_n = insert_slice %tmp_3_n_m into %empty_n
            scf.yield %tmp_3_n
        }
        %final_n = copy ins(%tmp_3_n) outs(%argR_n)
        %final = insert_slice %final_n into %argR
        scf.yield %final
    }
    return %final
}

## fusing copies in dim M
{
    %empty = empty()
    scf.for {
        %empty_n = extract_slice %empty
        %argB_n = extract_slice %argB
        %argC_n = extract_slice %argC
        %argD_n = extract_slice %argD
        %argR_n = extract_slice %argR
        %B_n = copy ins(%argB_n) outs(%empty_n)
        %C_n = copy ins(%argC_n) outs(%empty_n)
        %D_n = copy ins(%argD_n) outs(%empty_n)
        scf.for {
            %empty_m = extract_slice %empty
            %empty_n_m = extract_slice %empty_n
            %argA_m = extract_slice %argA
            %argC_n_m = extract_slice %argC_n
            %argD_n_m = extract_slice %argD_n
            %A_m = copy ins(%argA_m) outs(%empty_m)
            %C_n_m = copy ins(%argC_n_m) outs(%empty_n_m)
            %D_n_m = copy ins(%argD_n_m) outs(%empty_n_m)
            %tmp_1_n_m = matmul ins(%A_m, %B_n) outs(%empty_n_m)
            %tmp_2_n_m = elemwise ins(%tmp_1_n_m, %C_n_m) outs(%empty_n_m)
            %tmp_3_n_m = elemwise ins(%tmp_2_n_m, %D_n_m) outs(%empty_n_m)
            %argR_n_m = extract_slice %argR_n
            %final_n_m = copy ins(%tmp_3_n_m) outs(%argR_n_m)
            %final_n = insert_slice %final_n_m into %argR_n
            scf.yield %final_n
        }
        %final = insert_slice %final_n into %argR
        scf.yield %final
    }
    return %final
}


## tiling ops, fusing ops, fusing copies in dim K
{
    %empty = empty()
    scf.for {
        %empty_n = extract_slice %empty
        %argB_n = extract_slice %argB
        %argC_n = extract_slice %argC
        %argD_n = extract_slice %argD
        %argR_n = extract_slice %argR
        scf.for {
            %empty_m = extract_slice %empty
            %empty_n_m = extract_slice %empty_n
            %argA_m = extract_slice %argA
            %argC_n_m = extract_slice %argC_n
            %argD_n_m = extract_slice %argD_n
            %C_n_m = copy ins(%argC_n_m) outs(%empty_n_m)
            %D_n_m = copy ins(%argD_n_m) outs(%empty_n_m)
            scf.for (%arg0 = %empty_n_m){
                %empty_m_k = extract_slice %empty_m
                %empty_n_k = extract_slice %empty_n
                %argA_m_k = extract_slice %argA_m
                %argB_n_k = extract_slice %argB_n
                %A_m_k = copy ins(%argA_m_k) outs(%empty_m_k)
                %B_n_k = copy ins(%argB_n_k) outs(%empty_n_k)
                %tmp_1_n_m = matmul ins(%A_m_k, %B_n_k) outs(%arg0)
                scf.yield %tmp_1_n_m
            }
            %tmp_2_n_m = elemwise ins(%tmp_1_n_m, %C_n_m) outs(%empty_n_m)
            %tmp_3_n_m = elemwise ins(%tmp_2_n_m, %D_n_m) outs(%empty_n_m)
            %argR_n_m = extract_slice %argR_n
            %final_n_m = copy ins(%tmp_3_n_m) outs(%argR_n_m)
            %final_n = insert_slice %final_n_m into %argR_n
            scf.yield %final_n
        }
        %final = insert_slice %final_n into %argR
        scf.yield %final
    }
    return %final
}

## tiling ops in dim N_G
{
    %empty = empty()
    scf.for {
        %empty_n = extract_slice %empty
        %argB_n = extract_slice %argB
        %argC_n = extract_slice %argC
        %argD_n = extract_slice %argD
        %argR_n = extract_slice %argR
        scf.for {
            %empty_m = extract_slice %empty
            %empty_n_m = extract_slice %empty_n
            %argA_m = extract_slice %argA
            %argC_n_m = extract_slice %argC_n
            %argD_n_m = extract_slice %argD_n
            %C_n_m = copy ins(%argC_n_m) outs(%empty_n_m)
            %D_n_m = copy ins(%argD_n_m) outs(%empty_n_m)
            scf.for (%arg0 = %empty_n_m){
                %empty_m_k = extract_slice %empty_m
                %empty_n_k = extract_slice %empty_n
                %argA_m_k = extract_slice %argA_m
                %argB_n_k = extract_slice %argB_n
                %A_m_k = copy ins(%argA_m_k) outs(%empty_m_k)
                %B_n_k = copy ins(%argB_n_k) outs(%empty_n_k)
                scf.for {
                    %arg0_n = extract_slice %arg0
                    %B_n_k_n = extract_slice %B_n_k
                    %tmp_1_n_m_n = matmul ins(%A_m_k, %B_n_k_n) outs(%arg0_n)
                    %tmp_1_n_m = insert_slice %tmp_1_n_m_n into %arg0
                    scf.yield %tmp_1_n_m
                }
                scf.yield %tmp_1_n_m
            }
            %tmp_2_n_m = elemwise ins(%tmp_1_n_m, %C_n_m) outs(%empty_n_m)
            %tmp_3_n_m = elemwise ins(%tmp_2_n_m, %D_n_m) outs(%empty_n_m)
            %argR_n_m = extract_slice %argR_n
            %final_n_m = copy ins(%tmp_3_n_m) outs(%argR_n_m)
            %final_n = insert_slice %final_n_m into %argR_n
            scf.yield %final_n
        }
        %final = insert_slice %final_n into %argR
        scf.yield %final
    }
    return %final
}

## fusing copies in dim N_G
{
    %empty = empty()
    scf.for {
        %empty_n = extract_slice %empty
        %argB_n = extract_slice %argB
        %argC_n = extract_slice %argC
        %argD_n = extract_slice %argD
        %argR_n = extract_slice %argR
        scf.for {
            %empty_m = extract_slice %empty
            %empty_n_m = extract_slice %empty_n
            %argA_m = extract_slice %argA
            %argC_n_m = extract_slice %argC_n
            %argD_n_m = extract_slice %argD_n
            %C_n_m = copy ins(%argC_n_m) outs(%empty_n_m)
            %D_n_m = copy ins(%argD_n_m) outs(%empty_n_m)
            scf.for (%arg0 = %empty_n_m){
                %empty_m_k = extract_slice %empty_m
                %empty_n_k = extract_slice %empty_n
                %argA_m_k = extract_slice %argA_m
                %argB_n_k = extract_slice %argB_n
                %A_m_k = copy ins(%argA_m_k) outs(%empty_m_k) {A : DDR -> GSM}
                scf.for {
                    %arg0_n = extract_slice %arg0
                    %argB_n_k_n = extract_slice %argB_n_k
                    %empty_n_k_n = extract_slice %empty_n_k
                    %B_n_k_n = copy ins(%argB_n_k_n) outs(%empty_n_k_n) {B : DDR -> AM}
                    %tmp_1_n_m_n = matmul ins(%A_m_k, %B_n_k_n) outs(%arg0_n)
                    %tmp_1_n_m = insert_slice %tmp_1_n_m_n into %arg0
                    scf.yield %tmp_1_n_m
                }
                scf.yield %tmp_1_n_m
            }
            %tmp_2_n_m = elemwise ins(%tmp_1_n_m, %C_n_m) outs(%empty_n_m)
            %tmp_3_n_m = elemwise ins(%tmp_2_n_m, %D_n_m) outs(%empty_n_m)
            %argR_n_m = extract_slice %argR_n
            %final_n_m = copy ins(%tmp_3_n_m) outs(%argR_n_m)
            %final_n = insert_slice %final_n_m into %argR_n
            scf.yield %final_n
        }
        %final = insert_slice %final_n into %argR
        scf.yield %final
    }
    return %final
}

## tiling ops in dim M_G
{
    %empty = empty()
    scf.for {0:N_G:N} {
        %empty_n = extract_slice %empty
        %argB_n = extract_slice %argB
        %argC_n = extract_slice %argC
        %argD_n = extract_slice %argD
        %argR_n = extract_slice %argR
        scf.for {0:M_G:M} {
            %empty_m = extract_slice %empty
            %empty_n_m = extract_slice %empty_n
            %argA_m = extract_slice %argA
            %argC_n_m = extract_slice %argC_n
            %argD_n_m = extract_slice %argD_n
            %C_n_m = copy ins(%argC_n_m) outs(%empty_n_m)
            %D_n_m = copy ins(%argD_n_m) outs(%empty_n_m)
            scf.for (%arg0_n_m = %empty_n_m) {0:K_G:K} {
                %empty_m_k = extract_slice %empty_m
                %empty_n_k = extract_slice %empty_n
                %argA_m_k = extract_slice %argA_m
                %argB_n_k = extract_slice %argB_n
                %A_m_k = copy ins(%argA_m_k) outs(%empty_m_k) {A : DDR -> GSM}
                scf.for (%arg0_n_m) {0:N_A:N_G} {
                    %empty_n_k_n = extract_slice %empty_n_k
                    %empty_n_m_n = extract_slice %empty_n_m
                    %argB_n_k_n = extract_slice %argB_n_k
                    %arg0_n_m_n = extract_slice %arg0_n_m
                    %B_n_k_n = copy ins(%argB_n_k_n) outs(%empty_n_k_n) {B : DDR -> AM}
                    scf.for (%arg0_n_m_n) {0:M_A:M_G} {
                        %empty_n_m_n_m = extract_slice %empty_n_m_n
                        %A_m_k_m = extract_slice %A_m_k
                        %arg0_n_m_n_m_in = extract_slice %arg0_n_m_n
                        %arg0_n_m_n_m_out = matmul ins(%A_m_k_m, %B_n_k_n) outs(%arg0_n_m_n_m_in)
                        %tmp_1_n_m_n = insert_slice %arg0_n_m_n_m_out into %arg0_n_m_n
                        scf.yield %tmp_1_n_m_n
                    }
                    %tmp_1_n_m = insert_slice %tmp_1_n_m_n into %arg0_n_m
                    scf.yield %tmp_1_n_m
                }
                scf.yield %tmp_1_n_m
            }
            %tmp_2_n_m = elemwise ins(%tmp_1_n_m, %C_n_m) outs(%empty_n_m)
            %tmp_3_n_m = elemwise ins(%tmp_2_n_m, %D_n_m) outs(%empty_n_m)
            %argR_n_m = extract_slice %argR_n
            %final_n_m = copy ins(%tmp_3_n_m) outs(%argR_n_m)
            %final_n = insert_slice %final_n_m into %argR_n
            scf.yield %final_n
        }
        %final = insert_slice %final_n into %argR
        scf.yield %final
    }
    return %final
}


## adding copies in dim M_G
{
    %empty = empty()
    scf.for {0:N_G:N} {
        %empty_n = extract_slice %empty
        %argB_n = extract_slice %argB
        %argC_n = extract_slice %argC
        %argD_n = extract_slice %argD
        %argR_n = extract_slice %argR
        scf.for {0:M_G:M} {
            %empty_m = extract_slice %empty
            %empty_n_m = extract_slice %empty_n
            %argA_m = extract_slice %argA
            %argC_n_m = extract_slice %argC_n
            %argD_n_m = extract_slice %argD_n
            %C_n_m = copy ins(%argC_n_m) outs(%empty_n_m)
            %D_n_m = copy ins(%argD_n_m) outs(%empty_n_m)
            scf.for (%arg0_n_m = %empty_n_m) {0:K_G:K} {
                %empty_m_k = extract_slice %empty_m
                %empty_n_k = extract_slice %empty_n
                %argA_m_k = extract_slice %argA_m
                %argB_n_k = extract_slice %argB_n
                %A_m_k = copy ins(%argA_m_k) outs(%empty_m_k) {A : DDR -> GSM}
                scf.for (%arg0_n_m) {0:N_A:N_G} {
                    %empty_n_k_n = extract_slice %empty_n_k
                    %empty_n_m_n = extract_slice %empty_n_m
                    %argB_n_k_n = extract_slice %argB_n_k
                    %arg0_n_m_n = extract_slice %arg0_n_m
                    %B_n_k_n = copy ins(%argB_n_k_n) outs(%empty_n_k_n) {B : DDR -> AM}
                    scf.for (%arg0_n_m_n) {0:M_A:M_G} {
                        %empty_n_m_n_m = extract_slice %empty_n_m_n
                        %A_m_k_m = extract_slice %A_m_k
                        %arg0_n_m_n_m = extract_slice %arg0_n_m_n
                        %Ca_in = copy ins(%arg0_n_m_n_m) outs(%empty_n_m_n_m) {C : DDR -> AM}
                        %Ca_out = matmul ins(%A_m_k_m, %B_n_k_n) outs(%Ca_in)
                        %tmp_1_n_m_n_m = copy ins(%Ca_out) outs(%arg0_n_m_n_m) {C : AM -> DDR}
                        %tmp_1_n_m_n = insert_slice %tmp_1_n_m_n_m into %arg0_n_m_n
                        scf.yield %tmp_1_n_m_n
                    }
                    %tmp_1_n_m = insert_slice %tmp_1_n_m_n into %arg0_n_m
                    scf.yield %tmp_1_n_m
                }
                scf.yield %tmp_1_n_m
            }
            %tmp_2_n_m = elemwise ins(%tmp_1_n_m, %C_n_m) outs(%empty_n_m)
            %tmp_3_n_m = elemwise ins(%tmp_2_n_m, %D_n_m) outs(%empty_n_m)
            %argR_n_m = extract_slice %argR_n
            %final_n_m = copy ins(%tmp_3_n_m) outs(%argR_n_m)
            %final_n = insert_slice %final_n_m into %argR_n
            scf.yield %final_n
        }
        %final = insert_slice %final_n into %argR
        scf.yield %final
    }
    return %final
}

## tiling ops, adding copies in dim M_S
{
    %empty = empty()
    scf.for {0:N_G:N} {
        %empty_n = extract_slice %empty
        %argB_n = extract_slice %argB
        %argC_n = extract_slice %argC
        %argD_n = extract_slice %argD
        %argR_n = extract_slice %argR
        scf.for {0:M_G:M} {
            %empty_m = extract_slice %empty
            %empty_n_m = extract_slice %empty_n
            %argA_m = extract_slice %argA
            %argC_n_m = extract_slice %argC_n
            %argD_n_m = extract_slice %argD_n
            %C_n_m = copy ins(%argC_n_m) outs(%empty_n_m)
            %D_n_m = copy ins(%argD_n_m) outs(%empty_n_m)
            scf.for (%arg0_n_m = %empty_n_m) {0:K_G:K} {
                %empty_m_k = extract_slice %empty_m
                %empty_n_k = extract_slice %empty_n
                %argA_m_k = extract_slice %argA_m
                %argB_n_k = extract_slice %argB_n
                %A_m_k = copy ins(%argA_m_k) outs(%empty_m_k) {A : DDR -> GSM}
                scf.for (%arg0_n_m) {0:N_A:N_G} {
                    %empty_n_k_n = extract_slice %empty_n_k
                    %empty_n_m_n = extract_slice %empty_n_m
                    %argB_n_k_n = extract_slice %argB_n_k
                    %arg0_n_m_n = extract_slice %arg0_n_m
                    %B_n_k_n = copy ins(%argB_n_k_n) outs(%empty_n_k_n) {B : DDR -> AM}
                    scf.for (%arg0_n_m_n) {0:M_A:M_G} {
                        %empty_n_m_n_m = extract_slice %empty_n_m_n
                        %arg0_n_m_n_m = extract_slice %arg0_n_m_n
                        %Ca_in = copy ins(%arg0_n_m_n_m) outs(%empty_n_m_n_m) {C : DDR -> AM}
                        scf.for (%Ca_in) {0:M_S:M_A} {
                            %empty_m_k_m_m = extract_slice %empty_m_k {2d}
                            %A_m_k_m_m = extract_slice %A_m_k {2d}
                            %arg0_n_m_n_m_m = extract_slice %Ca_in
                            %As = copy ins(%A_m_k_m_m) outs(%empty_m_k_m_m) {A : GSM -> SM}
                            %tmp_1_n_m_n_m_m = matmul ins(%As, %B_n_k_n) outs(%arg0_n_m_n_m_m)
                            %Ca_out = insert_slice %tmp_1_n_m_n_m_m into %Ca_in
                            scf.yield %Ca_out
                        }
                        %tmp_1_n_m_n_m = copy ins(%Ca_out) outs(%arg0_n_m_n_m) {C : AM -> DDR}
                        %tmp_1_n_m_n = insert_slice %tmp_1_n_m_n_m into %arg0_n_m_n
                        scf.yield %tmp_1_n_m_n
                    }
                    %tmp_1_n_m = insert_slice %tmp_1_n_m_n into %arg0_n_m
                    scf.yield %tmp_1_n_m
                }
                scf.yield %tmp_1_n_m
            }
            %tmp_2_n_m = elemwise ins(%tmp_1_n_m, %C_n_m) outs(%empty_n_m)
            %tmp_3_n_m = elemwise ins(%tmp_2_n_m, %D_n_m) outs(%empty_n_m)
            %argR_n_m = extract_slice %argR_n
            %final_n_m = copy ins(%tmp_3_n_m) outs(%argR_n_m)
            %final_n = insert_slice %final_n_m into %argR_n
            scf.yield %final_n
        }
        %final = insert_slice %final_n into %argR
        scf.yield %final
    }
    return %final
}


// scf.for (%arg0_n_m_n) {0:M_A:M_G} {
//     %arg0_n_m_n_m = extract_slice %arg0_n_m_n
//     %Ca_in = copy ins(%arg0_n_m_n_m) outs(%empty) {C : DDR -> AM}
//     %empty = empty()
//     scf.for (%Ca_in) {0:M_S:M_A} {
//         %A_am_sm = extract_slice %A_am
//         %Ca_in_sm = extract_slice %Ca_in
//         %A_sm = copy ins(%A_am_sm) outs(%empty) {A : GSM -> SM}
//         %tmp = matmul ins(%A_sm, %B_am) outs(%Ca_in_sm)
//         %Ca_out = insert_slice %tmp into %Ca_in
//         scf.yield %Ca_out
//     }
//     %tmp_1_n_m_n_m = copy ins(%Ca_out) outs(%arg0_n_m_n_m) {C : AM -> DDR}
//     %tmp_1_n_m_n = insert_slice %tmp_1_n_m_n_m into %arg0_n_m_n
//     scf.yield %tmp_1_n_m_n
// }


## original version
%empty = empty()
scf.for (%C = %C0) {0:M_S:M_A} {
    %A_slice = extract_slice %A
    %A_buffer = copy ins(%A_slice) outs(%empty)
    %C_slice = extract_slice %C
    %output = matmul ins(%A_buffer, %B) outs(%C_slice)
    %C_new = insert_slice %output into %C
    scf.yield %C_new
}


## after double buffering
%empty = empty()

%A_slice_0 = extract_slice %A
%A_buffer_0 = copy ins(%A_slice_0) outs(%empty_0)
%C_slice_0 = extract_slice %C0

scf.for (%C = %C0, %A_buffer = %A_buffer_0, %C_slice = %C_slice_0) {M_S:M_S:M_A} {
    %output = matmul ins(%A_buffer, %B) outs(%C_slice)
    %C_new = insert_slice %output into %C

    %A_slice_next = extract_slice %A
    %A_buffer_next = copy ins(%A_slice_next) outs(%empty)
    %C_slice_next = extract_slice %C_new
    
    scf.yield %C_new, %A_buffer_next, %C_slice_next
}

%output_final = matmul ins(%A_buffer_next, %B) outs(%C_slice_next)
%C_final = insert_slice %output_final into %C_new


## after double buffering


#map = affine_map<(d0)[s0, s1] -> (-d0 + s1, s0)>
func.func @test_inner_loop(
    %A : tensor<?x?xf32>, %B : tensor<?x?xf32>, %C0 : tensor<?x?xf32>,
    %arg_m_tile)
{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m_len = tensor.dim %A, %c0 : tensor<?x?xf32>
    %n_len = tensor.dim %B, %c1 : tensor<?x?xf32>
    %k_len = tensor.dim %B, %c0 : tensor<?x?xf32>
    %m_tile = arith.index_cast %arg_m_tile : i64 to index

    %empty = tensor.empty(%m_tile, %k_len) : tensor<?x?xf32>
    %C_final = scf.for %pos = %c0 to %m_len step %m_tile iter_args(%C = %C0) -> (tensor<?x?xf32>) {
        %m_tile_cur = affine.min #map(%pos)[%m_tile, %m_len]
        %A_slice = tensor.extract_slice %A[%pos] [%m_tile_cur] [1] tensor<?x?xf32> to tensor<?x?xf32>
        %A_buffer = linalg.copy ins(%A_slice : tensor<?x?xf32>) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
        %C_slice = tensor.extract_slice %C[%pos] [%m_tile_cur] [1] tensor<?x?xf32> to tensor<?x?xf32>
        %output = linalg.matmul ins(%A_buffer, %B : tensor<?x?xf32>, tensor<?x?xf32>) outs(%C_slice : tensor<?x?xf32>) -> tensor<?x?xf32>
        %C_new = insert_slice %output into %C[%pos] [%m_tile_cur] [1] : tensor<?x?xf32> into tensor<?x?xf32>
        scf.yield %C_new : tensor<?x?xf32>
    }
}



// for i = 0:M_S:M_A
//     A(M_S*K_A) -> A_buffer
//     call micro_kernel(A_buffer, B, C_slice)
// end for

%empty = empty()

%A_slice_0 = extract_slice %A
%A_buffer_0 = copy ins(%A_slice_0) outs(%empty)
%C_slice_0 = extract_slice %C0

scf.for (%C = %C0, %A_buffer = %A_buffer_0, %C_slice = %C_slice_0) {M_S:M_S:M_A} {
    %A_slice_next = extract_slice %A
    %A_buffer_next = copy ins(%A_slice_next) outs(%empty)
    %C_slice_next = extract_slice %C

    %output = matmul ins(%A_buffer, %B) outs(%C_slice)
    %C_new = insert_slice %output into %C
    
    scf.yield %C_new, %A_buffer_next, %C_slice_next
}

%output_final = matmul ins(%A_buffer_next, %B) outs(%C_slice_next)
%C_final = insert_slice %output_final into %C_new




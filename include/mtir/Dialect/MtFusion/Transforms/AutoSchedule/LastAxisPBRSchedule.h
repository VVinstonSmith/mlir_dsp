//===- LastAxisPBRSchedule.h -- Schedule for Last Axis PBR Op ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MTIR_DIALECT_MTFUSION_TRANSFORMS_AUTOSCHEDULE_LASTAXISPBRSCHEDULE_H
#define MTIR_DIALECT_MTFUSION_TRANSFORMS_AUTOSCHEDULE_LASTAXISPBRSCHEDULE_H

#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/AutoScheduleBase.h"
#include "mtir/Dialect/MtFusion/IR/MtFusion.h"
#include "mtir/Dialect/MtFusion/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
struct LogicalResult;
class Location;
class OpBuilder;

namespace mtfusion {

class LastAxisPBRKernelInfo : public KernelInfo {
public:
  LastAxisPBRKernelInfo(MLIRContext *ctx)
      : KernelInfo(FusionKind::LastAxisPBR, ctx) {}

  static bool classof(const KernelInfo *T) {
    return T->getFusionKind() == FusionKind::LastAxisPBR;
  }

  /// Number of parallel dimensions that need to be tiled.
  size_t tileableParallelDimSize{0};
  /// Number of reduction dimensions that need to be tiled.
  /// Reduction
  size_t tilebaleReductionDimSize{0};
  /// Total number of tileable dimensions is the sum of the two.
  size_t totalTilableDimSize{0};

  /// The input value idx with the largest dimension size.
  size_t inputValueIdxWithHighestOrderDim{0};
};

/// Scheduler for kernels with last axis reduction operations and other
/// elemwise/broadcast operations.
class LastAxisPBRScheduler : public SchedulerBase {

public:
  explicit LastAxisPBRScheduler(func::FuncOp funcOpIn)
      : SchedulerBase(
            funcOpIn,
            std::make_unique<LastAxisPBRKernelInfo>(funcOpIn->getContext()),
            std::make_unique<TilingInfo>()){};

  /// Implementation of kernel analysis and verification.
  LogicalResult analyzeAndVerifyKernelImpl() override;

  /// Implementation of tiling case and schedule sequence generation.
  ///
  /// Tiling Case #0 ~ #N-2 (Split-N):
  /// Only tile the parallel dims.
  ///
  /// \code
  /// for block.idx in block.dim:
  ///   for ub.idx in ub_loop_cnt:
  ///     copyIn(ub_buffer_size, D)
  ///     compute(ub_buffer_size, D)
  ///     copyOut(ub_buffer_size, D)
  /// \endcode
  ///
  /// Tiling Case #N-1 (Split-D):
  ///
  /// \code
  /// for block.idx in block.dim:
  ///   for ub.idx in ub_n_loop_cnt:
  ///     for r.idx in ub_d_loop_cnt:
  ///       copyIn(1, rfactor_size)
  ///       compute(1, rfactor_size)
  ///
  ///     reduce(1, rfactor_size)
  ///     compute(1, 1)
  ///     copyOut(1, 1)
  ///
  ///     for d.idx in ub_d_loop_cnt:
  ///       copyIn(1, rfactor_size)
  ///       compute(1, rfactor_size)
  ///       copyOut(1, rfactor_size)
  /// \endcode
  ///
  /// Tiling Data is organized as:
  ///   1.   Tiling Key
  ///   2.   UB Tile Size in Parallel Dim 0
  ///   3.   UB Tile Size in Parallel Dim 1
  ///   ...
  ///   N.   UB Tile Size in Parallel Dim N-2
  ///   N+1. UB Tile Size in Reduction Dim N-1
  ///   N+2. Buffer size in bytes
  TilingComputeFn calculateTilingImpl() override;
  LogicalResult createScheduleImpl(TilingKey key,
                                   OpBuilder &opBuilder) override;
  LogicalResult createScheduleImplForSplitN(TilingKey key,
                                            OpBuilder &opBuilder);
  LogicalResult createScheduleImplForSplitD(OpBuilder &opBuilder);

private:
  /// Apply canonicalization patterns.
  ///
  /// Disabled `kSimplifyTrivialLoops` because loop handles might be invalidate
  /// if the tiled loop is trivial during compile-time
  void applyCanonicalization(OpBuilder &opBuilder);

  /// Return whether the current tiling key is a split-d tiling case.
  ///
  /// Tiling Case #N-1 is split-d tiling case.
  /// Tiling Cases #0 ~ #N-2 is split-n tiling case.
  bool isSplitDTilingCase(TilingKey key) const;

  size_t getReductionTilingFactorTilingDataIdx() const;

  enum class TilingAxesKind : uint8_t { kParallel, kReduction };

  /// For parallel ops, get the tiling factor for each axes acorrding to tiling
  /// axes kind, under the assumption that there is a single, last reduction
  /// axis.
  ///
  /// For tiling parallel axes, the parallel axes tiling factors are retrived
  /// from tiling data.
  ///
  /// For tiling reduction axes, the tiling factors are all zeros for 0 ~
  /// `parallelDims` parallel axes. The reduction axes tiling factor is retrived
  /// from tiling data.
  ValueHandleFoldResults getTilingFactorsForParallelOp(
      const TilingInfo *tilingInfo, ValueHandles tilingDataHandles,
      size_t parallelDims, TilingAxesKind tilingAxesKind,
      const SmallVector<TilingData *> &tilingData = {}) const;

  /// For reduction ops, tile reduction axes according to the compute tiling
  /// data, under the assumption that there is a single, last reduction axis.
  std::vector<int64_t>
  getTilingFactorsForReductionOp(const TilingInfo *tilingInfo,
                                 size_t parallelDims) const;

  //===--------------------------------------------------------------------===//
  // Helper functions for Split-D tiling kind.
  //===--------------------------------------------------------------------===//

  /// Annotate cache read ops by fusable group's name for Split-D tiling kind.
  void annotateFusableCacheReadsForSplitD(const KernelInfo *info,
                                          OpBuilder &opBuilder);

  /// Tile reduction op's reduction axis and fuse producers into the partial
  /// reduction loops.
  ForReductionTilingResult
  tileReductionAndFuseProducers(const LastAxisPBRKernelInfo *kernelInfo,
                                const TilingInfo *tilingInfo,
                                OpBuilder &opBuilder);

  /// Fuse producers into loops for Split-D tiling kind.
  /// \note The number of loops must match the number of producers
  void fuseProducersIntoAxisDLoop(
      ValueHandles forOps,
      const std::vector<KernelInfo::FusableProducers> &producersInfo,
      OpBuilder &opBuilder);
};

} // namespace mtfusion
} // namespace mlir

#endif // MTIR_DIALECT_MTFUSION_TRANSFORMS_AUTOSCHEDULE_LASTAXISPBRSCHEDULE_H
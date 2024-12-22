//===- KernelInfo.h --- Definition for Kernel Info --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MTIR_DIALECT_MTFUSION_TRANSFORMS_AUTOSCHEDULE_KERNELINFO_H
#define MTIR_DIALECT_MTFUSION_TRANSFORMS_AUTOSCHEDULE_KERNELINFO_H

#include "mtir/Dialect/MtFusion/IR/MtFusion.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace mtfusion {

//===----------------------------------------------------------------------===//
// KernelInfo
//===----------------------------------------------------------------------===//

/// Base structure for holding tiling-agnostic kernel information.
/// Kernel info for specific fusion scheduler should derive from this class.
class KernelInfo {
public:
  /// (alignment idx, alignment unit)
  using DimAndAlignment = std::pair<int, int>;

  enum class ConsumerType : uint8_t {
    kUnknown,
    kOutput,   // Output nodes (a.k.a operands of `func.return` op)
    kReduction // Reduction ops
  };

  /// Fusable producers represents a group of producers operations that can be
  /// fused into the containing loop of a consumer.
  struct FusableProducers {
    /// Index of the consumer op.
    ///
    /// For `REDUCTION` consumer, this is the index of the reduction op's
    /// topological ordering.
    /// For `OUTPUT` consumer, this is the index of the output's topological
    /// ordering.
    unsigned idx;
    /// Unique identifier of this group of producers.
    std::string groupName;
    /// List of producer operations.
    SetVector<Operation *> operations;
    /// List of producer that are block arguments.
    SetVector<unsigned> blockArguments;

    /// Dump producer information for debugging.
    void dump();
  };

  struct ReductionOpsInfo {
    /// Index to which this reduce op appeared in the payload IR.
    size_t idx;
    /// Total number of loops (reduction + parallel).
    int64_t numLoops{0};
    /// Reduction axes index.
    SetVector<int64_t> reductionDims;
    /// Number of results (for multi-reduce).
    int64_t numResults{0};
    /// Unique identifier of the reduce op in the paylod IR.
    std::string key{};
  };

  struct BroadcastOpsInfo {
    /// Total number of loops.
    int64_t numLoops{0};
    /// Broadcast axes index.
    SetVector<int64_t> broadcastDims;
  };

public:
  KernelInfo() = default;
  KernelInfo(FusionKind kind, MLIRContext *ctx) : kind_(kind), ctx_(ctx) {}
  KernelInfo &operator=(KernelInfo const &) = delete;
  virtual ~KernelInfo() = default;

  FusionKind getFusionKind() const { return kind_; }
  static bool classof(const KernelInfo *) { return true; }

  /// Returns the BlockArgument or reshaped value from `kernelInfo->inputValues`
  /// with max rank. Returns nullopt if no shaped type found.
  std::optional<std::pair<int64_t, int64_t>> getInputValueIdxWithMaxRank();

  /// Get the number of bits of the smallest tensor element type in kernel.
  int64_t getSmallestElementTypeBits();

  /// Get the alignment dimension and unit.
  /// \note Currently only a single dimension of broadcast and reduce op needs
  ///       alignment.
  std::optional<DimAndAlignment> getAlignments();

  MLIRContext *getContext() { return ctx_; };

public:
  /// Number of inputs.
  size_t numInputs{0};
  /// Number of outputs.
  size_t numOutputs{0};
  /// Topological ordering of the output values.
  SmallVector<int64_t> outputOrdering{};
  /// Indices to function arguments that are "tied to" function return values.
  SetVector<int64_t> funcArgIdxWithTiedReturnValue{};
  /// Indices to function arguments that are reshaped before use.
  SetVector<int64_t> funcArgWithReshapeIndices;
  /// Mapping from the index of the function return value to the index of the
  /// tied function arguments.
  DenseMap<int64_t, int64_t> returnValueIdx2TiedFuncArg;
  /// Indices to function returns values are reshaped values.
  SetVector<int64_t> returnValueWithReshapeIndices;
  /// Indices to function arguments that needs cache reading.
  SetVector<int64_t> cacheReadFuncArgIndices;
  /// Original kernel name.
  std::string baseKernelName{};
  /// Smallest element type of the tensors in the kernel.
  Type smallestElementType{Type()};
  /// Kernel function's input types.
  SmallVector<Type> inputTypes{};
  /// Kernel function's output types.
  SmallVector<Type> outputTypes{};
  /// Unscheduled, original kernel function.
  func::FuncOp originalKernel{nullptr};
  /// Maximum number of buffers that need to co-exist on local memory at the
  /// same time.
  int64_t maxBufferCnt{0};

  /// Reduction ops and their info.
  /// \note The map is ordered by reduce op's order in payload IR.
  std::map<Operation *, ReductionOpsInfo> reductionOps{};

  /// Broadcast ops and their info.
  /// \note The map is ordered by broadcast op's order in payload IR.
  std::map<Operation *, BroadcastOpsInfo> broadcastOps{};

  /// Mapping from consumer type to the list of fusable producers.
  std::map<ConsumerType, std::vector<FusableProducers>> fusableProducerInfos{};

  /// Kernel function's inputs, contains two types of values following the order
  /// of origin kernel arguments:
  /// 1. if the kernel arg is not reshaped, use the origin `BlockArgument` value
  /// 2. if the kernel arg is reshaped, use the result value from reshaped op
  SmallVector<Value> inputValues{};

  /// Kernel function's return values, contains two types of values following
  /// the order of origin kernel outputs (i.e. operands of `func.return` op):
  /// 1. if the kernel output is not reshaped, use the original value
  /// 2. if the kernel output is reshaped, use the value before reshaped
  SmallVector<Value> outputValues;

private:
  /// Get the alignment dimension and factor for reduce ops.
  std::optional<DimAndAlignment> getAlignmentsForReduceOp();

  /// Get the alignment dimension and factor for broadcast ops.
  std::optional<DimAndAlignment> getAlignmentsForBroadcastOp();

private:
  /// Underlying fusion kind.
  FusionKind kind_;
  MLIRContext *ctx_{nullptr};
};
} // namespace mtfusion
} // namespace mlir

#endif // MTIR_DIALECT_MTFUSION_TRANSFORMS_AUTOSCHEDULE_KERNELINFO_H

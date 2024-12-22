//===- AutoScheduleBase.h -- Auto scheduler basic definitions ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MTIR_DIALECT_MTFUSION_TRANSFORMS_AUTOSCHEDULE_AUTOSCHEDULEBASE_H
#define MTIR_DIALECT_MTFUSION_TRANSFORMS_AUTOSCHEDULE_AUTOSCHEDULEBASE_H

#include "mtir/Dialect/MtFusion/IR/MtFusion.h"
#include "mtir/Dialect/MtFusion/TransformOps/MtFusionTransformOps.h"
#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/KernelInfo.h"
#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/TilingUtils.h"
#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/ValueHandle.h"
#include "mtir/Dialect/MtFusion/Transforms/Passes.h"

// #include "mlir/Dialect/GPU/IR/GPUOpsAttributes.h.inc"

#include "mlir/Dialect/Bufferization/Transforms/BufferUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/DeviceMappingInterface.h"

namespace mlir {
struct LogicalResult;
class Location;
class OpBuilder;

namespace transform {
class NamedSequenceOp;
class TransformHandleTypeInterface;
} // namespace transform

namespace func {
class FuncOp;
} // namespace func

namespace mtfusion {

//===----------------------------------------------------------------------===//
// KernelInfoCollector
//===----------------------------------------------------------------------===//

class KernelInfoCollector {
public:
  explicit KernelInfoCollector(KernelInfo *info) : info_(info) {}
  explicit KernelInfoCollector(KernelInfo *info, AutoScheduleOptions options)
      : info_(info), scheduleOptions_(options) {}
  virtual ~KernelInfoCollector() = default;

  /// Main entry to collect information.
  /// Will call \c visitFuncImpl and \c postVisitFuncImpl.
  LogicalResult run();

protected:
  /// Safe functions to get kernel info pointer.
  KernelInfo *getInfo();
  KernelInfo *getInfo() const;

  AutoScheduleOptions getScheduleOptions() const { return scheduleOptions_; }

private:
  /// Visit function by traversing the operations in pre-order, with callbacks
  /// to various types of operations.
  LogicalResult visitFuncImpl(func::FuncOp f);

  /// Implementation of post processing logic. Can be overwritten accordingly by
  /// derived classes.
  virtual LogicalResult postVisitFuncImpl(func::FuncOp f);

  /// Various visitors call by \c visitFuncImpl. Should not be called directly.
  LogicalResult visitLinalgOp(Operation *op);

  /// Actual implementation of various visitors. Can be overwritten accordingly
  /// by derived classes.
  virtual LogicalResult visitLinalgOpImpl(Operation *op) { return success(); };

  /// Analyze and count the maximum number of buffers that must co-exists on
  /// local memory at the same time. The number of max buffer is in terms of
  /// the tensor with the smallest type.
  ///
  /// The analysis is based on
  ///   a) Whether the operation support in-place reuse
  ///   b) Whether the operations' operand will enable multi-buffer optimization
  ///   c) Whether the operation requires additional buffers to store
  ///      intermediate results
  // LogicalResult
  // countMaxBuffer(const utils::MultiBufferMap &multiBufferCnt = {});

  /// Analayze producer group information for consumers that contains reduction
  /// axes.
  void analyzeProducersForConsumerWithReductionAxes();
  void analyzeProducersForReductionOps();
  void analyzeProducersForOutputsWithReductionAxes();

  using FusableProducerTestFn =
      std::function<std::pair<bool, bool>(Operation *)>;
  /// Utility function to trace back consumer op's producers that can be fused
  /// to the same containing loop.
  static void traceBackToFusableProducersForConsumer(
      Operation *consumer, KernelInfo::FusableProducers &producerInfo,
      const FusableProducerTestFn &isFusableProducer);

  /// Return a pair of boolean values. The first value indicates whether the
  /// the operation is a fusable producer. The second value indicates that
  /// the search should continue.
  static std::pair<bool, bool> isFusableProducerForConsumerWithReductionAxes(
      Operation *op, const KernelInfo::ReductionOpsInfo &reductionOpsInfo);

private:
  /// Pointer to kernel info.
  KernelInfo *info_{nullptr};
  /// Auto schedule options.
  AutoScheduleOptions scheduleOptions_;
};

namespace detail {
/// Struct to return the result of cache read/write.
struct CacheIOResult {
  ValueHandle *cachedOps;
};

/// Struct to return the result of tiling ops using forall.
struct ForallTilingResult {
  ValueHandles loops;
};

/// Struct to return the result of tiling ops using for.
struct ForTilingResult {
  // When tiling ops using for, the number of loops returned depends
  // on the number of "tile-able" axes.
  SmallVector<ValueHandles> loops;
};

/// Struct to return the result of tiling reduction ops using for.
struct ForReductionTilingResult {
  /// The partial reduction tiled op generated.
  ValueHandles partialReductionOp;
  /// The final reduction operation merging all the partial reductions.
  ValueHandles finalReductionOp;
  /// The fill op used to initialize the neutral element.
  /// We support tiling multi-reduce ops (i.e., reduce with multiple results),
  /// each reduction will have its own init op.
  SmallVector<ValueHandles> reductionInitOp;
  /// The loop operations that iterate over the tiles.
  ValueHandles loops;
};

/// Struct to return the results of tiling a loop.
struct LoopTileResult {
  ValueHandle *outerLoop;
  ValueHandle *innerLoop;
};

/// Enum class for holding transform patterns.
enum class TransformPatternKind : uint8_t {
  CSE = 0,                               // ApplyPatternsOp {apply_cse}
  CANONICALIZATION,                      // ApplyCanonicalizationPatternsOp
  MERGE_CONSECUTIVE_INSERT_EXTRACT_SLICE // ApplyMergeConsecutiveInsertExtractSlicePatternsOp
};

/// Enum class for holding canonicalization patterns.
enum class CanonicalizationPatternKind : uint8_t {
  kSimplifyTrivialLoops = 0, // SimplifyTrivialLoops
};

/// Struct for specifying options when getting kernel inputs/outputs.
struct GetKernelIOOptions {
  /// The positions of the kernel input/output.
  SmallVector<int64_t> positionList{};
  /// Whether the raw position is the kernel input/output to exclude.
  bool isInverted{false};
  /// For getting kernel inputs, this is the positions of the input arguments
  /// that are reshaped. If set, the return handle points to the reshaped kernel
  /// input.
  /// For getting kernel outputs, this is the positions of the kernel
  /// outputs that are reshape op's results. If set, the return handle points to
  /// the value before reshaping.
  /// \Note Cannot be used when \c isInverted is set to true.
  SetVector<int64_t> findReshapePosition{};
};

/// Struct for specifying options for matching IR values.
struct MatchOptions {
  /// Whether to reverse the order of payload objects in \c target.
  bool needsReverse{false};
  /// If set, will only match operations that are ancestors of
  /// \c childHandleOrValue.
  std::optional<std::variant<ValueHandle *, Value>> childHandleOrValue{};
};

/// Enum class for loop tile mode.
enum class LoopTileMode : uint8_t { kFactorMode = 0, kNPartMode };

/// Struct for specifying options for loop tiling.
struct LoopTileOptions {
  /// The tiling mode.
  LoopTileMode mode{LoopTileMode::kFactorMode};
  /// Whether reorder the tiled axis.
  bool isReorderMode{false};
};

/// Struct for specifying options for mapping scf.for to scf.forall.
struct MapForToForallOptions {
  /// Device mapping attribute for the `scf.forall` op.
  std::optional<DeviceMappingAttrInterface> mapping{std::nullopt};
  /// Whether the transformation is effectively immediately. If not, only an
  /// attribute is added to the `scf.for` op.
  bool annotateOnly{false};
};

/// Struct for specifying options for set buffer size.
struct SetBufferSizeOptions {
  transform::SetBufferSizeMode mode{transform::SetBufferSizeMode::kPerByte};
  Type referenceType{Type()};
};
} // namespace detail

//===----------------------------------------------------------------------===//
// SchedulerBase
//===----------------------------------------------------------------------===//

/// Base class for auto scheduler.
/// Work flow:
///                          +---------------+
///                          | target kernel |
///                          |  fusion_kind  |
///                          +---------------+
///                                 |           @analyzeAndVerifyKernel
///  |----------------------------------------------------------------------|
///  |                            /  \          @calculateTiling            |
///  |                            ....                                      |
///  |     +-------------------+        +------------------+                |
///  |     |  tiling case #0  |         |  tiling case #N  |                |
///  |     +-------------------+        +------------------+                |
///  |                 \                        /    @selectTiling          |
///  |                 |                       |     @createScheduleImpl    |
///  |          +-------------+           +-------------+                   |
///  |          | schedule #i |           | schedule #k |                   |
///  |          +-------------+           +-------------+                   |
///  |                |                          |      @applyScheduleImpl  |
///  |     +---------------------+      +---------------------+             |
///  |     | scheduled kernel #0 |      | scheduled kernel #N |             |
///  |     +---------------------+      +---------------------+             |
///  |----------------------------------------------------------------------|
class SchedulerBase {
public:
  explicit SchedulerBase(func::FuncOp f, FusionKind kind);

  explicit SchedulerBase(func::FuncOp f, std::unique_ptr<KernelInfo> kernelInfo,
                         std::unique_ptr<TilingInfo> tilingInfo);

  virtual ~SchedulerBase();

  /// Main entry point to do auto-scheduling.
  virtual LogicalResult runOnOperation(OpBuilder &opBuilder);

  /// Apply schedule to outlineFunc
  static LogicalResult applySchedule(func::FuncOp &funcOp,
                                     OpBuilder &opBuilder);

  /// Get and set auto schedule options.
  static AutoScheduleOptions getAutoScheduleOptions() { return options_; }
  static void setAutoScheduleOptions(const AutoScheduleOptions &options) {
    options_ = options;
  }

protected:
  //===--------------------------------------------------------------------===//
  // Type defs.
  //===--------------------------------------------------------------------===//
  using CacheIOResult = detail::CacheIOResult;
  using ForallTilingResult = detail::ForallTilingResult;
  using ForTilingResult = detail::ForTilingResult;
  using ForReductionTilingResult = detail::ForReductionTilingResult;
  using TransformPatternKind = detail::TransformPatternKind;
  using CanonicalizationPatternKind = detail::CanonicalizationPatternKind;
  using SetBufferSizeMode = transform::SetBufferSizeMode;
  using GetKernelIOOptions = detail::GetKernelIOOptions;
  using MatchOptions = detail::MatchOptions;
  using NamedValueHandleArgs = detail::NamedValueHandleArgs;
  using LoopTileResult = detail::LoopTileResult;
  using LoopTileMode = detail::LoopTileMode;
  using LoopTileOptions = detail::LoopTileOptions;
  using MapForToForallOptions = detail::MapForToForallOptions;
  using SetBufferSizeOptions = detail::SetBufferSizeOptions;

  /// Implementation of kernel analysis and verification.
  virtual LogicalResult analyzeAndVerifyKernelImpl();

  /// Implementation of host tiling calculation logic.
  virtual TilingComputeFn calculateTilingImpl() = 0;

  /// Implementation of creating a schedule from the input tiling key.
  virtual LogicalResult createScheduleImpl(TilingKey key,
                                           OpBuilder &opBuilder) = 0;

  /// Run pre-schedule procedure (e.g., kernel info collection and
  /// verification).
  LogicalResult runPreScheduleProcedure(OpBuilder &opBuilder);

  /// Run schedule procedure (including tiling calculation and schedule
  /// operation).
  LogicalResult runScheduleProcedure(OpBuilder &opBuilder);

  /// Cache input and output values.
  virtual LogicalResult cacheIO(OpBuilder &opBuilder);

  //===--------------------------------------------------------------------===//
  // Basic Schedule API.
  //===--------------------------------------------------------------------===//

  /// Get value from handle.
  ///
  /// \param handle Pointer to a value handle instance.
  /// \param opBuilder Reference to IRBuilder instance.
  /// \return Value corresponding to the input handle.
  /// \note User should guarantee that the input handle is valid, otherwise
  ///       a runtime error is produced.
  Value getValue(ValueHandle *handle, OpBuilder &opBuilder);

  /// Get values from handles.
  ///
  /// \param handles Vector of pointer to a value handle instance.
  /// \param opBuilder Reference to IRBuilder instance.
  /// \return Values corresponding to the input handles.
  SmallVector<Value> getValues(const ValueHandles &handle,
                               OpBuilder &opBuilder);

  /// Match and return IR values with \c identifier of type \c type, with
  /// additional constraints/options specified in \c options.
  ///
  /// \param target Target to perform matching.
  /// \param identifier Identifier name.
  /// \param type Identifier type.
  /// \param opBuilder Reference to IRBuilder instance.
  /// \param options Match options.
  /// \return Values corresponding to the input handles.
  Value matchByIdentifier(Value target, StringRef identifier,
                          IdentifierType type, OpBuilder &opBuilder,
                          const MatchOptions &options = MatchOptions());

  /// Merge handles whose type is `handleType` and return the merged
  /// handle's value.
  ///
  /// \param handles Vector of handle values to merge.
  /// \param handleType Handle's type.
  /// \param opBuilder Reference to IRBuilder instance.
  /// \return Value holding the merged handles.
  Value mergeHandles(const SmallVectorImpl<Value> &handles,
                     transform::TransformHandleTypeInterface handleType,
                     OpBuilder &opBuilder);

  /// Annotate the IR values corresponding to \c target with \c attrName.
  ///
  /// \param target Target value to annotate.
  /// \param attrName Attribute name to add to operation's attribute list.
  /// \param opBuilder Reference to IRBuilder instance.
  void annotateByAttr(Value target, StringRef attrName, OpBuilder &opBuilder);

  /// Get handle to the kernel function.
  ///
  /// \param opBuilder Reference to IRBuilder instance.
  /// \return RegularValueHandle to `func.func` op.
  ValueHandle *getFuncHandle(OpBuilder &opBuilder);

  /// Get handle value to the kernel function.
  ///
  /// \param opBuilder Reference to IRBuilder instance.
  /// \return Value to a handle of `func.func` op.
  Value getFuncValue(OpBuilder &opBuilder);

  /// Get handles to the outputs of the kernel.
  ///
  /// \param opBuilder Reference to IRBuilder instance.
  /// \param options Options for getting kernel outputs.
  /// \return RegularValueHandles to the producing op of kernel function's
  ///         return values.
  ValueHandles
  getKernelOutputs(OpBuilder &opBuilder,
                   const GetKernelIOOptions &options = GetKernelIOOptions());

  /// Get handles to the inputs of the kernel.
  ///
  /// \param opBuilder Reference to IRBuilder instance.
  /// \param options Options for getting kernel inputs.
  /// \return RegularValueHandles to the kernel function's input block argument.
  ValueHandles
  getKernelInputs(OpBuilder &opBuilder,
                  const GetKernelIOOptions &options = GetKernelIOOptions());

  /// Get handle to the tiling data.
  ///
  /// \param d Tiling data pointer.
  /// \param opBuilder Reference to IRBuilder instance.
  /// \return FuncArgHandle to the kernel function's block argument that
  ///         corresponds to the tiling data.
  ValueHandle *getTilingDataHandle(TilingData *d, OpBuilder &opBuilder);

  /// Get handles to each tiling data in tiling struct \c s.
  ///
  /// \param s A series of tiling data pointer.
  /// \param opBuilder Reference to IRBuilder instance.
  /// \return FuncArgHandles to the kernel function's block arguments that
  ///         correspond to the tiling data in tiling struct.
  ValueHandles getTilingStructHandles(SmallVector<TilingData *> s,
                                      OpBuilder &opBuilder);

  /// Get handle to ops with identifier named \c identifier of type \c type
  /// in the kernel, with additional constraints/options specified in
  /// \c options.
  ///
  /// \param identifier Identifier name.
  /// \param type Identifier type.
  /// \param opBuilder Reference to IRBuilder instance.
  /// \param options Match options.
  /// \return NamedValueHandle to the target ops.
  ValueHandle *
  getOpsWithIdentifier(StringRef identifier, IdentifierType type,
                       OpBuilder &opBuilder,
                       const MatchOptions &options = MatchOptions());

  /// Perform cache read on kernel inputs.
  ///
  /// After cache read, an unique tag name will be added to the cached op.
  /// For example:
  /// ```
  /// func.func @foo(%arg0):
  ///   linalg.copy ins(%arg0) outs(...) {__arg0__}
  /// ```
  ///
  /// \param opBuilder Reference to IRBuilder instance.
  /// \return NamedValueHandle to cached ops. Note that the handle points to
  ///         ALL cached ops. If you wish to obtain a more fine-grained control
  ///         over each ops, you can match by the attributed name returned by
  ///         `getCacheReadTag`.
  CacheIOResult cacheRead(OpBuilder &opBuilder);

  /// Get a unique identifier to the cached op by the function argument index.
  std::string getCacheReadTag(size_t funcArgIdx);

  /// Perform cache write on kernel outputs.
  ///
  /// \param opBuilder Reference to IRBuilder instance.
  /// \return NamedValueHandle to the cached ops.
  CacheIOResult cacheWrite(OpBuilder &opBuilder);

  /// Tile the target linalg ops using \c scf.forall ops by a
  /// factor of \c blockDim. The block axis is tied to \c hivm.block<x>
  ///
  /// Before tiling:
  ///   linalg.op
  ///
  /// After tiling:
  ///   scf.forall %arg in (blockDim):
  ///     tiled linalg.op
  ///   mapping [hivm.block<x>]
  ///
  /// \param targets Value handles to linalg ops.
  /// \param blockDim Number of blocks.
  /// \param opBuilder Reference to IRBuilder instance.
  /// \return NamedValueHandles to `scf.forall` ops.
  /// \note The input `targets` handles are updated to the tiled linalg ops
  ///       and can be reused without invalidation.
  ForallTilingResult tileUsingForAll(ValueHandles &targets, int64_t blockDim,
                                     OpBuilder &opBuilder);

  /// Tile the target linalg ops using \c scf.for ops by \c tileSizes.
  ///
  /// \param targets Value handles to linalg ops.
  /// \param tileSize Value handles to dynamic tile sizes.
  /// \param opBuilder Reference to IRBuilder instance.
  /// \return NamedValueHandles to `scf.for` ops.
  /// \note The input `targets` handles are updated to the tiled linalg ops
  /// and
  ///       can be reused without invalidation.
  ForTilingResult tileUsingFor(ValueHandles &targets,
                               ValueHandleFoldResults &tileSizes,
                               OpBuilder &opBuilder);

  /// Tile the target linalg reduction op using \c scf.for ops by \c
  /// tileSizes.
  ///
  /// \param targets Value handles to \c linalg.reduce ops.
  /// \param tileSizes Static tile sizes.
  /// \param opBuilder Reference to IRBuilder instance.
  /// \param multiReduceNum The number of multi-reduced tensors.
  /// \return ForReductionTilingResult
  /// \note The input \c targets handles are invalidated.
  ForReductionTilingResult
  tileReductionUsingFor(ValueHandles &targets,
                        const std::vector<int64_t> &tileSizes,
                        OpBuilder &opBuilder, int64_t multiReduceNum = 1);

  /// Fuse independent loops together.
  ///
  /// \param loops Value handles to loops of the same type (i.e., all
  ///        `scf.for` or all `scf.forall`)
  /// \param opBuilder Reference to IRBuilder instance.
  /// \return NamedValueHandles to the fused loop.
  /// \note The input `loops` handles are invalidated.
  ValueHandle *fuseLoops(ValueHandles &loops, OpBuilder &opBuilder);

  /// Coalesces the perfect loop nest enclosed by \c outerMostLoop
  ///
  /// \param outerMostLoop Value handle to the outer most loop (must be either
  ///                      `scf.for` or `affine.for` loop)
  /// \param opBuilder Reference to IRBuilder instance.
  /// \return NamedValueHandles to the coalesced loop.
  /// \note The input \c outerMostLoop handle is invalidated.
  ValueHandle *coalesceLoops(ValueHandle *outerMostLoop, OpBuilder &opBuilder);

  /// Tile the given loop by a factor of \c tileSize.
  ///
  /// IR before tiling:
  ///   for i : l to u step s
  ///     use(i)
  ///
  /// When tiling mode is `LoopTile::kFactor` (default):
  ///
  /// IR after tiling loop i by a factor of x:
  ///   for i.o : l to u step x
  ///    for i.i : 0 to min(u - i.o, x) step s
  ///      use(i.o + i.i)
  ///
  /// When no-min-max-bounds option is enabled:
  ///   for i.o : l to u step x
  ///    for i.i : 0 to x step s
  ///     if (i.o + i.i < u)
  ///       use(i.o + i.i)
  ///
  /// When tiling mode is `LoopTile::kNPart`:
  ///
  /// IR after tiling loop i by a factor of x:
  ///   for i.o 0 to x step 1
  ///     for i.i 0 to min(ceilDiv(u, x), u - i.o*(ceilDiv(u, x))) step 1
  ///       use(i.o*(ceilDiv(u, x)) + i.i)
  /// Note that this requires the loop to be normalized before tiling. And it
  /// cannot be used together with no-min-max-bounds option.
  ///
  /// \param targetLoop Value handle to the target `scf.for` loop.
  /// \param tileSize Static tile size.
  /// \param opBuilder Reference to IRBuilder instance.
  /// \param options Loop tiling options.
  /// \return Handles to the outer and inner loop after tiling.
  /// \note The input \c targetLoop handle is invalidated.
  LoopTileResult tileLoop(ValueHandle *targetLoop, int64_t tileSize,
                          OpBuilder &opBuilder,
                          const LoopTileOptions &options = LoopTileOptions());

  /// Normalize the given loop (i.e., has step 1 while preserving trip count)
  ///
  /// \param targetLoop Value handle to the target `scf.for` loop.
  /// \note The input \c targetLoop handle is updated to the loop after being
  ///       normalized.
  void normalizeLoop(ValueHandle *targetLoop, OpBuilder &opBuilder);

  /// TODO: Add return value to this API.
  /// Fuse target ops into containing ops one by one.
  ///
  /// When target op has multiple users in the containing op, the producer
  /// will be tiled according to the union of the users.
  ///
  /// \param targetOps Handles to fuse.
  /// \param containingLoops Handles to the initial containing ops.
  /// \param opBuilder Reference to IRBuilder instance.
  /// \param duplicateProducers Whether to duplicate producer when it is used
  ///        in multiple containing ops.
  /// \param applyCanonicalizeAfterEachFusion Whether to apply canonicalize
  ///        patterns to the IR after each fusion.
  /// \note If `applyCanonicalizeAfterEachFusion` is set to true, all input
  ///       handles are invalidated.
  ///       Otherwise, the handles in `containingLoop` are automatically
  ///       updated. The handles in `targetOps` are automatically updated if
  ///       and only if `len(containingLoop) == 1`.
  void fuseIntoContaining(ValueHandles &targetOps, ValueHandles &containingLoop,
                          OpBuilder &opBuilder, bool duplicateProducers = false,
                          bool applyCanonicalizeAfterEachFusion = true);

  /// Split `handle` into `splitSize` parts.
  ///
  /// \param handle Target value handle.
  /// \param splitSize Number of parts to split.
  /// \param opBuilder Reference to IRBuilder instance.
  /// \return RegularValueHandles to the splitted handles.
  /// \note Runtime error will occur if the handle cannot be split into the
  ///       request parts.
  ValueHandles splitHandle(ValueHandle *handle, size_t splitSize,
                           OpBuilder &opBuilder);

  /// Apply canonicalize pass.
  ///
  /// \param opBuilder Reference to IRBuilder instance.
  /// \note This function resets all handles.
  void applyCanonicalization(OpBuilder &opBuilder);

  /// Apply common subexpression elimination pass.
  ///
  /// \param opBuilder Reference to IRBuilder instance.
  /// \note This function resets all handles.
  void applyCSE(OpBuilder &opBuilder);

  /// Apply `patterns` to `target`.
  ///
  /// \param target Target handle to apply patterns.
  /// \param patterns List of `TransformPatternKind` to apply.
  /// \param opBuilder Reference to IRBuilder instance.
  /// \param disablePatterns List of `CanonicalizationPatternKind` to disable.
  void applyPatterns(
      ValueHandle *target, const SmallVector<TransformPatternKind> &patterns,
      OpBuilder &opBuilder,
      const SmallVector<CanonicalizationPatternKind> &disablePatterns = {});

  /// Apply one-shot-bufferization to the kernel function.
  ///
  /// \param opBuilder Reference to IRBuilder instance.
  void applyOneShotBufferization(OpBuilder &opBuilder);

  /// Map `scf.forall` to HIVM Block ops.
  ///
  /// \param opBuilder Reference to IRBuilder instance.
  // void mapForallToHIVMBlocks(OpBuilder &opBuilder);

  /// Map `scf.for` to `scf.forall` op, with optional mapping.
  ///
  /// \param targetLoop Handle to `scf.for` op.
  /// \param mapping Optional mapping for`scf.forall` op.
  /// \param opBuilder Reference to IRBuilder instance.
  /// \param options Map for to forall options.
  /// \return NamedValueHandles to the `scf.forall` op if `options.annotateOnly`
  ///         is set to true. Otherwise, its the NamedValueHandles to the
  ///         `scf.for` op after annotation.
  /// \note The input `targetLoop` handle is invalidated.
  ValueHandle *mapForToForall(
      ValueHandle *targetLoop, OpBuilder &opBuilder,
      const MapForToForallOptions &options = MapForToForallOptions());

  using RegionBuilderFn =
      llvm::function_ref<void(ImplicitLocOpBuilder &, Block &)>;

  /// Construct `transform.foreachOp` and return its results.
  ///
  /// \param target Target to apply `transform.foreachOp`
  /// \param resultTypes `transform.foreachOp`'s result types.
  /// \param regionBuilder Lambda to build `transform.foreachOp`'s body.
  /// \param opBuilder Reference to IRBuilder instance.
  ResultRange createForEachOp(Value target, TypeRange resultTypes,
                              RegionBuilderFn regionBuilder,
                              OpBuilder &opBuilder);

  /// Set the size of the `targets` to `bufferSize`.
  ///
  /// If the payload operation is `memref.alloc` or `memeref.alloca`, the
  /// transformation takes place immediately.
  /// Otherwise, the target op is only annotated with the `bufferSize`, and
  /// the actual transformation will happen later on.
  ///
  /// \param targets Value handles to target ops.
  /// \param bufferSize Static buffer size.
  /// \param options
  /// \param opBuilder Reference to IRBuilder instance.
  /// \note The input `targets` handles are invalidated.
  void
  setBufferSize(ValueHandles &targets, int64_t bufferSize, OpBuilder &opBuilder,
                const SetBufferSizeOptions &options = SetBufferSizeOptions());

  //===--------------------------------------------------------------------===//
  // APIs to run pre/post process passes.
  //===--------------------------------------------------------------------===//

  /// Apply op flattening pass to \c target.
  // LogicalResult applyOpFlattenPass(Operation *target,
  //                                  const FlattenOpsOptions &options = {});

  /// Apply op fusion and outline pass to \c target.
  FailureOr<SmallVector<func::FuncOp>>
  applyOpFusionOutline(func::FuncOp target,
                       const MtFusionOpFusionOptions &options = {});

  //===--------------------------------------------------------------------===//
  // Value Handle related API.
  //===--------------------------------------------------------------------===//

  /// Create and record handle.
  template <class T, class... Args>
  T *record(Value v, OpBuilder &b, Args &&...args) {
    return handleRecord_->record<T>(
        recordImpl(v, b, std::forward<Args>(args)...));
  }

  /// Reset all recorded handles.
  /// \note Different value handle kind have different implementation.
  void resetAllHandles() { return handleRecord_->resetAllHandles(); }

  //===--------------------------------------------------------------------===//
  // Getter methods.
  //===--------------------------------------------------------------------===//

  /// Get the handle to transform sequence's block argument.
  Value getTransformSeqHandle() { return transformSeqBlockHandle_; }
  /// Get the enclosing module of the kernel function.
  ModuleOp getModule() { return module_; }
  /// Get a pointer to kernel info.
  KernelInfo *getKernelInfo() { return kernelInfo_.get(); }
  KernelInfo *getKernelInfo() const { return kernelInfo_.get(); }
  /// Get pointer to the tiling info.
  TilingInfo *getTilingInfo() { return tilingInfo_.get(); };
  TilingInfo *getTilingInfo() const { return tilingInfo_.get(); };
  /// Get MLIR Context.
  MLIRContext *getContext() { return module_->getContext(); };
  MLIRContext *getContext() const { return module_->getContext(); };
  /// Get reference to the handle record.
  HandleRecord *getHandleRecord() { return handleRecord_.get(); }
  /// Get the original kernel.
  func::FuncOp getOriginalKernel() {
    assert(originalKernel_);
    return originalKernel_;
  }
  /// Get the to-be-scheduled kernel.
  func::FuncOp getToBeScheduledKernel() {
    assert(toBeScheduledKernel_);
    return toBeScheduledKernel_;
  }
  /// Get the name to the original kernel.
  std::string getOriginalKernelName() {
    return getOriginalKernel().getSymName().str();
  }
  /// Get the name to the to-be-scheduled kernel.
  std::string getToBeScheduledKernelName() {
    return getToBeScheduledKernel().getSymName().str();
  }
  unsigned getBlockDim() { return options_.blockDim; }
  bool getEnableAutoMultiBuffer() { return options_.enableAutoMultiBuffer; }
  int64_t getMaxBufferCntTuning() { return options_.maxBufferCntTuning; }

  //===--------------------------------------------------------------------===//
  // Setter methods.
  //===--------------------------------------------------------------------===//

  /// Update the handle to transform sequence's block argument.
  void setTransformSeqHandle(Value newHandle) {
    transformSeqBlockHandle_ = newHandle;
  }
  /// Set the to-be-scheduled kernel.
  void setToBeScheduledKernel(func::FuncOp f) { toBeScheduledKernel_ = f; }
  /// Set tiling info.
  void setTilingInfo(TilingInfo &&info) {
    tilingInfo_ = std::make_unique<TilingInfo>(std::move(info));
  }
  /// Set the original kernel.
  void setOriginalKernel(func::FuncOp f) { originalKernel_ = f; }

private:
  /// Run analysis on kernel function and verify constraints.
  LogicalResult analyzeAndVerifyKernel();

  /// Calculate tiling struct for all tiling cases.
  LogicalResult calculateTiling(OpBuilder &opBuilder);

  /// Prune and select tiling cases if possible.
  LogicalResult selectTiling();

  /// Create one or more tiling cases and apply schedules.
  LogicalResult createAndApplySchedules(OpBuilder &opBuilder);

  /// Apply one specific schedule according to the input tiling info.
  LogicalResult applyScheduleImpl(OpBuilder &opBuilder);

  /// Prepare kernel function for scheduling and init schedule sequence.
  LogicalResult initSchedule(TilingKey key, OpBuilder &opBuilder);

  /// Reset things after doing schedule.
  void cleanUpAfterSchedule();

  /// Create switch cases for entry function to call scheduled functions
  /// according to tiling key and the callers of device kernels.
  LogicalResult fixCallSitesAndCaller(OpBuilder &opBuilder);

  // TODO: Refactor to use tiling utils.

  /// Caller information.
  struct CallerInfo {
    func::FuncOp caller;
    /// Callers original argument number (before appending tiling data)
    size_t callerOriginalArgNumber;
    /// Function called by the caller.
    func::FuncOp callee;
    /// Call sites within the caller calling callee.
    SmallVector<func::CallOp> callSites;
  };

  /// Get callee's caller's information.
  static void getCallerInfo(func::FuncOp callee, ModuleOp enclosingModule,
                            DenseMap<func::FuncOp, CallerInfo> &info);

  /// Information needed to construct callee's arguments.
  struct CallSiteArgBuilderInfo {
    /// Mapping from tiling index (in ordered present in tiling struct) to the
    /// caller's function argument index.
    DenseMap<size_t, size_t> tilingIdx2CallerArgIdx{};
    /// Mapping from the index of constant tiling data to the constant value
    /// in callee.
    DenseMap<size_t, int64_t> calleeArgIdx2ConstValue{};
    /// Whether callee is the original kernel.
    bool calleeIsOriginalKernel{false};
  };

  /// Fix the call sites by replacing arguments.
  void doFixCallSite(CallerInfo &callerInfo, func::CallOp callSite,
                     CallSiteArgBuilderInfo &builderInfo,
                     DenseMap<Operation *, Operation *> &irMap,
                     OpBuilder &opBuilder);

  /// Construct new call site arguments.
  static SmallVector<Value>
  getNewArgsForCallSite(func::FuncOp caller, func::CallOp oldCallSite,
                        const CallSiteArgBuilderInfo &info,
                        OpBuilder &opBuilder);

  /// Dump current schedule and kernel function for debugging purposes.
  void dumpKernelAndSchedule();

  /// Helper function to convert `tensor.empty` to
  /// `bufferization.alloc_tensor`.
  ///
  /// \note This function is invoked in `applyOneShotBufferization` and should
  ///       not be called separately.
  void bufferizeEmptyTensor(OpBuilder &opBuilder);

  /// Create and record NamedValueHandle.
  NamedValueHandle *recordImpl(Value target, OpBuilder &opBuilder,
                               const NamedValueHandleArgs &args);

  /// Create and record RegularValueHandle.
  RegularValueHandle *recordImpl(Value target, OpBuilder &opBuilder);

  /// Create and record FuncArgHandle.
  FuncArgHandle *recordImpl(Value target, OpBuilder &opBuilder,
                            size_t funcArgNum);

  std::pair<SmallVector<int64_t>, SmallVector<Value>>
  unpackFoldResults(ValueHandleFoldResults &values, OpBuilder &opBuilder);

private:
  /// Module enclosing the to-be-scheduled kernel.
  ModuleOp module_{nullptr};
  /// Original kernel function without scheduling.
  func::FuncOp originalKernel_{nullptr};
  /// Kernel function that will be scheduled.
  func::FuncOp toBeScheduledKernel_{nullptr};
  /// The transform sequence block argument value.
  Value transformSeqBlockHandle_;
  /// Information regarding the to-be-scheduled kernel.
  std::unique_ptr<KernelInfo> kernelInfo_{nullptr};
  /// Information regarding the tiling.
  std::unique_ptr<TilingInfo> tilingInfo_{nullptr};
  /// Record keeping all allocated value handles.
  std::unique_ptr<HandleRecord> handleRecord_{nullptr};
  /// Underlying fusion kind.
  FusionKind kind_;
  /// Schedule options.
  static AutoScheduleOptions options_;
};

} // namespace mtfusion
} // namespace mlir

#endif // MTIR_DIALECT_MTFUSION_TRANSFORMS_AUTOSCHEDULE_AUTOSCHEDULEBASE_H
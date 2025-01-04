//===- TilingBase.h -- Predefined Tiling basic definitions ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MTIR_DIALECT_MTFUSION_TRANSFORMS_PREDEFINEDTILING_TILINGBASE_H
#define MTIR_DIALECT_MTFUSION_TRANSFORMS_PREDEFINEDTILING_TILINGBASE_H

#include "mtir/Dialect/MtFusion/IR/MtFusion.h"
#include "mtir/Dialect/MtFusion/TransformOps/MtFusionTransformOps.h"
#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/KernelInfo.h"
#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/TilingUtils.h"
#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/ValueHandle.h"
#include "mtir/Dialect/MtFusion/Transforms/Passes.h"

#include "mlir/Dialect/Bufferization/Transforms/BufferUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/DeviceMappingInterface.h"
#include "llvm/ADT/SetVector.h"

#include <string>
#include <set>
#include <map>

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

namespace detail {
/// Struct to return the result of cache read/write.
struct CacheIOResult {
  ValueHandle *cachedOps;
};

struct MatmulResult {
  ValueHandle *matmulOps;
};

struct TilingDataResult {
  ValueHandle *tilingData;
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
// TilingItem
//===----------------------------------------------------------------------===//

class TilingItem {
public:
  TilingItem(std::string axis = "m", int nthreads = 1,
      SmallVector<std::string> copyMat = {},
      SmallVector<std::string> copyDst = {}) :
      axis_(axis), nthreads_(nthreads), copyMat_(copyMat), copyDst_(copyDst) {}

  static std::set<std::string> optNameSet;
  static std::set<std::string> axisSet;
  static std::set<std::string> matrixSet;
  static std::set<std::string> memorySet;

  static std::map<std::string, mtfusion::Axis> axisToEnum;
  static std::map<std::string, mtfusion::Matrix> matrixToEnum;
  static std::map<std::string, mtfusion::Cache> memoryToEnum;

  std::string getAxis() { return axis_; }
  void setAxis(std::string axis) { axis_ = axis; } 

  int64_t getNthreads() { return nthreads_; }
  void setNthreads(int64_t nthreads) { nthreads_ = nthreads; }

  int64_t getCopyNum() { return copyNum_; }
  void setCopyNum(int64_t copyNum) { copyNum_ = copyNum; }

  SmallVector<std::string>& getCopyMat() { return copyMat_; }
  void setCopyMat(SmallVector<std::string>& copyMat) { copyMat_ = copyMat; }
  void appendCopyMat(std::string& singleMat) { copyMat_.push_back(singleMat); }

  SmallVector<std::string>& getCopyDst() { return copyDst_; }
  void setCopyDst(SmallVector<std::string>& copyDst) { copyDst_ = copyDst; }
  void appendCopyDst(std::string& singleDst) { copyDst_.push_back(singleDst); }

  void getArgAttrs(SmallVector<NamedAttribute>& argAttrs,
      OpBuilder &opBuilder, MLIRContext* ctx);

  bool verify();
  void print();

private:
  std::string axis_; // m, n, k
  int64_t nthreads_ = 0;
  SmallVector<std::string> copyMat_; // A, B, C
  SmallVector<std::string> copyDst_; // GSM, AM, SM
  int64_t copyNum_ = 0;
};

using TilingSeq = SmallVector<TilingItem>;

//===----------------------------------------------------------------------===//
// TilerBase
//===----------------------------------------------------------------------===//

/// Base class for tiler.
class TilerBase {
public:
  explicit TilerBase(func::FuncOp f, FusionKind kind);

  explicit TilerBase(func::FuncOp f,
                    std::unique_ptr<KernelInfo> kernelInfo,
                    std::unique_ptr<TilingInfo> tilingInfo);
                    
  explicit TilerBase(func::FuncOp f, 
                    mtfusion::TilingSeq& tilingSeq,
                    FusionKind kind);

  explicit TilerBase(func::FuncOp f,
                    mtfusion::TilingSeq& tilingSeq,
                    std::unique_ptr<KernelInfo> kernelInfo,
                    std::unique_ptr<TilingInfo> tilingInfo);

  virtual ~TilerBase();

  /// Main entry point to do auto-scheduling.
  virtual LogicalResult runOnOperation(OpBuilder &opBuilder);

  /// Apply tiling to outlineFunc
  static LogicalResult applyTiling(func::FuncOp &funcOp,
      mtfusion::TilingSeq &tilingSeq, OpBuilder &opBuilder);

protected:
  //===--------------------------------------------------------------------===//
  // Type defs.
  //===--------------------------------------------------------------------===//
  using MatmulResult = detail::MatmulResult;
  using TilingDataResult = detail::TilingDataResult;
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

  /// Implementation of creating tiling
  virtual LogicalResult createTilingImpl(OpBuilder &opBuilder) = 0;

  //===--------------------------------------------------------------------===//
  // Basic Tiling API.
  //===--------------------------------------------------------------------===//

  /// Get value from handle.
  Value getValue(ValueHandle *handle, OpBuilder &opBuilder);

  /// Get values from handles.
  SmallVector<Value> getValues(const ValueHandles &handle,
                               OpBuilder &opBuilder);

  /// Match and return IR values with identifier of type type, with
  /// additional constraints/options specified in options.
  Value matchByIdentifier(Value target, ArrayRef<StringRef> identifiers,
                          IdentifierType type, OpBuilder &opBuilder,
                          const MatchOptions &options = MatchOptions());
  Value matchOperations(Value target, ArrayRef<StringRef> identifiers,
                        OpBuilder &opBuilder);

  /// Merge handles whose type is `handleType` and return the merged
  /// handle's value.
  Value mergeHandles(const SmallVectorImpl<Value> &handles,
                     transform::TransformHandleTypeInterface handleType,
                     OpBuilder &opBuilder);

  /// Annotate the IR values corresponding to target with attrName.
  void annotateByAttr(Value target, StringRef attrName, OpBuilder &opBuilder);
  void annotateByAttr(Value target, StringRef attrName, Attribute value, OpBuilder &opBuilder);

  /// Get handle to the kernel function.
  ValueHandle *getFuncHandle(OpBuilder &opBuilder);

  /// Get handle value to the kernel function.
  Value getFuncValue(OpBuilder &opBuilder);

  /// Get handles to the outputs of the kernel.
  ValueHandles
  getKernelOutputs(OpBuilder &opBuilder,
                   const GetKernelIOOptions &options = GetKernelIOOptions());

  /// Get handles to the inputs of the kernel.
  ValueHandles
  getKernelInputs(OpBuilder &opBuilder,
                  const GetKernelIOOptions &options = GetKernelIOOptions());

  /// Get handles to the operand of ops.
  ValueHandles getOpOperand(const ValueHandles &opHandles, 
      int64_t operandPos, OpBuilder &opBuilder);

  ValueHandles getOpResult(const ValueHandles &opHandles, 
      int64_t resultPos, OpBuilder &opBuilder);

  /// Get handle to the tiling data.
  ValueHandles getTilingDataHandles(
      SmallVector<size_t>& tilingArgsPos, OpBuilder &opBuilder);

  /// Get handle to ops with identifier named \c identifier of type \c type
  /// in the kernel, with additional constraints/options specified in
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
  CacheIOResult cacheRead(OpBuilder &opBuilder);
  CacheIOResult cacheRead(const ValueHandles &inputHandles,
      StringRef cacheReadTagName, OpBuilder &opBuilder);
  CacheIOResult cacheRead(const ValueHandles &opHandles, 
      int64_t oprIdx, 
      StringRef cacheReadTagName, //ArrayRef<Attribute> copyDstAttrs,
      OpBuilder &opBuilder);

  /// Get a unique identifier to the cached op by the function argument index.
  std::string getCacheReadTag(size_t funcArgIdx);

  /// Perform cache write on kernel outputs.
  CacheIOResult cacheWrite(OpBuilder &opBuilder);
  CacheIOResult cacheWrite(const ValueHandles &opHandles,
      int64_t oprIdx,
      StringRef cacheWriteTagName,
      OpBuilder &opBuilder);

  /// Tile the target linalg ops using scf.for ops by tileSizes.
  ForTilingResult tileUsingFor(ValueHandles &targets,
                               ValueHandleFoldResults &tileSizes,
                               OpBuilder &opBuilder);
  ForTilingResult tileSingleLayer(ValueHandles &targets,
                                  ValueHandleFoldResult tileSize,
                                  OpBuilder &opBuilder);

  /// Tile the target linalg reduction op using scf.for ops by tileSizes.
  ForReductionTilingResult
  tileReductionUsingFor(ValueHandles &targets,
                        const std::vector<int64_t> &tileSizes,
                        OpBuilder &opBuilder, int64_t multiReduceNum = 1);

  /// Fuse independent loops together.
  ValueHandle *fuseLoops(ValueHandles &loops, OpBuilder &opBuilder);

  /// Coalesces the perfect loop nest enclosed by outerMostLoop
  ValueHandle *coalesceLoops(ValueHandle *outerMostLoop, OpBuilder &opBuilder);

  /// Tile the given loop by a factor of tileSize.
  LoopTileResult tileLoop(ValueHandle *targetLoop, int64_t tileSize,
                          OpBuilder &opBuilder,
                          const LoopTileOptions &options = LoopTileOptions());

  /// Normalize the given loop (i.e., has step 1 while preserving trip count)
  void normalizeLoop(ValueHandle *targetLoop, OpBuilder &opBuilder);

  /// Fuse target ops into containing ops one by one.
  /// When target op has multiple users in the containing op, the producer
  /// will be tiled according to the union of the users.
  void fuseIntoContaining(ValueHandles &targetOps, ValueHandles &containingLoop,
                          OpBuilder &opBuilder, bool duplicateProducers = false,
                          bool applyCanonicalizeAfterEachFusion = true);

  /// Split `handle` into `splitSize` parts.
  ValueHandles splitHandle(ValueHandle *handle, size_t splitSize,
                           OpBuilder &opBuilder);

  /// Apply canonicalize pass.
  void applyCanonicalization(OpBuilder &opBuilder);
  void applyCanonToTarget(OpBuilder &opBuilder, Value target);

  /// Apply common subexpression elimination pass.
  void applyCSE(OpBuilder &opBuilder);

  /// Apply `patterns` to `target`.
  void applyPatterns(
      ValueHandle *target, const SmallVector<TransformPatternKind> &patterns,
      OpBuilder &opBuilder,
      const SmallVector<CanonicalizationPatternKind> &disablePatterns = {});

  /// Apply one-shot-bufferization to the kernel function.
  void applyOneShotBufferization(OpBuilder &opBuilder);

  using RegionBuilderFn =
      llvm::function_ref<void(ImplicitLocOpBuilder &, Block &)>;

  /// Construct `transform.foreachOp` and return its results.
  ResultRange createForEachOp(Value target, TypeRange resultTypes,
                              RegionBuilderFn regionBuilder,
                              OpBuilder &opBuilder);

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

  /// Get the sequence of tiling strategy.
  mtfusion::TilingSeq getTilingSeq() { return tilingSeq_; }
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
  /// Get the to-be-tiled kernel.
  func::FuncOp getToBeTiledKernel() {
    assert(toBeTiledKernel_);
    return toBeTiledKernel_;
  }
  /// Get the name to the original kernel.
  std::string getOriginalKernelName() {
    return getOriginalKernel().getSymName().str();
  }
  /// Get the name to the to-be-tiled kernel.
  std::string getToBeTiledKernelName() {
    return getToBeTiledKernel().getSymName().str();
  }

  //===--------------------------------------------------------------------===//
  // Setter methods.
  //===--------------------------------------------------------------------===//

  /// Update the handle to transform sequence's block argument.
  void setTransformSeqHandle(Value newHandle) {
    transformSeqBlockHandle_ = newHandle;
  }
  /// Set the to-be-tiled kernel.
  void setToBeTiledKernel(func::FuncOp f) { toBeTiledKernel_ = f; }
  /// Set tiling info.
  void setTilingInfo(TilingInfo &&info) {
    tilingInfo_ = std::make_unique<TilingInfo>(std::move(info));
  }
  /// Set the original kernel.
  void setOriginalKernel(func::FuncOp f) { originalKernel_ = f; }

private:
  /// Create a tiling case and apply tiling.
  LogicalResult createAndApplyTiling(OpBuilder &opBuilder);

  /// Prepare kernel function for tiling and init tiling sequence.
  LogicalResult initTiling(OpBuilder &opBuilder);

  /// Apply one specific Tiling plan according to the input tiling info.
  LogicalResult applyTilingImpl(OpBuilder &opBuilder);

  /// Reset things after doing schedule.
  void cleanUpAfterTiling();

  /// Create switch cases for entry function to call scheduled functions
  /// according to tiling key and the callers of device kernels.
  LogicalResult fixCallSitesAndCaller(OpBuilder &opBuilder);

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
    // DenseMap<size_t, int64_t> calleeArgIdx2ConstValue{};
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

  /// Helper function to convert `tensor.empty` to
  /// `bufferization.alloc_tensor`.
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

  std::pair<int64_t, Value>
  unpackFoldResult(ValueHandleFoldResult &v, OpBuilder &opBuilder);

private:
  /// Tiling strategy sequence.
  mtfusion::TilingSeq tilingSeq_;
  /// Module enclosing the to-be-tiled kernel.
  ModuleOp module_{nullptr};
  /// Original kernel function without scheduling.
  func::FuncOp originalKernel_{nullptr};
  /// Kernel function that will be tiled.
  func::FuncOp toBeTiledKernel_{nullptr};
  /// The transform sequence block argument value.
  Value transformSeqBlockHandle_;
  /// Information regarding the to-be-tiled kernel.
  std::unique_ptr<KernelInfo> kernelInfo_{nullptr};
  /// Information regarding the tiling.
  std::unique_ptr<TilingInfo> tilingInfo_{nullptr};
  /// Record keeping all allocated value handles.
  std::unique_ptr<HandleRecord> handleRecord_{nullptr};
  /// Underlying fusion kind.
  FusionKind kind_;
};

} // namespace mtfusion
} // namespace mlir

#endif // MTIR_DIALECT_MTFUSION_TRANSFORMS_PREDEFINEDTILING_TILINGBASE_H
//===- AutoScheduleBase.cpp -- Auto-schedule fused kernels ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements auto scheduler's basic functionality.
//
//===----------------------------------------------------------------------===//

#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/AutoScheduleBase.h"
#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/AnyPBSchedule.h"
#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/LastAxisPBRSchedule.h"
#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/PureElemwiseSchedule.h"
#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/ShallowCVSchedule.h"
#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/MixCVSchedule.h"

#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/AutoScheduleAttrDefs.h"
// #include "mtir/Dialect/Annotation/IR/Annotation.h"
#include "mtir/Dialect/MtFusion/IR/MtFusion.h"
#include "mtir/Dialect/MtFusion/TransformOps/MtFusionTransformOps.h"
#include "mtir/Dialect/MtFusion/Transforms/Passes.h"
#include "mtir/Dialect/MtFusion/Utils/Utils.h"
#include "mtir/Dialect/Utils/Util.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#include <iostream>
using namespace std;

#define DEBUG_TYPE "mtfusion-auto-schedule"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] [Base Scheduler] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_AUTOSCHEDULE
#include "mtir/Dialect/MtFusion/Transforms/Passes.h.inc"

} // namespace mlir

using namespace mlir;
using namespace mlir::mtfusion;

namespace {

inline bool compareTopologicalOrdering(const std::pair<Value, size_t> &v1,
                                       const std::pair<Value, size_t> &v2) {
  auto *v1DefiningOp = std::get<0>(v1).getDefiningOp();
  auto *v2DefiningOp = std::get<0>(v2).getDefiningOp();
  assert(v1DefiningOp != nullptr);
  assert(v2DefiningOp != nullptr);
  assert(v1DefiningOp->getBlock() == v2DefiningOp->getBlock());
  return v1DefiningOp->isBeforeInBlock(v2DefiningOp);
}

SmallVector<int64_t> getReturnValueTopologicalOrdering(
    const SmallVectorImpl<Value> &funcReturnValues) {
  // Get the topological ordering of the kernel outputs
  SmallVector<int64_t> sequence =
      llvm::to_vector(llvm::seq<int64_t>(0, funcReturnValues.size()));
  SmallVector<std::pair<Value, size_t>> funcReturnValuesAndOrdering;
  for (auto [value, idx] : llvm::zip(funcReturnValues, sequence)) {
    funcReturnValuesAndOrdering.push_back({value, idx});
  }
  llvm::sort(funcReturnValuesAndOrdering, compareTopologicalOrdering);
  return llvm::map_to_vector(
      funcReturnValuesAndOrdering,
      [](const std::pair<Value, int64_t> &v) { return std::get<1>(v); });
}

transform::SequenceOp initScheduleSequence(OpBuilder &opBuilder) {
  OpBuilder::InsertionGuard g(opBuilder);
  // create transform sequence op with name
  auto seqOp = opBuilder.create<transform::SequenceOp>(
      opBuilder.getUnknownLoc(), TypeRange(),
      transform::FailurePropagationMode::Propagate,
      opBuilder.getType<transform::AnyOpType>(),
      [](OpBuilder &b, Location nested, Value rootH) {
        b.create<transform::YieldOp>(nested, ValueRange());
      });
  return seqOp;
}

/// Set a unit attribute named \c attrName to \c op.
void setNamedUnitAttr(Operation *op, StringRef attrName) {
  assert(op != nullptr);
  op->setAttr(attrName, UnitAttr::get(op->getContext()));
}

/// Collect shaped type arguments used by reshape op.
SmallVector<Value> getMaybeReshapedInputs(ArrayRef<BlockArgument> inputs) {
  SmallVector<Value> result;
  result.append(inputs.begin(), inputs.end());
  for (auto [idx, arg] : llvm::enumerate(inputs)) {
    Type argType = arg.getType();
    if (!isa<ShapedType>(argType)) {
      continue;
    }
    auto maybeArgReshaped =
        mtfusion::traceReshapeOrSliceSingleConsumerOrSelf(arg);
    LDBG("maybeArgReshaped [" << idx << "]: " << maybeArgReshaped);
    if (!mtfusion::isReshapeOrSliceOp(maybeArgReshaped.getDefiningOp())) {
      continue;
    }
    result[idx] = maybeArgReshaped;
  }
  return result;
}

SmallVector<Value> getMaybeOutputsBeforeReshape(func::FuncOp funcOp) {
  SmallVector<Value> result;
  funcOp->walk([&result](func::ReturnOp retOp) {
    result = llvm::to_vector(retOp->getOperands());
  });
  for (auto [idx, output] : llvm::enumerate(result)) {
    auto maybeOutputsBeforeReshape =
        mtfusion::traceReshapeOrSliceSingleProducerOrSelf(output);
    LDBG("maybeOutputsBeforeReshape [" << idx
                                       << "]: " << maybeOutputsBeforeReshape);
    result[idx] = maybeOutputsBeforeReshape;
  }
  return result;
}

} // namespace

//===----------------------------------------------------------------------===//
// KernelInfoCollector
//===----------------------------------------------------------------------===//

KernelInfo *KernelInfoCollector::getInfo() {
  assert(info_ != nullptr);
  return info_;
}

KernelInfo *KernelInfoCollector::getInfo() const {
  assert(info_ != nullptr);
  return info_;
}

LogicalResult KernelInfoCollector::run() {
  func::FuncOp f = this->getInfo()->originalKernel;
  if (failed(visitFuncImpl(f)))
    return failure();
  return postVisitFuncImpl(f);
}

LogicalResult KernelInfoCollector::visitFuncImpl(func::FuncOp f) {
  auto walkResult = f.getOperation()->walk([&](Operation *op) {
    auto visitStatus = TypeSwitch<Operation *, LogicalResult>(op)
                           .Case<linalg::LinalgOp>(
                               [&](Operation *op) { return visitLinalgOp(op); })
                           .Default([&](Operation *) { return success(); });
    if (failed(visitStatus)) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(walkResult.wasInterrupted());
}

LogicalResult
KernelInfoCollector::postVisitFuncImpl([[maybe_unused]] func::FuncOp f) {
  // Collect producer group info
  analyzeProducersForConsumerWithReductionAxes();

  // Mark multi buffer
  auto kernelInputs = getMaybeReshapedInputs(f.getArguments());
  // utils::MultiBufferMap multiBufferCnt;
  if (getScheduleOptions().enableAutoMultiBuffer) {
    for (auto ioValues : kernelInputs) {
      // get IOValues copied result
      auto copyOpUsers =
          llvm::make_filter_range(ioValues.getUsers(), [](Operation *user) {
            return isa<linalg::CopyOp>(user);
          });
      if (llvm::hasSingleElement(copyOpUsers)) {
        Operation *copyOp = *(copyOpUsers.begin());
        LDBG("Auto multi-buffer value: " << copyOp->getResult(0));
        // multiBufferCnt[copyOp->getResult(0)] = 2;
      }
    }
  }
  // if (failed(countMaxBuffer(multiBufferCnt)))
  //   return failure();

  return success();
}

LogicalResult KernelInfoCollector::visitLinalgOp(Operation *op) {
  // Collect the element type with smallest bits
  auto elementTypes = llvm::map_to_vector(
      llvm::make_filter_range(op->getOperandTypes(), utils::hasRank),
      [](const Type &t) { return cast<ShapedType>(t).getElementType(); });
  auto localMinType = utils::getSmallestElementType(elementTypes);
  auto globalMinType = getInfo()->smallestElementType;
  if (globalMinType == Type() || localMinType.getIntOrFloatBitWidth() <
                                     globalMinType.getIntOrFloatBitWidth()) {
    getInfo()->smallestElementType = localMinType;
  } else {
    getInfo()->smallestElementType = globalMinType;
  }
  if (auto reduceOp = dyn_cast<linalg::ReduceOp>(op)) {
    KernelInfo::ReductionOpsInfo info;
    info.idx = getInfo()->reductionOps.size();
    info.numLoops = reduceOp.getNumLoops();
    info.numResults = reduceOp->getNumResults();
    info.reductionDims.insert(reduceOp.getDimensions().begin(),
                              reduceOp.getDimensions().end());
    info.key = llvm::formatv(kReductionOpIdxFormat, info.idx).str();
    setNamedUnitAttr(op, info.key);
    getInfo()->reductionOps[reduceOp] = std::move(info);
  }
  if (auto brcOp = dyn_cast<linalg::BroadcastOp>(op)) {
    KernelInfo::BroadcastOpsInfo info;
    info.numLoops = brcOp.getNumLoops();
    info.broadcastDims.insert(brcOp.getDimensions().begin(),
                              brcOp.getDimensions().end());
    getInfo()->broadcastOps[brcOp] = std::move(info);
  }
  if (!isa<linalg::CopyOp>(op)) {
    // Tag all linalg ops as intermediate producers so that they can be
    // matched during scheduling.
    setNamedUnitAttr(op, kIntermediateProducerTagName);
  }
  return visitLinalgOpImpl(op);
}

void KernelInfoCollector::traceBackToFusableProducersForConsumer(
    Operation *consumer, KernelInfo::FusableProducers &producerInfo,
    const FusableProducerTestFn &isFusableProducer) {
  auto dpsOp = dyn_cast<DestinationStyleOpInterface>(consumer);
  if (!dpsOp)
    return;
  auto workList = dpsOp.getDpsInputOperands();
  for (OpOperand *opOperand : workList) {
    // For cases like:
    //
    // func.func @foo(%arg)
    //   %reshaped = tensor.reshape(%arg)
    //   linalg.some_op ins(%reshaped)
    //
    // and the current opOperand is `%reshaped`, we need to try to traceback
    // to the value before reshape.
    // However, this is only valid in case the reshape source is a block
    // argument. Otherwise, we cannot handle it because reshape op is not
    // tilable.
    //
    // TODO: Need to reconsider the plan to outline a function that is
    // reshape/slicing-free
    Value maybeReshapeSource =
        traceReshapeOrSliceSingleProducerOrSelf(opOperand->get());
    if (auto blockArg = maybeReshapeSource.dyn_cast<BlockArgument>()) {
      LDBG("block argument: " << blockArg.getArgNumber() << " is fusable");
      producerInfo.blockArguments.insert(blockArg.getArgNumber());
      continue;
    }
    auto *nextOperation = maybeReshapeSource.getDefiningOp();
    if (producerInfo.operations.contains(nextOperation))
      continue;

    auto [isFusable, shouldContinue] = isFusableProducer(nextOperation);
    if (isFusable) {
      LDBG("operation is fusable :" << *nextOperation);
      producerInfo.operations.insert(nextOperation);
    }
    if (shouldContinue) {
      traceBackToFusableProducersForConsumer(nextOperation, producerInfo,
                                             isFusableProducer);
    }
  }
}

void KernelInfoCollector::analyzeProducersForConsumerWithReductionAxes() {
  if (getInfo()->reductionOps.empty())
    return;
  analyzeProducersForReductionOps();
  analyzeProducersForOutputsWithReductionAxes();
}

std::pair<bool, bool>
KernelInfoCollector::isFusableProducerForConsumerWithReductionAxes(
    Operation *op, const KernelInfo::ReductionOpsInfo &reductionOpsInfo) {
  if (!isa<DestinationStyleOpInterface>(op))
    return {false, false};

  auto results = op->getResults();
  // Currently don't support multi-output operations.
  if (!llvm::hasSingleElement(results))
    return {false, false};

  auto maybeShapedType = dyn_cast<ShapedType>(results.front().getType());
  if (!maybeShapedType)
    return {false, false};

  // Cannot fuse another reduce op
  if (isa<linalg::ReduceOp>(op))
    return {false, false};

  // For broadcast ops, it's fusable if the dimensions matches reduction op.
  // However, we should only continue to fuse its producers if the broadcasted
  // dims are not the reduction dims.
  if (auto brcOp = dyn_cast<linalg::BroadcastOp>(op)) {
    auto brcDims = brcOp.getDimensions();
    auto reductionDims = reductionOpsInfo.reductionDims;
    reductionDims.set_subtract(llvm::to_vector(brcDims));
    return {/*isFusable=*/brcOp.getNumLoops() == reductionOpsInfo.numLoops,
            /*shouldContinue=*/!reductionDims.empty()};
  }

  return {true, true};
}

void KernelInfoCollector::analyzeProducersForReductionOps() {
  KernelInfo *kernelInfo = getInfo();
  auto &fusableProducerInfos = kernelInfo->fusableProducerInfos;

  std::vector<KernelInfo::FusableProducers> fp;
  for (auto [reduceOp, info] : kernelInfo->reductionOps) {
    KernelInfo::FusableProducers g;
    auto isFusableProducerTestFunc =
        std::bind(isFusableProducerForConsumerWithReductionAxes,
                  std::placeholders::_1, info);
    traceBackToFusableProducersForConsumer(reduceOp, g,
                                           isFusableProducerTestFunc);

    g.idx = info.idx;
    g.groupName =
        llvm::formatv(kReductionFusableProducerFormat, info.idx).str();
    llvm::for_each(g.operations, std::bind(setNamedUnitAttr,
                                           std::placeholders::_1, g.groupName));
    g.dump();
    fp.emplace_back(g);
  }
  fusableProducerInfos[KernelInfo::ConsumerType::kReduction] = fp;
}

void KernelInfoCollector::analyzeProducersForOutputsWithReductionAxes() {
  KernelInfo *kernelInfo = getInfo();
  auto &fusableProducerInfos = kernelInfo->fusableProducerInfos;

  std::vector<KernelInfo::FusableProducers> fp;
  // Assume that all reduction ops have the same rank before reduction.
  // TODO: Support other cases ?
  assert(!kernelInfo->reductionOps.empty());
  const auto reductionInfo = (*(kernelInfo->reductionOps.begin())).second;
  auto isFusableProducerTestFunc =
      std::bind(isFusableProducerForConsumerWithReductionAxes,
                std::placeholders::_1, reductionInfo);

  for (auto [topoIdx, outputIdx] :
       llvm::enumerate(kernelInfo->outputOrdering)) {
    auto returnVal = kernelInfo->outputValues[outputIdx];
    auto returnValRank = cast<ShapedType>(returnVal.getType()).getRank();
    // Only need to consider return values that contains reduction axes.
    if (returnValRank != reductionInfo.numLoops)
      continue;

    KernelInfo::FusableProducers g;
    auto *returnDefiningOp = returnVal.getDefiningOp();
    // The return value's defining op is also a fusable producer because it
    // will be cache written to.
    g.operations.insert(returnDefiningOp);
    traceBackToFusableProducersForConsumer(returnDefiningOp, g,
                                           isFusableProducerTestFunc);

    g.idx = topoIdx;
    g.groupName =
        llvm::formatv(kReturnValueFusableProducerFormat, topoIdx).str();
    llvm::for_each(g.operations, std::bind(setNamedUnitAttr,
                                           std::placeholders::_1, g.groupName));
    g.dump();
    fp.emplace_back(g);
  }
  fusableProducerInfos[KernelInfo::ConsumerType::kOutput] = fp;
}

// LogicalResult KernelInfoCollector::countMaxBuffer(
//     const utils::MultiBufferMap &multiBufferCnt) {
//   KernelInfo *info = getInfo();
//   OpBuilder opBuilder(info->originalKernel.getContext());

//   // Broadcast and reduction ops should be aligned.
//   auto alignmentAttr =
//       hivm::AlignKindAttr::get(opBuilder.getContext(), hivm::AlignKind::ALIGN);

//   SetVector<Operation *> reductionOps;
//   for (auto [key, _] : info->reductionOps) {
//     reductionOps.insert(key);
//   }
//   SetVector<Operation *> broadcastOps;
//   for (auto [key, _] : info->broadcastOps) {
//     broadcastOps.insert(key);
//   }
//   SmallVector<Operation *> markOps;
//   for (Operation *op : llvm::concat<Operation *>(
//            llvm::to_vector(broadcastOps), llvm::to_vector(reductionOps))) {
//     opBuilder.setInsertionPointAfter(op);
//     // Extra buffer size is inferred from broadcast/reduction op's result.
//     auto markOp =
//         opBuilder.create<annotation::MarkOp>(op->getLoc(), op->getResult(0));
//     markOp->setAttr(hivm::AlignKindAttr::name, alignmentAttr);
//     markOps.push_back(markOp);
//   }
//   LDBG("---Counting max buffer ...");
//   LDBG("-----Func after annotation: \n" << info->originalKernel);
//   std::optional<int64_t> maxBufferCnt =
//       utils::countMaxBuffer(info->originalKernel,
//                             /*printLiveRange=*/false,
//                             /*multiBufferCnt=*/multiBufferCnt);
//   // Erase marks after counting max buffer because it keeps tensor's from being
//   // dce.
//   for (auto *op : llvm::reverse(markOps))
//     op->erase();

//   if (maxBufferCnt.has_value()) {
//     int64_t maxBufferCntInitVal = maxBufferCnt.value();
//     assert(maxBufferCntInitVal > 1);

//     int tuningDelta = getScheduleOptions().maxBufferCntTuning;
//     int64_t maxBufferCntAfterTuning = maxBufferCntInitVal + tuningDelta;
//     maxBufferCntAfterTuning = std::max((int64_t)1, maxBufferCntAfterTuning);
//     info->maxBufferCnt = maxBufferCntAfterTuning;
//     LDBG("-----Max buffer count is: " << info->maxBufferCnt);
//     return success();
//   }
//   return info->originalKernel.emitError("Max buffer count is zero!");
// }

//===----------------------------------------------------------------------===//
// SchedulerBase
//===----------------------------------------------------------------------===//

/// Init static data members.
AutoScheduleOptions SchedulerBase::options_ = AutoScheduleOptions();

SchedulerBase::SchedulerBase(func::FuncOp f, FusionKind kind) {
  kernelInfo_ = std::make_unique<KernelInfo>();
  tilingInfo_ = std::make_unique<TilingInfo>();
  handleRecord_ = std::make_unique<HandleRecord>();
  originalKernel_ = f;
  module_ = f.getOperation()->getParentOfType<ModuleOp>();
  kind_ = kind;
}

SchedulerBase::SchedulerBase(func::FuncOp f,
                             std::unique_ptr<KernelInfo> kernelInfo,
                             std::unique_ptr<TilingInfo> tilingInfo) {
  kernelInfo_ = std::move(kernelInfo);
  tilingInfo_ = std::move(tilingInfo);
  handleRecord_ = std::make_unique<HandleRecord>();
  originalKernel_ = f;
  module_ = f.getOperation()->getParentOfType<ModuleOp>();
  kind_ = kernelInfo_->getFusionKind();
}

SchedulerBase::~SchedulerBase() {
  kernelInfo_.reset();
  tilingInfo_.reset();
  handleRecord_.reset();
}

LogicalResult SchedulerBase::runPreScheduleProcedure(OpBuilder &opBuilder) {
  func::FuncOp currentFunc = getOriginalKernel();
  cout<<"### Entering SchedulerBase::runPreScheduleProcedure"<<endl;

  if (failed(cacheIO(opBuilder)))
    return currentFunc->emitWarning("Failed to cache inputs/outputs.");
  cout<<"### After cacheIO"<<endl;

  if (failed(analyzeAndVerifyKernel()))
    return currentFunc->emitWarning("Failed to analyze and verify kernel.");
  return success();
}

LogicalResult SchedulerBase::runScheduleProcedure(OpBuilder &opBuilder) {
  cout<<"### Entering SchedulerBase::runScheduleProcedure"<<endl;
  func::FuncOp currentFunc = getOriginalKernel(); 

  if (failed(calculateTiling(opBuilder)))
    return currentFunc->emitWarning("Failed to calculate tiling.");
  
  if (failed(selectTiling()))
    return currentFunc->emitWarning("Failed to select tiling.");

  if (failed(createAndApplySchedules(opBuilder)))
    return currentFunc->emitWarning("Failed to create and apply schedule.");
  return success();
}

LogicalResult SchedulerBase::runOnOperation(OpBuilder &opBuilder) {
  cout<<"### Entering SchedulerBase::runOnOperation"<<endl;

  if (failed(runPreScheduleProcedure(opBuilder)))
    return failure();

  if (failed(runScheduleProcedure(opBuilder)))
    return failure();
  return success();
}

LogicalResult SchedulerBase::analyzeAndVerifyKernel() {
  KernelInfo *info = getKernelInfo();
  info->originalKernel = getOriginalKernel();
  auto funcArgs = info->originalKernel.getArguments();
  FunctionType funcType = info->originalKernel.getFunctionType();
  info->numInputs = funcType.getNumInputs();
  info->baseKernelName = info->originalKernel.getSymName().str();
  info->numOutputs = funcType.getNumResults();
  info->inputValues = getMaybeReshapedInputs(funcArgs);
  info->outputValues = getMaybeOutputsBeforeReshape(info->originalKernel);
  info->outputOrdering = getReturnValueTopologicalOrdering(info->outputValues);
  auto getType = [](Value v) { return v.getType(); };
  info->inputTypes = llvm::map_to_vector(info->inputValues, getType);
  info->outputTypes = llvm::map_to_vector(info->outputValues, getType);

  // Get the list of block arguments that are used as dps init operands, and
  // whose tied result value is also the kernel return value.
  // The return value should enable `cache_write_to_output_init` option when
  // performing cache write.
  for (auto [idx, ba] : llvm::enumerate(funcArgs)) {
    bool funcArgIsReshaped = false;
    bool funcResultIsReshaped = false;
    if (auto resultIdx = mtfusion::getFuncArgTiedResultReturnIdx(
            ba, funcArgIsReshaped, funcResultIsReshaped)) {
      LDBG("Arg number : " << idx << " is tied to result number: "
                           << resultIdx.value());
      info->funcArgIdxWithTiedReturnValue.insert(idx);
      info->returnValueIdx2TiedFuncArg.insert({resultIdx.value(), idx});
      if (funcResultIsReshaped) {
        LDBG("Result number : " << idx << " is reshaped before return");
        info->returnValueWithReshapeIndices.insert(resultIdx.value());
      }
    }
    if (funcArgIsReshaped) {
      LDBG("Arg number : " << idx << " is reshaped before use");
      info->funcArgWithReshapeIndices.insert(idx);
    }
  }

  // The block arguments that need to do cache read are:
  //   - shaped arguments
  //   - arguments that are not tied to results
  auto funcArgIndices = llvm::to_vector(llvm::seq<int64_t>(0, info->numInputs));
  auto filteredIndices =
      llvm::make_filter_range(funcArgIndices, [&info](int64_t idx) {
        return !info->funcArgIdxWithTiedReturnValue.contains(idx) &&
               isa<ShapedType>(info->inputTypes[idx]);
      });
  info->cacheReadFuncArgIndices =
      SetVector<int64_t>{filteredIndices.begin(), filteredIndices.end()};

  return analyzeAndVerifyKernelImpl();
}

LogicalResult SchedulerBase::analyzeAndVerifyKernelImpl() {
  return KernelInfoCollector(getKernelInfo(), getAutoScheduleOptions()).run();
}

LogicalResult SchedulerBase::cacheIO(OpBuilder &opBuilder) {
  auto originalKernel = getOriginalKernel();
  // Perform cache IO
  cacheFuncIO(originalKernel, /*annotate=*/true);
  reorderOpsByBFS(originalKernel);
  /// Move result's tied init operands to function arguments if necessary.
  /// This is to handle cases like:
  /// ```mlir
  ///   func @foo(%arg)
  ///     return %arg
  /// ```
  /// which does not have any computation ops before auto-schedule. Need to
  /// perform the transformation again to the kernel with cache IO.
  PassManager pm(getContext());
  SmallVector<std::string> includeSymbols = {originalKernel.getSymName().str()};
  pm.addPass(mtfusion::createTensorResToOutParamsPass(includeSymbols));
  if (failed(pm.run(getModule())))
    return failure();
  return success();
}

LogicalResult SchedulerBase::calculateTiling(OpBuilder &opBuilder) {
  cout<<"### Entering SchedulerBase::calculateTiling"<<endl;
  OpBuilder::InsertionGuard g(opBuilder);
  TilingInfo *tilingInfo = getTilingInfo();
  MLIRContext *ctx = getContext();

  // Step 1. Get tiling compute function.
  TilingComputeFn fn = calculateTilingImpl();

  // Bail out if the derived scheduler does not require tiling
  if (!fn)
    return success();

  // Step 2. Create host tiling function.
  func::FuncOp originalKernel = getOriginalKernel();
  opBuilder.setInsertionPoint(originalKernel);
  FunctionType t =
      FunctionType::get(ctx,
                        /*inputs=*/originalKernel.getFunctionType().getInputs(),
                        /*results=*/
                        SmallVector<Type>());
  auto hostTilingFunc = opBuilder.create<func::FuncOp>(
      originalKernel.getLoc(),
      /*name=*/originalKernel.getSymName().str() + "_tiling_func",
      /*type=*/t);
  Block *entryBlock = hostTilingFunc.addEntryBlock();
  hostTilingFunc->setAttr(
      mtfusion::FuncKindAttr::name,
      mtfusion::FuncKindAttr::get(ctx, mtfusion::FuncKind::Host));

  // Step 3. Record host tiling function.
  tilingInfo->setHostTilingFunc(hostTilingFunc);

  // Step 4. Set block dim information.
  // TODO: obtain num core info from platform config
  tilingInfo->setBlockDim(SchedulerBase::getBlockDim());

  // Step 5. Construct ExprBuilder and set insertion point into host tiling
  // func.
  ExprBuilder exprBuilder(tilingInfo, getKernelInfo(), ctx);
  exprBuilder.setInsertionPointToStart(entryBlock);

  // Step 6. Evaluate tiling computation function and produce IR.
  auto returns =
      tilingInfo->evaluateTilingComputation(fn, getKernelInfo(), &exprBuilder);

  // Step 7: Return tiling data.
  opBuilder.setInsertionPointToEnd(entryBlock);
  opBuilder.create<func::ReturnOp>(originalKernel.getLoc(), returns);

  // Step 8: Update function type because for some fusion kind, the number of
  // tiling keys is kernel-dependent.
  hostTilingFunc.setFunctionType(t.clone(
      /*inputs=*/originalKernel.getFunctionType().getInputs(),
      /*results=*/SmallVector<Type>(returns.size(),
                                    opBuilder.getIntegerType(64))));
  LDBG("--Generated Tiling Func: \n" << *hostTilingFunc);
  return success();
}

LogicalResult SchedulerBase::selectTiling() {
  cout<<"### Entering SchedulerBase::selectTiling"<<endl;
  TilingInfo *tilingInfo = getTilingInfo();

  // Try to simplify host tiling func
  if (failed(tilingInfo->trySimplifyTilingFunc())) {
    LDBG("Failed to simplify host tiling func");
    return success();
  }

  LDBG("--Simplified Tiling Func: \n" << *tilingInfo->getHostTilingFunc());

  TilingData *tilingKey = tilingInfo->getTilingKey();
  // Cannot constantize tiling, all cases need to be generated
  if (!tilingKey->isConst()) {
    LDBG("Cannot constantize tiling");
    return success();
  }

  int64_t selectedTilingKey = tilingKey->getConst();
  LDBG("Selected tiling key: " << selectedTilingKey);
  // Prune tiling
  tilingInfo->pruneTilingExcept(selectedTilingKey);
  return success();
}

LogicalResult SchedulerBase::createAndApplySchedules(OpBuilder &opBuilder) {
  // iterate over tiling cases
  TilingInfo *info = getTilingInfo();

  for (TilingKey key : info->getTilingCases()) {
    LDBG("Creating schedule for tiling key: " << key);
    if (failed(initSchedule(key, opBuilder))) {
      return failure();
    }
    if (failed(createScheduleImpl(key, opBuilder))) {
      return failure();
    }
    LDBG("Dumping kernel and schedule for tiling key: " << key);
    dumpKernelAndSchedule();
    if (failed(applyScheduleImpl(opBuilder))) {
      return failure();
    }
    cleanUpAfterSchedule();
  }

  if (failed(fixCallSitesAndCaller(opBuilder)))
    return failure();

  LDBG("Removing original func...");
  getOriginalKernel()->erase();
  return success();
}

LogicalResult SchedulerBase::applySchedule(func::FuncOp &funcOp,
                                           OpBuilder &opBuilder) {
  auto fusionKindAttr =
      funcOp->getAttrOfType<FusionKindAttr>(FusionKindAttr::name);
  if (!fusionKindAttr || !mtfusion::isDevice(funcOp)) {
    LDBG("Unknown kernel fusion kind");
    return success();
  }
  auto fusionKind = fusionKindAttr.getFusionKind();
  std::unique_ptr<SchedulerBase> scheduler;
  switch (fusionKind) {
  case FusionKind::PureElemwise:
  case FusionKind::AnyPB:
    scheduler = std::make_unique<AnyPBScheduler>(funcOp);
    break;
  case FusionKind::LastAxisPBR:
    scheduler = std::make_unique<LastAxisPBRScheduler>(funcOp);
    break;
  case FusionKind::MixCV:
    scheduler = std::make_unique<MixCVScheduler>(funcOp);
    break;
  case FusionKind::ShallowCV:
    scheduler = std::make_unique<ShallowCVScheduler>(funcOp);
    break;
  case FusionKind::Unknown:
  default:
    return funcOp.emitError("Unknown kernel fusion kind");
  }
  return scheduler->runOnOperation(opBuilder);
}

LogicalResult SchedulerBase::applyScheduleImpl(OpBuilder &opBuilder) {
  PassManager pm(getContext());
  pm.addPass(
      mtfusion::createAutoScheduleInterpreterPass(getToBeScheduledKernelName()));
  pm.addPass(
      mtfusion::createEraseAutoSchedulePass(getToBeScheduledKernelName()));

  if (failed(pm.run(getModule())))
    return failure();
  return success();
}

void SchedulerBase::dumpKernelAndSchedule() {
  Operation *transformSeqOp = nullptr;
  getModule().walk([&](Operation *nestedOp) {
    if (!nestedOp->hasAttrOfType<StringAttr>(kTransformDialectTagAttrName))
      return WalkResult::advance();

    if (isa<transform::TransformOpInterface>(nestedOp) &&
        nestedOp->getAttrOfType<StringAttr>(kTransformDialectTagAttrName)
                .str() ==
            auto_schedule::getTransformRootTag(getToBeScheduledKernelName())) {
      assert(transformSeqOp == nullptr &&
             "transform seq with duplicate tags found!");
      transformSeqOp = nestedOp;
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });
  assert(transformSeqOp && "cannot find target transform seq");
  LDBG("---Current to-be-scheduled kernel func: \n"
       << *getToBeScheduledKernel());
  LDBG("---Current transform sequence: \n" << *transformSeqOp);
}

void SchedulerBase::cleanUpAfterSchedule() {
  getHandleRecord()->clear();
  setToBeScheduledKernel(nullptr);
  setTransformSeqHandle(Value());
  for (auto *td : getTilingInfo()->getTilingStruct())
    td->setHandle(nullptr);
}

LogicalResult SchedulerBase::initSchedule(TilingKey key, OpBuilder &opBuilder) {
  TilingInfo *tilingInfo = getTilingInfo();
  assert(tilingInfo != nullptr);

  // Step 1: Construct new kernel name with tiling key post-fix.
  func::FuncOp originalKernel = getOriginalKernel();
  auto toBeScheduledKernelName =
      originalKernel.getSymName().str() + "_" + std::to_string(key);
  auto module = getModule();
  if (module.lookupSymbol(opBuilder.getStringAttr(toBeScheduledKernelName))) {
    return originalKernel->emitError(
        "Duplicate kernel name during auto-scheduling process");
  }

  // Step 2: Clone original func and set it as the to-be-scheduled function.
  opBuilder.setInsertionPoint(originalKernel);
  func::FuncOp toBeScheduleKernel =
      cast<func::FuncOp>(opBuilder.clone(*originalKernel));
  toBeScheduleKernel.setSymName(toBeScheduledKernelName);
  this->setToBeScheduledKernel(toBeScheduleKernel);
  // Bind tiling key to the to-be-scheduled kernel.
  tilingInfo->recordKernelFunc(key, toBeScheduleKernel);

  // Step 3. Insert tiling data to to-be-scheduled kernel function and bind
  // tiling data to kernel argument
  auto argIdx = toBeScheduleKernel.getNumArguments();
  for (auto *iter = tilingInfo->tilingDataBegin();
       iter != tilingInfo->tilingDataEnd(); iter++) {
    TilingData *td = iter->get();
    td->setPos(argIdx);

    SmallVector<NamedAttribute> argAttrs = {getTilingDataAttr(opBuilder)};
    if (iter == tilingInfo->tilingDataBegin())
      argAttrs.push_back(getTilingKeyAttr(opBuilder));

    toBeScheduleKernel.insertArgument(argIdx, td->getType(),
                                      opBuilder.getDictionaryAttr(argAttrs),
                                      toBeScheduleKernel.getLoc());
    argIdx++;
  }

  // Step 4. Insert transform sequence right after the to-be-scheduled kernel.
  opBuilder.setInsertionPointAfter(toBeScheduleKernel);
  auto seqOp = initScheduleSequence(opBuilder);
  auto *transformBody = seqOp.getBodyBlock();
  // Set insertion point to transform sequence body
  opBuilder.setInsertionPointToStart(transformBody);
  // Record transform sequence block argument
  setTransformSeqHandle(transformBody->getArguments().front());

  // Step 5. Set attributes to various functions
  auto blockDimIntAttr = opBuilder.getIntegerAttr(opBuilder.getIntegerType(64),
                                                  tilingInfo->getBlockDim());
  toBeScheduleKernel->setAttr(mtfusion::BlockDimAttr::name, blockDimIntAttr);
  toBeScheduleKernel->setAttr(
      mtfusion::TilingFuncAttr::name,
      opBuilder.getStringAttr(tilingInfo->getHostTilingFunc().getSymName()));

  // Set transform root and payload root tags
  toBeScheduleKernel->setAttr(
      kTransformDialectTagAttrName,
      opBuilder.getStringAttr(
          auto_schedule::getPayloadRootTag(toBeScheduledKernelName)));
  seqOp->setAttr(kTransformDialectTagAttrName,
                 opBuilder.getStringAttr(auto_schedule::getTransformRootTag(
                     toBeScheduledKernelName)));

  return success();
}

void SchedulerBase::getCallerInfo(func::FuncOp callee, ModuleOp enclosingModule,
                                  DenseMap<func::FuncOp, CallerInfo> &info) {
  std::optional<SymbolTable::UseRange> maybeUses =
      callee.getSymbolUses(enclosingModule);
  for (SymbolTable::SymbolUse use : maybeUses.value()) {
    func::CallOp callSite = cast<func::CallOp>(use.getUser());
    auto callerOp = callSite->getParentOfType<func::FuncOp>();
    auto &callerInfo = info[callerOp];
    callerInfo.caller = callerOp;
    callerInfo.callerOriginalArgNumber = callerOp.getNumArguments();
    callerInfo.callee = callee;
    callerInfo.callSites.push_back(callSite);
  }
}

SmallVector<Value> SchedulerBase::getNewArgsForCallSite(
    func::FuncOp caller, func::CallOp oldCallSite,
    const SchedulerBase::CallSiteArgBuilderInfo &info, OpBuilder &opBuilder) {
  auto oldCallArgs = oldCallSite->getOperands();
  size_t oldArgCount = oldCallArgs.size();
  size_t tilingStructSize = info.tilingIdx2CallerArgIdx.size();
  size_t newArgCount = oldArgCount + tilingStructSize;

  SmallVector<Value> newCallArgs;
  newCallArgs.reserve(newArgCount);
  // By convention, tiling data is appended after existing args, but the order
  // to which they're added matches the tiling struct order
  newCallArgs.append(oldCallArgs.begin(), oldCallArgs.end());
  newCallArgs.append(SmallVector<Value>(tilingStructSize, Value()));

  for (size_t idx = oldArgCount; idx < newArgCount; idx++) {
    if (!info.calleeIsOriginalKernel ||
        !info.calleeArgIdx2ConstValue.contains(idx)) {
      // If the callee is not the original device kernel function,
      // or if the tiling data is not constant, the new call args should
      // be the caller's function argument.
      auto tilingIdx = idx - oldArgCount;
      assert(info.tilingIdx2CallerArgIdx.contains(tilingIdx));
      newCallArgs[idx] =
          caller.getArgument(info.tilingIdx2CallerArgIdx.at(tilingIdx));
      continue;
    }
    Dialect *arithDialect =
        opBuilder.getContext()->getLoadedDialect<arith::ArithDialect>();
    assert(arithDialect);
    auto newCallArg =
        arithDialect
            ->materializeConstant(opBuilder,
                                  opBuilder.getI64IntegerAttr(
                                      info.calleeArgIdx2ConstValue.at(idx)),
                                  opBuilder.getI64Type(), oldCallSite->getLoc())
            ->getResult(0);
    newCallArgs[idx] = newCallArg;
  }
  return newCallArgs;
}

void SchedulerBase::doFixCallSite(CallerInfo &callerInfo, func::CallOp callSite,
                                  CallSiteArgBuilderInfo &builderInfo,
                                  DenseMap<Operation *, Operation *> &irMap,
                                  OpBuilder &opBuilder) {
  OpBuilder::InsertionGuard g(opBuilder);

  TilingInfo *tilingInfo = getTilingInfo();
  auto tilingKey2Kernel = tilingInfo->getTilingKey2KernelMap();
  assert(!tilingKey2Kernel.empty());

  opBuilder.setInsertionPoint(callSite);
  auto newArgs = getNewArgsForCallSite(callerInfo.caller, callSite, builderInfo,
                                       opBuilder);

  // If the callee is not the original kernel, generate a new call
  // with the same callee but new args and bail out.
  if (!builderInfo.calleeIsOriginalKernel) {
    func::CallOp newCallSite = opBuilder.create<func::CallOp>(
        callSite.getLoc(), callSite.getResultTypes(), callSite.getCallee(),
        newArgs);
    irMap.insert(std::make_pair(callSite, newCallSite));
    return;
  }
  auto tilingKeys = tilingInfo->getTilingCases();
  // If there is only one tiling key, generate a new call with new args and
  // bail out.
  if (tilingKey2Kernel.size() == 1) {
    func::FuncOp kernelFunc = tilingKey2Kernel.at(tilingKeys[0]);
    func::CallOp newCallSite = opBuilder.create<func::CallOp>(
        callSite.getLoc(), callSite.getResultTypes(), kernelFunc.getSymName(),
        newArgs);
    irMap.insert(std::make_pair(callSite, newCallSite));
    return;
  }
  // Otherwise, create switch cases to different tiling cases.
  // Assume that tiling key is first extra argument in the caller.
  Value tilingKey =
      callerInfo.caller.getArgument(callerInfo.callerOriginalArgNumber);
  tilingKey = castToIndex(tilingKey, opBuilder);

  scf::IndexSwitchOp switchOp = opBuilder.create<scf::IndexSwitchOp>(
      callSite.getLoc(), callSite.getResultTypes(), tilingKey,
      tilingKeys.getArrayRef(), tilingKeys.size());
  for (Region &region : switchOp.getCaseRegions())
    region.emplaceBlock();

  for (size_t i = 0, e = switchOp.getNumCases(); i != e; ++i) {
    opBuilder.setInsertionPointToStart(&switchOp.getCaseBlock(i));
    func::FuncOp kernelFunc = tilingKey2Kernel.at(tilingKeys[i]);
    func::CallOp newCallSite = opBuilder.create<func::CallOp>(
        callSite.getLoc(), callSite.getResultTypes(), kernelFunc.getSymName(),
        newArgs);
    opBuilder.create<scf::YieldOp>(callSite->getLoc(),
                                   newCallSite.getResults());
  }

  switchOp.getDefaultRegion().emplaceBlock();
  opBuilder.setInsertionPointToStart(&switchOp.getDefaultBlock());
  Value constFalse = opBuilder.create<arith::ConstantOp>(
      callSite.getLoc(), opBuilder.getI1Type(), opBuilder.getBoolAttr(false));
  opBuilder.create<cf::AssertOp>(callSite.getLoc(), constFalse,
                                 "Invalid tiling key");

  Value undefValue = opBuilder.create<ub::PoisonOp>(callSite.getLoc(),
                                                    callSite.getResultTypes());
  opBuilder.create<scf::YieldOp>(callSite.getLoc(), undefValue);
  irMap.insert(std::make_pair(callSite, switchOp));
}

/// Fix direct and indirect callers of the unscheduled kernel.
///
/// Original IR:
/// \code
/// func.func private @unschedule_kernel()
/// func.func private @schedule_kernel_1(%tiling_key, %cst_td, %var_td)
/// func.func private @schedule_kernel_2(%tiling_key, %cst_td, %var_td)
///
/// func.func @nested_caller() {
///    func.call @unschedule_kernel()
/// }
/// func.func @caller() {
///   func.call @nested_caller()
/// }
/// \endcode
///
/// After fixing call sites:
/// \code
/// func.func private @unschedule_kernel()
/// func.func private @schedule_kernel_1(%tiling_key, %cst_td, %var_td)
/// func.func private @schedule_kernel_2(%tiling_key, %cst_td, %var_td)
///
/// func.func @nested_caller(%tiling_key, %cst_td, %var_td) {
///    %cst_td = arith.const ...
///    %cst_tiling_key_1 = arith.const ...
///    %cst_tiling_key_2 = arith.const ...
///    scf.index_switch %tiling_key
///      case %cst_tiling_key_1 :
///        func.call @schedule_kernel_1(%tiling_key, %cst_td, %var_td)
///      case %cst_tiling_key_2 :
///        func.call @schedule_kernel_1(%tiling_key, %cst_td, %var_td)
/// }
/// func.func @caller(%tiling_key, %cst_td, %var_td) {
///   func.call @nested_caller(%tiling_key, %cst_td, %var_td)
/// }
/// \endcode
LogicalResult SchedulerBase::fixCallSitesAndCaller(OpBuilder &opBuilder) {
  OpBuilder::InsertionGuard g(opBuilder);
  LDBG("Fixing call sites of un-scheduled func...");

  TilingInfo *tilingInfo = getTilingInfo();
  auto tilingKey2Kernel = tilingInfo->getTilingKey2KernelMap();
  assert(!tilingKey2Kernel.empty());

  // Get position and value of constant tiling data
  DenseMap<size_t, int64_t> calleeArgIdx2ConstValue;
  for (auto *iter = tilingInfo->tilingDataBegin();
       iter != tilingInfo->tilingDataEnd(); iter++) {
    const TilingData *td = iter->get();
    if (!td->isConst()) {
      continue;
    }
    calleeArgIdx2ConstValue.insert({td->getPos(), td->getConst()});
  }
  size_t tilingStructSize = tilingInfo->size();

  // Get callers of the original, unscheduled kernel
  DenseMap<func::FuncOp, CallerInfo> workList;
  getCallerInfo(getOriginalKernel(), getModule(), workList);

  // Bail out on trivial case where there is no caller
  if (workList.empty()) {
    // TODO: If there is only one tiling case, don't modify the kernel name for
    // now. Modify this after we fully switch to support dynamic/static shape +
    // multiple tiling
    auto tilingCases = tilingInfo->getTilingCases();
    if (tilingCases.size() == 1)
      tilingKey2Kernel[tilingCases[0]].setSymName(getOriginalKernelName());
    return success();
  }

  // Repeatedly modify the caller and call site, until there is no caller.
  DenseMap<Operation *, Operation *> irMap;
  DenseSet<func::FuncOp> processedCaller;
  while (!workList.empty()) {
    auto &[caller, callerInfo] = *(workList.begin());
    if (processedCaller.contains(caller)) {
      LDBG("Cyclic call detected");
      return failure();
    }
    LDBG("Fixing call site in: \n" << *caller);

    size_t callerArgCount = caller.getNumArguments();
    getCallerInfo(caller, getModule(), workList);

    // Insert tiling data arguments to caller's enclosing func
    // TODO: Refactor
    DenseMap<size_t, size_t> tilingIdx2CallerArgIdx;
    for (size_t idx = callerArgCount; idx < callerArgCount + tilingStructSize;
         idx++) {
      SmallVector<NamedAttribute> argAttrs = {getTilingDataAttr(opBuilder)};
      if (idx == callerArgCount)
        argAttrs.push_back(getTilingKeyAttr(opBuilder));
      caller.insertArgument(idx, opBuilder.getI64Type(),
                            opBuilder.getDictionaryAttr(argAttrs),
                            caller.getLoc());
      tilingIdx2CallerArgIdx.insert({idx - callerArgCount, idx});
    }

    // Fix the call sites
    bool calleeIsOriginalKernel = callerInfo.callee == getOriginalKernel();
    CallSiteArgBuilderInfo builderInfo{tilingIdx2CallerArgIdx,
                                       calleeArgIdx2ConstValue,
                                       calleeIsOriginalKernel};
    for (func::CallOp callSite : callerInfo.callSites) {
      doFixCallSite(callerInfo, callSite, builderInfo, irMap, opBuilder);
    }

    processedCaller.insert(caller);
    workList.erase(caller);
  }

  for (auto &[oldOp, newOp] : irMap) {
    oldOp->replaceAllUsesWith(newOp);
    oldOp->erase();
  }

  return success();
}

NamedValueHandle *SchedulerBase::recordImpl(Value target, OpBuilder &opBuilder,
                                            const NamedValueHandleArgs &args) {
  // If the identifier type is operation name, then it's already unique.
  std::string newName = args.type == IdentifierType::kAttribute
                            ? getHandleRecord()->getAndRecordAttrName(args.name)
                            : args.name.str();

  if (args.needsReverse)
    target = opBuilder.create<transform::ReverseOp>(
        target.getLoc(),
        /*result=*/TypeRange{opBuilder.getType<transform::AnyOpType>()},
        /*target=*/target);

  if (args.needsAnnotate)
    opBuilder.create<transform::AnnotateOp>(
        target.getLoc(),
        /*target=*/target,
        /*name=*/opBuilder.getStringAttr(newName),
        /*param=*/Value{});

  return new NamedValueHandle(target, newName, args.type, HandleStatus::kValid,
                              args.needsReverse);
}

RegularValueHandle *
SchedulerBase::recordImpl(Value target, [[maybe_unused]] OpBuilder &opBuilder) {
  return new RegularValueHandle(target, HandleStatus::kValid);
}

FuncArgHandle *SchedulerBase::recordImpl(Value target,
                                         [[maybe_unused]] OpBuilder &opBuilder,
                                         size_t funcArgNum) {
  return new FuncArgHandle(target, funcArgNum, HandleStatus::kValid);
}

// LogicalResult
// SchedulerBase::applyOpFlattenPass(Operation *target,
//                                   const FlattenOpsOptions &options) {
//   PassManager pm(getContext());
//   pm.addPass(mtfusion::createFlattenOpsPass(options));
//   if (failed(pm.run(target)))
//     return failure();
//   RewritePatternSet patterns(getContext());
//   tensor::populateFoldTensorEmptyPatterns(patterns);
//   if (failed(applyPatternsAndFoldGreedily(target, std::move(patterns))))
//     return failure();
//   return success();
// }

FailureOr<SmallVector<func::FuncOp>>
SchedulerBase::applyOpFusionOutline(func::FuncOp target,
                                    const MtFusionOpFusionOptions &options) {

  SmallVector<func::FuncOp> outlinedFuncs;
  if (failed(outlineFusedFuncs(target, options, outlinedFuncs)))
    return failure();

  return outlinedFuncs;
}

namespace {
struct AutoSchedulePass : public impl::AutoScheduleBase<AutoSchedulePass> {
  using AutoScheduleBase<AutoSchedulePass>::AutoScheduleBase;

  explicit AutoSchedulePass(const AutoScheduleOptions &options)
      : AutoScheduleBase(options) {}

  void runOnOperation() override;
};

} // namespace

void AutoSchedulePass::runOnOperation() {
  AutoScheduleOptions options;
  options.blockDim = this->blockDim;
  options.enableAutoMultiBuffer = this->enableAutoMultiBuffer;
  options.maxBufferCntTuning = this->maxBufferCntTuning;
  SmallVector<func::FuncOp> funcList;
  getOperation()->walk([&](func::FuncOp func) { funcList.push_back(func); });

  SchedulerBase::setAutoScheduleOptions(options);
  for (auto &func : funcList) {
    OpBuilder opBuilder(&getContext());
    if (succeeded(SchedulerBase::applySchedule(func, opBuilder)))
      continue;

    func->emitOpError("Failed to create and apply schedule.");
    return signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::mtfusion::createMtFusionAutoSchedulePass(
    const AutoScheduleOptions &options) {
  return std::make_unique<AutoSchedulePass>(options);
}
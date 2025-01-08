//===- TilingOperations.cpp -- Predefined tiling operation Impl.---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements tiling operations.
//
//===----------------------------------------------------------------------===//

#include "mtir/Dialect/MtFusion/Transforms/PredefinedTiling/TilingBase.h"
#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/AutoScheduleAttrDefs.h"
#include "mtir/Dialect/MtFusion/IR/MtFusion.h"
#include "mtir/Dialect/MtFusion/TransformOps/MtFusionTransformOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/TransformOps/BufferizationTransformOps.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "mtfusion-predefined-tiling"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] [Base Tiler] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::mtfusion;

namespace {
std::string stringfyCanonicalizationPatternKind(
    mtfusion::detail::CanonicalizationPatternKind kind) {
  switch (kind) {
  case mtfusion::detail::CanonicalizationPatternKind::kSimplifyTrivialLoops:
    return "SimplifyTrivialLoops";
  default:
    llvm_unreachable("Unsupported transform pattern kin");
    return "";
  }
}
} // namespace

Value TilerBase::getValue(ValueHandle *handle, OpBuilder &opBuilder) {
  assert(handle != nullptr);
  if (auto *h = dyn_cast<RegularValueHandle>(handle)) {
    return h->get();
  }
  if (auto *h = dyn_cast<NamedValueHandle>(handle)) {
    return h->get(getTransformSeqHandle(), opBuilder);
  }
  if (auto *h = dyn_cast<FuncArgHandle>(handle)) {
    return h->get(getFuncValue(opBuilder), opBuilder);
  }
  llvm_unreachable("Not implemented!");
}

SmallVector<Value> TilerBase::getValues(const ValueHandles &handle,
                                        OpBuilder &opBuilder) {
  return llvm::map_to_vector(handle, [this, &opBuilder](ValueHandle *handle) {
    return this->getValue(handle, opBuilder);
  });
}

std::pair<SmallVector<int64_t>, SmallVector<Value>>
TilerBase::unpackFoldResults(ValueHandleFoldResults &values,
                            OpBuilder &opBuilder) {
  SmallVector<int64_t> staticSizes;
  SmallVector<Value> dynamicSizes;
  for (auto &v : values) {
    auto maybeConstInteger = v.getConstInteger();
    if (maybeConstInteger) {
      staticSizes.push_back(maybeConstInteger.value());
      continue;
    }
    staticSizes.push_back(ShapedType::kDynamic);
    dynamicSizes.push_back(getValue(*v.getValueHandle(), opBuilder));
  }
  return {staticSizes, dynamicSizes};
}

std::pair<int64_t, Value>
TilerBase::unpackFoldResult(ValueHandleFoldResult &v,
                            OpBuilder &opBuilder) {
  auto maybeConstInteger = v.getConstInteger();
  if (maybeConstInteger) {
    return {maybeConstInteger.value(), Value()};
  }
  int64_t staticSize = ShapedType::kDynamic;
  Value dynamicSize = getValue(*v.getValueHandle(), opBuilder);
  return {staticSize, dynamicSize};
}

Value TilerBase::getFuncValue(OpBuilder &opBuilder) {
  auto matchTarget = getTransformSeqHandle();
  return opBuilder.create<transform::MatchOp>(
      matchTarget.getLoc(), matchTarget,
      ArrayRef<StringRef>({func::FuncOp::getOperationName()}));
}

ValueHandle *TilerBase::getFuncHandle(OpBuilder &opBuilder) {
  return record<RegularValueHandle>(getFuncValue(opBuilder), opBuilder);
}

ValueHandles
TilerBase::getKernelOutputs(OpBuilder &opBuilder,
                            const GetKernelIOOptions &options) {
  if (options.isInverted) {
    assert(options.findReshapePosition.empty() &&
           "isInverted cannot be used with findReshapePosition");
  }
  auto funcValue = getFuncValue(opBuilder);
  ValueHandles handles;
  for (size_t operandIdx : getKernelInfo()->outputOrdering) {
    // TODO: The result is matched one-by-one because split handle op cannot
    // split transform any value typed inputs.
    auto resultHandle = opBuilder.create<transform::GetFuncResultOp>(
        funcValue.getLoc(),
        /*outputs=*/opBuilder.getType<transform::AnyValueType>(),
        /*target=*/funcValue,
        /*raw_position_list=*/
        opBuilder.getDenseI64ArrayAttr({static_cast<int64_t>(operandIdx)}),
        /*is_inverted=*/options.isInverted,
        /*is_all=*/false,
        /*find_reshape_producer=*/
        options.findReshapePosition.contains(operandIdx));
    handles.push_back(
        record<RegularValueHandle>(resultHandle.getResult(), opBuilder));
  }
  return handles;
}

ValueHandles TilerBase::getKernelInputs(OpBuilder &opBuilder,
                                        const GetKernelIOOptions &options) {
  if (options.isInverted) {
    assert(options.findReshapePosition.empty() &&
           "isInverted cannot be used with findReshapePosition");
  }
  auto funcValue = getFuncValue(opBuilder);
  ValueHandles handles;
  // TODO: The result is matched one-by-one because merge handle op cannot
  // merge transform any value typed inputs.
  for (auto operandIdx : options.positionList) {
    auto funcArgHandle = opBuilder.create<transform::GetFuncArgumentOp>(
        funcValue.getLoc(),
        /*outputs=*/opBuilder.getType<transform::AnyValueType>(),
        /*target=*/funcValue,
        /*raw_position_list=*/
        opBuilder.getDenseI64ArrayAttr({operandIdx}),
        /*is_inverted=*/options.isInverted,
        /*is_all=*/false,
        /*find_reshape_consumer*/
        options.findReshapePosition.contains(operandIdx));
    handles.push_back(
        record<RegularValueHandle>(funcArgHandle.getResult(), opBuilder));
  }
  return handles;
}

ValueHandles TilerBase::getOpOperand(const ValueHandles &opHandles, 
    int64_t operandPos, OpBuilder &opBuilder) {
  ValueHandles operandHandles;
  for(auto opHandle : opHandles) {
    auto targetOp = getValue(opHandle, opBuilder);
    auto operand = opBuilder.create<transform::GetOperandOp>(
        targetOp.getLoc(), 
        opBuilder.getType<transform::AnyValueType>(),
        targetOp, operandPos);
    operandHandles.push_back(
        record<RegularValueHandle>(operand.getResult(), opBuilder));
  }
  return operandHandles;
}

ValueHandles TilerBase::getOpResult(const ValueHandles &opHandles, 
    int64_t resultPos, OpBuilder &opBuilder) {
  ValueHandles resultHandles;
  for(auto opHandle : opHandles) {
    auto targetOp = getValue(opHandle, opBuilder);
    auto result = opBuilder.create<transform::GetResultOp>(
        targetOp.getLoc(), 
        opBuilder.getType<transform::AnyValueType>(),
        targetOp, resultPos);
    resultHandles.push_back(
        record<RegularValueHandle>(result.getResult(), opBuilder));
  }
  return resultHandles;
}

TilerBase::CacheIOResult
TilerBase::cacheRead(const ValueHandles &opHandles, 
      int64_t oprIdx, 
      StringRef cacheReadTagName, //ArrayRef<Attribute> copyDstAttrs,
      OpBuilder &opBuilder) {
  ValueHandles inputHandles = getOpOperand(
      opHandles, oprIdx, opBuilder);
  for (auto [idx, inputHandle] : llvm::enumerate(inputHandles)) {
    Value inputs = getValue(inputHandle, opBuilder);
    auto cachedOp =
        opBuilder.create<transform::CacheReadOp>(
            inputs.getLoc(),
            /*cached=*/opBuilder.getType<transform::AnyOpType>(),
            /*targets=*/inputs).getCached();
    annotateByAttr(cachedOp, cacheReadTagName, opBuilder);
    // auto consumeHandles = 
    //     opBuilder.create<transform::GetConsumersOfResult>(
    //         cachedOp.getLoc(), 
    //         opBuilder.getType<transform::AnyOpType>(),
    //         cachedOp, /*result_number*/0);
    // opHandles[idx]->setHandle(consumeHandles);
    // auto newOpHandle = 
    //     opBuilder.create<transform::SplitHandleOp>(
    //         consumeHandles.getLoc(), consumeHandles, 1
    //     ).getResults()[0];
    // opHandles[idx]->setHandle(newOpHandle);
  }
  auto matchTarget = getTransformSeqHandle();
  auto cachedOps = matchByIdentifier(matchTarget, cacheReadTagName,
                                    IdentifierType::kAttribute, opBuilder);
  /*cachedOps=*/
  CacheIOResult cacheIOResult = 
  {record<NamedValueHandle>(cachedOps, opBuilder,
                            NamedValueHandleArgs{cacheReadTagName,
                                                IdentifierType::kAttribute,
                                                /*needsAnnotate=*/false,
                                                /*needsReverse=*/false})};
  return cacheIOResult;
}

TilerBase::CacheIOResult
TilerBase::cacheWrite(const ValueHandles &opHandles,
      int64_t resIdx,
      StringRef cacheWriteTagName,
      OpBuilder &opBuilder) {
  ValueHandles outputHandles = getOpResult(
      opHandles, resIdx, opBuilder);
  for (auto [idx, outputHandle] : llvm::enumerate(outputHandles)) {
    Value outputs = getValue(outputHandle, opBuilder);
    auto cachedOp =
        opBuilder
            .create<transform::CacheWriteOp>(
                outputs.getLoc(),
                /*cached=*/opBuilder.getType<transform::AnyOpType>(),
                /*targets=*/outputs,
                /*output_only=*/true,
                /*cache_write_to_output_init=*/true)
            .getCached();
    annotateByAttr(cachedOp, cacheWriteTagName, opBuilder);
  }
  auto matchTarget = getTransformSeqHandle();
  auto cachedOps = matchByIdentifier(matchTarget, cacheWriteTagName,
                                    IdentifierType::kAttribute, opBuilder);
  /*cachedOps=*/
  CacheIOResult cacheIOResult = 
  {record<NamedValueHandle>(cachedOps, opBuilder,
                            NamedValueHandleArgs{cacheWriteTagName,
                                                IdentifierType::kAttribute,
                                                /*needsAnnotate=*/false,
                                                /*needsReverse=*/true})};
  return cacheIOResult;
}

std::pair<TilerBase::CacheIOResult, TilerBase::CacheIOResult>
TilerBase::cacheReadAndWrite(const ValueHandles &opHandles, int64_t resIdx,
    StringRef cacheReadTagName, StringRef cacheWriteTagName, OpBuilder &opBuilder) {
  ValueHandles outputHandles = getOpResult(
      opHandles, resIdx, opBuilder);
  for (auto [idx, outputHandle] : llvm::enumerate(outputHandles)) {
    Value outputs = getValue(outputHandle, opBuilder);
    auto readAndWriteOp = opBuilder.create<transform::CacheReadAndWriteOp>(
        outputs.getLoc(),
        opBuilder.getType<transform::AnyOpType>(),
        opBuilder.getType<transform::AnyOpType>(),
        outputs);
    auto copyReadOp = readAndWriteOp.getCopyReadOp();
    annotateByAttr(copyReadOp, cacheReadTagName, opBuilder);
    auto copyWriteOp = readAndWriteOp.getCopyWriteOp();
    annotateByAttr(copyWriteOp, cacheWriteTagName, opBuilder);
  }
  auto matchTarget = getTransformSeqHandle();
  auto cpyReadOps = matchByIdentifier(matchTarget, cacheReadTagName,
                                      IdentifierType::kAttribute, opBuilder);
  CacheIOResult cacheReadResult = 
  {record<NamedValueHandle>(cpyReadOps, opBuilder,
                            NamedValueHandleArgs{cacheReadTagName,
                                                IdentifierType::kAttribute,
                                                /*needsAnnotate=*/false,
                                                /*needsReverse=*/false})};
  auto cpyWriteOps = matchByIdentifier(matchTarget, cacheWriteTagName,
                                      IdentifierType::kAttribute, opBuilder);
  CacheIOResult cacheWriteResult = 
  {record<NamedValueHandle>(cpyWriteOps, opBuilder,
                            NamedValueHandleArgs{cacheWriteTagName,
                                                IdentifierType::kAttribute,
                                                /*needsAnnotate=*/false,
                                                /*needsReverse=*/false})};
  return {cacheReadResult, cacheWriteResult};
}

TilerBase::CacheIOResult TilerBase::cacheRead(OpBuilder &opBuilder) {
  auto kernelInputsHandles = getKernelInputs(
      opBuilder, GetKernelIOOptions{
                     /*positionList=*/
                     llvm::to_vector(getKernelInfo()->cacheReadFuncArgIndices),
                     /*isInverted=*/false,
                     /*findReshapePosition=*/
                     getKernelInfo()->funcArgWithReshapeIndices});

  for (auto [idx, kernelInputsHandle] : llvm::enumerate(kernelInputsHandles)) {
    Value inputs = getValue(kernelInputsHandle, opBuilder);
    auto cachedOp =
        opBuilder
            .create<transform::CacheReadOp>(
                inputs.getLoc(),
                /*cached=*/opBuilder.getType<transform::AnyOpType>(),
                /*targets=*/inputs)
            .getCached();
    annotateByAttr(cachedOp, kCacheReadTagName, opBuilder);
    annotateByAttr(cachedOp, getCacheReadTag(idx), opBuilder);
    kernelInputsHandle->invalidate();
  }
  auto matchTarget = getTransformSeqHandle();
  auto cachedOps = matchByIdentifier(matchTarget, kCacheReadTagName,
                                     IdentifierType::kAttribute, opBuilder);
  return CacheIOResult{
      /*cachedOps=*/
      record<NamedValueHandle>(cachedOps, opBuilder,
                               NamedValueHandleArgs{kCacheReadTagName,
                                                    IdentifierType::kAttribute,
                                                    /*needsAnnotate=*/false,
                                                    /*needsReverse=*/true})};
}

TilerBase::CacheIOResult TilerBase::cacheWrite(OpBuilder &opBuilder) {
  auto kernelOutputHandles = getKernelOutputs(
      opBuilder,
      GetKernelIOOptions{/*positionList=*/
                         getKernelInfo()->outputOrdering,
                         /*isInverted=*/false,
                         /*findReshapePosition=*/
                         getKernelInfo()->returnValueWithReshapeIndices});

  SmallVector<Value> cacheWriteOriginalOps;
  for (auto [originalResultIdx, outputHandle] :
       llvm::zip_equal(getKernelInfo()->outputOrdering, kernelOutputHandles)) {
    auto output = getValue(outputHandle, opBuilder);
    auto cachedWriteOp =
        opBuilder
            .create<transform::CacheWriteOp>(
                output.getLoc(),
                /*cached=*/opBuilder.getType<transform::AnyOpType>(),
                /*targets=*/output,
                /*output_only=*/true,
                /*cache_write_to_output_init=*/
                getKernelInfo()->returnValueIdx2TiedFuncArg.contains(
                    originalResultIdx))
            .getCached();
    annotateByAttr(cachedWriteOp, kCacheWriteTagName, opBuilder);
    outputHandle->invalidate();
  }
  auto matchTarget = getTransformSeqHandle();
  auto cachedOps = matchByIdentifier(matchTarget, kCacheWriteTagName,
                                     IdentifierType::kAttribute, opBuilder);
  return CacheIOResult{/*cachedOps=*/record<NamedValueHandle>(
      cachedOps, opBuilder,
      NamedValueHandleArgs{kCacheWriteTagName, IdentifierType::kAttribute,
                           /*needsAnnotate=*/false})};
}

ValueHandles TilerBase::getTilingDataHandles(
    SmallVector<size_t>& tilingArgsPos, OpBuilder &opBuilder) {
  ValueHandles tilingArgHandles;
  auto funcValue = getFuncValue(opBuilder);
  for(auto argIdx : tilingArgsPos) {
    auto funcArgHandles = opBuilder.create<transform::GetFuncArgumentOp>(
        funcValue.getLoc(),
        /*outputs=*/opBuilder.getType<transform::AnyValueType>(),
        /*target=*/funcValue,
        /*raw_position_list=*/
        SmallVector<int64_t>{static_cast<int64_t>(argIdx)},
        /*is_inverted=*/false);
    auto *handle = record<FuncArgHandle>(
        funcArgHandles.getOutputs(), opBuilder, argIdx);
    tilingArgHandles.push_back(handle);
  }
  return tilingArgHandles;
}

TilerBase::ForTilingResult
TilerBase::tileUsingFor(ValueHandles &targets,
                        ValueHandleFoldResults &tileSizes,
                        OpBuilder &opBuilder) {
  auto mapFn = [this, &opBuilder](Value tiledLoop) -> ValueHandle * {
    return record<NamedValueHandle>(
        tiledLoop, opBuilder,
        NamedValueHandleArgs{kTiledForTagName, IdentifierType::kAttribute});
  };
  auto [staticTileSizes, dynamicTileSizes] =
      unpackFoldResults(tileSizes, opBuilder);
  SmallVector<bool> scalableSizes(tileSizes.size(), false);
  SmallVector<Type> outputTypes(
      llvm::count_if(staticTileSizes,
                     [](int64_t tileSize) { return tileSize != 0; }),
      opBuilder.getType<transform::AnyOpType>());

  SmallVector<ValueHandles> loopHandles;
  for (auto *targetHandle : targets) {
    auto targetValue = getValue(targetHandle, opBuilder);
    auto forOp = opBuilder.create<transform::TileUsingForOp>(
        targetValue.getLoc(),
        /*tiled_linalg_op=*/opBuilder.getType<transform::AnyOpType>(),
        /*loops=*/outputTypes,
        /*target=*/targetValue,
        /*dynamic_sizes=*/dynamicTileSizes,
        /*static_sizes=*/staticTileSizes,
        /*interchange=*/ArrayRef<int64_t>{},
        /*scalable_sizes=*/scalableSizes);
    loopHandles.emplace_back(llvm::map_to_vector(forOp.getLoops(), mapFn));
    // Update original handle to hold the tiled op.
    targetHandle->setHandle(forOp.getTiledLinalgOp());
  }
  return ForTilingResult{/*loops=*/loopHandles};
}

TilerBase::ForTilingResult
TilerBase::tileSingleLayer(ValueHandles &targets,
                          ValueHandleFoldResult tileSize,
                          OpBuilder &opBuilder) {
  auto mapFn = [this, &opBuilder](Value tiledLoop) -> ValueHandle * {
    return record<NamedValueHandle>(
        tiledLoop, opBuilder,
        NamedValueHandleArgs{kTiledForTagName, IdentifierType::kAttribute});
  };
  auto [staticTileSize, dynamicTileSize] =
      unpackFoldResult(tileSize, opBuilder);
  bool scalableSize = (staticTileSize == ShapedType::kDynamic) ? false : true ;
  Type outputType = opBuilder.getType<transform::AnyOpType>();

  SmallVector<ValueHandles> loopHandles;
  for (auto *targetHandle : targets) {
    auto targetValue = getValue(targetHandle, opBuilder);
    auto forOp = opBuilder.create<transform::TileUsingForOp>(
        targetValue.getLoc(),
        /*tiled_linalg_op=*/opBuilder.getType<transform::AnyOpType>(),
        /*loops=*/outputType,
        /*target=*/targetValue,
        /*dynamic_sizes=*/dynamicTileSize,
        /*static_sizes=*/staticTileSize,
        /*interchange=*/ArrayRef<int64_t>{},
        /*scalable_sizes=*/scalableSize);
    loopHandles.emplace_back(llvm::map_to_vector(forOp.getLoops(), mapFn));
    // Update original handle to hold the tiled op.
    targetHandle->setHandle(forOp.getTiledLinalgOp());
  }
  return ForTilingResult{/*loops=*/loopHandles};
}

ValueHandle *TilerBase::fuseLoops(ValueHandles &loops,
                                  OpBuilder &opBuilder) {
  assert(!std::empty(loops) && "Should fuse more than one loops");
  auto *fusedLoopHandle = loops.front();
  auto fusedLoopValue = getValue(fusedLoopHandle, opBuilder);
  fusedLoopHandle->invalidate();

  for (auto *nextLoopHandle : llvm::drop_begin(loops)) {
    auto nextLoopValue = getValue(nextLoopHandle, opBuilder);
    fusedLoopValue =
        opBuilder
            .create<transform::LoopFuseSibling>(
                nextLoopValue.getLoc(),
                /*fused_loop=*/opBuilder.getType<transform::AnyOpType>(),
                /*target=*/fusedLoopValue,
                /*source=*/nextLoopValue)
            .getFusedLoop();
    nextLoopHandle->invalidate();
  }
  return record<NamedValueHandle>(
      fusedLoopValue, opBuilder,
      NamedValueHandleArgs{kFusedLoopTagName, IdentifierType::kAttribute});
}

void TilerBase::fuseIntoContaining(ValueHandles &targetOps,
                                  ValueHandles &containingLoops,
                                  OpBuilder &opBuilder,
                                  bool duplicateProducers,
                                  bool applyCanonicalizeAfterEachFusion) {
  SmallVector<Value> containingLoopValues =
      getValues(containingLoops, opBuilder);
  size_t numContainingLoop = containingLoopValues.size();

  SmallVector<Value> fusedLoops;
  for (auto *targetHandle : targetOps) {
    auto targetValue = getValue(targetHandle, opBuilder);
    if (applyCanonicalizeAfterEachFusion) {
      // Construct `transform::ForeachOp` to perform canonicalization before
      // fusing each target op into the containing op.
      // This is necessarily for complicated cases where the target ops
      // are used multiple times in the containing op.
      auto forEachRegionBuilderFn = [&](ImplicitLocOpBuilder &opBuilder,
                                        Block &block) -> void {
        auto blockArg = block.getArgument(0);

        // disabled patterns:
        //   a) kSimplifyTrivialLoops: in case trivial loops is simplified and
        //      lead to invalid loop handles
        applyPatterns(
            getFuncHandle(opBuilder),
            /*patterns=*/
            SmallVector<TransformPatternKind>{
                TransformPatternKind::CSE,
                TransformPatternKind::CANONICALIZATION,
                TransformPatternKind::MERGE_CONSECUTIVE_INSERT_EXTRACT_SLICE},
            opBuilder,
            /*disablePatterns=*/
            SmallVector<CanonicalizationPatternKind>{
                CanonicalizationPatternKind::kSimplifyTrivialLoops});

        auto fuseIntoContainingOp =
            opBuilder.create<transform::ExtendedFuseIntoContainingOp>(
                targetValue.getLoc(),
                /*fused_op=*/
                std::vector<Type>(numContainingLoop,
                                  opBuilder.getType<transform::AnyOpType>()),
                /*new_containing_op=*/
                std::vector<Type>(numContainingLoop,
                                  opBuilder.getType<transform::AnyOpType>()),
                /*producer_op=*/blockArg,
                /*containing_op=*/containingLoopValues,
                /*duplicate_producer=*/
                BoolAttr::get(opBuilder.getContext(), duplicateProducers));

        opBuilder.create<transform::YieldOp>(
            fuseIntoContainingOp.getLoc(), fuseIntoContainingOp->getResults());
      };

      std::vector<Type> forEachResultTypes(
          numContainingLoop * 2, opBuilder.getType<transform::AnyOpType>());
      auto forEachResults = createForEachOp(targetValue, forEachResultTypes,
                                            forEachRegionBuilderFn, opBuilder);

      fusedLoops = {forEachResults.begin(),
                    forEachResults.begin() + numContainingLoop};
      // TODO: Update containingLoopValue to ForeachOp's result
    } else {
      auto fuseIntoContainingOp =
          opBuilder.create<transform::ExtendedFuseIntoContainingOp>(
              targetValue.getLoc(),
              /*fused_op=*/
              std::vector<Type>(numContainingLoop,
                                opBuilder.getType<transform::AnyOpType>()),
              /*new_containing_op=*/
              std::vector<Type>(numContainingLoop,
                                opBuilder.getType<transform::AnyOpType>()),
              /*producer_op=*/targetValue,
              /*containing_op=*/containingLoopValues,
              /*duplicate_producer=*/
              BoolAttr::get(opBuilder.getContext(), duplicateProducers));
      fusedLoops = fuseIntoContainingOp.getFusedOp();
      containingLoopValues = fuseIntoContainingOp.getNewContainingOp();
    }
    if (numContainingLoop == 1) {
      targetHandle->setHandle(fusedLoops.front());
    } else {
      targetHandle->invalidate();
    }
  }
  // update handle to containing loop
  for (auto it : llvm::enumerate(containingLoops)) {
    if (applyCanonicalizeAfterEachFusion) {
      // Currently, for ForeachOp, the payload ops of the corresponding
      // YieldOp operand are merged and mapped to the same resulting handle.
      // Therefore, the result value of ForeachOp corresponding to the new
      // containing op will map to the same containing op many times. This is
      // a bit confusing for downstream users. So we invalidate the handles
      // for now.
      it.value()->invalidate();
    } else {
      it.value()->setHandle(containingLoopValues[it.index()]);
    }
  }
}

void TilerBase::applyCanonicalization(OpBuilder &opBuilder) {
  auto matchTarget = getTransformSeqHandle();
  matchTarget = opBuilder
                    .create<transform::ApplyRegisteredPassOp>(
                        matchTarget.getLoc(),
                        /*result=*/opBuilder.getType<transform::AnyOpType>(),
                        /*target=*/matchTarget,
                        /*pass_name=*/opBuilder.getStringAttr("canonicalize"))
                    .getResult();
  resetAllHandles();
  setTransformSeqHandle(matchTarget);
}

void TilerBase::applyCanonToTarget(OpBuilder &opBuilder, Value matchTarget) {
  matchTarget = opBuilder
                    .create<transform::ApplyRegisteredPassOp>(
                        matchTarget.getLoc(),
                        /*result=*/opBuilder.getType<transform::AnyOpType>(),
                        /*target=*/matchTarget,
                        /*pass_name=*/opBuilder.getStringAttr("canonicalize"))
                    .getResult();
  resetAllHandles();
  setTransformSeqHandle(matchTarget);
}

void TilerBase::applyCSE(OpBuilder &opBuilder) {
  auto matchTarget = getTransformSeqHandle();
  matchTarget = opBuilder
                    .create<transform::ApplyRegisteredPassOp>(
                        matchTarget.getLoc(),
                        /*result=*/opBuilder.getType<transform::AnyOpType>(),
                        /*target=*/matchTarget,
                        /*pass_name=*/opBuilder.getStringAttr("cse"))
                    .getResult();
  resetAllHandles();
  setTransformSeqHandle(matchTarget);
}

void TilerBase::applyPatterns(
    ValueHandle *target, const SmallVector<TransformPatternKind> &patterns,
    OpBuilder &opBuilder,
    const SmallVector<CanonicalizationPatternKind> &disablePatterns) {
  bool applyCSE = false;
  auto bodyBuilderFn = [&patterns, &applyCSE](OpBuilder &p, Location loc) {
    llvm::for_each(patterns, [&p, &loc, &applyCSE](TransformPatternKind k) {
      switch (k) {
      case TransformPatternKind::CSE:
        applyCSE = true;
        break;
      case TransformPatternKind::CANONICALIZATION:
        p.create<transform::ApplyCanonicalizationPatternsOp>(loc);
        break;
      case TransformPatternKind::MERGE_CONSECUTIVE_INSERT_EXTRACT_SLICE:
        p.create<transform::ApplyMergeConsecutiveInsertExtractSlicePatternsOp>(
            loc);
        break;
      default:
        break;
      }
    });
  };
  Value targetValue = getValue(target, opBuilder);
  auto applyPatternsOp = opBuilder.create<transform::ApplyPatternsOp>(
      targetValue.getLoc(),
      /*target=*/targetValue,
      /*bodyBuilder=*/bodyBuilderFn);

  // Set apply CSE
  applyPatternsOp.setApplyCse(applyCSE);

  // Add disable patterns
  SmallVector<Attribute> stringfiedDisablePatterns;
  for (auto k : disablePatterns) {
    stringfiedDisablePatterns.push_back(
        opBuilder.getStringAttr(stringfyCanonicalizationPatternKind(k)));
  }
  target->invalidate();
}

void TilerBase::bufferizeEmptyTensor(OpBuilder &opBuilder) {
  auto matchTarget = getTransformSeqHandle();
  matchTarget = opBuilder.create<transform::MatchOp>(
      matchTarget.getLoc(), matchTarget,
      ArrayRef<StringRef>({func::FuncOp::getOperationName()}));
  // Step 1: Construct MatchOp to find all EmptyOp in target.
  auto emptyTensors = opBuilder.create<transform::MatchOp>(
      matchTarget.getLoc(), matchTarget,
      ArrayRef<StringRef>({tensor::EmptyOp::getOperationName()}));
  // Step 2: Prepare input of EmptyTensorToAllocTensorOp.
  auto castResult =
      opBuilder
          .create<transform::CastOp>(
              emptyTensors.getLoc(),
              /*resultTypes=*/
              TypeRange{transform::OperationType::get(
                  opBuilder.getContext(), tensor::EmptyOp::getOperationName())},
              /*input=*/emptyTensors)
          .getOutput();
  // Step 3: Construct EmptyTensorToAllocTensorOp.
  opBuilder.create<transform::EmptyTensorToAllocTensorOp>(
      castResult.getLoc(), /*resultTypes=*/
      TypeRange{transform::OperationType::get(
          opBuilder.getContext(),
          bufferization::AllocTensorOp::getOperationName())},
      /*input=*/castResult);
}

void TilerBase::applyOneShotBufferization(OpBuilder &opBuilder) {
  // Convert `tensor.empty` to `bufferization.alloc_tensor` op first.
  bufferizeEmptyTensor(opBuilder);
  auto matchTarget = getTransformSeqHandle();
  matchTarget = opBuilder.create<transform::MatchOp>(
      matchTarget.getLoc(), matchTarget,
      ArrayRef<StringRef>({func::FuncOp::getOperationName()}));
  matchTarget = opBuilder
                    .create<transform::OneShotBufferizeOp>(
                        matchTarget.getLoc(),
                        /*transformed=*/
                        opBuilder.getType<transform::AnyOpType>(),
                        /*target=*/matchTarget)
                    .getTransformed();
  resetAllHandles();
  setTransformSeqHandle(matchTarget);
}

Value TilerBase::matchOperations(Value target,
                                ArrayRef<StringRef> identifiers,
                                OpBuilder &opBuilder) {
  SmallVector<Type> resultTypes = {opBuilder.getType<transform::AnyOpType>()};
  SmallVector<Attribute> stringAttrs;
  for(auto str : identifiers) {
    stringAttrs.push_back(opBuilder.getStringAttr(str));
  }
  auto ops = opBuilder.getArrayAttr(stringAttrs);
  auto opAttrs = DictionaryAttr{};
  auto interface = transform::MatchInterfaceEnumAttr{};
  auto filterResultType = TypeAttr{};
  auto filterOperandTypes = ArrayAttr{};
  Value matchResult = opBuilder.create<transform::MatchOp>(
      target.getLoc(), /*resultTypes=*/resultTypes,
      /*target=*/target, /*ops=*/ops,
      /*interface=*/interface, /*op_attrs=*/opAttrs,
      /*filter_result_type=*/filterResultType,
      /*filter_operand_types=*/filterOperandTypes).getResults();
  return matchResult;            
}

Value TilerBase::matchByIdentifier(Value target, ArrayRef<StringRef> identifiers,
                                  IdentifierType type,
                                  OpBuilder &opBuilder,
                                  const MatchOptions &options) {
  SmallVector<Type> resultTypes = {opBuilder.getType<transform::AnyOpType>()};

  auto ops = ArrayAttr();
  if (type == IdentifierType::kOperation) {
    SmallVector<Attribute> stringAttrs;
    for(auto id : identifiers)
      stringAttrs.push_back(opBuilder.getStringAttr(id));
    ops = opBuilder.getArrayAttr(stringAttrs);
  }

  auto opAttrs = DictionaryAttr{};
  if (type == IdentifierType::kAttribute) {
    SmallVector<NamedAttribute> namedAttrs;
    for(auto id : identifiers)
      namedAttrs.push_back(opBuilder.getNamedAttr(id, opBuilder.getUnitAttr()));
    opAttrs = opBuilder.getDictionaryAttr(namedAttrs);
  }

  auto interface = transform::MatchInterfaceEnumAttr{};
  auto filterResultType = TypeAttr{};
  auto filterOperandTypes = ArrayAttr{};

  Value matchResult;
  if (!options.childHandleOrValue.has_value()) {
    matchResult = opBuilder
                      .create<transform::MatchOp>(
                          target.getLoc(),
                          /*resultTypes=*/resultTypes,
                          /*target=*/target,
                          /*ops=*/ops,
                          /*interface=*/interface,
                          /*op_attrs=*/
                          opAttrs,
                          /*filter_result_type=*/filterResultType,
                          /*filter_operand_types=*/filterOperandTypes)
                      .getResults();
  } else {
    std::variant<ValueHandle *, Value> val = options.childHandleOrValue.value();
    Value childValue;
    if (std::holds_alternative<Value>(val))
      childValue = std::get<Value>(val);
    else if (std::holds_alternative<ValueHandle *>(val)) {
      auto *valHandle = std::get<ValueHandle *>(val);
      childValue = getValue(valHandle, opBuilder);
    } else {
      llvm_unreachable("Not implemented!");
    }
    matchResult = opBuilder
                      .create<transform::MatchAncestorOfOp>(
                          target.getLoc(),
                          /*resultTypes=*/resultTypes,
                          /*target=*/target,
                          /*child=*/childValue,
                          /*ops=*/ops,
                          /*interface=*/interface,
                          /*op_attrs=*/
                          opAttrs,
                          /*filter_result_type=*/filterResultType,
                          /*filter_operand_types=*/filterOperandTypes)
                      .getResults();
  }

  if (options.needsReverse)
    matchResult = opBuilder.create<transform::ReverseOp>(
        matchResult.getLoc(),
        /*result=*/TypeRange{opBuilder.getType<transform::AnyOpType>()},
        /*target=*/matchResult);

  return matchResult;
}

void TilerBase::annotateByAttr(Value target, StringRef attrName,
                              OpBuilder &opBuilder) {
  opBuilder.create<transform::AnnotateOp>(
      target.getLoc(),
      /*target=*/target,
      /*name=*/opBuilder.getStringAttr(attrName),
      /*param=*/Value{});
}

void TilerBase::annotateByAttr(Value target, StringRef attrName,
                              Attribute attr, OpBuilder &opBuilder) {
  auto annotateOp = opBuilder.create<transform::AnnotateOp>(
      target.getLoc(),
      TypeRange{},//opBuilder.getType<transform::AnyOpType>(),
      /*target=*/ValueRange{target},
      NamedAttribute{opBuilder.getStringAttr(attrName), attr}
  );
  annotateOp.dump();
}

ValueHandles TilerBase::splitHandle(ValueHandle *handle, size_t splitSize,
                                    OpBuilder &opBuilder) {
  Value handleValue = getValue(handle, opBuilder);
  auto results = opBuilder
                     .create<transform::SplitHandleOp>(
                         handleValue.getLoc(),
                         /*handle=*/handleValue,
                         /*numResultHandles=*/static_cast<int64_t>(splitSize))
                     .getResults();
  return llvm::map_to_vector(results, [this, &opBuilder](Value result) {
    return (ValueHandle *)record<RegularValueHandle>(result, opBuilder);
  });
}

ResultRange TilerBase::createForEachOp(Value target, TypeRange resultTypes,
                                      RegionBuilderFn regionBuilder,
                                      OpBuilder &opBuilder) {
  OpBuilder::InsertionGuard guard(opBuilder);
  auto foreach = opBuilder.create<transform::ForeachOp>(target.getLoc(),
                                                        /*results=*/resultTypes,
                                                        /*target=*/target);
  Region &body = foreach.getBody();
  Block *block = opBuilder.createBlock(
      &body, /*insertPt=*/{}, {opBuilder.getType<transform::AnyOpType>()},
      {foreach.getLoc()});
  ImplicitLocOpBuilder b(opBuilder.getUnknownLoc(), opBuilder);
  regionBuilder(b, *block);
  transform::ForeachOp::ensureTerminator(body, opBuilder, foreach.getLoc());
  return foreach->getResults();
}

Value TilerBase::mergeHandles(
    const SmallVectorImpl<Value> &handles,
    transform::TransformHandleTypeInterface handleType, OpBuilder &opBuilder) {
  assert(!handles.empty());
  return opBuilder.create<transform::MergeHandlesOp>(handles.front().getLoc(),
                                                     /*result=*/handleType,
                                                     /*target=*/handles);
}

ValueHandle *TilerBase::getOpsWithIdentifier(StringRef identifier,
                                            IdentifierType type,
                                            OpBuilder &opBuilder,
                                            const MatchOptions &options) {
  auto matchTarget = getTransformSeqHandle();
  auto targetOps =
      matchByIdentifier(matchTarget, identifier, type, opBuilder, options);
  // Don't need to annotate because the ops are matched by op name.
  return record<NamedValueHandle>(
      targetOps, opBuilder,
      NamedValueHandleArgs{identifier, type,
                           /*needsAnnotate=*/false});
}

std::string TilerBase::getCacheReadTag(size_t funcArgIdx) {
  return llvm::formatv(kFuncArgIdxFormat, funcArgIdx).str();
}
//===- MixCVSchedule.cpp -- Auto-schedule fused kernels ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements auto schedule policy for mix cv kernels.
//
//===----------------------------------------------------------------------===//

#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/MixCVSchedule.h"
#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/AutoScheduleBase.h"
#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/BlockPureElemwiseSchedule.h"
#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/AutoScheduleAttrDefs.h"
#include "mtir/Dialect/MtFusion/IR/MtFusion.h"
#include "mtir/Dialect/MtFusion/TransformOps/MtFusionTransformOps.h"
#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/ValueHandle.h"
#include "mtir/Dialect/MtFusion/Transforms/OpFusion/FusableBlockOutliner.h"
#include "mtir/Dialect/MtFusion/Transforms/OpFusion/FusableHelper.h"
#include "mtir/Dialect/MtFusion/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"

#include <utility>

#define DEBUG_TYPE "mtfusion-auto-schedule"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] [Mix CV] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::mtfusion;

namespace {

constexpr static size_t kMixCVTilingCnt = 10;

constexpr static llvm::StringLiteral kPostVectorFuncTagName =
    "post_vector_func";

constexpr static llvm::StringLiteral kGenPostVectorFuncName =
    "_mtir_gen_vector_epilogue_func";

constexpr static llvm::StringLiteral kErasedThisArg = "erase_this_arg";

constexpr static llvm::StringLiteral kTilingInfoNames[kMixCVTilingCnt]{
    "tiling_key",        "block_size_m",   "block_size_n",   "block_size_k",
    "process_size_m",    "process_size_n", "process_size_k", "swizzle_offset",
    "swizzle_direction", "p_tiles"};

/// Tiling Key
static constexpr int64_t kTilingCaseKeysAttched[1] = {
    /* kTilingCaseKeyBm128n256k256 */
    300,
};

struct BlockShapeTilingData {
  int64_t m;
  int64_t n;
  int64_t k;
};

struct ProcessShapeTilingData {
  int64_t m;
  int64_t n;
  int64_t k;
};

struct SwizzleDefaultTilingData {
  int64_t offset;
  int64_t direction;
};

struct EpilogueTilingData {
  int64_t pTile;
};

struct MixCVTilingConfig {
  BlockShapeTilingData block;
  ProcessShapeTilingData process;
  SwizzleDefaultTilingData swizzle;
  EpilogueTilingData epilogue;
};

/// Tiling Struct Default configs
static constexpr MixCVTilingConfig kMixCVDefaultTilingInfo[1] = {
    /* kTilingCaseKeyBm128n256k256 */
    {{128, 256, 256}, {128, 256, 64}, {1, 0}, {4}}};
} // namespace

//===----------------------------------------------------------------------===//
// MixCVScheduler
//===----------------------------------------------------------------------===//

TilingComputeFn MixCVScheduler::calculateTilingImpl() {
  return [](KernelInfo *kernelInfo,
            ExprBuilder *opBuilder) -> TilingFnResultTy {
    // auto maxBufferCnt = utils::countMaxBuffer(kernelInfo->originalKernel,
    //                                           /*printLiveRange=*/false);
    // assert(maxBufferCnt > 0 && "buffer count should be greater than zero!");

    OpBuilder::InsertionGuard g(*opBuilder);
    // Calculate tiling data.
    MLIRContext *ctx = opBuilder->getContext();
    TilingCases c;
    TilingStruct s;
    for (auto [tilingKey, tilingConfig] :
         llvm::zip(kTilingCaseKeysAttched, kMixCVDefaultTilingInfo)) {
      // Set tiling keys.
      c.insert(tilingKey);
      Expr tilingKeyExpr = opBuilder->createConstExpr(tilingKey);
      Expr blockTileM = opBuilder->createConstExpr(tilingConfig.block.m);
      Expr blockTileN = opBuilder->createConstExpr(tilingConfig.block.n);
      Expr blockTileK = opBuilder->createConstExpr(tilingConfig.block.k);
      Expr processTileM = opBuilder->createConstExpr(tilingConfig.process.m);
      Expr processTileN = opBuilder->createConstExpr(tilingConfig.process.n);
      Expr processTileK = opBuilder->createConstExpr(tilingConfig.process.k);
      Expr swizzleOffset =
          opBuilder->createConstExpr(tilingConfig.swizzle.offset);
      Expr swizzleDirection =
          opBuilder->createConstExpr(tilingConfig.swizzle.direction);
      Expr epiloguePTile =
          opBuilder->createConstExpr(tilingConfig.epilogue.pTile);

      auto tilingDataType = IntegerType::get(ctx, 64);
      TilingData tilingData0 =
          TilingData(std::move(tilingKeyExpr), tilingDataType);
      TilingData tilingData1 =
          TilingData(std::move(blockTileM), tilingDataType);
      TilingData tilingData2 =
          TilingData(std::move(blockTileN), tilingDataType);
      TilingData tilingData3 =
          TilingData(std::move(blockTileK), tilingDataType);
      TilingData tilingData4 =
          TilingData(std::move(processTileM), tilingDataType);
      TilingData tilingData5 =
          TilingData(std::move(processTileN), tilingDataType);
      TilingData tilingData6 =
          TilingData(std::move(processTileK), tilingDataType);
      TilingData tilingData7 =
          TilingData(std::move(swizzleOffset), tilingDataType);
      TilingData tilingData8 =
          TilingData(std::move(swizzleDirection), tilingDataType);
      TilingData tilingData9 =
          TilingData(std::move(epiloguePTile), tilingDataType);

      // Build tiling struct.
      s.push_back(std::make_unique<TilingData>(std::move(tilingData0)));
      s.push_back(std::make_unique<TilingData>(std::move(tilingData1)));
      s.push_back(std::make_unique<TilingData>(std::move(tilingData2)));
      s.push_back(std::make_unique<TilingData>(std::move(tilingData3)));
      s.push_back(std::make_unique<TilingData>(std::move(tilingData4)));
      s.push_back(std::make_unique<TilingData>(std::move(tilingData5)));
      s.push_back(std::make_unique<TilingData>(std::move(tilingData6)));
      s.push_back(std::make_unique<TilingData>(std::move(tilingData7)));
      s.push_back(std::make_unique<TilingData>(std::move(tilingData8)));
      s.push_back(std::make_unique<TilingData>(std::move(tilingData9)));
    }

    return TilingFnResultTy{std::move(c), std::move(s)};
  };
}

//===----------------------------------------------------------------------===//
// Implementation of MixCVScheduler schedule functions.
//===----------------------------------------------------------------------===//

void getElementWiseOps(func::FuncOp &funcOp,
                       SmallVector<Operation *> &elemwiseOps) {
  funcOp.walk([&](Operation *ops) {
    if (isa<linalg::ElemwiseBinaryOp>(ops) ||
        isa<linalg::ElemwiseUnaryOp>(ops) // ||
        // isa<mtfusion::ElemwiseBinaryOp>(ops) ||
        // isa<mtfusion::ElemwiseUnaryOp>(ops)
        ) {
      elemwiseOps.push_back(ops);
    }
  });
}

bool outlinePureElemWiseOps(func::FuncOp &funcOp, func::FuncOp &outlineFunc,
                            SmallVector<Operation *> &elemwiseOps) {
  OpBuilder opBuilder(funcOp->getContext());
  opBuilder.setInsertionPoint(funcOp);
  assert(elemwiseOps.size() > 0);

  // outline pure elementwise ops
  MtFusionOpFusionOptions options;
  options.outputMode = OutputMode::Multiple;
  options.fusionMode = FusionKind::PureElemwise;
  options.alwaysInline = true;
  bool isSingleOpt = elemwiseOps.size() == 1;
  opfusion::FusableHelper fusableHelper(options.fusionMode, true);
  opfusion::FusableBlocks fusableBlocks = {
      opfusion::FusableBlock(elemwiseOps, &fusableHelper, isSingleOpt)};
  opfusion::FusableBlockOutliner outliner(fusableBlocks, options.outputMode,
                                          options.alwaysInline);
  if (!outliner.outline("_vv_outlined_"))
    return false;
  assert(outliner.getOutlinedFuncs().size() == 1);
  outlineFunc = outliner.getOutlinedFuncs().front();
  return true;
}

void removeMixCVUnunsedOps(func::FuncOp &funcOp,
                           SmallVector<Operation *> &matmulOps) {
  OpBuilder opBuilder(funcOp.getContext());
  auto returnOp = cast<func::ReturnOp>(funcOp.getBody().front().back());

  SmallVector<Operation *> callOps;
  funcOp.walk([&](Operation *ops) {
    if (isa<func::CallOp>(ops)) {
      callOps.push_back(ops);
    }
    if (isa<linalg::MatmulOp>(ops) || isa<linalg::MatmulTransposeAOp>(ops) ||
        isa<linalg::MatmulTransposeBOp>(ops)) {
      matmulOps.push_back(ops);
    }
  });
  opBuilder.setInsertionPoint(returnOp);
  opBuilder.create<func::ReturnOp>(returnOp->getLoc(),
                                   ValueRange{matmulOps.front()->getResult(0)});
  returnOp->erase();
  for (Operation *ops : callOps) {
    ops->erase();
  }
}

void createOperationWithValueMap(func::FuncOp &funcOp,
                                 llvm::MapVector<Value, Value> &valueMap,
                                 SmallVector<Operation *> &elemWiseOps) {
  OpBuilder opBuilder(funcOp.getContext());
  // replace original elemwise op
  for (auto *elemWiseOp : elemWiseOps) {
    opBuilder.setInsertionPoint(elemWiseOp);
    Operation *replaceOp = opBuilder.clone(*elemWiseOp);
    SmallVector<Value> replaceOperands;
    for (auto operand : replaceOp->getOperands()) {
      replaceOperands.push_back(valueMap.lookup(operand));
    }
    replaceOp->setOperands(replaceOperands);

    replaceOp->getResult(0).setType(replaceOperands.back().getType());
    valueMap.insert_or_assign(elemWiseOp->getResult(0),
                              replaceOp->getResult(0));
  }
}

void insertDynamicTensorArgument(func::FuncOp &funcOp,
                                 llvm::MapVector<Value, Value> &valueMap) {
  OpBuilder opBuilder(funcOp->getContext());
  size_t numOfCurArguments = funcOp.getNumArguments();
  auto loc = funcOp.getArguments().back().getLoc();
  // insert dynamic argument
  for (size_t i = 0; i < numOfCurArguments; i++) {
    auto curArgment = funcOp.getArgument(i);
    if (isa<RankedTensorType>(curArgment.getType())) {
      auto replaceType = RankedTensorType::get(
          SmallVector<int64_t>{ShapedType::kDynamic, ShapedType::kDynamic},
          getElementTypeOrSelf(curArgment.getType()));
      funcOp.setArgAttr(i, kErasedThisArg, opBuilder.getUnitAttr());
      funcOp.insertArgument(numOfCurArguments + i, replaceType,
                            DictionaryAttr{}, loc);
      valueMap.insert_or_assign(curArgment,
                                funcOp.getArgument(numOfCurArguments + i));
    }
  }
  BlockArgument out = funcOp.getArguments().back();
  // convert inner ops(tensor.empty) to dynamic
  funcOp.walk([&](Operation *ops) {
    opBuilder.setInsertionPointAfter(ops);
    if (isa<tensor::EmptyOp>(ops)) {
      auto tensorType = ops->getResult(0).getType().cast<TensorType>();
      auto replaceType = RankedTensorType::get(
          SmallVector<int64_t>{ShapedType::kDynamic, ShapedType::kDynamic},
          tensorType.getElementType());
      auto dim0 = opBuilder.create<tensor::DimOp>(ops->getLoc(), out, 0);
      auto dim1 = opBuilder.create<tensor::DimOp>(ops->getLoc(), out, 1);
      auto replaceOps = opBuilder.create<tensor::EmptyOp>(
          ops->getLoc(), TypeRange{replaceType}, ValueRange{dim0, dim1});
      valueMap.insert_or_assign(ops->getResult(0), replaceOps->getResult(0));
    }
  });
}

void eraseTargetOps(SmallVector<Operation *> &ops) {
  for (auto iter = ops.rbegin(); iter != ops.rend(); iter++) {
    (*iter)->erase();
  }
}

void eraseFuncArgsWithAttr(func::FuncOp &funcOp, llvm::StringRef attr) {
  BitVector indicesToErase(funcOp.getNumArguments());
  for (auto argIndex : llvm::seq<int>(0, funcOp.getNumArguments()))
    if (funcOp.getArgAttr(argIndex, attr))
      indicesToErase.set(argIndex);
  funcOp.eraseArguments(indicesToErase);
}

void updatePostVecFuncInfo(func::FuncOp &funcOp, const std::string &funcName) {
  OpBuilder opBuilder(funcOp.getContext());
  // update func symbol name
  funcOp.setSymName(funcName);
  // insert dynamic argument
  llvm::MapVector<Value, Value> valueMap;
  insertDynamicTensorArgument(funcOp, valueMap);
  // create dynamic elemwise ops
  SmallVector<Operation *> oriElemWiseOps;
  getElementWiseOps(funcOp, oriElemWiseOps);
  createOperationWithValueMap(funcOp, valueMap, oriElemWiseOps);
  // update return ops && update func signature
  auto returnOp = cast<func::ReturnOp>(funcOp.getBody().front().back());
  auto replaceValue = valueMap.lookup(returnOp->getOperand(0));
  opBuilder.setInsertionPoint(returnOp);
  opBuilder.create<func::ReturnOp>(returnOp->getLoc(),
                                   ValueRange{replaceValue});
  returnOp->erase();
  eraseTargetOps(oriElemWiseOps);
  eraseFuncArgsWithAttr(funcOp, kErasedThisArg);
  FunctionType newFuncTy = funcOp.getFunctionType().clone(
      funcOp.getArgumentTypes(), replaceValue.getType());
  funcOp.setType(newFuncTy);
}

void applyCSEAndCanonicalize(func::FuncOp funcOp) {
  PassManager pm(funcOp->getContext());
  CanonicalizerOptions options;
  // options.enableExtendedPattern = true;
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  if (failed(pm.run(funcOp))) {
    funcOp->emitError("Apply Canonicalizer && CSE error");
  }
}

void updateMixCVFunc(func::FuncOp &funcOp, TilingInfo *tilingInfo,
                     KernelInfo *kernelInfo, const std::string &postVecFuncName,
                     bool singleCube = false) {
  OpBuilder opBuilder(funcOp.getContext());
  SmallVector<Operation *> matmulOps;
  // annotate func args for v
  for (auto argIdx : kernelInfo->cacheReadFuncArgIndices) {
    auto argument = funcOp.getArgument(argIdx);
    for (auto *userOp : argument.getUsers()) {
      if (isa<func::CallOp>(userOp)) {
        funcOp.setArgAttr(argIdx, FuncArgForVOpAttr::name,
                          opBuilder.getUnitAttr());
      }
    }
  }
  // remove unused ops, update ret op
  removeMixCVUnunsedOps(funcOp, matmulOps);
  if (!singleCube) {
    // annotate matmul attr
    matmulOps.front()->setAttr(kPostVectorFuncTagName,
                               opBuilder.getStringAttr(postVecFuncName));
  }
  // annotate tiling attr
  for (size_t idx = 1; idx < tilingInfo->getTilingStruct().size(); idx++) {
    auto *tilingData = tilingInfo->getTilingData(idx);
    assert(tilingData->isConst() && "target tiling size should be const");
    matmulOps.front()->setAttr(
        kTilingInfoNames[idx],
        opBuilder.getI64IntegerAttr(tilingData->getConst()));
  }
}

LogicalResult MixCVScheduler::createScheduleImpl(TilingKey key,
                                                 OpBuilder &opBuilder) {
  TilingInfo *tilingInfo = getTilingInfo();
  assert(tilingInfo != nullptr);
  func::FuncOp funcOp = getToBeScheduledKernel();
  func::FuncOp postVecFunc;
  SmallVector<Operation *> elemwiseOps;
  getElementWiseOps(funcOp, elemwiseOps);
  auto postVecFuncName = getOriginalKernelName() + kGenPostVectorFuncName.str();
  if (elemwiseOps.size() == 0) {
    // single cube: only mark tiling params
    updateMixCVFunc(funcOp, tilingInfo, getKernelInfo(), postVecFuncName, true);
  } else {
    // Step 1: outline pure element wise ops
    if (!outlinePureElemWiseOps(funcOp, postVecFunc, elemwiseOps)) {
      return funcOp->emitError("Failed to apply outline fusion.");
    }
    // Step 2: update mix cv func: remove unused ops && annotate matmul attr
    updateMixCVFunc(funcOp, tilingInfo, getKernelInfo(), postVecFuncName,
                    false);
    // Step 3: update post vector func info
    updatePostVecFuncInfo(postVecFunc, postVecFuncName);
    // Step 4: apply CSE && Canonicalize for post vector func
    applyCSEAndCanonicalize(postVecFunc);
    // Step 5: apply schedule for post vec func
    OpBuilder vOpBuilder(postVecFunc);
    std::unique_ptr<SchedulerBase> vScheduler;
    vScheduler = std::make_unique<BlockPureElemwiseScheduler>(postVecFunc);
    if (failed(vScheduler->runOnOperation(vOpBuilder))) {
      postVecFunc.emitError(
          "Failed to apply block pure elemwise auto schedule.");
    }
  }

  return success();
}
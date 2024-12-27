  //===--------- MultiBuffering.cpp - MultiBuffering Pass -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mtir/Dialect/MtFusion/IR/MtFusion.h"
#include "mtir/Dialect/MtFusion/Transforms/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include <iostream>
using namespace std;

namespace mlir {
#define GEN_PASS_DEF_MULTIBUFFERING
#include "mtir/Dialect/MtFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mtfusion;

namespace {

static MemRefType makeStridedLayoutDynamic(MemRefType type) {
  return MemRefType::Builder(type).setLayout(StridedLayoutAttr::get(
      type.getContext(), ShapedType::kDynamic,
      SmallVector<int64_t>(type.getRank(), ShapedType::kDynamic)));
}

static memref::SubViewOp createReducedSubviewFromAlloc(OpBuilder& builder, Location loc,
        memref::AllocOp allocOp, Value buffNum) {
    auto oldShape = SmallVector<int64_t>(
        allocOp.getType().getNumDynamicDims(), ShapedType::kDynamic);
    MemRefType oldType = MemRefType::get(oldShape, allocOp.getType().getElementType());
    auto dynSizes = allocOp.getDynamicSizes();

    SmallVector<OpFoldResult> oldOffsets(oldType.getRank(), builder.getIndexAttr(0));
    SmallVector<OpFoldResult> newOffsets = {getAsOpFoldResult(buffNum)};
    newOffsets.append(oldOffsets);
    SmallVector<OpFoldResult> newSizes = {builder.getIndexAttr(1)};
    newSizes.append(dynSizes.begin(), dynSizes.end());
    SmallVector<OpFoldResult> newStrides(oldType.getRank() + 1, builder.getIndexAttr(1));
    auto bufferSubview = builder.create<memref::SubViewOp>(
        loc, makeStridedLayoutDynamic(oldType), 
        allocOp, newOffsets, newSizes, newStrides);
    return bufferSubview;
}

static memref::AllocOp liftUpAllocOp(OpBuilder& builder, memref::AllocOp allocOp) {
  Operation* lastProducer = allocOp.getOperand(0).getDefiningOp();
  for(auto operand : allocOp.getOperands()) {
    auto producer = operand.getDefiningOp();
    if(lastProducer->isBeforeInBlock(producer)) {
      lastProducer = producer;
    }
  }
  OpBuilder::InsertionGuard insGuard(builder);
  builder.setInsertionPointAfter(lastProducer);
  auto newAllocOp = cast<memref::AllocOp>(builder.clone(*allocOp));
  allocOp.getResult().replaceAllUsesWith(newAllocOp.getResult());
  allocOp.erase();
  return newAllocOp;
}

static void fuseInCopyOpProducer(OpBuilder& builder, linalg::CopyOp copyOp) {
  OpBuilder::InsertionGuard insGuard(builder);
  builder.setInsertionPointToStart(&copyOp->getParentRegion()->front());
  auto producer = copyOp.getInputs()[0].getDefiningOp();
  auto newProducer = builder.clone(*producer);
  producer->getResult(0).replaceAllUsesWith(newProducer->getResult(0));
  producer->erase();
}

scf::ForOp multiBufferize(scf::ForOp forOp, int level) {
  OpBuilder builder(forOp.getContext());
  OpBuilder::InsertionGuard insGuard(builder);
  builder.setInsertionPoint(forOp);
  auto curLoc = forOp.getLoc();

  auto posStart = forOp.getLowerBound();
  auto posEnd = forOp.getUpperBound();
  auto tileSize = forOp.getStep();
  auto c0 = builder.create<arith::ConstantIndexOp>(curLoc, 0);

  IRMapping prelogueMapping;
  prelogueMapping.map(forOp.getInductionVar(), posStart);

  SetVector<linalg::CopyOp> readCpyOps, writeCpyOps;
  SetVector<memref::AllocOp> allocSet;
  DenseMap<memref::AllocOp, int64_t> bufferFactorMap;

  /// Find allocOps from CopyOps.
  forOp.walk([&](linalg::CopyOp copyOp) {
    if(copyOp->getParentOp() != forOp)
      return WalkResult::skip();
    for(auto dst : copyOp.getDpsInits()) {
      auto op = dst.getDefiningOp();
      if(auto allocOp = dyn_cast<memref::AllocOp>(op)) {
        if(!allocSet.contains(allocOp)) {
          allocSet.insert(allocOp);
          bufferFactorMap[allocOp] = 2;
        }
        readCpyOps.insert(copyOp);
      }
    }
    for(auto src : copyOp.getDpsInputs()) {
      auto op = src.getDefiningOp();
      if(auto allocOp = dyn_cast<memref::AllocOp>(op)) {
        if(!allocSet.contains(allocOp)) {
          allocSet.insert(allocOp);
          bufferFactorMap[allocOp] = 2;
        } else {
          bufferFactorMap[allocOp] = 3;
        }
        writeCpyOps.insert(copyOp);
      }
    }
    return WalkResult::advance();
  });
  // cout<<"readCpyOps:"<<endl; for(auto op : readCpyOps) op->dump();
  // cout<<"writeCpyOps:"<<endl; for(auto op : writeCpyOps) op->dump();
  // cout<<"allocSet:"<<endl; for(auto op : allocSet) op->dump();

  DenseMap<memref::AllocOp, memref::AllocOp> allocMap;

  /// Create multi-buffer allocOps and map to original allocOps.
  for(auto allocOp : allocSet) {
    auto oldType = allocOp.getType();
    auto oldShape = oldType.getShape();
    SmallVector<int64_t> newShape = {bufferFactorMap[allocOp]};
    newShape.append(oldShape.begin(), oldShape.end());
    
    auto newType = MemRefType::get(newShape, oldType.getElementType());
    auto dynSizes = allocOp.getDynamicSizes();
    auto alignAttr = allocOp.getAlignmentAttr();
    auto newAllocOp = liftUpAllocOp(builder, 
        builder.create<memref::AllocOp>(
            curLoc, newType, dynSizes, alignAttr
    ));
    allocMap.insert(std::pair(allocOp, newAllocOp));
    
    auto buffer_0 = createReducedSubviewFromAlloc(builder, curLoc, newAllocOp, c0);
    prelogueMapping.map(allocOp.getResult(), buffer_0.getResult());
  }

  SetVector<Value> readValueSet, writeValueSet;

  /// Find Values from SubviewOps and CopyOps to be renewed.
  forOp.walk([&](Operation *op) {
    if(op->getParentOp() != forOp)
      return WalkResult::skip();
    if(isa<memref::AllocOp>(op) || isa<scf::YieldOp>(op))
      return WalkResult::skip();
    if(isa<scf::ForOp>(op) ||
        (isa<linalg::LinalgOp>(op) && !isa<linalg::CopyOp>(op)))
      return WalkResult::skip();
    if(auto copyOp = dyn_cast<linalg::CopyOp>(op)) {
      if(writeCpyOps.contains(copyOp)) {
        for(auto operand : copyOp.getOperands()) {
          writeValueSet.insert(operand);
        }
        return WalkResult::skip();
      }
    }

    Operation *newOp = builder.clone(*op, prelogueMapping);
    prelogueMapping.map(op->getResults(), newOp->getResults());

    if(auto subviewOp = dyn_cast<memref::SubViewOp>(op)){
      auto res = subviewOp.getResult();
      readValueSet.insert(res);
    } else if(auto copyOp = dyn_cast<linalg::CopyOp>(op)) { // in readCpyOps
      auto dst = copyOp.getDpsInits()[0];
      readValueSet.insert(dst);
    }
    return WalkResult::advance();
  });

  /// Create iter_args from readValueSet and writeValueSet.
  SmallVector<Value> iterArgs;
  auto prelogueValueMap = prelogueMapping.getValueMap();
  iterArgs.append(llvm::map_to_vector(readValueSet, [&](Value oldVal){
    return prelogueValueMap[oldVal];
  }));
  iterArgs.append(llvm::map_to_vector(writeValueSet, [&](Value oldVal){
    return prelogueValueMap[oldVal];
  }));

  /// After create the prelogue, create the main loop.
  auto newForOp = builder.create<scf::ForOp>(curLoc, 
      posStart, posEnd, tileSize, iterArgs);

  builder.setInsertionPointToStart(newForOp.getBody());

  /// Compute the position, tile size, buffer id of the next iteration.
  auto pos_next = builder.create<arith::AddIOp>(
      curLoc, newForOp.getInductionVar(), tileSize);
  auto nextExist = builder.create<arith::CmpIOp>(
      curLoc, mlir::arith::CmpIPredicate::slt, pos_next, posEnd);

  IRMapping loopBodyMapping;
  loopBodyMapping.map(forOp.getInductionVar(), pos_next);

  auto absPos_next = builder.create<arith::SubIOp>(curLoc, pos_next, posStart);
  auto idx_next = builder.create<arith::DivSIOp>(curLoc, absPos_next, tileSize);
  for(auto allocOp : allocSet) {
    auto newAllocOp = allocMap[allocOp];
    auto bufferId_next = builder.create<arith::RemSIOp>(curLoc, idx_next,
        builder.create<arith::ConstantIndexOp>(curLoc, bufferFactorMap[allocOp]));
    auto buffer_next = createReducedSubviewFromAlloc(builder, curLoc, newAllocOp, bufferId_next);
    loopBodyMapping.map(allocOp.getResult(), buffer_next.getResult());
  }

  SetVector<Operation*> prefetchOps, postStoreOps;

  /// Prefetch data for the next iteration.
  forOp.walk([&](Operation *op) {
    if(op->getParentOp() != forOp || isa<scf::YieldOp>(op))
      return WalkResult::skip();
    if((isa<linalg::LinalgOp>(op) && !isa<linalg::CopyOp>(op))
        || isa<scf::ForOp>(op)) // skip computing ops
      return WalkResult::skip();
    if(isa<linalg::CopyOp>(op) && 
        writeCpyOps.count(cast<linalg::CopyOp>(op))) // skip write copy ops
      return WalkResult::skip();

    Operation *newOp = builder.clone(*op, loopBodyMapping);
    loopBodyMapping.map(op->getResults(), newOp->getResults());

    if(isa<memref::SubViewOp>(op) || isa<linalg::CopyOp>(op)) {
      auto keyVal = isa<memref::SubViewOp>(op) ?
          op->getResult(0) : cast<linalg::CopyOp>(op).getDpsInits()[0]; 
      if(readValueSet.count(keyVal)) {
        prefetchOps.insert(newOp);
      }
    }
    return WalkResult::advance();
  });

  SetVector<Value> loopBodyYieldVals;
  
  for(int i = 0; i < readValueSet.size(); i++) {
    /// Map prefetched data to scf.yield values. 
    loopBodyYieldVals.insert(
        loopBodyMapping.getValueMap().at(readValueSet[i]));
    /// Map data used by computation to scf.for region args.
    loopBodyMapping.map(readValueSet[i], newForOp.getRegionIterArgs()[i]);
  }

  /// Map old iterator to the position of current iteration. 
  loopBodyMapping.map(forOp.getInductionVar(), newForOp.getInductionVar());
  
  /// Generate computation and store ops with scf.for region args.
  bool isBeforeCompute = true, isBeforeStore = true;
  forOp.walk([&](Operation *op) {
    if(op->getParentOp() != forOp || isa<scf::YieldOp>(op))
      return WalkResult::skip();
    if(isa<memref::AllocOp>(op) || isa<memref::SubViewOp>(op) || 
        (isBeforeCompute && isa<linalg::CopyOp>(op)))
      return WalkResult::skip(); // skip alloc, subview, copy(before computation) ops.
    if((isa<linalg::LinalgOp>(op) && !isa<linalg::CopyOp>(op))
        || isa<scf::ForOp>(op)) { // meet computation ops
      isBeforeCompute = false;
    }

    if(isBeforeStore && isa<linalg::CopyOp>(op)) { // writing ops
      for(int i=0; i<writeValueSet.size(); i++){
        /// Map computing data to scf.yield values. 
        loopBodyYieldVals.insert(
            loopBodyMapping.getValueMap().at(writeValueSet[i]));
        /// Map data used by store ops to scf.for region args.
        loopBodyMapping.map(writeValueSet[i], 
            newForOp.getRegionIterArgs()[i + readValueSet.size()]);
      }
      isBeforeStore = false;
    }

    Operation *newOp = builder.clone(*op, loopBodyMapping);
    loopBodyMapping.map(op->getResults(), newOp->getResults());

    if(isa<linalg::CopyOp>(op)) {
      postStoreOps.insert(newOp);
    }
    return WalkResult::advance();
  });

  /// Add scf.yielf for the new created scf.for
  builder.create<scf::YieldOp>(curLoc, loopBodyYieldVals.getArrayRef());

  forOp.erase(); // remove old scf.for

  /// Create scf.if for prefetch ops.
  builder.setInsertionPointAfter(prefetchOps.back());
  SmallVector<Type> preIfRetTypes = 
      llvm::map_to_vector(prefetchOps, [&](Operation* op) -> Type {
    if(auto subviewOp = dyn_cast<memref::SubViewOp>(op)) {
      return subviewOp.getResult().getType();
    } else if(auto copyOp = dyn_cast<linalg::CopyOp>(op)) {
      return copyOp.getDpsInits()[0].getType();
    }
  });
  auto preIfOp = builder.create<scf::IfOp>(
      curLoc, preIfRetTypes, nextExist, true/*withElseRegion*/);

  /// Move prefetch ops into scf.if
  SmallVector<Value> preIfYieldVals;
  IRMapping preIfBlockMapping;
  builder.setInsertionPointToStart(&preIfOp.getBodyRegion().front());
  SmallPtrSet<Operation *, 4> usesNotReplaced{prefetchOps.begin(), prefetchOps.end()};
  for(int i = 0; i < prefetchOps.size(); i++) {
    auto oldOp = prefetchOps[i];
    auto newOp = builder.clone(*oldOp, preIfBlockMapping);
    Value newResult;
    if(auto subviewOp = dyn_cast<memref::SubViewOp>(oldOp)) {
      subviewOp.getResult().replaceAllUsesExcept(
          preIfOp.getResult(i), usesNotReplaced);
      newResult = cast<memref::SubViewOp>(newOp).getResult();
      preIfBlockMapping.map(subviewOp.getResult(), newResult);
    } else if(auto copyOp = dyn_cast<linalg::CopyOp>(oldOp)) {
      usesNotReplaced.insert(newOp);
      copyOp.getDpsInits()[0].replaceAllUsesExcept(
          preIfOp.getResult(i), usesNotReplaced);
      newResult = cast<linalg::CopyOp>(newOp).getDpsInits()[0];
    }
    preIfYieldVals.push_back(newResult);
  }
  /// Add yield op to scf.if
  builder.create<scf::YieldOp>(curLoc, preIfYieldVals);
  for(auto oldOp : reverse(prefetchOps)) {
    oldOp->erase();
  }
  /// Add yield op to scf.else
  builder.setInsertionPointToStart(&preIfOp.getElseRegion().front());
  builder.create<scf::YieldOp>(curLoc, 
      ArrayRef<Value>{newForOp.getRegionIterArgs().begin(),
      newForOp.getRegionIterArgs().begin() + readValueSet.size()});

  if(!writeCpyOps.empty()) {
    /// Create scf.if for postStore ops.
    builder.setInsertionPointAfter(postStoreOps.back());
    auto lastExist = builder.create<arith::CmpIOp>(curLoc, 
        mlir::arith::CmpIPredicate::ne, newForOp.getInductionVar(), posStart);
    auto postIfOp = builder.create<scf::IfOp>(curLoc, lastExist);

    /// Move postStore ops into scf.if
    builder.setInsertionPointToStart(&postIfOp.getBodyRegion().front());
    for(int i = 0; i < postStoreOps.size(); i++) {
      auto oldOp = cast<linalg::CopyOp>(postStoreOps[i]);
      builder.clone(*oldOp);
    }

    /// Add data postStore ops after current loop
    builder.setInsertionPointAfter(newForOp);
    for(int i = 0; i < postStoreOps.size(); i++) {
      auto oldOp = cast<linalg::CopyOp>(postStoreOps[i]);
      auto newOp = builder.clone(*oldOp);
      for(int id = 0; id < oldOp.getNumOperands(); id++) {
        auto retVal = newForOp.getResult(
            cast<BlockArgument>(oldOp.getOperand(id)).getArgNumber() - 1);
        newOp->setOperand(id, retVal);
      }
      oldOp.erase();
    }
  }

  return newForOp;
}

void recursiveTraverseForOps(scf::ForOp forOpRoot, int level){
  scf::ForOp newForOpRoot = multiBufferize(forOpRoot, level);
  
  newForOpRoot.walk([&](scf::ForOp forOp){
    if(forOp->getParentOp() != newForOpRoot)
      return WalkResult::skip();
    
    recursiveTraverseForOps(forOp, level + 1);
    
    return WalkResult::advance();
  });
}

} // namepsace

namespace mlir {
class MultiBufferingPass : public impl::MultiBufferingBase<MultiBufferingPass> {
public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    funcOp.walk([&](scf::ForOp forOp) {
      if(forOp->getParentOp() != funcOp)
        return WalkResult::skip();

      recursiveTraverseForOps(forOp, 0 /*level*/);

      return WalkResult::advance();
    });
  }
};
} // namespace mlir

std::unique_ptr<Pass> mlir::mtfusion::createMultiBufferingPass() {
  return std::make_unique<MultiBufferingPass>();
}
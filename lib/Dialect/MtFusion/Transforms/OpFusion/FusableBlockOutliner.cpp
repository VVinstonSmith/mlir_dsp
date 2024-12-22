//===- FusableBlockOutliner.cpp - Separate fusable blocks from its func ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mtir/Dialect/MtFusion/Transforms/OpFusion/FusableBlockOutliner.h"
#include "mtir/Dialect/MtFusion/IR/MtFusion.h"
#include "mtir/Dialect/MtFusion/Utils/Utils.h"
#include "mtir/Dialect/MtFusion/Transforms/OpFusion/FusableHelper.h"

#include "llvm/ADT/PriorityQueue.h"
#include "llvm/Support/Debug.h"

#include "mlir/Transforms/TopologicalSortUtils.h"
// #include "mlir/Analysis/TopologicalSortUtils.h"

#define DEBUG_TYPE "mtfusion-fuse"

namespace mlir {
namespace mtfusion {
namespace opfusion {

FusableBlockOutliner::FusableBlockOutliner(FusableBlocks &fusableBlocks,
                                           OutputMode outputMode,
                                           bool alwaysInline,
                                           bool skipOutliningReshape)
    : fusableBlocks_(fusableBlocks), alwaysInline_(alwaysInline) {
  if (outputMode == OutputMode::Multiple)
    return;

  LLVM_DEBUG(
      llvm::dbgs()
          << "Separating main fusable block for single output fusion\n";);
  FusableBlocks newFusableBlocks;
  for (FusableBlock &curBlock : fusableBlocks) {
    if (skipOutliningReshape) { // 如果既跳过OutliningReshape，又没有important op pattern，那么跳过该block.
      bool hasImportant = false;
      for (auto op : curBlock.getOps()) {
        if (FusableHelper::isImportantPattern(op)) {
          hasImportant = true;
          break;
        }
      }
      if (!hasImportant)
        continue;
    }
    SetVector<Operation *> opSet(curBlock.getOps().begin(),
                                 curBlock.getOps().end());

    DenseMap<Operation *, size_t> topoOrder;
    auto allOps = curBlock.getOps();
    LLVM_DEBUG(llvm::dbgs() << "\n Blocks:\n";);
    for (size_t i = 0; i < allOps.size(); i++) {
      LLVM_DEBUG(llvm::dbgs() << *allOps[i] << "\n");
      topoOrder[allOps[i]] = i;
    }
    SetVector<Operation *> currentOutputs;
    for (Value v : curBlock.getOutputs()) {
      // re-fusion non picked fused blocks for single mode is not implemented
      Operation *outOp = v.getDefiningOp();
      currentOutputs.insert(outOp);
    }
    using PairTopoOperation = std::pair<size_t, Operation *>;
    // Gather reachable ops from outOp (backward), outOp is not included.

    // The following invariants hold:
    // - If a fusable blocks A is schedulable, then its connected subgraph
    // inside is schedulable
    // - One node can't be in multiple blocks in SingleMode
    auto collectFusedOps = [&](Operation *outOp) -> SmallVector<Operation *> {
      // Priority queue is needed to maintain the topological order
      // we need it to compute something like
      // A --> B
      // |  /
      // v L
      // C --> outFuse
      //
      // If the order of traversal C, A, B
      // When it's time to relax A, it will not be included
      // because B is not fused to C yet.
      //
      // Need to relax B first, thus forcing topological order
      llvm::PriorityQueue<PairTopoOperation> dijkstraQueue;
      dijkstraQueue.push(PairTopoOperation(topoOrder[outOp], outOp));
      DenseSet<Operation *> newOpped;
      newOpped.insert(outOp);
      while (!dijkstraQueue.empty()) {
        Operation *curOp = dijkstraQueue.top().second;
        dijkstraQueue.pop();
        for (const Value &operand : curOp->getOperands()) {
          Operation *defOp = operand.getDefiningOp();
          if (!defOp)
            continue;
          if (!opSet.contains(defOp))
            continue;

          LLVM_DEBUG(llvm::dbgs() << "Relaxing " << *defOp << "\n");
          // If its within the fused block
          // A --> B --> outFuse
          // |  /
          // v L
          // C --> outFuse
          // will just take all
          bool safeToFuse = true;
          if (outputMode == OutputMode::SingleAggressive) {
            // Always safe to fuse
            if (currentOutputs.contains(defOp))
              safeToFuse = false;
          } else if (outputMode == OutputMode::Single) {
            // Check if the usage for this is all inside the newOpped
            for (const Value &res : defOp->getResults()) {
              if (!safeToFuse)
                break;
              for (Operation *opUser : res.getUsers()) {
                if (!newOpped.contains(opUser)) {
                  safeToFuse = false;
                  break;
                }
              }
            }
          } else {
            assert("outputMode not handled");
          }
          if (safeToFuse && !newOpped.contains(defOp)) {
            dijkstraQueue.push(PairTopoOperation(topoOrder[defOp], defOp));
            newOpped.insert(defOp);
          }
        }
      }

      SmallVector<Operation *> sortedNewOp(newOpped.begin(), newOpped.end());
      std::sort(sortedNewOp.begin(), sortedNewOp.end(),
                [&](Operation *opA, Operation *opB) {
                  return topoOrder[opA] < topoOrder[opB];
                });

      return sortedNewOp;
    };
    auto tmpOut = curBlock.getOutputs();
    for (Value v : tmpOut) {
      Operation *outOp = v.getDefiningOp();
      LLVM_DEBUG(llvm::dbgs() << "Separating for out " << *outOp << "\n");
      // re-fusion non picked fused blocks for single mode is not implemented
      const SmallVector<Operation *> &fusedOps = collectFusedOps(outOp);

      if (fusedOps.size() == 1)
        continue;
      SmallVector<Operation *, 1> tmpOutOp = {outOp};
      // Generating new ops
      newFusableBlocks.emplace_back(fusedOps, curBlock.fusableHelper_,
                                    tmpOutOp);
    }
  }
  fusableBlocks_.swap(newFusableBlocks);
}

SmallVector<func::FuncOp> FusableBlockOutliner::getOutlinedFuncs() const {
  return outlinedFuncs_;
}

bool FusableBlockOutliner::outline(const std::string &prefixOutline) {
  for (FusableBlock &curBlock : fusableBlocks_) {
    func::FuncOp fusedFunc = outlineFunc(curBlock, prefixOutline);
    if (!fusedFunc)
      return false;

    outlinedFuncs_.push_back(fusedFunc);

    func::CallOp fusionInvoke = createInvoke(fusedFunc, curBlock);
    if (!fusionInvoke)
      return false;
  }

  return true;
}

void FusableBlockOutliner::setOutlineFuncAttributes(
    func::FuncOp &func, const FusionKind &fusionKind, OpBuilder &builder,
    bool isCallerHost) {
  mtfusion::trySetFuncKind(func, mtfusion::FuncKind::Device);
  func->setAttr(FusionKindAttr::name,
                FusionKindAttr::get(func->getContext(), fusionKind));
  if (isCallerHost) {
    func->setAttr(mtfusion::EntryAttr::name,
                  builder.getUnitAttr());
  }
}

std::string FusableBlockOutliner::getNewFusionName(llvm::StringRef symbolName,
                                                   llvm::StringRef prefixName) {
  return symbolName.str() + prefixName.str() + "_" + std::to_string(funcCnt_++);
}

void FusableBlockOutliner::eraseTriviallyDeadOps(ArrayRef<Operation *> ops) {
  for (auto I = ops.rbegin(), E = ops.rend(); I != E; ++I) {
    Operation *curOp = *I;
    if (isOpTriviallyDead(curOp))
      curOp->erase();
  }
}

func::FuncOp
FusableBlockOutliner::outlineFunc(FusableBlock &curBlock,
                                  const std::string &prefixOutline) {
  func::FuncOp parF = curBlock.getParentOfType<func::FuncOp>();
  OpBuilder curBuilder(parF.getContext());
  OpBuilder::InsertionGuard insGuard(curBuilder);
  curBuilder.setInsertionPoint(parF);
  // Create function prototype
  FunctionType funcTy = FunctionType::get(
      parF.getContext(), TypeRange(ValueRange(curBlock.getInputs())),
      TypeRange(ValueRange(curBlock.getOutputs())));
  func::FuncOp newFunc = curBuilder.create<func::FuncOp>(
      curBlock.getLoc(), getNewFusionName(parF.getSymName(), prefixOutline),
      funcTy);
  setOutlineFuncAttributes(newFunc, curBlock.fusableHelper_->getFusionKind(),
                           curBuilder, mtfusion::isHost(parF));
  if (alwaysInline_) {
    newFunc->setAttr(mtfusion::AlwaysInlineAttr::getMnemonic(),
                     mtfusion::AlwaysInlineAttr::get(newFunc.getContext()));
  }

  // Create function body
  Block *entryBB = newFunc.addEntryBlock();
  curBuilder.setInsertionPointToStart(entryBB);

  // Clone operations and replace usages
  IRMapping curMap;
  for (auto [oldIn, newIn] :
       llvm::zip(curBlock.getInputs(), entryBB->getArguments())) {
    curMap.map(oldIn, newIn);
  }

  SetVector<Operation *> newOps;
  for (Operation *op : curBlock.getOpWithAuxs())
    newOps.insert(curBuilder.clone(*op, curMap));

  SetVector<Value> outs;
  for (Value out : curBlock.getOutputs()) {
    assert(curMap.getValueMap().contains(out));
    outs.insert(curMap.getValueMap().at(out));
  }

  curBuilder.create<func::ReturnOp>(curBlock.getLoc(),
                                    ValueRange(outs.getArrayRef()));

  eraseTriviallyDeadOps(newOps.getArrayRef());
  mtfusion::trySetFuncKind(newFunc, mtfusion::FuncKind::Device);
  return newFunc;
}

func::CallOp FusableBlockOutliner::createInvoke(func::FuncOp newFunc,
                                                FusableBlock &fusionBlock) {
  OpBuilder curBuilder(newFunc.getContext());
  OpBuilder::InsertionGuard insGuard(curBuilder);

  curBuilder.setInsertionPointAfter(fusionBlock.getLastOp());

  func::CallOp newInvoke = curBuilder.create<func::CallOp>(
      fusionBlock.getLoc(), newFunc, fusionBlock.getInputs());
  for (auto [oldOut, newOut] :
       llvm::zip(fusionBlock.getOutputs(), newInvoke->getResults()))
    ((Value)oldOut).replaceAllUsesWith(newOut);

  Block *curBlock = curBuilder.getInsertionBlock();
  if (!curBlock->verifyOpOrder())
    sortTopologically(curBlock);

  eraseTriviallyDeadOps(fusionBlock.getOpWithAuxs());

  func::FuncOp curFunc = newInvoke->template getParentOfType<func::FuncOp>();

  // Will assume it's not heterogeneous if it's not defined
  mtfusion::trySetFuncKind(curFunc, mtfusion::FuncKind::Device);

  return newInvoke;
}
} // namespace opfusion
} // namespace mtfusion
} // namespace mlir
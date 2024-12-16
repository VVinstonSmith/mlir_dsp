//===- FusableBlock.cpp - Fusable block contains fusable ops --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mtir/Dialect/MtFusion/Transforms/OpFusion/FusableBlock.h"
#include "mtir/Dialect/MtFusion/Transforms/OpFusion/FusableHelper.h"

#define DEBUG_TYPE "mtfusion-fuse"

namespace mlir {
namespace mtfusion {
namespace opfusion {

void FusableBlock::dump() {
  llvm::dbgs() << "FusableBlock {\n";
  llvm::dbgs() << "ins:\n";
  for (Value val : getInputs()) {
    val.print(llvm::dbgs());
    llvm::dbgs() << "\n";
  }

  llvm::dbgs() << "outs:\n";
  for (Value val : getOutputs()) {
    val.print(llvm::dbgs());
    llvm::dbgs() << "\n";
  }

  llvm::dbgs() << "Ops:\n";
  for (Operation *op : getOps()) {
    op->print(llvm::dbgs());
    llvm::dbgs() << "\n";
  }

  llvm::dbgs() << "OpWithAuxs:\n";
  for (Operation *op : getOpWithAuxs()) {
    op->print(llvm::dbgs());
    llvm::dbgs() << "\n";
  }

  llvm::dbgs() << "}\n";
}

void FusableBlock::visitOutValues() {
  SetVector<Operation *> observedOuts =
      outsModification_.empty() ? ops_ : outsModification_;
  for (Operation *op : observedOuts) {
    for (const Value &res : op->getResults()) {
      for (Operation *opUser : res.getUsers()) {
        if (!ops_.contains(opUser)) {
          outs_.insert(res);
        }
      }
    }
  }
  fillNonEdgeOps();
  assert(!outs_.empty());
}

void FusableBlock::fillNonEdgeOps() { 
  for (Operation *op : ops_) {
    for (const Value &res : op->getResults()) {
      if (!outs_.count(res)) {
        nonEdgeOps_.insert(res.getDefiningOp());
      }
    }
  }
}

bool FusableBlock::shouldIncludeOp(Operation *defOp, Operation *parentOp,
                                   bool isStoppingBuffer) {
  // Check if defOp has the same parent and meets inclusion criteria
  return defOp->getParentOp() == parentOp && !opWithAuxs_.contains(defOp) && 
         (fusableHelper_->includeAuxiliary(defOp) ||
          (!isStoppingBuffer && fusableHelper_->includeBuffer(defOp)));
}

void FusableBlock::processOperand(const Value &operand, Operation *parentOp,
                                  bool isStoppingBuffer,
                                  SetVector<Operation *> &newOps) {
  if (Operation *defOp = operand.getDefiningOp()) {
    LLVM_DEBUG(llvm::dbgs() << "Checking defOp: " << *defOp << "\n";);
    if (shouldIncludeOp(defOp, parentOp, isStoppingBuffer)) {
      newOps.insert(defOp);
    }
  }
}

void FusableBlock::processRegionOperands(Operation *op, bool isStoppingBuffer,
                                         SetVector<Operation *> &newOps) {
  for (Region &region : op->getRegions()) {
    region.walk([&](Operation *regionOp) {
      for (const Value &operand : regionOp->getOperands()) {
        processOperand(operand, op->getParentOp(), isStoppingBuffer, newOps);
      } 
    });
  }
}

void FusableBlock::visitAuxiliaryOps() {
  assert(opWithAuxs_.empty());
  if (outs_.empty())
    visitOutValues();
  opWithAuxs_ = ops_;

  // Include auxiliary ops
  {
    DenseSet<Operation *> visited;
    do {
      SetVector<Operation *> newOps;
      for (Operation *op : opWithAuxs_) {
        bool isStoppingBuffer =
            (fusableHelper_->moveOutToParam() && !nonEdgeOps_.contains(op));
        if (visited.contains(op))
          continue;

        // Process direct operands
        for (const Value &operand : op->getOperands()) {
          processOperand(operand, op->getParentOp(), isStoppingBuffer, newOps);
        }

        // Process region operands
        processRegionOperands(op, isStoppingBuffer, newOps);

        visited.insert(op);
      }

      if (newOps.empty())
        break;

      newOps.insert(opWithAuxs_.begin(), opWithAuxs_.end());
      opWithAuxs_.swap(newOps);
    } while (true);
  }
}

void FusableBlock::visitInValues() {
  for (Operation *op : getOpWithAuxs()) {
    for (const Value &operand : op->getOperands()) {
      if (Operation *defOp = operand.getDefiningOp()) {
        if (!opWithAuxs_.contains(defOp)) {
          ins_.insert(operand);
        }
      } else {
        ins_.insert(operand);
      }
    }
  }
  assert(!ins_.empty());
}
} // namespace opfusion
} // namespace mtfusion
} // namespace mlir
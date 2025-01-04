  //===--------- RedundantExtractRemoval.cpp - RedundantExtractRemoval Pass -----------------===//
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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include <iostream>
using namespace std;

namespace mlir {
#define GEN_PASS_DEF_REDUNDANTEXTRACTREMOVAL
#include "mtir/Dialect/MtFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mtfusion;

namespace {

bool isSameSize(OffsetSizeAndStrideOpInterface op1,
    OffsetSizeAndStrideOpInterface op2) {
  if(op1.getSizes().size() != op2.getSizes().size())
    return false;
  size_t rank = op1.getSizes().size();
  for(size_t i = 0; i < rank; i++) {
    if(op1.isDynamicSize(i) && op2.isDynamicSize(i)) {
      if(op1.getDynamicSize(i) != op2.getDynamicSize(i))
        return false;
    } else if(!op1.isDynamicSize(i) && !op2.isDynamicSize(i)) {
      if(op1.getStaticSize(i) != op2.getStaticSize(i))
        return false;
    } else {
      return false;
    }
  }
  return true;
}

bool isSameSize(tensor::EmptyOp emptyOp, tensor::ExtractSliceOp sliceOp) {
  auto emptyType = emptyOp.getType();
  if(sliceOp.getSizes().size() != emptyType.getRank())
    return false;
  size_t rank = sliceOp.getSizes().size();
  for(size_t i = 0, dynIdx = 0; i < rank; i++) {
    if(sliceOp.isDynamicSize(i) && emptyType.isDynamicDim(i)) {
      if(sliceOp.getDynamicSize(i) != emptyOp.getDynamicSizes()[dynIdx++])
        return false;
    } else if(!sliceOp.isDynamicSize(i) && !emptyType.isDynamicDim(i)) {
      if(sliceOp.getStaticSize(i) != emptyType.getDimSize(i))
        return false;
    } else {
      return false;
    }
  }
  return true;
}

bool isSameSize(memref::AllocOp allocOp, memref::SubViewOp subviewOp) {
  auto allocType = allocOp.getType();
  if(subviewOp.getSizes().size() != allocType.getRank())
    return false;
  size_t rank = subviewOp.getSizes().size();
  for(size_t i = 0, dynIdx = 0; i < rank; i++) {
    if(subviewOp.isDynamicSize(i) && allocType.isDynamicDim(i)) {
      if(subviewOp.getDynamicSize(i) != allocOp.getDynamicSizes()[dynIdx++])
        return false;
    } else if(!subviewOp.isDynamicSize(i) && !allocType.isDynamicDim(i)) {
      if(subviewOp.getStaticSize(i) != allocType.getDimSize(i))
        return false;
    } else {
      return false;
    }
  }
  return true;
}

// void replaceParentWithChild(Operation* parentOp, Operation* childOp) {
//   if(auto childSliceOp = dyn_cast<tensor::ExtractSliceOp>(childOp)) {
//     if(auto parentSliceOp = dyn_cast<tensor::ExtractSliceOp>(parentOp)) {
//     } else if(parentEmptyOp = dyn_cast<tensor::EmptyOp>(parentOp)) {
//     }
//   } else if(auto childSubviewOp = dyn_cast<memref::SubViewOp>(childOp)) {
//   }
// }

void removeRedundantExtract(scf::ForOp forOp) {
  OpBuilder builder(forOp.getContext());
  forOp.walk([&](Operation* op) {
    if(op->getParentOp() != forOp)
      return WalkResult::skip();
    if(!isa<memref::SubViewOp>(op) && !isa<tensor::ExtractSliceOp>(op))
      return WalkResult::skip();
    // if current op is a tensor.extract_slice
    if(auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(op)) {
      Operation* parentOp = sliceOp.getSource().getDefiningOp();
      if(!parentOp) {
        // if current op's src value is a scf.for iter_arg.
        for(auto [idx, regionIterArg] : llvm::enumerate(forOp.getRegionIterArgs())) {
          if(regionIterArg == sliceOp.getSource()) {
            parentOp = forOp.getInitArgs()[idx].getDefiningOp();
            break;
          }
        }
      }
      if(!parentOp)
        return WalkResult::skip();
      // if the parent op is a linalg.copy
      while(auto copyOp = dyn_cast<linalg::CopyOp>(parentOp)) {
        parentOp = copyOp.getInputs()[0].getDefiningOp();
      }
      // if the parent op is a tensor.extract_slice
      if(auto parentSliceOp = dyn_cast<tensor::ExtractSliceOp>(parentOp)) {
        if(isSameSize(parentSliceOp, sliceOp)) {
          sliceOp.replaceAllUsesWith(parentSliceOp.getResult());
          sliceOp.erase();
        }
      }
      // if the parent op is a tensor.empty
      else if(auto emptyOp = dyn_cast<tensor::EmptyOp>(parentOp)) {
        if(isSameSize(emptyOp, sliceOp)) {
          sliceOp.replaceAllUsesWith(emptyOp.getResult());
          sliceOp.erase();
        }
      }
    }
    // if current op is a memref.subview
    else if(auto subviewOp = dyn_cast<memref::SubViewOp>(op)) {
      // if current op has a parent op.
      if(auto parentOp = subviewOp.getSource().getDefiningOp()) {
        // if the parent op is a memref.alloc
        if(auto allocOp = dyn_cast<memref::AllocOp>(parentOp)) {
          if(isSameSize(allocOp, subviewOp)) {
            subviewOp.replaceAllUsesWith(allocOp.getResult());
            subviewOp.erase();
          }
        }
        // if the parent op is a memref.subview
        else if(auto parentSubviewOp = dyn_cast<memref::SubViewOp>(parentOp)) {
          if(isSameSize(parentSubviewOp, subviewOp)) {
            subviewOp.replaceAllUsesWith(parentSubviewOp.getResult());
            subviewOp.erase();
          }
        }
      }
    }
    return WalkResult::advance();
  });
}

void recursiveTraverseForOps(scf::ForOp forOpRoot, int level){
  removeRedundantExtract(forOpRoot);

  forOpRoot.walk([&](scf::ForOp forOp){
    if(forOp->getParentOp() != forOpRoot)
      return WalkResult::skip();
    
    recursiveTraverseForOps(forOp, level + 1);
    
    return WalkResult::advance();
  });
}

} // namepsace

namespace mlir {
class RedundantExtractRemovalPass
    : public impl::RedundantExtractRemovalBase<RedundantExtractRemovalPass> {
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

std::unique_ptr<Pass> mlir::mtfusion::createRedundantExtractRemovalPass() {
  return std::make_unique<RedundantExtractRemovalPass>();
}
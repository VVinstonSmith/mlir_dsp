  //===--------- ThreadParallelization.cpp - ThreadParallelization Pass -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mtir/Dialect/MtFusion/IR/MtFusion.h"
#include "mtir/Dialect/MtFusion/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"

#include <iostream>
using namespace std;

namespace mlir {
#define GEN_PASS_DEF_THREADPARALLELIZATION
#include "mtir/Dialect/MtFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mtfusion;

namespace {

SmallVector<std::pair<Value, scf::ForOp>> getForOpUses(Value val) {
  SmallVector<std::pair<Value, scf::ForOp>> ans;
  for(auto& operand : val.getUses()) {
    Operation* ownerOp = operand.getOwner();
    if(auto forOp = dyn_cast<scf::ForOp>(ownerOp)) {
      if(forOp.getStep() == operand.get()) {
         ans.push_back({operand.get(), cast<scf::ForOp>(forOp)});
      }   
    } else {
      for(auto res : ownerOp->getResults()) {
        ans.append(getForOpUses(res));
      }
    }
  }
  return ans;
}

void parallelizeforOp(scf::ForOp forOp, unsigned nthreads) {
  OpBuilder builder(forOp.getContext());
  OpBuilder::InsertionGuard insGuard(builder);
  builder.setInsertionPoint(forOp);
  auto curLoc = forOp.getLoc();

  auto cst_n = builder.create<arith::ConstantIndexOp>(curLoc, nthreads);
  auto oldStep = forOp.getStep();
  auto newStep = builder.create<arith::MulIOp>(curLoc, oldStep, cst_n);

  forOp.setStep(newStep);
}

} // namepsace

namespace mlir {
class ThreadParallelizationPass
    : public impl::ThreadParallelizationBase<ThreadParallelizationPass> {
public:
  void runOnOperation() override {

    ThreadParallelizationOptions options;
    options.nthreads = this->nthreads;

    func::FuncOp funcOp = getOperation();

    for(size_t argIdx = 0; argIdx < funcOp.getNumArguments(); argIdx++) {
      if(!funcOp.getArgAttr(argIdx, mtfusion::TilingDataAttr::name))
        continue;
      // If this tiling argument needs nthread-parallelization.
      if(auto nthreadsAttr = funcOp.getArgAttr(
          argIdx, mtfusion::NthreadsAttr::name)) {
        auto nthreads = (options.nthreads == 0) ? 
            cast<NthreadsAttr>(nthreadsAttr).getNthreads() : options.nthreads;
        
        auto tilingDataUses = getForOpUses(funcOp.getArgument(argIdx));
        
        for(auto [stepVal, forOp] : tilingDataUses) {
          parallelizeforOp(forOp, nthreads);
        }
      }
    }
  }
};
} // namespace mlir

std::unique_ptr<Pass> mlir::mtfusion::createThreadParallelizationPass() {
  return std::make_unique<ThreadParallelizationPass>();
}
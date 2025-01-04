  //===--------- BufferNormalize.cpp - BufferNormalize Pass -----------------===//
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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"

#include <iostream>
#include <map>
#include <utility>
using namespace std;

namespace mlir {
#define GEN_PASS_DEF_BUFFERNORMALIZE
#include "mtir/Dialect/MtFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mtfusion;

namespace {

const std::map<std::string, mtfusion::Cache> StrToCache = {
    {"DDR", Cache::DDR}, {"GSM", Cache::GSM},
    {"AM", Cache::AM}, {"SM", Cache::SM}};

std::pair<mtfusion::Cache, mtfusion::Cache> getSrcAndDst(StringRef str) {
  std::string src, dst;
  auto it = str.begin();
  while(it != str.end() && *it == ' ')
    ++it;
  while(it != str.end() && *it != ':') {
    src.push_back(*it);
    ++it;
  }
  ++it;
  while(it != str.end() && *it == ' ')
    ++it;
  while(it != str.end() && *it != ':') {
    dst.push_back(*it);
    ++it;
  }
  while(!src.empty() && src.back() == ' ')
    src.pop_back();
  while(!dst.empty() && dst.back() == ' ')
    dst.pop_back();
  assert(StrToCache.count(src) && StrToCache.count(dst));
  return {StrToCache.at(src), StrToCache.at(dst)};
}

memref::AllocOp searchAllocFromValue(Value val) {
  while(auto parentOp = val.getDefiningOp()) {
    if(auto allocOp = dyn_cast<memref::AllocOp>(parentOp)) {
      return allocOp;
    } 
    if(auto subviewOp = dyn_cast<memref::SubViewOp>(parentOp)) {
      val = subviewOp.getSource();
    }
  }
  return memref::AllocOp();
}

void replaceAllocSizesWithForStep(memref::AllocOp allocOp) {
  for(auto [oprIdx, operand] :
      llvm::enumerate(allocOp.getOperands())) {
    auto parentOp = operand.getDefiningOp();
    if(isa<affine::AffineApplyOp>(parentOp) || 
        isa<affine::AffineMinOp>(parentOp) ||
        isa<affine::AffineMaxOp>(parentOp)) { 
      for(auto affineOpr : parentOp->getOperands()) {
        bool flag = false;
        for(auto op : affineOpr.getUsers()) {
          if(auto forOp = dyn_cast<scf::ForOp>(op)) {
            if(affineOpr == forOp.getStep()) {
              allocOp.setOperand(oprIdx, affineOpr);
              flag = true;
              break;
            }
          }
        } if(flag) break;
      }
    }
  }
}

} // namepsace

namespace mlir {
class BufferNormalizePass
    : public impl::BufferNormalizeBase<BufferNormalizePass> {
public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    OpBuilder builder(funcOp.getContext());
    /// Eleminate redundant memref.copy.
    funcOp.walk([&](memref::CopyOp memCopyOp) {
      auto memCopySrc = memCopyOp.getSource();
      for(auto user : memCopySrc.getUsers()) {
        if(auto linalgCopy = dyn_cast<linalg::CopyOp>(user)) {
          auto linalgCopyOutOpr = linalgCopy.getOutputs()[0];
          if(memCopySrc == linalgCopyOutOpr) {
            linalgCopy.setDpsInitOperand(0, memCopyOp.getTarget());
          }
        } 
      }
      memCopyOp.erase();
      return WalkResult::advance();
    });
    /// Normalize memref.alloc.
    funcOp.walk([&](linalg::CopyOp copyOp) {
      auto attrNames = copyOp->getAttrs();
      if(attrNames.empty())
        return WalkResult::skip();
      /// Read memory level from stringAttr. 
      auto [_, dstStr] = getSrcAndDst(attrNames.front().getName());
      if(auto allocOp = searchAllocFromValue(copyOp.getDpsInits()[0])) {
        /// Add tag to memref.alloc according to memLevel attr
        allocOp->setAttr(
          mtfusion::MemLevelAttr::name,
          mtfusion::MemLevelAttr::get(funcOp.getContext(), dstStr));
        /// Replace the sizes of memref.alloc with the step of scf.for
        replaceAllocSizesWithForStep(allocOp);
      }
      return WalkResult::advance();
    });
  }
};
} // namespace mlir

std::unique_ptr<Pass> mlir::mtfusion::createBufferNormalizePass() {
  return std::make_unique<BufferNormalizePass>();
}
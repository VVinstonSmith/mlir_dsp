//===- CopyRemovalPass.cpp ---- Redundant Copy Removal Rransform Pass -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mtir/Dialect/MtFusion/IR/MtFusion.h"
#include "mtir/Dialect/MtFusion/Transforms/Passes.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallSet.h"

namespace mlir {
#define GEN_PASS_DEF_REDUNDANTCOPYREMOVAL
#include "mtir/Dialect/MtFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::mtfusion;

/// This pass removes redundant copy operations. Additionally, it
/// removes leftover definition and deallocation operations by erasing the
/// copy operation.
struct RedundantCopyRemovalPass
    : public impl::RedundantCopyRemovalBase<RedundantCopyRemovalPass> {
public:
  void runOnOperation() override;

private:
  /// Returns the allocation operation for `value` if it exists.
  /// nullptr otherwise.
  Operation *getAllocationOp(Value value) {
    if (Operation *op = value.getDefiningOp()) {
      if (auto effects = dyn_cast<MemoryEffectOpInterface>(op))
        if (effects.hasEffect<mlir::MemoryEffects::Allocate>())
          return op;
    }
    return nullptr;
  }

  /// Returns the deallocation operation for `value` if it exists.
  /// nullptr otherwise.
  Operation *getDeallocationOp(Value value) const {
    auto valueUsers = value.getUsers();
    auto it = llvm::find_if(valueUsers, [&](Operation *op) {
      auto effects = dyn_cast<MemoryEffectOpInterface>(op);
      return effects && effects.hasEffect<mlir::MemoryEffects::Free>();
    });
    return (it == valueUsers.end() ? nullptr : *it);
  }

  /// Check whether the `val` is used by `op`.
  static bool doesOpUseVal(Value val, Operation *op) {
    return llvm::is_contained(op->getOperands(), val);
  }

  /// Check if an op that lies on one of the paths between `start`
  /// and `end` and satisfies `checkPropertiesOfOperation`.
  bool hasInterveningOp(const Value val, Operation *start, Operation *end,
                        std::function<bool(Value, Operation *)>
                            checkPropertiesOfOperation) const {
    // Check for all paths from operation `fromp` to operation `untilOp` for the
    // given property.
    std::function<bool(Operation *, Operation *)> recur =
        [&](Operation *fromOp, Operation *untilOp) {
          auto fromOpBlock = fromOp->getBlock();
          for (auto iter = ++fromOp->getIterator(), end = fromOpBlock->end();
               iter != end && &*iter != untilOp; ++iter) {
            if (checkPropertiesOfOperation(val, &*iter)) {
              return true;
            }
          }
          return false;
        };
    return recur(start, end);
  }

  void removeCopy(CopyOpInterface copyOp,
                  llvm::SmallPtrSet<Operation *, 4> &opsToErase) {
    Value src = copyOp.getSource();
    Value dest = copyOp.getTarget();
    /// Constraints:
    /// 1) The `destination` op should be MemoryEffects::Allocate op or function
    /// argument.
    /// 2) If the `destination` op is MemoryEffects::Allocate op, there should
    /// not exist any users of `destination` op before the copy op. We replace
    /// the dest by src.
    /// Input:
    /// func() {
    ///   %source = alloc/alloca()
    ///   %destination = alloc/alloca()
    ///   write_to(%source)
    ///   copy(%source, %destination)
    ///   return %destination
    /// }
    ///
    /// Output:
    /// func(){
    ///   %source = alloc/alloca()
    ///   write_to(%source)
    ///   return %source
    /// }

    /// 3) If the `destination` op is function argument, which means there
    /// should not exist any users of `source` op after the copy op. We replace
    /// the src by dest.
    /// Input:
    /// func(%destination : memref) {
    ///   %source = alloc()
    ///   write_to(%source)
    ///   copy(%source, %destination)
    ///   dealloc(%source)
    ///   return
    /// }
    ///
    /// Output:
    /// func(%destination : memref){
    ///   write_to(%destination)
    ///   return
    /// }

    Operation *destDefOp = getAllocationOp(dest);
    Operation *srcDefOp = getAllocationOp(src);

    if ((destDefOp == nullptr && !llvm::isa<BlockArgument>(dest)) ||
        (destDefOp != nullptr &&
         !mlir::hasEffect<mlir::MemoryEffects::Allocate>(destDefOp))) {
      return;
    }
    Operation *srcDeallocOp = getDeallocationOp(src);
    Operation *destDeallocOp = getDeallocationOp(dest);
    Operation *firstOpUsingDest = &dest.getParentRegion()->front().front();
    Operation *lastOpUsingSrc = &src.getParentRegion()->back().back();

    if (hasInterveningOp(src, copyOp,
                         srcDeallocOp ? srcDeallocOp : lastOpUsingSrc,
                         &doesOpUseVal) ||
        hasInterveningOp(dest, destDefOp ? destDefOp : firstOpUsingDest, copyOp,
                         &doesOpUseVal)) {
      return;
    }

    if (destDefOp) {
      // replace dst by src
      opsToErase.insert(destDefOp);
      opsToErase.insert(copyOp);
      if (destDeallocOp)
        opsToErase.insert(destDeallocOp);
      dest.replaceAllUsesWith(src);
      return;
    }

    if (srcDefOp &&
        dest.getParentBlock()->getParentOp()->isAncestor(srcDefOp)) {
      // replace src by dst
      opsToErase.insert(srcDefOp);
      opsToErase.insert(copyOp);
      if (srcDeallocOp)
        opsToErase.insert(srcDeallocOp);
      src.replaceAllUsesWith(dest);
      return;
    }
  }
};

void RedundantCopyRemovalPass::runOnOperation() {
  func::FuncOp func = getOperation();
  llvm::SmallPtrSet<Operation *, 4> opsToErase;
  Liveness live(func);
  func.walk([&](CopyOpInterface copyOp) { removeCopy(copyOp, opsToErase); });
  for (Operation *op : opsToErase) {
    assert(op->use_empty() &&
           "uses remaining for copy ops, memref allocation and deallocation "
           "ops that should have ready to be erased");
    op->erase();
  }
  return;
}

std::unique_ptr<Pass> mlir::mtfusion::createRedundantCopyRemovalPass() {
  return std::make_unique<RedundantCopyRemovalPass>();
}
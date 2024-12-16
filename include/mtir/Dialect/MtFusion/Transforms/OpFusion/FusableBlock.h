//===- FusableBlock.h --------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mtir/Dialect/MtFusion/Transforms/OpFusion/FusableHelper.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <iostream>

#ifndef MTIR_DIALECT_MTFUSION_TRANSFORMS_OPFUSION_FUSABLEBLOCK_H
#define MTIR_DIALECT_MTFUSION_TRANSFORMS_OPFUSION_FUSABLEBLOCK_H

namespace mlir {
namespace mtfusion {
namespace opfusion {
class FusableBlock {
public:
  explicit FusableBlock(const llvm::ArrayRef<Operation *> ops,
                        const FusableHelper *fusableHelper,
                        bool isSingleOpt = false)
      : fusableHelper_(fusableHelper), ops_(ops.begin(), ops.end()) {
    if (isSingleOpt)
      assert(ops_.size() == 1);
    else
      assert(ops_.size() > 1);
  };
  explicit FusableBlock(const llvm::ArrayRef<Operation *> ops,
                        const FusableHelper *fusableHelper,
                        const llvm::ArrayRef<Operation *> mod)
      : fusableHelper_(fusableHelper), ops_(ops.begin(), ops.end()),
        outsModification_(mod.begin(), mod.end()) { 
    assert(ops_.size() > 1);

    // std::cout<<"outsModification_ : "<<std::endl;                 
    // for(auto op : outsModification_){
    //   op->dump();
    // }
    // std::cout<<std::endl; 
  };

  Operation *getLastOp() { return getOutputs().back().getDefiningOp(); }
  template <typename T>
  T getParentOfType() const {
    return getOps().back()->getParentOfType<T>();
  }
  Location getLoc() const { return getOps().back()->getLoc(); }

  llvm::ArrayRef<Operation *> getOps() const { return ops_.getArrayRef(); }

  llvm::ArrayRef<Value> getInputs() {
    if (ins_.empty())
      visitInValues();
    return ins_.getArrayRef();
  }

  llvm::ArrayRef<Value> getOutputs() {
    if (outs_.empty())
      visitOutValues();
    return outs_.getArrayRef();
  }

  llvm::ArrayRef<Operation *> getOpWithAuxs() {
    if (opWithAuxs_.empty())
      visitAuxiliaryOps();
    return opWithAuxs_.getArrayRef();
  }
  void dump();

  const FusableHelper *fusableHelper_;

private:
  void visitOutValues();
  void fillNonEdgeOps();
  void visitAuxiliaryOps();
  void visitInValues();
  bool shouldIncludeOp(Operation *defOp, Operation *parentOp,
                       bool isStoppingBuffer);
  void processOperand(const Value &operand, Operation *parentOp,
                      bool isStoppingBuffer, SetVector<Operation *> &newOps);
  void processRegionOperands(Operation *op, bool isStoppingBuffer,
                             SetVector<Operation *> &newOps);

  mutable llvm::SetVector<Operation *> ops_;
  mutable llvm::SetVector<Operation *> outsModification_;
  mutable llvm::SetVector<Operation *> opWithAuxs_; // 用于存储与主要操作相关的辅助操作。辅助操作可以包括但不限于内存分配、释放、数据转换等，它们对于主要操作的正确执行是必要的，但本身并不直接参与主要的计算逻辑
  mutable llvm::SetVector<Operation *> nonEdgeOps_;
  mutable llvm::SetVector<Value> ins_;
  mutable llvm::SetVector<Value> outs_;
};
} // namespace opfusion
} // namespace mtfusion
} // namespace mlir

#endif // MTIR_DIALECT_MTFUSION_TRANSFORMS_OPFUSION_FUSABLEBLOCK_H

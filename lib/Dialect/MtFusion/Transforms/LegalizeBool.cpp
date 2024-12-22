//===--------- LegalizeBool.cpp - Legalize bool type Pass -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mtir/Dialect/MtFusion/IR/MtFusion.h"
#include "mtir/Dialect/MtFusion/Transforms/Passes.h"
#include "mtir/Dialect/MtFusion/Utils/Utils.h"
#include "mtir/Dialect/Utils/Util.h"

namespace mlir {
#define GEN_PASS_DEF_LEGALIZEBOOLPASS
#include "mtir/Dialect/MtFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mtfusion;

class LegalizeBoolPass : public impl::LegalizeBoolPassBase<LegalizeBoolPass> {
public:
  void runOnOperation() override;

private:
  LogicalResult convertEntryKernel(func::FuncOp func, OpBuilder &builder) {
    FunctionType oldType = func.getFunctionType();
    llvm::SmallVector<Type, 4> newInputTypes;
    llvm::SmallVector<Type, 4> newResultTypes;

    // Convert Input Type
    for (Type type : oldType.getInputs()) {
      newInputTypes.push_back(convertBoolToInt8(type));
    }

    // Convert Result Type
    for (Type type : oldType.getResults()) {
      newResultTypes.push_back(convertBoolToInt8(type));
    }

    // Create new function type
    FunctionType newType =
        builder.getFunctionType(newInputTypes, newResultTypes);
    func.setType(newType);

    // Update function body
    if (!func.empty()) {
      Block &entryBlock = func.getBody().front();

      // Update block argument types
      for (unsigned i = 0; i < func.getNumArguments(); ++i) {
        BlockArgument arg = entryBlock.getArgument(i);
        Type newType = newInputTypes[i];
        arg.setType(newType);
      }

      builder.setInsertionPointToStart(&entryBlock);

      // Cast updated i8 input argument to Int1
      Type i1Type = builder.getI1Type();
      Type i8Type = builder.getI8Type();
      for (unsigned i = 0; i < func.getNumArguments(); ++i) {
        if (isI1ElemType(oldType.getInput(i))) {
          Value arg = func.getArgument(i);
          auto mode =
              mlir::utils::selectRoundMode<mtfusion::RoundMode>(i8Type, i1Type);
          Value castResult =
              mtfusion::castTo(builder, /*src=*/arg, /*targetElemType=*/i1Type,
                              /*roundMode=*/mode);
          arg.replaceAllUsesExcept(castResult, castResult.getDefiningOp());
        }
      }

      // Sign extend boolean return value to Int8
      func.walk([&](func::ReturnOp returnOp) {
        builder.setInsertionPoint(returnOp);
        llvm::SmallVector<Value, 4> newOperands;
        for (Value operand : returnOp.getOperands()) {
          if (isI1ElemType(operand.getType())) {
            auto extOp = builder.create<arith::ExtUIOp>(
                returnOp.getLoc(), builder.getI8Type(), operand);
            newOperands.push_back(extOp.getResult());
          } else {
            newOperands.push_back(operand);
          }
        }
        returnOp->setOperands(newOperands);
      });
    }
    return success();
  }

  bool isIntegerElemType(Type type, unsigned width) const {
    auto elemTy = getElementTypeOrSelf(type);
    return elemTy.isInteger(width);
  }

  bool isI1ElemType(Type type) const { return isIntegerElemType(type, 1); }
  bool isI8ElemType(Type type) const { return isIntegerElemType(type, 8); }

  Type convertBoolToInt8(Type type) {
    if (!isI1ElemType(type)) {
      return type;
    }
    Type typeInt8 = IntegerType::get(type.getContext(), 8);
    if (type.isInteger(1)) {
      // scalar type i1
      return typeInt8;
    }
    auto shapedType = dyn_cast_or_null<ShapedType>(type);
    if (!shapedType) {
      return type;
    }
    // shaped type i1
    return shapedType.clone(typeInt8);
  }
};

void LegalizeBoolPass::runOnOperation() {
  MLIRContext *context = &getContext();
  OpBuilder builder(context);

  func::FuncOp entryFunc = getOperation();
  if (!entryFunc->hasAttr(mtfusion::EntryAttr::name)) {
    return;
  }

  if (failed(convertEntryKernel(entryFunc, builder))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mtfusion::createLegalizeBoolPass() {
  return std::make_unique<LegalizeBoolPass>();
}


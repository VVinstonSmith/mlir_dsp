//===- FusableBlockOutliner.h ------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mtir/Dialect/MtFusion/Transforms/OpFusion/FusableBlock.h"
#include "mtir/Dialect/MtFusion/Transforms/Passes.h"

#ifndef MTIR_DIALECT_MTFUSION_TRANSFORMS_OPFUSION_FUSABLEBLOCKOUTLINER_H
#define MTIR_DIALECT_MTFUSION_TRANSFORMS_OPFUSION_FUSABLEBLOCKOUTLINER_H
namespace mlir {
namespace mtfusion {
namespace opfusion {
class FusableBlockOutliner {
public:
  FusableBlockOutliner(FusableBlocks &fusableBlocks, OutputMode outputMode,
                       bool alwaysInline, bool skipReshapingOp = false);

  SmallVector<func::FuncOp> getOutlinedFuncs() const;

  static void setOutlineFuncAttributes(func::FuncOp &func,
                                       const FusionKind &fusionKind,
                                       OpBuilder &builder, bool isCallerHost);
  bool outline(const std::string &prefixOutline = "");

private:
  size_t funcCnt_{0};
  std::string getNewFusionName(llvm::StringRef symbolName,
                               llvm::StringRef prefixName);
  void eraseTriviallyDeadOps(ArrayRef<Operation *> ops);
  func::FuncOp outlineFunc(FusableBlock &curBlock,
                           const std::string &prefixOutline = "");
  func::CallOp createInvoke(func::FuncOp newFunc, FusableBlock &fusionBlock);

  FusableBlocks &fusableBlocks_;
  const bool alwaysInline_;
  SmallVector<func::FuncOp> outlinedFuncs_;
};
} // namespace opfusion
} // namespace mtfusion
} // namespace mlir

#endif // MTIR_DIALECT_MTFUSION_TRANSFORMS_OPFUSION_FUSABLEBLOCKOUTLINER_H

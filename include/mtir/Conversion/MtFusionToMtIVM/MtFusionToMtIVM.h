//===- MtFusionToMtIVM.h - MtFusion to MtIVM/LLVM conversion ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MTIR_CONVERSION_MTFUSIONTOMTIVM_MTFUSIONTOMTIVM_H
#define MTIR_CONVERSION_MTFUSIONTOMTIVM_MTFUSIONTOMTIVM_H

#include <memory>

namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_CONVERTMTFUSIONTOMTIVM
#include "mtir/Conversion/Passes.h.inc"

namespace mtivm {
void populateMtFusionToMtIVMConversionPatterns(LLVMTypeConverter &converter,
                                             RewritePatternSet &patterns);
} // namespace mtivm

/// Creates a pass to convert the MtFusion dialect to the MtIVM dialect.
std::unique_ptr<Pass> createMtFusionToMtIVMConversionPass();

} // namespace mlir

#endif // MTIR_CONVERSION_MTFUSIONTOMTIVM_MTFUSIONTOMTIVM_H
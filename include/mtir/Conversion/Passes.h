//===- Passes.h - Conversion Pass Construction and Registration -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MTIR_CONVERSION_PASSES_H
#define MTIR_CONVERSION_PASSES_H

#include "mtir/Conversion/MtFusionToMtIVM/MtFusionToMtIVM.h"

#include "mlir/Pass/Pass.h"

#if MTIR_TORCH_CONVERSIONS_ENABLED
#include "mtir/Conversion/TorchToMtFusion/TorchToMtFusion.h"
#endif

namespace mtir {

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "mtir/Conversion/Passes.h.inc"

} // namespace mtir

#endif // MTIR_CONVERSION_PASSES_H
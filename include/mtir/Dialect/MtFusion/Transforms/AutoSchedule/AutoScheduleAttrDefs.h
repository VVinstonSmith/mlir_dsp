//===- AutoScheduleAttrDefs.h - Auto Schedule attributes --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines constant attributes used in Auto Schedule.
//
//===----------------------------------------------------------------------===//

#ifndef MTIR_DIALECT_MTFUSION_TRANSFORMS_AUTOSCHEDULE_AUTOSCHEDULEATTRDEFS_H
#define MTIR_DIALECT_MTFUSION_TRANSFORMS_AUTOSCHEDULE_AUTOSCHEDULEATTRDEFS_H

#include "llvm/ADT/StringRef.h"

// TODO: obtain ub size from platform config
static constexpr int64_t kUBMaxSizeInBits = 192 * 1024 * 8;
static constexpr int64_t kUBAlignSizeInBytes = 32;
static constexpr int64_t kNumBitsInByte = 8;

/// Name of the attribute used for targeting the transform dialect interpreter
/// at specific operations.
constexpr static llvm::StringLiteral kTransformDialectTagAttrName =
    "transform.target_tag";

constexpr static char kReductionOpIdxFormat[] = "__reduction{0}__";

constexpr static char kReductionFusableProducerFormat[] =
    "__reduction{0}_fusable_producer__";

constexpr static char kReturnValueFusableProducerFormat[] =
    "__result{0}_fusable_producer__";

constexpr static char kFuncArgIdxFormat[] = "__arg{0}__";

constexpr static llvm::StringLiteral kIntermediateProducerTagName =
    "__intermediate_producer__";

constexpr static llvm::StringLiteral kCacheReadTagName = "__cache_read__";

constexpr static llvm::StringLiteral kCacheWriteTagName = "__cache_write__";

constexpr static llvm::StringLiteral kTiledForAllTagName = "__tiled_forall__";

constexpr static llvm::StringLiteral kTiledForTagName = "__tiled_for__";

constexpr static llvm::StringLiteral kFusedLoopTagName = "__fused_loop__";

constexpr static llvm::StringLiteral kForallLoopTagName = "__forall__";

constexpr static llvm::StringLiteral kCoalescedLoopTagName =
    "__coalesced_loop__";

constexpr static llvm::StringLiteral kTileReductionPartialReductionOpTagName =
    "__partial_reduction_op__";

constexpr static llvm::StringLiteral kTileReductionFinalReductionOpTagName =
    "__final_reduction_op__";

constexpr static llvm::StringLiteral kTileReductionInitOpTagName =
    "__reduction_init_op__";

constexpr static llvm::StringLiteral kTileReductionLoopTagName =
    "__reduction_loop__";

#endif // MTIR_DIALECT_MTFUSION_TRANSFORMS_AUTOSCHEDULE_AUTOSCHEDULEATTRDEFS_H


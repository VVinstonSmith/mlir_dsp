//===- Util.cpp ---MtIR Dialect Uitls-------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mtir/Dialect/Utils/Util.h"

namespace mlir {

namespace utils {

std::optional<size_t> getShapeRank(const Type &type) {
    if(auto shapedType = type.dyn_cast<ShapedType>()){
        return shapedType.getRank() - shapedType.getNumDynamicDims();
    }
    return std::nullopt;
}

std::optional<size_t> getShapeRank(const Value &v) {
    if(auto shapedType = v.getType().dyn_cast<ShapedType>()){
        return shapedType.getRank() - shapedType.getNumDynamicDims();
    }
    return std::nullopt;
}

} // namespace utils

namespace mtfusion {
namespace util {

bool hasDynamicShapeOperand(Operation *op) {
    for(auto operand : op->getOperands()){
        if(auto shapedType = operand.getType().dyn_cast<ShapedType>()){
            if(!shapedType.hasStaticShape())
                return false;
        }
    }
    return true;
}

} // namespace util
} // namespace mtfusion

} // namespace mlir
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

/// Returns true if input type is a shaped type with known rank.
bool hasRank(const Type &type) {
  if(auto shapedType = dyn_cast<ShapedType>(type)) {
    return shapedType.hasRank();
  }
  return false;
}

/// Returns true if input type is shaped.
bool isShaped(const Type &type) {
  return type.isa<ShapedType>();
}

/// Returns true if value is scalar or zero rank tensor or one-size tensor
bool isScalarLike(Value value) {
    Type type = value.getType();
    if (type.isa<IntegerType, FloatType>()) {
        return true;
    }
    if (auto tensorType = type.dyn_cast<TensorType>()) {
        if (!tensorType.hasStaticShape()) {
            return false;
        }
        return llvm::all_of(tensorType.getShape(), [](int64_t size) { return size == 1; });
    }
    return false;
}

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
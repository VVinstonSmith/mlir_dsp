//===- Util.h ---MtIR Dialect Uitls-------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MTIR_DIALECT_UTILS_UTIL_H
#define MTIR_DIALECT_UTILS_UTIL_H

#include <utility>

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/TypeRange.h"
#include <numeric>
namespace mlir {

namespace utils {
constexpr const uint8_t kBitsToByte = 8;
namespace debugger {

// Type trait to check if T is an LLVM-style container
template <typename T, typename = void>
struct IsLLVMContainer : std::false_type {};

template <typename T>
struct IsLLVMContainer<T, std::void_t<decltype(std::declval<T>().begin()),
                                      decltype(std::declval<T>().end()),
                                      decltype(std::declval<T>().size())>>
    : std::true_type {};

// Type trait to check if T supports indexing
template <typename T, typename = void>
struct HasSubscript : std::false_type {};

template <typename T>
struct HasSubscript<T, std::void_t<decltype(std::declval<T>()[0])>>
    : std::true_type {};

template <typename T>
std::string to_string(const T &container, int indent = 0, bool useEndl = false);

template <typename T>
std::string toStrHelper(const T &value, int indent, bool useEndl) {
  if constexpr (IsLLVMContainer<T>::value) {
    return to_string(value, indent + 2, useEndl);
  } else {
    return std::to_string(value);
  }
}
template <typename T>
std::string to_string(const T &container, int indent, bool useEndl) {
  std::ostringstream oss;
  std::string indentation(indent, ' ');

  auto appendEl = [&](const auto &element, bool isLast) {
    if (useEndl)
      oss << indentation;
    oss << toStrHelper(element, indent, useEndl);
    if (!isLast)
      oss << ", ";
    if (useEndl)
      oss << "\n";
  };

  if (useEndl)
    oss << indentation;
  else
    oss << "[";

  if (!container.empty()) {
    if (useEndl)
      oss << "\n";
    if constexpr (HasSubscript<T>::value) {
      for (size_t i = 0; i < container.size(); ++i) {
        appendEl(container[i], i == container.size() - 1);
      }
    } else {
      auto it = container.begin();
      auto end = container.end();
      while (it != end) {
        appendEl(*it, std::next(it) == end);
        ++it;
      }
    }
    if (useEndl)
      oss << indentation;
  }
  oss << "]";
  return oss.str();
}

} // namespace debugger

// Currently dtype cast rules:
// (1-RINT  ) f32 -> f16/bf16/f32
// (2-NORMAL) f16/f32 -> f32
// (3-TRUNC ) float -> int
// (4-TRUNC ) int -> float
// (5-NORMAL) int -> int
// (6-NORMAL) others
template <typename T>
T selectRoundMode(Type inType, Type outType) {
  if (inType.isF32()) {
    if (outType.isF16() || outType.isBF16() || outType.isF32()) {
      return T::RINT;
    }
  }

  if (outType.isF32()) {
    if (inType.isF16() || inType.isBF16()) {
      return T::NORMAL;
    }
  }

  if (inType.isInteger(8) && outType.isF16()) {
    return T::NORMAL;
  }

  if (isa<mlir::FloatType>(inType) && outType.isInteger()) {
    return T::TRUNC;
  }

  if (inType.isInteger() && isa<mlir::FloatType>(outType)) {
    return T::TRUNC;
  }

  if (inType.isInteger() && outType.isInteger()) {
    return T::NORMAL;
  }
  llvm_unreachable("unsupported type cast.");
}

inline Type getMostElementType(
    SmallVector<Type> types,
    const std::function<bool(const Type &, const Type &)> &comparator) {
  llvm::sort(types, comparator);
  return types.front();
}

/// Return the type that has the smallest bits.
/// \note The input types must be an int or float type.
inline Type getSmallestElementType(SmallVector<Type> types) {
  return getMostElementType(
      std::move(types), [](const Type &lhs, const Type &rhs) -> bool {
        return lhs.getIntOrFloatBitWidth() < rhs.getIntOrFloatBitWidth();
      });
}

/// Return the type that has the largest bits.
/// \note The input types must be an int or float type.
inline Type getLargestElementType(SmallVector<Type> types) {
  return getMostElementType(
      std::move(types), [](const Type &lhs, const Type &rhs) -> bool {
        return lhs.getIntOrFloatBitWidth() > rhs.getIntOrFloatBitWidth();
      });
}
/// Return true if the input operation is `memref.alloc` or `memref.alloca`
bool isAllocLikeOp(Operation *op);

/// Return true if the input value is the SSA result value of `memref.alloc` or
/// `memref.alloca` op.
bool isAllocLikeOp(Value val);

/// Set buffer size to alloc like ops by constructing a new, static-shape
/// alloc. The new alloc is viewed to the original shape.
/// \note Assertion is raised if `op` is not `memref.alloc` or `memref.alloca`
memref::ViewOp createAllocWithSettingBufferSize(Operation *op,
                                                int64_t bufferSize,
                                                RewriterBase &opBuilder);

/// Returns true if input type is a shaped type with known rank.
bool hasRank(const Type &type);

/// Returns the shape rank if exist
std::optional<size_t> getShapeRank(const Type &type);
std::optional<size_t> getShapeRank(const Value &v);

using DimensionShape = SmallVector<int64_t>;
std::optional<std::pair<size_t, DimensionShape>>
getValueShapeInfo(const Value &v);

/// Returns true if input type is shaped.
bool isShaped(const Type &type);

/// Returns true if none of the value is dynamic.
/// \note This should only be applied to shapes/strides/offsets.
bool isFullyStatic(const SmallVector<int64_t> &values);

/// Returns shape of shaped type with known rank.
SmallVector<int64_t> getShape(const Type &type);

/// Get total size of a given array.
std::optional<int64_t> getStaticTotalSize(const ArrayRef<int64_t> &shapes);

SmallVector<int64_t>
getReassociationMapping(ArrayRef<ReassociationIndices> reassociation);

SmallVector<int64_t> getNewIndexing(ArrayRef<int64_t> oldIndexing,
                                    ArrayRef<int64_t> mapping);

void sortReassociation(MutableArrayRef<ReassociationIndices> reassociation);

SmallVector<int64_t> inversePermutation(ArrayRef<int64_t> perm);

/// Returns true if value is scalar or zero rank tensor or one-size tensor
bool isScalarLike(Value value);

// /// Return true if op is annotation mark op with attr `name`
// bool isAnnotationWithAttr(Operation *op, StringRef name);

// /// Search the users of value v to find first annotation op with attr `name`.
// std::optional<Operation *> getAnnotateOpWithAttr(Value v, StringRef name);

// /// Search the users of value v to find all the annotation ops with attr `name`.
// SmallVector<Operation *> getAllAnnotateOpsWithAttr(Value v, StringRef name);

// /// Search the users of each operand to find the annotation op with attr `name`.
// SmallVector<std::optional<Operation *>>
// getAnnotateOpWithAttrForEachOperand(const SmallVectorImpl<Value> &operands,
//                                     StringRef name);
} // namespace utils

// namespace hivm {
// namespace util {
// // TODO: platform information
// constexpr static unsigned int VL = 256;
// constexpr static unsigned int BL = VL / 8;
// const static int vectorBlockSizeBit = 256;
// const static int srcNumPerRepeatOfVBRCBIntrin = 8;

// constexpr static unsigned int INTR_BYTES_PER_BLOCK = 32;
// constexpr static unsigned int INTR_BYTES_PER_REPEAT = 256;

// /// Deduce Alignment information for DPS Op's init operand.
// ///
// /// If operand has memref semantic, we try to deduce the information from the
// /// memref type. Otherwise, we look for annotations on the tied result value. If
// /// there is conflicting annotations, a warning is produced.
// AlignKind deduceAlignmentForDPSInitOperand(OpOperand &operand);

// AlignKind deduceAlignmentForMemRefType(MemRefType vecType);

// mlir::Value tracebackMemref(mlir::Value memrefVal);

// /// Traceback `memrefVal` to its defining memref alloc if possible and return
// /// the MemRefType if it has static shape.
// std::optional<MemRefType> traceToGetStaticShapedType(mlir::Value memrefVal);

// std::optional<int64_t> traceToAllocMaxSize(mlir::Value memrefVal);

// enum class ReduceKind : uint32_t {
//   sum = 1,
//   prod = 2,
//   max = 3,
//   min = 4,
//   max_with_index = 5,
//   min_with_index = 6,
//   any = 7,
//   all = 8,
// };
// enum ReduceKind getReduceKind(linalg::ReduceOp reduceOp);

// /// Determine whether a value is signed or unsigned.
// bool shouldMapToUnsigned(IntegerType::SignednessSemantics val);

// /// Return the elementType as string for function name creation.
// std::string getTypeName(Location loc, Type type);

// /// Return the ConstantOp IntValue.
// FailureOr<std::string> stringfyConstantIntOpValue(Location loc, Value value);

// bool isTransposeWithLastAxis(ArrayRef<int64_t> permutation);

// int64_t getNumPerBlock(Type t);
// int64_t getNumPerRepeat(Type t);

// template <typename IRType, typename CType>
// bool isConst(TypedAttr v, CType t) {
//   if (isa<FloatAttr>(v)) {
//     auto srcTypeAttr = dyn_cast_or_null<FloatAttr>(v);
//     return srcTypeAttr.getValue() == APFloat(t);
//   }
//   if (isa<IntegerAttr>(v)) {
//     auto srcIntAttr = dyn_cast_or_null<IntegerAttr>(v);
//     auto intval = srcIntAttr.getInt();
//     return intval == t;
//   }
//   return false;
// }

// // Returns if the given source MemRef type is collapsible with the specified
// // reassociation indices. This function works as a strict extension based
// // on `memref::CollapseShapeOp::isGuaranteedCollapsible`, which has weak
// // constraints on the strides of trailing one-size dimensions.
// bool isGuaranteedCollapsibleStrictly(
//     MemRefType srcType, ArrayRef<ReassociationIndices> reassociation);

// /// Return the MemRefTypes
// SmallVector<MemRefType> getMemRefTypes(TypeRange types);

// /// Judge if all MemRefTypes has same rank value
// bool isAllSameRank(const SmallVectorImpl<MemRefType> &memrefTypes);

// /// Returns if the reassociations are identity that each indices group only
// /// contains a single dimension. e.g. `[[0], [1], [3]]` is indentity collapse.
// bool isIdentityCollapse(
//     const SmallVectorImpl<ReassociationIndices> &reassociations);

// /// Refine the reassociations into largest possible continuous parts,ensuring
// /// that all memrefTypes can be collapsed together.
// SmallVector<ReassociationIndices> getContinuousReassociation(
//     const SmallVectorImpl<MemRefType> &memrefTypes,
//     const SmallVectorImpl<ReassociationIndices> &reassociations);

// /// Refine the reassociations into continuous parts. Here reshape dim means
// /// reduce or broadcast dim which cannot be collapsed with non-reshape dim.
// SmallVector<ReassociationIndices>
// getContinuousReassociation(const SmallVectorImpl<MemRefType> &memrefTypes,
//                            ArrayRef<int64_t> reshapeDims = {},
//                            ArrayRef<int64_t> permutations = {});

// /// Combine reassociation index groups if the reassociation indices group is not
// /// transposed, and the shape of each memref type corresponding to each index in
// /// the reassociation indices group is 1
// SmallVector<ReassociationIndices> combineReassociationGroups(
//     const SmallVectorImpl<MemRefType> &memrefTypes,
//     const SmallVectorImpl<ReassociationIndices> &reassociations);

// template <typename TensorOrMemRefType,
//           typename = typename std::enable_if_t<
//               std::is_same_v<TensorOrMemRefType, TensorType> ||
//               std::is_same_v<TensorOrMemRefType, MemRefType>>>
// SmallVector<int> collectAlignUnits(ArrayRef<int32_t> alignDims,
//                                    ArrayRef<int32_t> alignBytes,
//                                    TensorOrMemRefType unalignedTy) {
//   int rank = unalignedTy.getRank();
//   int bitWidth = unalignedTy.getElementTypeBitWidth();
//   SmallVector<int> alignTargets(rank, 1);
//   assert(alignBytes.size() == alignDims.size());
//   for (int i = 0; i < alignDims.size(); ++i) {
//     int dim = alignDims[i];
//     assert(dim >= 0 && dim < rank);
//     int alignBits = alignBytes[i] * utils::kBitsToByte;
//     if (bitWidth % alignBits == 0) {
//       // If the alignment is smaller than type size, align to itself
//       continue;
//     }
//     assert(alignBits % bitWidth == 0 &&
//            "Alignment cannot satisfied by bitwidth");
//     alignTargets[dim] = std::lcm(alignBits / bitWidth, alignTargets[dim]);
//   }

//   int innerAlignedUnits = 1;
//   int shapeAccumulation = 1;
//   auto shapes = unalignedTy.getShape();
//   SmallVector<int> alignUnits(rank + 1, 1);
//   for (int dim = rank - 1; dim >= 0; --dim) {
//     // The alignment target forces the INNER dimension to get aligned
//     int newAlignedUnits = std::lcm(innerAlignedUnits, alignTargets[dim]);
//     if (shapeAccumulation % newAlignedUnits == 0) {
//       // already aligned
//       alignUnits[dim + 1] = 1;
//     } else {
//       alignUnits[dim + 1] = newAlignedUnits / innerAlignedUnits;
//     }
//     innerAlignedUnits = newAlignedUnits;
//     if (!ShapedType::isDynamic(shapes[dim])) {
//       shapeAccumulation =
//           shapeAccumulation * std::lcm(shapes[dim], alignUnits[dim + 1]);
//     }
//   }
//   // The outermost dimension needs no extra alignments
//   alignUnits[0] = 1;
//   return alignUnits;
// }

// } // namespace util
// } // namespace hivm
} // namespace mlir

namespace mlir {
namespace mtfusion {
namespace util {

bool hasDynamicShapeOperand(Operation *op);
bool isReachable(Operation *start, Operation *dest, DominanceInfo &dominators);

} // namespace util
} // namespace mtfusion
} // namespace mlir

#endif
//===- TilingUtils.h -- Utilities for Auto Schedule Tiling ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mtir/Dialect/MtFusion/IR/MtFusion.h"
#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/ValueHandle.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"

namespace mlir {
namespace mtfusion {

class KernelInfo;
class TilingInfo;

/// Get \c {mtfusion.tiling_data} unit attribute.
NamedAttribute getTilingDataAttr(OpBuilder &opBuilder);

/// Get \c {mtfusion.tiling_key} unit attribute.
NamedAttribute getTilingKeyAttr(OpBuilder &opBuilder);

//===----------------------------------------------------------------------===//
// Expr
//===----------------------------------------------------------------------===//

enum class ExprKind : uint8_t {
  // Regular expression, can represent a constant or the result of arithmetic
  // computation
  kRegular = 0,
  // A placeholder symbol for tensor dimension, either bound at runtime
  // (dynamic shape) or compile time (static shape).
  kDimSymbol
};

class ExprBuilder;

/// Base type for Expression.
///
/// Expression can be used to compute tiling values in host tiling functions.
/// Each expression is bound to a IR value that is constructed on-the-fly as
/// users expression computation logic using expressions.
///
/// The bound IR value is either
///   1) the result of `affine::ApplyOp` or
///   2) the result of `arith::CmpXXOp` or
///   3) the result of `tensor::DimOp`.
class Expr {
public:
  Expr() = default;
  virtual ~Expr() = default;
  explicit Expr(ExprKind kind) : kind_(kind){};
  explicit Expr(const Value &value, ExprKind kind, ExprBuilder *builder)
      : v_(value), kind_(kind), builder_(builder) {}

  // FIXME: Implement me
  Expr operator+(int64_t cst);
  Expr operator+(const Expr &other);
  Expr operator-(int64_t cst);
  Expr operator-(const Expr &other);
  Expr operator*(int64_t cst);
  Expr operator*(const Expr &other);
  Expr ceilDiv(uint64_t cst);
  Expr ceilDiv(const Expr &other);
  Expr floorDiv(uint64_t cst);
  Expr floorDiv(const Expr &other);
  Expr operator%(uint64_t cst);
  Expr operator%(const Expr &other);

  /// Returns the next integer that is greater than or equal to \p this and is a
  /// multiple of \p align. \p align must be non-zero.
  ///
  /// Examples:
  /// \code
  ///   alignTo(5, 8) = 8
  ///   alignTo(17, 8) = 24
  ///   alignTo(321, 255) = 510
  /// \endcode
  Expr alignTo(uint64_t align);
  Expr alignTo(const Expr &align);

  /// Returns the largest integer less than or equal to \p this and is a
  /// multiple of \p align. \p align must be non-zero.
  ///
  /// Examples:
  /// \code
  ///   alignTo(5, 8) = 0
  ///   alignTo(17, 8) = 16
  ///   alignTo(321, 255) = 255
  /// \endcode
  Expr alignDown(uint64_t align);
  Expr alignDown(const Expr &align);

  Expr operator>(int64_t cst);
  Expr operator>(const Expr &other);
  Expr operator>=(int64_t cst);
  Expr operator>=(const Expr &other);
  Expr operator<(int64_t cst);
  Expr operator<(const Expr &other);
  Expr operator<=(int64_t cst);
  Expr operator<=(const Expr &other);
  Expr operator==(int64_t cst);
  Expr operator==(const Expr &other);
  Expr operator!=(int64_t cst);
  Expr operator!=(const Expr &other);

  /// Get the underlying Value.
  Value getMaterializedValue() { return v_; }

  MLIRContext *getContext() const { return v_.getContext(); }

  /// Return the classification for this type.
  ExprKind getExprKind() const { return kind_; }

  static bool classof(const Expr *) { return true; }

  /// Get the underlying builder with location.
  ExprBuilder &getBuilder() { return *builder_; }

private:
  /// Underlying IR Value.
  Value v_;
  /// Expression kind.
  ExprKind kind_{ExprKind::kRegular};
  /// OpBuilder
  ExprBuilder *builder_{nullptr};
};

/// Expression representing a tensor's dimension size.
class DimSymbol : public Expr {
public:
  DimSymbol() : Expr(ExprKind::kDimSymbol) {}
  virtual ~DimSymbol() = default;
  explicit DimSymbol(const Value &value, ExprBuilder *builder)
      : Expr(value, ExprKind::kDimSymbol, builder) {}
  static bool classof(const Expr *e) {
    return e->getExprKind() == ExprKind::kDimSymbol;
  }
};

/// Commonly used operations for Expr.
Expr max(Expr lhs, int64_t rhs);
Expr min(Expr lhs, Expr rhs);
Expr select(Expr condition, Expr trueValue, Expr falseValue);

//===----------------------------------------------------------------------===//
// ExprBuilder
//===----------------------------------------------------------------------===//
class KernelInfo;

/// Expression Builder class.
///
/// Users can only create constant-value `Expr` or `DimSymbol` using this class.
/// The base auto scheduler is responsible for creating an instance of
/// `ExprBuilder` and setting the insertion point into the host tiling function.
class ExprBuilder : public OpBuilder {
public:
  explicit ExprBuilder(MLIRContext *ctx) : OpBuilder(ctx) {}
  explicit ExprBuilder(TilingInfo *info, KernelInfo *kernelInfo,
                       MLIRContext *ctx)
      : OpBuilder(ctx), tilingInfo_(info), kernelInfo_(kernelInfo) {}

  /// Create an `Expr` holding a int64_t constant value.
  Expr createConstExpr(int64_t cst);

  /// Create a `DimSymbol` that represents the `dimIdx`-th dimension size of
  /// the interested value of the current to-be-scheduled kernel.
  /// If the `tensorIdx`-th arg is not reshaped in origin kernel, directly
  /// create the dim symbol from the `tensorIdx`-th arg. Otherwise, create the
  /// dim symbol from the result value reshaped from the `tensorIdx`-th arg.
  Expr createDimSymbolExpr(size_t tensorIdx, size_t dimIdx);

  /// Create `DimSymbol`s that represents the dimension size of
  /// the interested value of the current to-be-scheduled kernel from `startDim`
  /// to `endDim` (with step = 1).
  SmallVector<Expr> createDimSymbolExprs(size_t tensorIdx, size_t startDim,
                                         size_t endDim);

private:
  /// Create a `DimSymbol` that represent the `dimIdx`-th dimension size of
  /// the `tensorValue` of the current to-be-scheduled kernel.
  /// The `tensorValue` can be tensor argument or tensor value reshaped from
  /// tensor argument.
  Expr createDimSymbolExpr(Value tensorValue, size_t dimIdx);

private:
  TilingInfo *tilingInfo_{nullptr};

  KernelInfo *kernelInfo_{nullptr};
};

//===----------------------------------------------------------------------===//
// TilingData
//===----------------------------------------------------------------------===//

/// The `TilingData` class represents a binding between host tiling data, device
/// kernel argument, and `ValueHandles` used by auto-scheduler's schedule
/// operations.
///
/// For example, consider the following IR :
/// \code
/// func.func private @host_tiling(...) -> (i64, i64)
/// func.func private @device_kernel(..., %tiling_data0 : i64, %tiling_data1:
/// i64)
/// \endcode
///
/// During schedule, if one which to use a tiling data's value to perform
/// scheduling, he/she can get a "handle" that points to device kernel argument
/// tied to that tiling data.
///
/// For instance, the following code snippet:
/// \code
///   TilingData *ubTileSize = tilingInfo->getTilingData(0);
///   ValueHandle* ubTilingDataHandle = getTilingDataHandle(*ubTileSize, ...);
///   tileUsingFor(targetOp, ubTilingDataHandle, ...);
/// \endcode
///
/// will generate the following schedule sequence:
/// \code
/// %arg_handle = transform.func.get_func_argument %arg0[N] ...
/// transform.structured.tile_using_for %target_op[%arg_handle] ...
/// \endcode
///
/// which finally produces the scheduled kernel:
/// \code
/// func.func @device_kernel(..., %tiling_data0 : i64, %tiling_data1: i64) {
///   scf.for %arg7 = %c0 to %4 step %tiling_data0 iter_args(...)
///   ...
/// }
/// \endcode
///
/// The underlying tiling data storage is either an `Expr` or a constant value
/// of `int64_t` type.
struct TilingData {
public:
  using TilingDataTy = std::variant<std::unique_ptr<Expr>, int64_t>;

  TilingData() = default;
  explicit TilingData(Expr &&data, Type t)
      : data_(TilingDataTy(std::make_unique<Expr>(data))), t_(t) {}

  /// Returns whether the tiling data is constant.
  bool isConst() const { return std::holds_alternative<int64_t>(data_); }

  /// Get `Expr` corresponding to the tiling data. Raise exception if the data
  /// is constantized.
  Expr *getExpr() const;

  /// Get constantized value of the tiling data. Rase exception if the data
  /// is not constantized.
  int64_t getConst() const;

  /// Query the tiling data type.
  Type getType() { return t_; }

  /// Getters and setters for the value handle pointer.
  ValueHandle *getHandle() { return vh_; }
  ValueHandle *getHandle() const { return vh_; }
  void setHandle(ValueHandle *vh) { vh_ = vh; }

  /// Set tiling data to expression or to constant value.
  void setData(int64_t newData) { data_ = TilingDataTy(newData); }
  void setData(Expr &&newData) {
    data_ = TilingDataTy(std::make_unique<Expr>(newData));
  }

  /// Getters and setters for the tiling data's position index within kernel
  /// function's input argument.s
  size_t getPos() const { return pos_; }
  void setPos(size_t pos) { pos_ = pos; }

private:
  /// Tiling data storage.
  TilingDataTy data_;
  /// Type of the tiling data.
  Type t_;
  /// Position within the to-be-scheduled function.
  size_t pos_;
  /// Bound value handle pointer using during scheduling.
  ValueHandle *vh_{nullptr};
};

/// Tiling Key is a compile-time constant value of type `int64_t`. It should be
/// a unique identifier of a tiling case. The exact meaning of each key is
/// determined by the scheduler.
using TilingKey = int64_t;

//===----------------------------------------------------------------------===//
// TilingCases
//===----------------------------------------------------------------------===//

/// Tiling Cases are a collection of unique Tiling Keys. Each key corresponds
/// to a schedule implementation.
using TilingCases = SetVector<TilingKey>;

//===----------------------------------------------------------------------===//
// TilingStruct
//===----------------------------------------------------------------------===//

using TilingDataPtr = std::unique_ptr<TilingData>;

/// Tiling Struct is a series of Tiling Data. The order of tiling data
/// corresponds to the order of values returned by the host tiling function.
using TilingStruct = SmallVector<TilingDataPtr>;

//===----------------------------------------------------------------------===//
// TilingComputeFn
//===----------------------------------------------------------------------===//

using TilingFnResultTy = std::pair<TilingCases, TilingStruct>;
/// Tiling computation function is a lambda function that computes the tiling
/// data using information from kernel information.
/// It returns `TilingCases` and `TilingStruct`.
using TilingComputeFn =
    llvm::function_ref<TilingFnResultTy(KernelInfo *, ExprBuilder *)>;

//===----------------------------------------------------------------------===//
// TilingInfo
//===----------------------------------------------------------------------===//

/// Data structure for holding tiling information.
class TilingInfo {
public:
  using tiling_data_iterator = SmallVectorImpl<TilingDataPtr>::iterator;

  TilingInfo() = default;
  virtual ~TilingInfo() = default;
  TilingInfo &operator=(TilingInfo const &) = delete;

  explicit TilingInfo(size_t tilingSize)
      : struct_(SmallVector<TilingDataPtr>(tilingSize)) {}

  TilingInfo(TilingInfo &&other) {
    this->struct_ = std::move(other.struct_);
    this->caseKeys_ = std::move(other.caseKeys_);
    this->blockDim_ = other.blockDim_;
    this->hostTilingFunc_ = other.hostTilingFunc_;
    this->tilingComputeFn_ = other.tilingComputeFn_;
  }

  tiling_data_iterator tilingDataBegin() { return struct_.begin(); }
  tiling_data_iterator tilingDataEnd() { return struct_.end(); }

  /// Return whether tiling struct is empty.
  bool empty() { return size() == 0; }

  /// Get the number of tiling data.
  size_t size() { return struct_.size(); }
  size_t size() const { return struct_.size(); }

  /// Get pointers to all tiling data.
  SmallVector<TilingData *> getTilingStruct();

  /// Get pointer to tiling key.
  TilingData *getTilingKey();

  /// Get pointer to the `idx`-th tiling data.
  TilingData *getTilingData(unsigned idx);
  TilingData *getTilingData(unsigned idx) const;

  /// Get all tiling cases.
  TilingCases getTilingCases() { return caseKeys_; }

  /// Getter and setter for block dim.
  uint32_t getBlockDim() { return blockDim_; }
  void setBlockDim(uint32_t blockDim) { blockDim_ = blockDim; }

  /// Getter and setter to host tiling function.
  func::FuncOp getHostTilingFunc() { return hostTilingFunc_; }
  void setHostTilingFunc(func::FuncOp f) { hostTilingFunc_ = f; }

  /// Get the `idx`-th function argument of host tiling func.
  BlockArgument getHostTilingFuncArg(size_t idx);

  /// Remove all tiling keys from tiling cases except for `keepKey`.
  void pruneTilingExcept(int64_t keepKey);

  /// Try to simply host tiling function.
  LogicalResult trySimplifyTilingFunc();

  /// Evaluate tiling computation function to generate IR values.
  SmallVector<Value> evaluateTilingComputation(TilingComputeFn fn,
                                               KernelInfo *kernelInfo,
                                               ExprBuilder *builder);

  /// Get mapping from tiling key to device kernel function.
  llvm::DenseMap<TilingKey, func::FuncOp> getTilingKey2KernelMap();

  /// Record assocation between tiling key and to-be-scheduled kernel function.
  void recordKernelFunc(TilingKey key, func::FuncOp f);

private:
  /// Block dimension.
  uint32_t blockDim_{0};
  /// Reference to tiling computation lambda function.
  TilingComputeFn tilingComputeFn_;
  /// Tiling data.
  TilingStruct struct_;
  /// Tiling cases.
  TilingCases caseKeys_;
  /// Pointer to host tiling function.
  func::FuncOp hostTilingFunc_{nullptr};
  /// Tiling key to scheduled device kernel.
  llvm::DenseMap<TilingKey, func::FuncOp> tilingKey2Kernel_;
};

namespace tiling {

/// Construct an array of `Expr`s that represents the accumulated number of
/// elements up to a certain dimension.
///
/// Input:
/// [dim_0, dim_1, ... dim_{N-1}]
///
/// Output:
/// [dim_0,
///  dim_0 * dim_1, ...,
///  dim_0 * ... * dim_{N-2} * dim_{N-1}]
SmallVector<Expr> getAccumulatedDims(SmallVector<Expr> dims);
} // namespace tiling

} // namespace mtfusion
} // namespace mlir
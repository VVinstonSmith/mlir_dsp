//===- MtFusionOps.cpp - Implementation of MtFusion Dialect Ops ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mtir/Dialect/MtFusion/IR/MtFusion.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

using namespace mlir;
using namespace mlir::mtfusion;

//===----------------------------------------------------------------------===//
// Support for named MtFusion ops defined in ods-gen.
//===----------------------------------------------------------------------===//

using RegionBuilderFn = llvm::function_ref<void(ImplicitLocOpBuilder &, Block &,
                                                ArrayRef<NamedAttribute>)>;

/// Fills the region of a structured operation using the provided
/// `regionBuilder`. The method is used by both named structured ops created by
/// ods-gen and by manually defined C++ ops. It is called by both builders and
/// parsers and creates a block with arguments corresponding to the elemental
/// types of `inputTypes` and `outputTypes`. All output types are asserted to be
/// ShapedType.
static void fillStructuredOpRegion(OpBuilder &opBuilder, Region &region,
                                   TypeRange inputTypes, TypeRange outputTypes,
                                   ArrayRef<NamedAttribute> attrs,
                                   RegionBuilderFn regionBuilder) {
  assert(llvm::all_of(outputTypes,
                      [](Type t) { return llvm::isa<ShapedType>(t); }));

  SmallVector<Type, 8> argTypes;
  SmallVector<Location, 8> argLocs;
  for (auto containers : {inputTypes, outputTypes}) {
    for (auto t : containers) {
      argTypes.push_back(
          isa<MemRefType, RankedTensorType>(t) ? getElementTypeOrSelf(t) : t);

      // TODO: Pass in a proper location here.
      argLocs.push_back(opBuilder.getUnknownLoc());
    }
  }

  // RAII.
  OpBuilder::InsertionGuard guard(opBuilder);
  Block *body =
      opBuilder.createBlock(&region, /*insertPt=*/{}, argTypes, argLocs);

  opBuilder.setInsertionPointToStart(body);
  ImplicitLocOpBuilder b(opBuilder.getUnknownLoc(), opBuilder);
  regionBuilder(b, *body, attrs);

  // indexing_maps is an auto-generated method.

  // iterator_types is an auto-generated method.
}

/// Creates a structured operation given `inputs`, `outputs`, and `attributes`.
/// The result types are derived automatically if `resultTensorTypes` is none.
/// The body of the operation is filled using `regionBuilder`. All ods-gen
/// created structured operations use the method to implement their builders.
static void buildStructuredOp(OpBuilder &b, OperationState &state,
                              std::optional<TypeRange> resultTensorTypes,
                              ValueRange inputs, ValueRange outputs,
                              ArrayRef<NamedAttribute> attributes,
                              RegionBuilderFn regionBuilder) {
  // Derive the result types if needed.
  SmallVector<Type> derivedResultTypes =
      resultTensorTypes.value_or(TypeRange());
  if (!resultTensorTypes)
    copy_if(outputs.getTypes(), std::back_inserter(derivedResultTypes),
            [](Type type) { return llvm::isa<RankedTensorType>(type); });

  state.addOperands(inputs);
  state.addOperands(outputs);
  state.addTypes(derivedResultTypes);
  state.addAttributes(attributes);
  state.addAttribute(
      "operandSegmentSizes",
      b.getDenseI32ArrayAttr({static_cast<int32_t>(inputs.size()),
                              static_cast<int32_t>(outputs.size())}));

  // Create and fill the region of the structured operation.
  Region &region = *state.addRegion();
  fillStructuredOpRegion(b, region, TypeRange(inputs), TypeRange(outputs),
                         state.attributes.getAttrs(), regionBuilder);
}

/// Common parsing used for both named structured ops created by ods-gen and by
/// manually defined C++ ops. Does not handle regions.
static ParseResult
parseCommonStructuredOpParts(OpAsmParser &parser, OperationState &result,
                             SmallVectorImpl<Type> &inputTypes,
                             SmallVectorImpl<Type> &outputTypes,
                             bool addOperandSegmentSizes = true) {
  SMLoc attrsLoc, inputsOperandsLoc, outputsOperandsLoc;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> inputsOperands,
      outputsOperands;

  if (succeeded(parser.parseOptionalLess())) {
    if (parser.parseAttribute(result.propertiesAttr) || parser.parseGreater())
      return failure();
  }
  attrsLoc = parser.getCurrentLocation();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (succeeded(parser.parseOptionalKeyword("ins"))) {
    if (parser.parseLParen())
      return failure();

    inputsOperandsLoc = parser.getCurrentLocation();
    if (parser.parseOperandList(inputsOperands) ||
        parser.parseColonTypeList(inputTypes) || parser.parseRParen())
      return failure();
  }

  if (succeeded(parser.parseOptionalKeyword("outs"))) {
    outputsOperandsLoc = parser.getCurrentLocation();
    if (parser.parseLParen() || parser.parseOperandList(outputsOperands) ||
        parser.parseColonTypeList(outputTypes) || parser.parseRParen())
      return failure();
  }

  if (parser.resolveOperands(inputsOperands, inputTypes, inputsOperandsLoc,
                             result.operands) ||
      parser.resolveOperands(outputsOperands, outputTypes, outputsOperandsLoc,
                             result.operands))
    return failure();

  if (addOperandSegmentSizes) {
    // This is a bit complex because we're trying to be backward compatible with
    // operation syntax that mix the inherent attributes and the discardable
    // ones in the same dictionary. If the properties are used, we append the
    // operandSegmentSizes there directly. Otherwise we append it to the
    // discardable attributes dictionary where it is handled by the generic
    // Operation::create(...) method.
    if (result.propertiesAttr) {
      NamedAttrList attrs = llvm::cast<DictionaryAttr>(result.propertiesAttr);
      attrs.append("operandSegmentSizes",
                   parser.getBuilder().getDenseI32ArrayAttr(
                       {static_cast<int32_t>(inputsOperands.size()),
                        static_cast<int32_t>(outputsOperands.size())}));
      result.propertiesAttr = attrs.getDictionary(parser.getContext());
    } else {
      result.addAttribute("operandSegmentSizes",
                          parser.getBuilder().getDenseI32ArrayAttr(
                              {static_cast<int32_t>(inputsOperands.size()),
                               static_cast<int32_t>(outputsOperands.size())}));
    }
  }
  if (!result.propertiesAttr) {
    std::optional<RegisteredOperationName> info =
        result.name.getRegisteredInfo();
    if (info) {
      if (failed(info->verifyInherentAttrs(result.attributes, [&]() {
            return parser.emitError(attrsLoc)
                   << "'" << result.name.getStringRef() << "' op ";
          })))
        return failure();
    }
  }
  return success();
}

static void printCommonStructuredOpParts(OpAsmPrinter &p, ValueRange inputs,
                                         ValueRange outputs) {
  if (!inputs.empty())
    p << " ins(" << inputs << " : " << inputs.getTypes() << ")";
  if (!outputs.empty())
    p << " outs(" << outputs << " : " << outputs.getTypes() << ")";
}

//===----------------------------------------------------------------------===//
// Specific parsing and printing for named structured ops created by ods-gen.
//===----------------------------------------------------------------------===//

static ParseResult parseNamedStructuredOpRegion(
    OpAsmParser &parser, Region &region, unsigned numRegionArgs,
    TypeRange inputTypes, TypeRange outputTypes, ArrayRef<NamedAttribute> attrs,
    RegionBuilderFn regionBuilder) {
  if (numRegionArgs != inputTypes.size() + outputTypes.size()) {
    return parser.emitError(
        parser.getCurrentLocation(),
        llvm::formatv("[parseNamedStructuredOpRegion] ods-gen generated "
                      "region expects {0} args, got {1}",
                      numRegionArgs, inputTypes.size() + outputTypes.size()));
  }

  OpBuilder opBuilder(parser.getContext());
  fillStructuredOpRegion(opBuilder, region, inputTypes, outputTypes, attrs,
                         regionBuilder);
  return success();
}

static ParseResult
parseNamedStructuredOpResults(OpAsmParser &parser,
                              SmallVectorImpl<Type> &resultTypes) {
  if (parser.parseOptionalArrowTypeList(resultTypes))
    return failure();
  return success();
}

static ParseResult parseNamedStructuredOp(OpAsmParser &parser,
                                          OperationState &result,
                                          unsigned numRegionArgs,
                                          RegionBuilderFn regionBuilder) {
  // TODO: Enable when ods-gen supports captures.
  SmallVector<Type, 1> inputTypes, outputTypes;
  if (parseCommonStructuredOpParts(parser, result, inputTypes, outputTypes))
    return failure();

  // TODO: consider merging results parsing into region parsing.
  // Need to wait for declarative assembly resolution to decide.
  SmallVector<Type, 1> outputTensorsTypes;
  if (parseNamedStructuredOpResults(parser, outputTensorsTypes))
    return failure();
  result.addTypes(outputTensorsTypes);

  std::unique_ptr<Region> region = std::make_unique<Region>();
  if (parseNamedStructuredOpRegion(parser, *region, numRegionArgs, inputTypes,
                                   outputTypes, result.attributes.getAttrs(),
                                   regionBuilder))
    return failure();
  result.addRegion(std::move(region));

  return success();
}

static void printNamedStructuredOpResults(OpAsmPrinter &p,
                                          TypeRange resultTypes) {
  if (resultTypes.empty())
    return;
  p.printOptionalArrowTypeList(resultTypes);
}

static void printNamedStructuredOp(OpAsmPrinter &p, Operation *op,
                                   ValueRange inputs, ValueRange outputs) {
  p.printOptionalAttrDict(
      op->getAttrs(),
      /*elidedAttrs=*/{"operandSegmentSizes",
                       // See generated code in
                       // MtFusionNamedStructuredOps.yamlgen.cpp.inc
                       "mtfusion.memoized_indexing_maps"});

  // Printing is shared with generic ops, except for the region and
  // attributes.
  printCommonStructuredOpParts(p, inputs, outputs);

  // Results printing.
  printNamedStructuredOpResults(p, op->getResultTypes());

  // Region is elided.
}

//===----------------------------------------------------------------------===//
// Region builder helper.
// TODO: Move this to a utility library.
// The public methods on this class are referenced directly from generated code.
// Helper build the unary, binary, and type conversion functions defined by the
// DSL. See MtFusionNamedStructuredOps.yamlgen.cpp.inc for the code that uses
// this class.
//
// Implementations of the math functions must be polymorphic over numeric types,
// internally performing necessary casts. If the function application makes no
// sense, then the only recourse is to assert and return nullptr. This can be
// extended later if it becomes possible to fail construction of the region. The
// invariant should be enforced at a higher level.
//
// TODO: These helpers are currently type polymorphic over the class of integer
// and floating point types, but they will not internally cast within bit
// widths of a class (mixed precision such as i8->i32) or across classes
// (i.e. mixed float and integer). Many such combinations are ambiguous or need
// to be handled with care and work is being considered to extend the op
// language to make such cases explicit. In the mean-time, violating this will
// fail verification, which is deemed acceptable.
//===----------------------------------------------------------------------===//

namespace {

class RegionBuilderHelper {
public:
  RegionBuilderHelper(MLIRContext *context, Block &block)
      : context(context), block(block) {}

  // Build the unary functions defined by OpDSL.
  Value buildUnaryFn(UnaryFn unaryFn, Value arg) {
    OpBuilder builder = getBuilder();
    switch (unaryFn) {
    case UnaryFn::sqrt:
      return builder.create<math::SqrtOp>(arg.getLoc(), arg);
    case UnaryFn::rsqrt:
      return builder.create<math::RsqrtOp>(arg.getLoc(), arg);
    case UnaryFn::tanh:
      return builder.create<math::TanhOp>(arg.getLoc(), arg);
    case UnaryFn::sin:
      return builder.create<math::SinOp>(arg.getLoc(), arg);
    case UnaryFn::cos:
      return builder.create<math::CosOp>(arg.getLoc(), arg);
    case UnaryFn::absi:
      return builder.create<math::AbsIOp>(arg.getLoc(), arg);
    case UnaryFn::erf:
      return builder.create<math::ErfOp>(arg.getLoc(), arg);
    case UnaryFn::log2:
      return builder.create<math::Log2Op>(arg.getLoc(), arg);
    case UnaryFn::log10:
      return builder.create<math::Log10Op>(arg.getLoc(), arg);
    case UnaryFn::log1p:
      return builder.create<math::Log1pOp>(arg.getLoc(), arg);
    case UnaryFn::exp2:
      return builder.create<math::Exp2Op>(arg.getLoc(), arg);
    case UnaryFn::expm1:
      return builder.create<math::ExpM1Op>(arg.getLoc(), arg);
    case UnaryFn::relu:
      if (isFloatingPoint(arg)) {
        Type type = arg.getType();
        Value zero = builder.create<arith::ConstantOp>(
            arg.getLoc(), type, builder.getFloatAttr(type, 0.0));
        return builder.create<arith::MaximumFOp>(arg.getLoc(), zero, arg);
      } else if (isInteger(arg)) {
        Type type = arg.getType();
        Value zero = builder.create<arith::ConstantOp>(
            arg.getLoc(), type, builder.getIntegerAttr(type, 0));
        return builder.create<arith::MaxSIOp>(arg.getLoc(), zero, arg);
      } else {
        llvm_unreachable("unsupported type for relu");
      }
    case UnaryFn::rec:
      if (isFloatingPoint(arg)) {
        Type type = arg.getType();
        Value one = builder.create<arith::ConstantOp>(
            arg.getLoc(), type, builder.getFloatAttr(type, 1.0));
        return builder.create<arith::DivFOp>(arg.getLoc(), one, arg);
      } else if (isInteger(arg)) {
        Type type = arg.getType();
        Value one = builder.create<arith::ConstantOp>(
            arg.getLoc(), type, builder.getIntegerAttr(type, 1));
        return builder.create<arith::DivSIOp>(arg.getLoc(), one, arg);
      } else {
        llvm_unreachable("unsupported type for reciprocal");
      }
    case UnaryFn::vnot:
      if (isInteger(arg)) {
        Type type = arg.getType();
        Value zero = builder.create<arith::ConstantOp>(
            arg.getLoc(), type, builder.getIntegerAttr(type, 0));
        return builder.create<arith::XOrIOp>(arg.getLoc(), zero, arg);
      } else {
        llvm_unreachable("unsupported type for not");
      }
    }
    llvm_unreachable("unsupported unary function");
  }

  // Build the binary functions defined by OpDSL.
  Value buildBinaryFn(BinaryFn binaryFn, Value arg0, Value arg1) {
    bool allComplex = isComplex(arg0) && isComplex(arg1);
    bool allFloatingPoint = isFloatingPoint(arg0) && isFloatingPoint(arg1);
    bool allInteger = isInteger(arg0) && isInteger(arg1);
    bool allBool = allInteger && arg0.getType().getIntOrFloatBitWidth() == 1 &&
                   arg1.getType().getIntOrFloatBitWidth() == 1;
    if (!allComplex && !allFloatingPoint && !allInteger)
      llvm_unreachable("unsupported non numeric type");
    OpBuilder builder = getBuilder();
    switch (binaryFn) {
    case BinaryFn::vor:
      if (allInteger)
        return builder.create<arith::OrIOp>(arg0.getLoc(), arg0, arg1);
      else
        llvm_unreachable("unsupported type for vor");
    case BinaryFn::vxor:
      if (allInteger)
        return builder.create<arith::XOrIOp>(arg0.getLoc(), arg0, arg1);
      else
        llvm_unreachable("unsupported type for vxor");
    case BinaryFn::vand:
      if (allInteger)
        return builder.create<arith::AndIOp>(arg0.getLoc(), arg0, arg1);
      else
        llvm_unreachable("unsupported type for vand");
    case BinaryFn::minf:
      if (allFloatingPoint)
        return builder.create<arith::MinNumFOp>(arg0.getLoc(), arg0, arg1);
      else
        llvm_unreachable("unsupported type for vmin");
    case BinaryFn::maxf:
      if (allFloatingPoint)
        return builder.create<arith::MaxNumFOp>(arg0.getLoc(), arg0, arg1);
      else
        llvm_unreachable("unsupported type for vmax");
    case BinaryFn::powf:
      if (allFloatingPoint)
        return builder.create<math::PowFOp>(arg0.getLoc(), arg0, arg1);
      else
        llvm_unreachable("unsupported type for vpow");
    case BinaryFn::mod:
      if (allInteger)
        return builder.create<arith::RemSIOp>(arg0.getLoc(), arg0, arg1);
      else
        llvm_unreachable("unsupported type for mod");
    }
    llvm_unreachable("unsupported binary function");
  }

  // Build the compare functions defined by OpDSL.
  Value buildCompareFn(CompareFn compareFn, Value arg0, Value arg1) {
    bool allComplex = isComplex(arg0) && isComplex(arg1);
    bool allFloatingPoint = isFloatingPoint(arg0) && isFloatingPoint(arg1);
    bool allInteger = isInteger(arg0) && isInteger(arg1);
    bool allBool = allInteger && arg0.getType().getIntOrFloatBitWidth() == 1 &&
                   arg1.getType().getIntOrFloatBitWidth() == 1;
    if (!allComplex && !allFloatingPoint && !allInteger)
      llvm_unreachable("unsupported non numeric type");
    OpBuilder builder = getBuilder();
    switch (compareFn) {
    case CompareFn::veq:
      if (allInteger)
        return builder.create<arith::CmpIOp>(
            arg0.getLoc(), arith::CmpIPredicate::eq, arg0, arg1);
      else if (allFloatingPoint)
        return builder.create<arith::CmpFOp>(
            arg0.getLoc(), arith::CmpFPredicate::OEQ, arg0, arg1);
      else
        llvm_unreachable("unsupported type for veq");
    case CompareFn::vne:
      if (allInteger)
        return builder.create<arith::CmpIOp>(
            arg0.getLoc(), arith::CmpIPredicate::ne, arg0, arg1);
      else if (allFloatingPoint)
        return builder.create<arith::CmpFOp>(
            arg0.getLoc(), arith::CmpFPredicate::UNE, arg0, arg1);
      else
        llvm_unreachable("unsupported type for vne");
    case CompareFn::vle:
      if (allInteger)
        return builder.create<arith::CmpIOp>(
            arg0.getLoc(), arith::CmpIPredicate::sle, arg0, arg1);
      if (allFloatingPoint)
        return builder.create<arith::CmpFOp>(
            arg0.getLoc(), arith::CmpFPredicate::OLE, arg0, arg1);
      else
        llvm_unreachable("unsupported type for vle");
    case CompareFn::vlt:
      if (allInteger)
        return builder.create<arith::CmpIOp>(
            arg0.getLoc(), arith::CmpIPredicate::slt, arg0, arg1);
      if (allFloatingPoint)
        return builder.create<arith::CmpFOp>(
            arg0.getLoc(), arith::CmpFPredicate::OLT, arg0, arg1);
      else
        llvm_unreachable("unsupported type for vlt");
    case CompareFn::vge:
      if (allInteger)
        return builder.create<arith::CmpIOp>(
            arg0.getLoc(), arith::CmpIPredicate::sge, arg0, arg1);
      if (allFloatingPoint)
        return builder.create<arith::CmpFOp>(
            arg0.getLoc(), arith::CmpFPredicate::OGE, arg0, arg1);
      else
        llvm_unreachable("unsupported type for vge");
    case CompareFn::vgt:
      if (allInteger)
        return builder.create<arith::CmpIOp>(
            arg0.getLoc(), arith::CmpIPredicate::sgt, arg0, arg1);
      if (allFloatingPoint)
        return builder.create<arith::CmpFOp>(
            arg0.getLoc(), arith::CmpFPredicate::OGT, arg0, arg1);
      else
        llvm_unreachable("unsupported type for vgt");
    }
    llvm_unreachable("unsupported binary function");
  }

  // Build the Ternary functions defined by OpDSL.
  Value buildTernaryFn(TernaryFn ternaryFn, Value arg0, Value arg1,
                       Value arg2) {
    bool allComplex = isComplex(arg1) && isComplex(arg2);
    bool allFloatingPoint = isFloatingPoint(arg1) && isFloatingPoint(arg2);
    bool allInteger = isInteger(arg1) && isInteger(arg2);
    bool allBool = allInteger && arg1.getType().getIntOrFloatBitWidth() == 1 &&
                   arg2.getType().getIntOrFloatBitWidth() == 1;
    if (!allComplex && !allFloatingPoint && !allInteger)
      llvm_unreachable("unsupported non numeric type");
    OpBuilder builder = getBuilder();
    switch (ternaryFn) {
    case TernaryFn::select:
      if (allInteger || allFloatingPoint)
        return builder.create<arith::SelectOp>(arg0.getLoc(), arg0, arg1, arg2);
      else
        llvm_unreachable("unsupported type for select");
    }
    llvm_unreachable("unsupported select function");
  }

  // Build the type functions defined by OpDSL.
  Value buildTypeFn(TypeFn typeFn, Type toType, Value operand) {
    switch (typeFn) {
    case TypeFn::cast_signed:
      return cast(toType, operand, false);
    case TypeFn::cast_unsigned:
      return cast(toType, operand, true);
    }
    llvm_unreachable("unsupported type conversion function");
  }

  // Build the type functions defined by OpDSL.
  Value buildRoundMode(RoundMode round, Type toType, Value operand) {
    bool isUnsignedCast = false;
    return cast(toType, operand, isUnsignedCast);
  }

  void yieldOutputs(ValueRange values) {
    OpBuilder builder = getBuilder();
    Location loc = builder.getUnknownLoc();
    builder.create<linalg::YieldOp>(loc, values);
  }

  Value constant(const std::string &value) {
    OpBuilder builder = getBuilder();
    Location loc = builder.getUnknownLoc();
    Attribute valueAttr = parseAttribute(value, builder.getContext());
    return builder.create<arith::ConstantOp>(loc, ::cast<TypedAttr>(valueAttr));
  }

  Value index(int64_t dim) {
    OpBuilder builder = getBuilder();
    return builder.create<linalg::IndexOp>(builder.getUnknownLoc(), dim);
  }

  Type getIntegerType(unsigned width) {
    return IntegerType::get(context, width);
  }

  Type getFloat32Type() { return Float32Type::get(context); }
  Type getFloat64Type() { return Float64Type::get(context); }

private:
  // Generates operations to cast the given operand to a specified type.
  // If the cast cannot be performed, a warning will be issued and the
  // operand returned as-is (which will presumably yield a verification
  // issue downstream).
  Value cast(Type toType, Value operand, bool isUnsignedCast) {
    OpBuilder builder = getBuilder();
    auto loc = operand.getLoc();
    return convertScalarToDtype(builder, loc, operand, toType, isUnsignedCast);
  }

  bool isComplex(Value value) {
    return llvm::isa<ComplexType>(value.getType());
  }
  bool isFloatingPoint(Value value) {
    return llvm::isa<FloatType>(value.getType());
  }
  bool isInteger(Value value) {
    return llvm::isa<IntegerType>(value.getType());
  }

  OpBuilder getBuilder() {
    OpBuilder builder(context);
    builder.setInsertionPointToEnd(&block);
    return builder;
  }

  MLIRContext *context;
  Block &block;
};

} // namespace

// [Modified by Smith]
// static void getGenericEffectsImpl(
//     SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
//         &effects,
//     ValueRange results, const ValueRange inputOperands,
//     ValueRange outputOperands) {
//   for (auto operand : inputOperands) {
//     if (!llvm::isa<MemRefType>(operand.getType()))
//       continue;
//     effects.emplace_back(MemoryEffects::Read::get(), operand,
//                          SideEffects::DefaultResource::get());
//   }
//   for (auto operand : outputOperands) {
//     if (!llvm::isa<MemRefType>(operand.getType()))
//       continue;
//     effects.emplace_back(MemoryEffects::Read::get(), operand,
//                          SideEffects::DefaultResource::get());
//     effects.emplace_back(MemoryEffects::Write::get(), operand,
//                          SideEffects::DefaultResource::get());
//   }
// }

// #define GET_OP_CLASSES
// #include "mtir/Dialect/MtFusion/IR/MtFusionNamedStructuredOps.yamlgen.cpp.inc"

#define GET_OP_CLASSES
#include "mtir/Dialect/MtFusion/IR/MtFusionOps.cpp.inc"

#define GET_OP_CLASSES
#include "mtir/Dialect/MtFusion/IR/MtFusionStructuredOps.cpp.inc"

static LogicalResult appendMangledType(llvm::raw_string_ostream &ss, Type t) {
  if (auto memref = llvm::dyn_cast<MemRefType>(t)) {
    ss << "view";
    for (auto size : memref.getShape())
      if (size < 0)
        ss << "sx";
      else
        ss << size << "x";
    if (failed(appendMangledType(ss, memref.getElementType())))
      return failure();
    if (auto as = memref.getMemorySpace()) {
      if (auto attr = llvm::dyn_cast<IntegerAttr>(as))
        ss << "as" << attr.getInt();
      else
        return failure();
    }
    return success();
  }
  if (auto vec = llvm::dyn_cast<VectorType>(t)) {
    ss << "vector";
    llvm::interleave(
        vec.getShape(), [&](int64_t i) { ss << i; }, [&]() { ss << "x"; });
    if (failed(appendMangledType(ss, vec.getElementType())))
      return failure();
    return success();
  }
  if (t.isSignlessIntOrIndexOrFloat()) {
    ss << t;
    return success();
  }
  return failure();
}

std::string mlir::mtfusion::generateLibraryCallName(Operation *op) {
  assert(isa<linalg::LinalgOp>(op));
  std::string name(op->getName().getStringRef().str());
  std::string fun = "";
  for (NamedAttribute kv : op->getAttrs()) {
    if (UnaryFnAttr ufa = llvm::dyn_cast<UnaryFnAttr>(kv.getValue())) {
      fun = stringifyEnum(ufa.getValue()).str() + "_";
    } else if (BinaryFnAttr bfa = llvm::dyn_cast<BinaryFnAttr>(kv.getValue())) {
      fun = stringifyEnum(bfa.getValue()).str() + "_";
    }
  }
  name.reserve(128);
  std::replace(name.begin(), name.end(), '.', '_');
  llvm::raw_string_ostream ss(name);
  ss << "_" << fun;
  for (Type t : op->getOperandTypes()) {
    if (failed(appendMangledType(ss, t)))
      return std::string();
    ss << "_";
  }
  std::string res = ss.str();
  res.pop_back();
  return res;
}

/// Pattern to fold cast into emtpy.
///
/// Before:
/// tensor.empty(shape1, dtype1) + mtfusion.cast(dtype2)
///
/// After:
/// tensor.empty(shape1, dtype2)
///
/// Restrictions:
/// the output of cast op should be an empty op
struct FoldCastEmpty : OpRewritePattern<mtfusion::CastOp> {
  using OpRewritePattern<mtfusion::CastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mtfusion::CastOp castOp,
                                PatternRewriter &rewriter) const override {
    auto defEmptyOp = castOp.getInputs()[0].getDefiningOp<tensor::EmptyOp>();
    if (!defEmptyOp)
      return failure();
    auto output = castOp.getOutputs()[0];
    if (!output.getDefiningOp<tensor::EmptyOp>())
      return failure();
    rewriter.replaceOp(castOp, output);
    return success();
  }
};

struct SimplifyRank0Tensor
    : public OpInterfaceRewritePattern<mlir::linalg::LinalgOp> {
  using OpInterfaceRewritePattern<
      mlir::linalg::LinalgOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(mlir::linalg::LinalgOp op,
                                PatternRewriter &rewriter) const override {
    bool operOnRank0Tensor = llvm::all_of(op->getOperandTypes(), [](Type type) {
      return type.isa<ShapedType>() &&
             llvm::dyn_cast<ShapedType>(type).getRank() == 0;
    });

    if (!operOnRank0Tensor) {
      return failure();
    }

    // TODO: Add mtfusion::CopyOp
    if (isa<linalg::CopyOp>(op)) {
      return rewriter.notifyMatchFailure(op, "cannot simplify copy");
    }

    // Step 1. Set insert pos
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);

    // Step 2. Map arguments of block to op input/result.
    Block &block = op->getRegions().front().getBlocks().front();
    IRMapping mapping;
    auto inputs = op.getDpsInputs();
    auto results = op->getResults();
    auto arguments = block.getArguments();
    assert(arguments.size() == inputs.size() + results.size());
    int cnt = 0;
    for (Value in : inputs) {
      mapping.map(block.getArgument(cnt++), in);
    }
    for (Value res : results) {
      mapping.map(block.getArgument(cnt++), res);
    }

    // Step 3. Travese operations in block and convert scalar into rank0 tensor.
    for (auto &opInBlock : block.getOperations()) {
      if (opInBlock.hasTrait<OpTrait::IsTerminator>()) {
        continue;
      }

      auto *newOp = rewriter.clone(opInBlock, mapping);
      for (Value res : newOp->getResults()) {
        auto newType = RankedTensorType::get(ArrayRef<int64_t>{},
                                             getElementTypeOrSelf(res));
        res.setType(newType);
      }

      for (auto [newRes, oldRes] :
           llvm::zip(newOp->getResults(), opInBlock.getResults())) {
        mapping.map(oldRes, newRes);
      }
    }

    // Step 4. replace res with yieldop res.
    auto *terminator = block.getTerminator();
    assert(terminator);
    assert(isa<linalg::YieldOp>(terminator));
    auto yieldOp = cast<linalg::YieldOp>(terminator);
    for (auto [res, yieldOper] :
         llvm::zip(op->getResults(), yieldOp.getOperands())) {
      rewriter.replaceAllUsesWith(res, mapping.lookup(yieldOper));
    }

    rewriter.eraseOp(op);
    return success();
  }
};

bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;
  Type a = inputs.front(), b = outputs.front();
  auto aT = llvm::dyn_cast<TensorType>(a);
  auto bT = llvm::dyn_cast<TensorType>(b);
  if (!aT || !bT)
    return false;

  if (aT.getElementType() != bT.getElementType())
    return false;

  return succeeded(verifyCompatibleShape(aT, bT));
}

void CastOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResults()[0], "cast");
}

void CastOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<FoldCastEmpty>(context);
}

//===----------------------------------------------------------------------===//
// MtFusionDialect
//===----------------------------------------------------------------------===//
void MtFusionDialect::getCanonicalizationPatterns(
    RewritePatternSet &results) const {
  results.add<SimplifyRank0Tensor>(getContext());
}

#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/ValueHandle.h"
#include "mtir/Dialect/MtFusion/TransformOps/MtFusionTransformOps.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace mtfusion {

//===----------------------------------------------------------------------===//
// Implementation of ValueHandle
//===----------------------------------------------------------------------===//

Value ValueHandle::getImpl(Value matchTarget, OpBuilder &opBuilder) {
  llvm_unreachable("Not implemented!");
}

Value ValueHandle::getImpl() { llvm_unreachable("Not implemented!"); }

//===----------------------------------------------------------------------===//
// Implementation of ValueHandleFoldResult
//===----------------------------------------------------------------------===//

ValueHandleFoldResult::ValueHandleFoldResult(int64_t cst, MLIRContext *ctx)
    : PointerUnion<ValueHandle *, Attribute>(
          IntegerAttr::get(IntegerType::get(ctx, 64), cst)) {}

std::optional<int64_t> ValueHandleFoldResult::getConstInteger() {
  auto maybeAttr = this->dyn_cast<Attribute>();
  if (!maybeAttr)
    return std::nullopt;
  return getConstantIntValue(this->get<Attribute>());
}

std::optional<ValueHandle *> ValueHandleFoldResult::getValueHandle() {
  auto *maybeAttr = this->dyn_cast<ValueHandle *>();
  if (!maybeAttr)
    return std::nullopt;
  return this->get<ValueHandle *>();
}

//===----------------------------------------------------------------------===//
// Implementation of RegularValueHandle
//===----------------------------------------------------------------------===//

Value RegularValueHandle::getImpl() {
  if (this->status_ == HandleStatus::kValid)
    return handle_;
  llvm_unreachable("Invalid handle!");
}

//===----------------------------------------------------------------------===//
// Implementation of NamedValueHandle
//===----------------------------------------------------------------------===//

Value NamedValueHandle::getImpl(Value matchTarget, OpBuilder &opBuilder) {
  if (this->status_ == HandleStatus::kValid)
    return handle_;

  if (this->status_ != HandleStatus::kNeedsRematch) {
    llvm_unreachable("Invalid handle!");
    return Value{};
  }

  TypedValue<transform::TransformHandleTypeInterface> matchResult;
  switch (this->type_) {
  case (IdentifierType::kAttribute):
    matchResult = opBuilder
                      .create<transform::MatchOp>(
                          matchTarget.getLoc(),
                          /*resultTypes=*/
                          // TypeRange{opBuilder.getType<transform::AnyOpType>()},
                          opBuilder.getType<transform::AnyOpType>(),
                          /*target=*/matchTarget,
                          /*ops=*/ArrayAttr{},
                          /*interface=*/transform::MatchInterfaceEnumAttr{},
                          /*op_attrs=*/
                          opBuilder.getDictionaryAttr(
                              ArrayRef<NamedAttribute>{opBuilder.getNamedAttr(
                                  this->name_, opBuilder.getUnitAttr())}),
                          /*filter_result_type=*/TypeAttr{},
                          /*filter_operand_types=*/ArrayAttr{})
                      .getResults();
    break;
  case (IdentifierType::kOperation):
    matchResult =
        opBuilder
            .create<transform::MatchOp>(
                matchTarget.getLoc(),
                /*resultTypes=*/
                TypeRange{opBuilder.getType<transform::AnyOpType>()},
                /*target=*/matchTarget,
                /*ops=*/
                opBuilder.getArrayAttr({opBuilder.getStringAttr(this->name_)}),
                /*interface=*/transform::MatchInterfaceEnumAttr{},
                /*op_attrs=*/
                DictionaryAttr{},
                /*filter_result_type=*/TypeAttr{},
                /*filter_operand_types=*/ArrayAttr{})
            .getResults();
    break;
  default:
    llvm_unreachable("Not implemented!");
    return Value();
  }

  if (this->needsReverse_) {
    matchResult = opBuilder.create<transform::ReverseOp>(
        matchResult.getLoc(),
        /*result=*/TypeRange{opBuilder.getType<transform::AnyOpType>()},
        /*target=*/matchResult);
  }
  this->handle_ = matchResult;
  this->status_ = HandleStatus::kValid;
  return handle_;
}

//===----------------------------------------------------------------------===//
// Implementation of FuncArgHandle
//===----------------------------------------------------------------------===//

Value FuncArgHandle::getImpl(Value funcValue, OpBuilder &opBuilder) {
  if (this->status_ == HandleStatus::kValid)
    return handle_;

  if (this->status_ == HandleStatus::kNeedsRematch) {
    auto getFuncArgOp = opBuilder.create<transform::GetFuncArgumentOp>(
        funcValue.getLoc(),
        /*outputs=*/opBuilder.getType<transform::AnyValueType>(),
        /*target=*/funcValue,
        /*raw_position_list=*/
        SmallVector<int64_t>{static_cast<int64_t>(this->funcArgNum_)},
        /*is_inverted=*/false);
    this->handle_ = getFuncArgOp;
    this->status_ = HandleStatus::kValid;
    return handle_;
  }
  llvm_unreachable("Invalid handle!");
  return {};
}

//===----------------------------------------------------------------------===//
// Implementation of HandleRecord
//===----------------------------------------------------------------------===//

std::string HandleRecord::getAndRecordAttrName(StringRef oldAttrName) {
  auto iter = attributeCount_.find(oldAttrName);
  std::string newTag = oldAttrName.data();
  size_t count = 1;
  if (iter != attributeCount_.end()) {
    count = iter->second;
    newTag = newTag + "_" + std::to_string(count++);
  }
  // update old attr with new count
  attributeCount_.insert_or_assign(oldAttrName, count);
  return newTag;
}

void HandleRecord::resetAllHandles() {
  for (ValueHandle *h : allocatedHandles_) {
    llvm::TypeSwitch<ValueHandle *, void>(h)
        .Case([](NamedValueHandle *h) {
          h->setStatus(HandleStatus::kNeedsRematch);
        })
        .Case(
            [](RegularValueHandle *h) { h->setStatus(HandleStatus::kInvalid); })
        .Case(
            [](FuncArgHandle *h) { h->setStatus(HandleStatus::kNeedsRematch); })
        .Default([](ValueHandle *) { llvm_unreachable("Not implemented!"); });
  }
}

void HandleRecord::clear() {
  for (ValueHandle *h : allocatedHandles_) {
    delete h;
  }
  allocatedHandles_.clear();
  attributeCount_.clear();
}

void setStatusTo(ValueHandles &vhs, HandleStatus status) {
  llvm::for_each(vhs, [&status](ValueHandle *vh) { vh->setStatus(status); });
}

} // namespace mtfusion
} // namespace mlir
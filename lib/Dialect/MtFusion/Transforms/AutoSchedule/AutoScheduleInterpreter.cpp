//===- AutoScheduleInterpreter.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a test pass that interprets Transform dialect operations in
// the module.
//
//===----------------------------------------------------------------------===//

#include "mtir/Dialect/MtFusion/Transforms/Passes.h"
#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/AutoScheduleAttrDefs.h"
#include "mtir/Dialect/MtFusion/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterPassBase.h"

using namespace mlir;
using namespace mlir::mtfusion;
using namespace mlir::detail;

namespace {

template <typename Derived>
class OpPassWrapper : public PassWrapper<Derived, OperationPass<>> {};

class AutoScheduleInterpreterPass
    : public transform::TransformInterpreterPassBase<
          AutoScheduleInterpreterPass, OpPassWrapper> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AutoScheduleInterpreterPass)

  using AutoScheduleInterpreterPassBase =
      transform::TransformInterpreterPassBase<AutoScheduleInterpreterPass,
                                              OpPassWrapper>;

  AutoScheduleInterpreterPass() = default;
  AutoScheduleInterpreterPass(const AutoScheduleInterpreterPass &pass)
      : TransformInterpreterPassBase(pass) {
    this->enforceSingleToplevelTransformOp =
        pass.enforceSingleToplevelTransformOp;
    this->transformFileName = pass.transformFileName;
    this->transformLibraryPaths = pass.transformLibraryPaths;
    this->enableExpensiveChecks = pass.enableExpensiveChecks;
    this->debugPayloadRootTag = pass.debugPayloadRootTag;
    this->debugTransformRootTag = pass.debugTransformRootTag;
  }
  AutoScheduleInterpreterPass(
      const transform::TransformOptions &transformOptions,
      const std::string &kernelName)
      : AutoScheduleInterpreterPassBase(transformOptions) {
    this->debugPayloadRootTag = auto_schedule::getPayloadRootTag(kernelName);
    this->debugTransformRootTag =
        auto_schedule::getTransformRootTag(kernelName);
  }

  StringRef getArgument() const override {
    return "mtfusion-auto-schedule-interpreter";
  }

  StringRef getDescription() const override {
    return "apply auto schedule transform sequence to target kernel";
  }

  static StringLiteral getBinaryName() { return "mtir-opt"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<transform::TransformDialect>();
  }

  void runOnOperation() override {
    options = options.enableExpensiveChecks(this->enableExpensiveChecks);
    options = options.enableEnforceSingleToplevelTransformOp(
        this->enforceSingleToplevelTransformOp);

    if (!this->transformFileName.empty()) {
      getOperation()->emitError("Does not support transformFileName yet.");
      return signalPassFailure();
    }

    if (!this->transformLibraryPaths.empty()) {
      getOperation()->emitError("Does not support transformLibraryPaths yet.");
      return signalPassFailure();
    }

    if (failed(transform::detail::interpreterBaseRunOnOperationImpl(
            getOperation(), getArgument(), getSharedTransformModule(),
            getTransformLibraryModule(), RaggedArray<transform::MappedValue>{},
            options, transformFileName, transformLibraryPaths,
            debugPayloadRootTag, debugTransformRootTag, getBinaryName())))
      return signalPassFailure();
  }

  Option<bool> enableExpensiveChecks{
      *this, "enable-expensive-checks", llvm::cl::init(false),
      llvm::cl::desc("Perform expensive checks to better report errors in the "
                     "transform IR")};
  Option<bool> enforceSingleToplevelTransformOp{
      *this, "enforce-single-top-level-transform-op", llvm::cl::init(true),
      llvm::cl::desc("Ensure that only a single top-level transform op is "
                     "present in the IR.")};

  // Options required for passes derived from TransformInterpreterPassBase.
  Option<std::string> transformFileName{
      *this, "transform-file-name", llvm::cl::init(""),
      llvm::cl::desc(
          "Optional filename containing a transform dialect specification to "
          "apply. If left empty, the IR is assumed to contain one top-level "
          "transform dialect operation somewhere in the module.")};
  Option<std::string> debugPayloadRootTag{
      *this, "debug-payload-root-tag", llvm::cl::init(""),
      llvm::cl::desc(
          "Select the operation with 'transform.target_tag' attribute having "
          "the given value as payload IR root. If empty select the pass anchor "
          "operation as the payload IR root.")};
  Option<std::string> debugTransformRootTag{
      *this, "debug-transform-root-tag", llvm::cl::init(""),
      llvm::cl::desc(
          "Select the operation with 'transform.target_tag' attribute having "
          "the given value as container IR for top-level transform ops. This "
          "allows user control on what transformation to apply. If empty, "
          "select the container of the top-level transform op.")};
  ListOption<std::string> transformLibraryPaths{
      *this, "transform-library-paths", llvm::cl::ZeroOrMore,
      llvm::cl::desc("Optional paths to files with modules that should be "
                     "merged into the transform module to provide the "
                     "definitions of external named sequences.")};
};

struct EraseAutoSchedulePass
    : public PassWrapper<EraseAutoSchedulePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EraseAutoSchedulePass)

  EraseAutoSchedulePass() = default;
  explicit EraseAutoSchedulePass(const EraseAutoSchedulePass &pass) {
    this->kernelName = pass.kernelName;
  };
  explicit EraseAutoSchedulePass(const std::string &kernelName) {
    this->kernelName = kernelName;
  }

  StringRef getArgument() const final { return "mtfusion-erase-auto-schedule"; }

  StringRef getDescription() const final {
    return "erase auto schedule transform sequence from the IR";
  }

  void runOnOperation() override {
    auto targetPayloadRootTag = auto_schedule::getPayloadRootTag(kernelName);
    auto targetTransformRootTag =
        auto_schedule::getTransformRootTag(kernelName);
    getOperation()->walk<WalkOrder::PreOrder>([&](Operation *nestedOp) {
      if (!nestedOp->hasAttrOfType<StringAttr>(kTransformDialectTagAttrName)) {
        return WalkResult::advance();
      }
      if (isa<transform::TransformOpInterface>(nestedOp) &&
          nestedOp->getAttrOfType<StringAttr>(kTransformDialectTagAttrName)
                  .str() == targetTransformRootTag) {
        nestedOp->erase();
        return WalkResult::skip();
      }
      if (isa<func::FuncOp>(nestedOp) &&
          nestedOp->getAttrOfType<StringAttr>(kTransformDialectTagAttrName)
                  .str() == targetPayloadRootTag) {
        nestedOp->removeAttr(kTransformDialectTagAttrName);
        return WalkResult::skip();
      }
      return WalkResult::advance();
    });
  }

  Option<std::string> kernelName{
      *this, "kernel-name", llvm::cl::init(""),
      llvm::cl::desc("Erase transform sequence for target kernel")};
};

} // namespace

namespace mlir {
namespace mtfusion {
void registerEraseAutoSchedulePass() {
  PassRegistration<EraseAutoSchedulePass> reg;
}

void registerAutoScheduleInterpreterPass() {
  PassRegistration<AutoScheduleInterpreterPass> reg;
}

std::unique_ptr<Pass>
createAutoScheduleInterpreterPass(const std::string &kernelName) {
  return std::make_unique<AutoScheduleInterpreterPass>(
      transform::TransformOptions{}, kernelName);
}

std::unique_ptr<Pass>
createEraseAutoSchedulePass(const std::string &kernelName) {
  return std::make_unique<EraseAutoSchedulePass>(kernelName);
}

} // namespace mtfusion
} // namespace mlir
//===- PredefinedTilingBase.cpp -- Auto-schedule fused kernels ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the basic functionality of predefined tiling.
//
//===----------------------------------------------------------------------===//

#include "mtir/Dialect/MtFusion/Transforms/PredefinedTiling/TilingBase.h"
#include "mtir/Dialect/MtFusion/Transforms/PredefinedTiling/DeepCVTiling.h"
#include "mtir/Dialect/MtFusion/TransformOps/MtFusionTransformOps.h"
#include "mtir/Dialect/MtFusion/Utils/Utils.h"
#include "mtir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"

#include <iostream>
using namespace std;

namespace mlir {
#define GEN_PASS_DEF_PREDEFINEDTILING
#include "mtir/Dialect/MtFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mtfusion;
using namespace mlir::mtfusion;

namespace {

void output_err(const std::string& msg, const std::string& value = "") {
  std::cout << msg << value << std::endl;
}

bool isNumber(const std::string& str){
  for(auto ch : str){
    if(ch < '0' || ch >'9')
      return false;
  } return true;
}

transform::SequenceOp initTilingSequence(OpBuilder &opBuilder) {
  OpBuilder::InsertionGuard g(opBuilder);
  // create transform sequence op with name
  auto seqOp = opBuilder.create<transform::SequenceOp>(
      opBuilder.getUnknownLoc(), TypeRange(),
      transform::FailurePropagationMode::Propagate,
      opBuilder.getType<transform::AnyOpType>(),
      [](OpBuilder &builder, Location nested, Value rootH) {
        builder.create<transform::YieldOp>(nested, ValueRange());
      });
  return seqOp;
}

} // namespace mlir

//===----------------------------------------------------------------------===//
// TilingItem
//===----------------------------------------------------------------------===//

void TilingItem::getArgAttrs(SmallVector<NamedAttribute>& argAttrs,
    OpBuilder &opBuilder, MLIRContext* ctx) {
  argAttrs.push_back(NamedAttribute{
      opBuilder.getStringAttr(mtfusion::TilingDataAttr::name),
      opBuilder.getUnitAttr()});
  argAttrs.push_back(NamedAttribute{
      opBuilder.getStringAttr(mtfusion::TilingAxisAttr::name),
      TilingAxisAttr::get(ctx, axisToEnum[getAxis()])});
  if(getNthreads() > 1) {
    argAttrs.push_back(NamedAttribute{
      opBuilder.getStringAttr(mtfusion::NthreadsAttr::name),
      NthreadsAttr::get(ctx, getNthreads())});
  }
  if(getCopyNum() != 0) {
    auto matAttrs = llvm::map_to_vector(getCopyMat(), [&](std::string& matStr){
      return matrixToEnum[matStr];
    });
    argAttrs.push_back(NamedAttribute{
      opBuilder.getStringAttr(mtfusion::CopyMatAttr::name),
      CopyMatAttr::get(ctx, matAttrs)});

    auto dstAttrs = llvm::map_to_vector(getCopyDst(), [&](std::string& memStr){
      return memoryToEnum[memStr];
    });
    argAttrs.push_back(NamedAttribute{
      opBuilder.getStringAttr(mtfusion::CopyDstAttr::name),
      CopyDstAttr::get(ctx, dstAttrs)});
  }
}

void TilingItem::print() {
  std::cout << "tilingItem info:" << std::endl;
  std::cout << "axis:" << axis_ << std::endl;
  std::cout << "nthreads:" << nthreads_ << std::endl;
  for(size_t i = 0; i < copyMat_.size(); i++) {
    std::cout << "copy-mat=" << copyMat_[i] << " copy-dst=" << copyDst_[i] << std::endl;
  }
}

bool TilingItem::verify() {
  if(getCopyMat().size() == getCopyDst().size()) {
    setCopyNum(getCopyMat().size());
    return true;
  } return false;
}

//===----------------------------------------------------------------------===//
// TilerBase
//===----------------------------------------------------------------------===//

std::set<std::string> TilingItem::optNameSet = {
    "axis", "copy-mat", "copy-dst", "nthreads"};
std::set<std::string> TilingItem::axisSet = {"M", "N", "K", "m", "n", "k"};
std::set<std::string> TilingItem::matrixSet = {"A", "B", "C", "a", "b", "c"};
std::set<std::string> TilingItem::memorySet = {"GSM", "AM", "SM", "gsm", "am", "sm"};

std::map<std::string, mtfusion::Axis> TilingItem::axisToEnum = {
  {"M", mtfusion::Axis::M}, {"N", mtfusion::Axis::N}, {"K", mtfusion::Axis::K},
  {"m", mtfusion::Axis::M}, {"n", mtfusion::Axis::N}, {"k", mtfusion::Axis::K}};
std::map<std::string, mtfusion::Matrix> TilingItem::matrixToEnum = {
  {"A", mtfusion::Matrix::MatA}, {"B", mtfusion::Matrix::MatB}, {"C", mtfusion::Matrix::MatC},
  {"a", mtfusion::Matrix::MatA}, {"b", mtfusion::Matrix::MatB}, {"c", mtfusion::Matrix::MatC}};
std::map<std::string, mtfusion::Cache> TilingItem::memoryToEnum = {
  {"GSM", mtfusion::Cache::GSM}, {"AM", mtfusion::Cache::AM}, {"SM", mtfusion::Cache::SM},
  {"gsm", mtfusion::Cache::GSM}, {"am", mtfusion::Cache::AM}, {"sm", mtfusion::Cache::SM}};

TilerBase::TilerBase(func::FuncOp f, FusionKind kind) {
  kernelInfo_ = std::make_unique<KernelInfo>();
  tilingInfo_ = std::make_unique<TilingInfo>();
  handleRecord_ = std::make_unique<HandleRecord>();
  originalKernel_ = f;
  module_ = f.getOperation()->getParentOfType<ModuleOp>();
  kind_ = kind;
}

TilerBase::TilerBase(func::FuncOp f,
      std::unique_ptr<KernelInfo> kernelInfo,
      std::unique_ptr<TilingInfo> tilingInfo) {
  kernelInfo_ = std::move(kernelInfo);
  tilingInfo_ = std::move(tilingInfo);
  handleRecord_ = std::make_unique<HandleRecord>();
  originalKernel_ = f;
  module_ = f.getOperation()->getParentOfType<ModuleOp>();
  kind_ = kernelInfo_->getFusionKind();
}

TilerBase::TilerBase(func::FuncOp f, 
      mtfusion::TilingSeq& tilingSeq, 
      FusionKind kind) : tilingSeq_(tilingSeq) {
  kernelInfo_ = std::make_unique<KernelInfo>();
  tilingInfo_ = std::make_unique<TilingInfo>();
  handleRecord_ = std::make_unique<HandleRecord>();
  originalKernel_ = f;
  module_ = f.getOperation()->getParentOfType<ModuleOp>();
  kind_ = kind;
}

TilerBase::TilerBase(func::FuncOp f,
      mtfusion::TilingSeq& tilingSeq, 
      std::unique_ptr<KernelInfo> kernelInfo,
      std::unique_ptr<TilingInfo> tilingInfo) : tilingSeq_(tilingSeq) {
  kernelInfo_ = std::move(kernelInfo);
  tilingInfo_ = std::move(tilingInfo);
  handleRecord_ = std::make_unique<HandleRecord>();
  originalKernel_ = f;
  module_ = f.getOperation()->getParentOfType<ModuleOp>();
  kind_ = kernelInfo_->getFusionKind();
}

TilerBase::~TilerBase() {
  kernelInfo_.reset();
  tilingInfo_.reset();
  handleRecord_.reset();
}

LogicalResult TilerBase::runOnOperation(OpBuilder &opBuilder) {
  func::FuncOp currentFunc = getOriginalKernel();
  if (failed(createAndApplyTiling(opBuilder)))
    return currentFunc->emitWarning("Failed to create and apply schedule.");
  return success();
}

LogicalResult TilerBase::createAndApplyTiling(OpBuilder &opBuilder) {
  if(failed(initTiling(opBuilder))) {
    return failure();
  }
  if (failed(createTilingImpl(opBuilder))) {
    return failure();
  }
  if (failed(applyTilingImpl(opBuilder))) {
    return failure();
  }
  cleanUpAfterTiling();
  if (failed(fixCallSitesAndCaller(opBuilder))) {
    return failure();
  }
  getOriginalKernel()->erase();
  return success();
}

LogicalResult TilerBase::applyTiling(func::FuncOp &funcOp,
      mtfusion::TilingSeq& tilingSeq, OpBuilder &opBuilder) {
  auto fusionKindAttr = 
      funcOp->getAttrOfType<FusionKindAttr>(FusionKindAttr::name);
  auto fusionKind = fusionKindAttr.getFusionKind();
  std::unique_ptr<TilerBase> tiler;
  switch (fusionKind) {
    case FusionKind::MixCV:
      tiler = std::make_unique<DeepCVTiler>(funcOp, tilingSeq);
      break;
    default:
      return funcOp.emitError("Unknown kernel fusion kind");
  }
  return tiler->runOnOperation(opBuilder);
}

LogicalResult TilerBase::applyTilingImpl(OpBuilder &opBuilder) {
  PassManager pm(getContext());
  pm.addPass(
      mtfusion::createAutoScheduleInterpreterPass(getToBeTiledKernelName()));
  pm.addPass(
      mtfusion::createEraseAutoSchedulePass(getToBeTiledKernelName()));
  if (failed(pm.run(getModule())))
    return failure();
  return success();
}

void TilerBase::cleanUpAfterTiling() {
  getHandleRecord()->clear();
  // setToBeTiledKernel(nullptr);
  setTransformSeqHandle(Value());
  for (auto *td : getTilingInfo()->getTilingStruct())
    td->setHandle(nullptr);
}

LogicalResult TilerBase::initTiling(OpBuilder &opBuilder) {
  /// Step 1: Construct new kernel name with tiling key post-fix.
  func::FuncOp originalKernel = getOriginalKernel();
  auto toBeTiledKernelName =
      originalKernel.getSymName().str() + "_tiling";
  auto module = getModule();
  if (module.lookupSymbol(opBuilder.getStringAttr(toBeTiledKernelName))) {
    return originalKernel->emitError(
        "Duplicate kernel name during auto-scheduling process");
  }

  /// Step 2: Clone original func and set it as the to-be-scheduled function.
  opBuilder.setInsertionPoint(originalKernel);
  func::FuncOp toBeTiledKernel =
      cast<func::FuncOp>(opBuilder.clone(*originalKernel));
  toBeTiledKernel.setSymName(toBeTiledKernelName);
  this->setToBeTiledKernel(toBeTiledKernel);

  /// Step 3. Insert tiling data to to-be-scheduled kernel function and bind
  /// tiling data to kernel argument.
  opBuilder.setInsertionPointToStart(&toBeTiledKernel.getBlocks().front());
  // Create memref to store tiling arguments.
  // auto memrefType = MemRefType::get(
  //     getTilingSeq().size(), opBuilder.getI64Type());
  // auto tilingDataAlloca = opBuilder.create<memref::AllocaOp>(
  //     toBeTiledKernel.getLoc(), memrefType);
  // Traverse each tiling arguments.
  // auto tilingDataPosStart = toBeTiledKernel.getNumArguments();
  auto argIdx = toBeTiledKernel.getNumArguments();
  for(auto& tilingItem : this->getTilingSeq()) {
    // Generate attr from tiling item.
    SmallVector<NamedAttribute> argAttrs;
    tilingItem.getArgAttrs(argAttrs, opBuilder, toBeTiledKernel->getContext());
    // Insert tiling argument.
    toBeTiledKernel.insertArgument(argIdx, opBuilder.getI64Type(),
        opBuilder.getDictionaryAttr(argAttrs), toBeTiledKernel.getLoc());
    // Cast tiling arg from i64 to index.
    auto castOp = opBuilder.create<index::CastSOp>(toBeTiledKernel.getLoc(),
        opBuilder.getIndexType(), toBeTiledKernel.getArgument(argIdx));
    // castOp->setAttrs(opBuilder.getDictionaryAttr(argAttrs));
    castOp->setAttr(
      opBuilder.getStringAttr(mtfusion::TilingDataAttr::name),
      opBuilder.getUnitAttr()); 
    // Store function argument and load it agian.
    // auto idx = opBuilder.create<arith::ConstantOp>(
    //   tilingDataAlloca.getLoc(), opBuilder.getIndexAttr(argIdx - tilingDataPosStart));
    // opBuilder.create<memref::StoreOp>(tilingDataAlloca.getLoc(), 
    //     toBeTiledKernel.getArgument(argIdx), tilingDataAlloca, ValueRange{idx});
    // auto loadOp = opBuilder.create<memref::LoadOp>(tilingDataAlloca.getLoc(),
    //     tilingDataAlloca, ValueRange{idx});
    // loadOp->setAttrs(dicAttr);
    argIdx++;
  }

  /// Step 4. Insert transform sequence right after the to-be-scheduled kernel.
  opBuilder.setInsertionPointAfter(toBeTiledKernel);
  auto seqOp = initTilingSequence(opBuilder);
  auto *transformBody = seqOp.getBodyBlock();
  // Set insertion point to transform sequence body
  opBuilder.setInsertionPointToStart(transformBody);
  // Record transform sequence block argument
  setTransformSeqHandle(transformBody->getArguments().front());

  // Set transform root and payload root tags.
  llvm::StringLiteral kTransformDialectTagAttrName = "transform.target_tag";
  // toBeTiledKernelName + "_payload"
  toBeTiledKernel->setAttr(
      kTransformDialectTagAttrName,
      opBuilder.getStringAttr(
          auto_schedule::getPayloadRootTag(toBeTiledKernelName)));
  // toBeTiledKernelName + "_transform"
  seqOp->setAttr(kTransformDialectTagAttrName,
                 opBuilder.getStringAttr(auto_schedule::getTransformRootTag(
                     toBeTiledKernelName)));
  return success();
}

void TilerBase::getCallerInfo(func::FuncOp callee, ModuleOp enclosingModule,
                              DenseMap<func::FuncOp, CallerInfo> &info) {
  std::optional<SymbolTable::UseRange> maybeUses =
      callee.getSymbolUses(enclosingModule);
  for (SymbolTable::SymbolUse use : maybeUses.value()) {
    func::CallOp callSite = cast<func::CallOp>(use.getUser());
    auto callerOp = callSite->getParentOfType<func::FuncOp>();
    auto &callerInfo = info[callerOp];
    callerInfo.caller = callerOp;
    callerInfo.callerOriginalArgNumber = callerOp.getNumArguments();
    callerInfo.callee = callee;
    callerInfo.callSites.push_back(callSite);
  }
}

SmallVector<Value>TilerBase::getNewArgsForCallSite(
    func::FuncOp caller, func::CallOp oldCallSite,
    const TilerBase::CallSiteArgBuilderInfo &info, OpBuilder &opBuilder) {
  auto oldCallArgs = oldCallSite->getOperands();
  size_t oldArgCount = oldCallArgs.size();
  size_t tilingStructSize = info.tilingIdx2CallerArgIdx.size();
  size_t newArgCount = oldArgCount + tilingStructSize;

  SmallVector<Value> newCallArgs;
  newCallArgs.reserve(newArgCount);
  // By convention, tiling data is appended after existing args, but the order
  // to which they're added matches the tiling struct order
  newCallArgs.append(oldCallArgs.begin(), oldCallArgs.end());
  newCallArgs.append(SmallVector<Value>(tilingStructSize, Value()));

  for (size_t idx = oldArgCount; idx < newArgCount; idx++) {
    auto tilingIdx = idx - oldArgCount;
    assert(info.tilingIdx2CallerArgIdx.contains(tilingIdx));
    newCallArgs[idx] =
        caller.getArgument(info.tilingIdx2CallerArgIdx.at(tilingIdx));
  }
  return newCallArgs;
}

void TilerBase::doFixCallSite(CallerInfo &callerInfo, func::CallOp callSite,
                              CallSiteArgBuilderInfo &builderInfo,
                              DenseMap<Operation *, Operation *> &irMap,
                              OpBuilder &opBuilder) {
  OpBuilder::InsertionGuard g(opBuilder);
  opBuilder.setInsertionPoint(callSite);
  auto newArgs = getNewArgsForCallSite(callerInfo.caller, callSite, builderInfo,
                                       opBuilder);
  func::CallOp newCallSite = opBuilder.create<func::CallOp>(
      callSite.getLoc(), callSite.getResultTypes(), 
      getToBeTiledKernel().getSymName(), newArgs);
  irMap.insert(std::make_pair(callSite, newCallSite));
}

LogicalResult TilerBase::fixCallSitesAndCaller(OpBuilder &opBuilder) {
  OpBuilder::InsertionGuard g(opBuilder);

  // Get callers of the original, unscheduled kernel
  DenseMap<func::FuncOp, CallerInfo> workList;
  getCallerInfo(getOriginalKernel(), getModule(), workList);

  // Bail out on trivial case where there is no caller
  if (workList.empty())
    return failure();

  // Repeatedly modify the caller and call site, until there is no caller.
  DenseMap<Operation *, Operation *> irMap;
  DenseSet<func::FuncOp> processedCaller;
  while (!workList.empty()) {
    auto &[caller, callerInfo] = *(workList.begin());
    if (processedCaller.contains(caller)) {
      return failure();
    }

    size_t callerArgCount = caller.getNumArguments();
    getCallerInfo(caller, getModule(), workList);

    // Insert tiling data arguments to caller's enclosing func.
    DenseMap<size_t, size_t> tilingIdx2CallerArgIdx;
    for (size_t idx = callerArgCount; 
        idx < callerArgCount + this->getTilingSeq().size(); idx++) {
      SmallVector<NamedAttribute> argAttrs = {getTilingDataAttr(opBuilder)};
      caller.insertArgument(idx, opBuilder.getI64Type(),
                            opBuilder.getDictionaryAttr(argAttrs),
                            caller.getLoc());
      tilingIdx2CallerArgIdx.insert({idx - callerArgCount, idx});
    }

    // Fix the call sites
    bool calleeIsOriginalKernel = callerInfo.callee == getOriginalKernel();
    CallSiteArgBuilderInfo builderInfo{tilingIdx2CallerArgIdx,
                                       calleeIsOriginalKernel};
    for (func::CallOp callSite : callerInfo.callSites) {
      doFixCallSite(callerInfo, callSite, builderInfo, irMap, opBuilder);
    }
    processedCaller.insert(caller);
    workList.erase(caller);
  }
  for (auto &[oldOp, newOp] : irMap) {
    oldOp->replaceAllUsesWith(newOp);
    oldOp->erase();
  }
  return success();
}

NamedValueHandle *TilerBase::recordImpl(Value target, OpBuilder &opBuilder,
                                            const NamedValueHandleArgs &args) {
  // If the identifier type is operation name, then it's already unique.
  std::string newName = args.type == IdentifierType::kAttribute
                            ? getHandleRecord()->getAndRecordAttrName(args.name)
                            : args.name.str();
  if (args.needsReverse)
    target = opBuilder.create<transform::ReverseOp>(
        target.getLoc(),
        /*result=*/TypeRange{opBuilder.getType<transform::AnyOpType>()},
        /*target=*/target);
  if (args.needsAnnotate)
    opBuilder.create<transform::AnnotateOp>(
        target.getLoc(),
        /*target=*/target,
        /*name=*/opBuilder.getStringAttr(newName),
        /*param=*/Value{});
  return new NamedValueHandle(target, newName, args.type, HandleStatus::kValid,
                              args.needsReverse);
}

RegularValueHandle *
TilerBase::recordImpl(Value target, [[maybe_unused]] OpBuilder &opBuilder) {
  return new RegularValueHandle(target, HandleStatus::kValid);
}

FuncArgHandle *TilerBase::recordImpl(Value target,
                                    [[maybe_unused]] OpBuilder &opBuilder,
                                    size_t funcArgNum) {
  return new FuncArgHandle(target, funcArgNum, HandleStatus::kValid);
}

//===----------------------------------------------------------------------===//
// Static functions
//===----------------------------------------------------------------------===//

static bool readSingleOption(const std::string& str, 
                            TilingItem& tilingItem) {
  auto it_0 = str.begin();
  while(*it_0 == '-')
    ++it_0;
  /// read optName
  auto it = it_0;
  while(it != str.end() && *it != '=')
    ++it;
  if(it == str.end()){
    output_err("No '=' has been found.");
    return false;
  }
  std::string optName = {it_0, it++};
  if(!TilingItem::optNameSet.count(optName)) {
    output_err("Error option name: ", optName);
    return false;
  }
  /// read optInputs
  it_0 = it;
  if(it_0 == str.end()){
    output_err("Option lacks an input.");
    return false;
  }
  while(1) {
    while(it != str.end() && *it != ',')
      ++it;
    std::string curOptInput = {it_0, it};
    if(optName == "axis") {
      if(!TilingItem::axisSet.count(curOptInput)) {
        output_err("Error axis name:", curOptInput);
        return false;
      }
      tilingItem.setAxis(curOptInput);
    } else if(optName == "nthreads") {
      if(!isNumber(curOptInput)){
        output_err("nthreads must be a number, can not be ", curOptInput);
        return false;
      }
      tilingItem.setNthreads(atoi(curOptInput.c_str()));
    } else if(optName == "copy-mat") {
      if(!TilingItem::matrixSet.count(curOptInput)) {
        output_err("Error matrix name: ", curOptInput);
        return false;
      }
      tilingItem.appendCopyMat(curOptInput);
    } else if(optName == "copy-dst") {
      if(!TilingItem::memorySet.count(curOptInput)) {
        output_err("Error memory hieracy name: ", curOptInput);
        return false;
      }
      tilingItem.appendCopyDst(curOptInput);
    }
    if(it == str.end())
      break;
    it_0 = ++it;
  }
  return true;
}

static bool readTilingSeqFromOptions(mtfusion::TilingSeq& tilingSeq, 
                                    const std::string& tilingSeqStr) {
  auto it = tilingSeqStr.begin();
  while(1) {
    while(it != tilingSeqStr.end() && *it != '{')
      ++it;
    if(it == tilingSeqStr.end()) {
      if(!tilingSeq.empty())
        break;
      output_err("There isn't any tiling item.");
      return false;
    }
    TilingItem tilingItem;
    while(1) {
      while(++it != tilingSeqStr.end() && *it == ' ')
        ++it;
      auto it_0 = it;
      while(it != tilingSeqStr.end() && *it != ' ' && *it != '}')
        it++;
      if(it == tilingSeqStr.end()) {
        output_err("'}' is missed.");
        return false;
      } 
      if(!readSingleOption({it_0, it}, tilingItem))
        return false;
      if(*it == '}')
        break;
    }
    if(!tilingItem.verify()) {
      output_err("Tiling item verification failed.");
      return false;
    }
    tilingSeq.push_back(tilingItem);
  }
  return true;
}

static void printTilingSeq(mtfusion::TilingSeq& tilingSeq) {
  std::cout << "--------------------------" << std::endl;
  for(size_t i = 0; i < tilingSeq.size(); i++) {
    std::cout << "No." << i << " ";
    tilingSeq[i].print();
    std::cout << "--------------------------" << std::endl;
  }
}

namespace mlir {
class PredefinedTilingPass: public impl::PredefinedTilingBase<PredefinedTilingPass> {
public:
  using PredefinedTilingBase<PredefinedTilingPass>::PredefinedTilingBase;

  explicit PredefinedTilingPass(const PredefinedTilingOptions &options)
      : PredefinedTilingBase(options) {}

  void runOnOperation() override;
};
} // namespace mlir

void PredefinedTilingPass::runOnOperation() {
  std::string tilingSeqOptStr = this->tilingSeqStr;
  mtfusion::TilingSeq tilingSeq;
  if(!readTilingSeqFromOptions(tilingSeq, tilingSeqOptStr))
    return;
  // printTilingSeq(tilingSeq);
  getOperation()->walk([&](func::FuncOp funcOp) {
    auto funcKindAttr =
      funcOp->getAttrOfType<mtfusion::FuncKindAttr>(mtfusion::FuncKindAttr::name);
    if (!funcKindAttr || funcKindAttr.getFunctionKind() != FuncKind::Device)
      return WalkResult::skip();  // must be a device function
    auto fusionKindAttr = 
      funcOp->getAttrOfType<mtfusion::FusionKindAttr>(mtfusion::FusionKindAttr::name);
    if (!fusionKindAttr || 
        fusionKindAttr.getFusionKind() != this->fusionMode)
      return WalkResult::skip(); // funcOp.fusionKind must equal to pass.option.fusionMode

    OpBuilder opBuilder(&getContext());
    if(!succeeded(TilerBase::applyTiling(funcOp, tilingSeq, opBuilder))) {
      funcOp->emitOpError("Failed to create and apply tiling.");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
}

std::unique_ptr<Pass> mlir::mtfusion::createPredefinedTilingPass() {
  return std::make_unique<PredefinedTilingPass>();
}

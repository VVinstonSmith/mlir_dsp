//===- FusableBlockAnalyzer.h ------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mtir/Dialect/MtFusion/Transforms/OpFusion/FusableBlock.h"
#include "mtir/Dialect/MtFusion/Transforms/OpFusion/FusableHelper.h"
#include "mtir/Dialect/MtFusion/Transforms/Passes.h"

#ifndef MTIR_DIALECT_MTFUSION_TRANSFORMS_OPFUSION_FUSABLEBLOCKANALYZER_H
#define MTIR_DIALECT_MTFUSION_TRANSFORMS_OPFUSION_FUSABLEBLOCKANALYZER_H

namespace mlir {
namespace mtfusion {
namespace opfusion {

FusableBlocks getFusableBlocks(func::FuncOp func,
                               FusableHelper &fusableHelper);

class FusableBlockAnalyzer {
public:
  FusableBlockAnalyzer(Block &block, const FusableHelper &fusableHelper);

  SmallVector<SetVector<Operation *>> fuseBlock();
  FusableBlocks getFusableBlocks();

private:
  typedef DenseMap<size_t, bool> AdjacentEdge;
  typedef SmallVector<AdjacentEdge> AdjacencyList;
  typedef SmallVector<DenseSet<size_t>> ReverseAdjacencyList;

  const FusableHelper *fusableHelper_;
  mutable SetVector<Operation *> ops_;
  DenseMap<Operation *, size_t> opIdx_;

  SmallVector<int, 8> disjointSet_;
  SmallVector<uint8_t, 8> setType_;
  // Record for last axis reduce op specially
  SmallVector<int, 8> opMaxRank_;

  AdjacencyList edge_;
  ReverseAdjacencyList revEdge_;

  bool sameSet(size_t a, size_t b);
  int find(size_t X);
  bool reInitEdges();
  void mergeEdge(int nodeA, int nodeB);
  void join(int nodeA, int nodeB);
  bool isRestrictedByDependency(int startNode, int endNode,
                                bool horizontal = false);
  bool verifyRulesAndJoin(int nodeU, int nodeV, bool horizontal = false);
};
} // namespace opfusion
} // namespace mtfusion
} // namespace mlir

#endif //MTIR_DIALECT_MTFUSION_TRANSFORMS_OPFUSION_FUSABLEBLOCKANALYZER_H

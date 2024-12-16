//===- FusableBlockAnalyzer.cpp - Generate fusable blocks by rules --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mtir/Dialect/MtFusion/Transforms/OpFusion/FusableBlockAnalyzer.h"
#include "mtir/Dialect/MtFusion/Transforms/OpFusion/FusableHelper.h"

#include "llvm/Support/Debug.h"

#include <queue>

#define DEBUG_TYPE "mtfusion-fuse"

namespace mlir {
namespace mtfusion {
namespace opfusion {

FusableBlocks getFusableBlocks(func::FuncOp func,
                               FusableHelper &fusableHelper) {
  FusableBlocks fusableBlocks;
  for (Block &block : func.getBody()) {
    FusableBlockAnalyzer analyzer(block, fusableHelper);
    // This uses append, because it can
    // return multiple fusable blocks from 1 block
    fusableBlocks.append(analyzer.getFusableBlocks());
  }
  return fusableBlocks;
}

FusableBlockAnalyzer::FusableBlockAnalyzer(Block &block,
                                           const FusableHelper &fusableHelper)
    : fusableHelper_(&fusableHelper) {
  for (Operation &op : block.getOperations())
    ops_.insert(&op);

  // Initialize OpIdx
  for (size_t idx = 0; idx < ops_.size(); ++idx) {
    opIdx_[ops_[idx]] = idx;
  }

  disjointSet_.resize(ops_.size(), -1);
  setType_.resize(ops_.size());
  opMaxRank_.resize(ops_.size());

  edge_.resize(ops_.size());
  revEdge_.resize(ops_.size());

  // Construct edge for blocks
  for (size_t idx = 0; idx < ops_.size(); ++idx) {
    Operation *op = ops_[idx];
    opMaxRank_[idx] = fusableHelper_->obtainLastReduceRank(op);
    setType_[idx] = fusableHelper_->obtainType(op);
  }
}

bool FusableBlockAnalyzer::verifyRulesAndJoin(int nodeU, int nodeV,
                                              bool horizontal) {
  // Try to join
  int parentU = find(nodeU);
  int parentV = find(nodeV);

  // Already in one set, skip fusing
  if (parentU == parentV)
    return false;

  LLVM_DEBUG(llvm::dbgs() << "Joining and checking by reduce dimensions\n";);
  // Reduce dimension checker
  if (fusableHelper_->
        isRestrictedByReduceRank(opMaxRank_[parentU], opMaxRank_[parentV])) {

    LLVM_DEBUG(llvm::dbgs() << "Not allowed by reduce dimensions\n";);
    return false;
  }

  // Node type checker
  if (fusableHelper_->isRestrictedByNodeType(setType_[parentU],
                                             setType_[parentV])) {
    LLVM_DEBUG(llvm::dbgs() << "Not allowed by node Type\n";);
    return false;
  }

  // Restricted by dependency
  if (isRestrictedByDependency(parentU, parentV, horizontal)) {
    LLVM_DEBUG(llvm::dbgs() << "Not allowed by fusion graph dependency\n";);
    return false;
  }

  if (horizontal) {
    // Check the opposite as well
    if (isRestrictedByDependency(parentV, parentU, horizontal)) {
      LLVM_DEBUG(llvm::dbgs()
                     << "Not allowed by opposite fusion graph dependency\n";);
      return false;
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Joining\n";);
  join(parentU, parentV);
  return true;
}

SmallVector<SetVector<Operation *>> FusableBlockAnalyzer::fuseBlock() {
  reInitEdges();
  for (size_t nodeU = 0; nodeU < ops_.size(); ++nodeU) {
    // Fusing
    Operation *op = ops_[nodeU];

    // Restricted the fusion if it's shallowCV and shape is dynamic
    if (fusableHelper_->isRestrictedByDynamicShape(op))
      continue;

    for (Operation *userOp : op->getUsers()) {
      const size_t nodeV = opIdx_[userOp];
      if (fusableHelper_->isFusable(op, userOp)) {
        // Verify graph and parent rules
        LLVM_DEBUG(llvm::dbgs()
                       << "Fusable nodes " << *userOp << " " << *op << "\n";);
        verifyRulesAndJoin(nodeU, nodeV);
      }
    }
  }

  int32_t horizontalFusionCount = 0;
  LLVM_DEBUG(llvm::dbgs() << "Horizontal fusion merging\n";);
  // Try to merge horizontal fusion
  // Should we merge this in the return value instead ??
  for (size_t nodeU = 0; nodeU < ops_.size(); ++nodeU) {
    if (horizontalFusionCount >= fusableHelper_->maxHorizontalFusion())
      break;
    // Fusing
    Operation *op = ops_[nodeU];
    // Restricted the fusion if it's shallowCV and shape is dynamic
    if (fusableHelper_->isRestrictedByDynamicShape(op))
      continue;
    for (size_t nodeV = nodeU + 1; nodeV < ops_.size(); ++nodeV) {
      if (horizontalFusionCount >= fusableHelper_->maxHorizontalFusion())
        break;
      Operation *opV = ops_[nodeV];
      if (fusableHelper_->isFusable(op, opV)) {
        // Verify graph and parent rules
        LLVM_DEBUG(llvm::dbgs()
                       << "Fusable nodes " << *op << " " << *opV << "\n";);
        horizontalFusionCount += verifyRulesAndJoin(nodeU, nodeV, true);
      }
    }
  }

  if (horizontalFusionCount >= fusableHelper_->maxHorizontalFusion()) {
    LLVM_DEBUG(llvm::dbgs() << "Maximum horizontal fusion has reached\n";);
  }

  SmallVector<SetVector<Operation *>> groups(ops_.size());
  for (size_t idx = 0; idx < ops_.size(); ++idx) {
    Operation *op = ops_[idx];
    groups[find(opIdx_.at(op))].insert(op);
  }

  SmallVector<SetVector<Operation *>> fusedGroups;
  for (const auto &group : groups)
    if (group.size() > 1)
      fusedGroups.push_back(group);

  return fusedGroups;
}

FusableBlocks FusableBlockAnalyzer::getFusableBlocks() {
  FusableBlocks res;
  for (const auto &ops : fuseBlock())
    res.emplace_back(ops.getArrayRef(), fusableHelper_);
  return res;
}

bool FusableBlockAnalyzer::sameSet(size_t a, size_t b) {
  return find(a) == find(b);
}

int FusableBlockAnalyzer::find(size_t X) {
  return disjointSet_[X] < 0 ? X : disjointSet_[X] = find(disjointSet_[X]);
}

bool FusableBlockAnalyzer::reInitEdges() {
  for (size_t idx = 0; idx < ops_.size(); ++idx) {
    edge_[idx].clear();
    revEdge_[idx].clear();
  }
  for (size_t i = 0; i < ops_.size(); ++i) {
    size_t idx = find(i);
    Operation *op = ops_[i];
    for (Operation *userOp : op->getUsers()) {
      if (!opIdx_.contains(userOp)) {
        // Check if there is inter block control flow
        continue;
      }
      const size_t nextIdx = find(opIdx_[userOp]);
      if (idx == nextIdx)
        continue;
      // If not exist, then default fusable
      bool existing = (edge_[idx].count(nextIdx) ? edge_[idx][nextIdx] : true);
      edge_[idx].insert(
          {nextIdx, existing && fusableHelper_->isFusable(op, userOp)});
      revEdge_[nextIdx].insert(idx);
    }
  }
  return true;
}

void FusableBlockAnalyzer::mergeEdge(int nodeA, int nodeB) {
  // Remove self loop
  edge_[nodeA].erase(nodeB);
  edge_[nodeB].erase(nodeA);
  // Merge all outdegree of nodeB into nodeA
  for (const auto &[targetNode, isFusable] : edge_[nodeB]) {
    // Make sure reverse edge of the target is available
    assert(revEdge_[targetNode].contains(nodeB));
    // Remove the indegree of the target and update with nodeA
    revEdge_[targetNode].erase(nodeB);
    revEdge_[targetNode].insert(nodeA);
    // If its restricted, then replace
    if (!isFusable) {
      edge_[nodeA][targetNode] = false;
    } else {
      // Else try to put in the value, would return false if non are made
      edge_[nodeA].try_emplace(targetNode, /* isFusable */ true);
    }
  }
  // Clear for B
  edge_[nodeB].clear();

  // Adjust inDegree of B and A, all indegree of B now points to A
  revEdge_[nodeA].erase(nodeB);
  revEdge_[nodeB].erase(nodeA);
  for (int prevValue : revEdge_[nodeB]) {
    auto findMergeB = edge_[prevValue].find(nodeB);
    // Make sure that edge prevValue -> nodeB exists
    assert(findMergeB != edge_[prevValue].end());
    const bool isFusable = findMergeB->second;
    if (!isFusable) {
      // If it was restricted, then make a restriction
      // from prevValue -> nodeA
      edge_[prevValue][nodeA] = false;
    } else {
      // Else, just try emplace if not exist
      edge_[prevValue].try_emplace(nodeA, /* isFusable */ true);
    }
    edge_[prevValue].erase(nodeB);
    // Make sure nodeA has back edge to prevvalue
    revEdge_[nodeA].insert(prevValue);
  }
  // Clear for nodeB
  revEdge_[nodeB].clear();
}

// disjointSet_[x] has 2 different states:
// If its < 0: then -disjointSet_[x] size of the set with head x.
// if its >= 0: then disjointSet_[x] is the parent of the set
//
// For example, the union find data structures:
// Set1: {5, 7 (head), 8, 3};
// Set2: {4 (head), 2, 0};
// Set3: {9, 10 (head)};
// Set4: {1 (head)};
// disjointSet_[9] = 10; (in set3, parent of 9 is 10)
// disjointSet_[1] = -1; (Set4 has size 1)
// disjointSet_[7] = -4; (Set1 has size 4)
// disjointSet_[10] = -2; (Set3 has size 2)
// disjointSet_[0] = 4; (Set1 has size 4)
void FusableBlockAnalyzer::join(int nodeA, int nodeB) {
  // Fetching parents of both nodes
  nodeA = find(nodeA);
  nodeB = find(nodeB);

  // If the same set, then skip merging
  if (nodeA == nodeB)
    return;

  // Take note before swapping
  int preA = nodeA;
  int preB = nodeB;

  // Merge small to large
  // This make Union Find Data Structure to have
  // amortized almost linear time complexity
  if (disjointSet_[nodeA] > disjointSet_[nodeB]) {
    std::swap(nodeA, nodeB);
  }

  mergeEdge(nodeA, nodeB);

  // Both value is the size, move all elements in B to A
  // Size of A gets added to size of B
  disjointSet_[nodeA] += disjointSet_[nodeB];

  // Assign parent of B to A
  disjointSet_[nodeB] = nodeA;

  if (opMaxRank_[nodeA] < 0) {
    opMaxRank_[nodeA] = opMaxRank_[nodeB];
  }

  setType_[nodeA] = fusableHelper_->adjustType(setType_[preA], setType_[preB]);
}

bool FusableBlockAnalyzer::isRestrictedByDependency(int startNode, int endNode,
                                                    bool horizontal) {

  if (!horizontal) {
    auto findEd = edge_[startNode].find(endNode);
    if (findEd == edge_[startNode].end() || findEd->second == false) {
      // If no fusable direct edge is found, then it must be restricted
      return true;
    } else if (edge_[startNode].size() == 1) {
      // Simple direct check, there is no cycle,
      // so it means indegree must equal to 1
      return false;
    }
  } else {
    auto findEd = edge_[startNode].find(endNode);

    if (findEd != edge_[startNode].end()) {
      LLVM_DEBUG(llvm::dbgs() << "This one found a path, but fails before\n";);
      // If fusable direct edge is found, then it must be restricted because it
      // fails the non horizontal test before
      return true;
    }
  }

  SmallVector<int> inDegree(ops_.size());
  BitVector isVisited(ops_.size(), false);

  // Find induced subgraph starting from startNode
  std::queue<int> visited;
  isVisited[startNode] = true;
  visited.push(startNode);

  // Two nodes are mergeable if the target
  // doesn't have any other dependency from the parent
  while (!visited.empty()) {
    int pos = visited.front();
    visited.pop();
    for (const auto &[targetNode, isFusable] : edge_[pos]) {
      (void)isFusable;
      inDegree[targetNode]++;
      if (isVisited.test(targetNode))
        continue;

      isVisited.set(targetNode);
      visited.push(targetNode);
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Got indegree " << inDegree[endNode] << "\n";);
  if (!horizontal)
    return inDegree[endNode] > 1;
  else
    return inDegree[endNode] > 0;
}

} // namespace opfusion
} // namespace mtfusion
} // namespace mlir
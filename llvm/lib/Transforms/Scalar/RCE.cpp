//===- CopyElimination.cpp - Eliminate calls of c++ copy/move ctors ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Ultimate Copy Elision proposal in C++ language
// See http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p0889r1.html
//
// The main idea is to remove unnecessary copy/move constructors if source
// object has no any uses except for call constructor. In this case we replace
// all uses of source object with destination one and remove it.
//
// TODO:
// minor:
//   1. Add pass class for new pass manager
//   2. Add option to disable subobjects
//===----------------------------------------------------------------------===//

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/CodeMoverUtils.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

#define DEBUG_TYPE "rce"

STATISTIC(NumErasedCMCtors, "Number of erased copy/move constructors");
STATISTIC(NumErasedLpads, "Number of erased landing pads");
STATISTIC(NumErasedIns, "Number of erased instructions (cleanup etc)");

static cl::opt<bool> ClNocaptureCtors(
    "nocapture-ctors", cl::init(false),
    cl::desc("Consider that all ctors don't capture their arguments"));

// debugging option to select limit for removing copy/move constructors
static cl::opt<int> ClLimit(
    "rce-limit", cl::init(-1),
    cl::desc("limit per translation unit for ultimate copy elision applying"),
    cl::Hidden);
static int CopyEliminationCnt = 0;

// option is necessary only to collects statistics
static cl::opt<bool> ClDisableCopyElimination(
    "disable-rce-pass", cl::init(false),
    cl::desc("disable llvm redundant copy elimination pass"), cl::ReallyHidden);

// enable additional verification functionality
static cl::opt<bool>
    ClVerify("rce-verify", cl::Hidden,
             cl::desc("Verify function, domtree and post-domtree after rce"),
#ifdef EXPENSIVE_CHECKS
             cl::init(true)
#else
             cl::init(false)
#endif
    );

namespace {

class RedundantCopyEliminationLegacyPass;

// using CopyList = std::vector<SmallVector<Instruction *, 2>>;
using CopyList = SmallVector<Instruction *, 32>;
using IntrinsicVector = SmallVector<Instruction *, 8>;

bool isApplyingLimitReached() {
  if (ClLimit > -1) {
    if (CopyEliminationCnt >= ClLimit) {
      // applying limit was reached
      LLVM_DEBUG(dbgs() << ">> Applying limit (" << CopyEliminationCnt
                        << ") is reached\n");
      return true;
    }
    ++CopyEliminationCnt;
  }

  return false;
}

// create unconditional branch instruction before terminator
auto makeUnconditionalBranch(Instruction *Term) {
  assert(Term->isTerminator() && "must be terminator");
  auto *BB = Term->getParent();
  auto *Target = *successors(BB).begin();

  SmallVector<DominatorTree::UpdateType, 4> Updates;
  for (auto *Succ : successors(BB)) {
    if (Target == Succ)
      continue;
    Succ->removePredecessor(BB, true);
    Updates.push_back({DominatorTree::Delete, BB, Succ});
  }

  BranchInst::Create(Target, Term);

  return Updates;
}

// find all loop headers in function
// auto findLoopHeaders(const Function &F) {
//  SmallVector<std::pair<const BasicBlock *, const BasicBlock *>, 16> Edges;
//  FindFunctionBackedges(F, Edges);
//
//  SmallPtrSet<BasicBlock *, 16> LoopHeaders;
//
//  for (const auto &Edge : Edges)
//    LoopHeaders.insert(const_cast<BasicBlock *>(Edge.second));
//
//  return LoopHeaders;
//}

///
/// BEGIN: helper functions to check dominance and reachability
///
// template <typename Cont>
// bool dominatesAny(const Cont &C, const Instruction *I,
//                  const DominatorTree *DT) {
//  auto CIDominatesI = [DT, I](const Instruction *CI) {
//    return DT->dominates(CI, I);
//  };
//
//  return any_of(C, CIDominatesI);
//}
//
// template <typename Cont>
// bool isDominatedAny(const Instruction *I, const Cont &C,
//                    const DominatorTree *DT) {
//  auto IDominatesCI = [DT, I](const Instruction *CI) {
//    return DT->dominates(I, CI);
//  };
//
//  return any_of(C, IDominatesCI);
//}
//
// template <typename Cont>
// bool reachesAny(const Cont &C, const Instruction *I, const DominatorTree *DT)
// {
//  auto CIReachesI = [DT, I](const Instruction *CI) {
//    return isPotentiallyReachable(CI, I, nullptr, DT);
//  };
//
//  return any_of(C, CIReachesI);
//}
//
// template <typename Cont>
// bool isReachedAny(const Instruction *I, const Cont &C,
//                  const DominatorTree *DT) {
//  auto IReachesCI = [DT, I](const Instruction *CI) {
//    return isPotentiallyReachable(I, CI, nullptr, DT);
//  };
//
//  return any_of(C, IReachesCI);
//}
///
/// END: helper functions to check dominance and reachability
///

///
/// BEGIN: helper functions to check intrinsic type
///
bool isLifetimeStart(const Value &V) {
  if (const auto *II = dyn_cast<IntrinsicInst>(&V))
    return II->getIntrinsicID() == Intrinsic::lifetime_start;

  return false;
}

bool isLifetimeEnd(const Value &V) {
  if (const auto *II = dyn_cast<IntrinsicInst>(&V))
    return II->getIntrinsicID() == Intrinsic::lifetime_end;

  return false;
}

bool isLifetimeStartOrEnd(const Value &V) {
  return isLifetimeStart(V) || isLifetimeEnd(V);
}

bool isCleanupStart(const Value &V) {
  if (const auto *II = dyn_cast<IntrinsicInst>(&V))
    return II->getIntrinsicID() == Intrinsic::cleanup_start;

  return false;
}

bool isCleanupEnd(const Value &V) {
  if (const auto *II = dyn_cast<IntrinsicInst>(&V))
    return II->getIntrinsicID() == Intrinsic::cleanup_end;

  return false;
}

bool isCleanupStartOrEnd(const Value &V) {
  return isCleanupStart(V) || isCleanupEnd(V);
}

bool isCopyStart(const Value &V) {
  if (const auto *II = dyn_cast<IntrinsicInst>(&V))
    return II->getIntrinsicID() == Intrinsic::copy_start;

  return false;
}

bool isCopyEnd(const Value &V) {
  if (const auto *II = dyn_cast<IntrinsicInst>(&V))
    return II->getIntrinsicID() == Intrinsic::copy_end;

  return false;
}

bool isCopyStartOrEnd(const Value &V) { return isCopyStart(V) || isCopyEnd(V); }

// Get operand of copy.start/copy.end intrinsic
Value *getCopyOperand(const Instruction &Copy, unsigned OpNum) {
  assert(isCopyStartOrEnd(Copy));
  // auto *Cast = cast<BitCastInst>(Copy.getOperand(OpNum));
  // return Cast->getOperand(0);
  return Copy.getOperand(OpNum);
}
///
/// END: helper functions to check intrinsic type
///

///
/// BEGIN: helper functions to check instructions properties
///
// NOTE: double-check
bool isTrivial(const Value &V) {
  if (isCleanupStartOrEnd(V) || isCopyStartOrEnd(V) || isLifetimeStartOrEnd(V))
    return true;

  if (const auto *I = dyn_cast<Instruction>(&V))
    return !I->mayReadOrWriteMemory();

  return false;
}
///
/// END: helper functions to check instructions properties
///

///
/// START: helper classes to detect begin/end of copy or cleanup of the object
///

/// @brief helper class for convenient generation comparison (==, < etc)
///        of metadata for cleanup/copy instructions
struct MetaGeneration {
  explicit MetaGeneration(const Instruction *I, unsigned MKind)
      : I(I), MKind(MKind) {}

  const Instruction *I;
  unsigned MKind;
};

MetaGeneration getMetaGen(const Instruction *I, unsigned MKind) {
  return MetaGeneration(I, MKind);
}

bool operator<(const MetaGeneration &MG1, const MetaGeneration &MG2) {
  const auto *I1 = MG1.I;
  const auto *I2 = MG2.I;
  unsigned MKind = MG1.MKind;

  assert((MG1.MKind == MG2.MKind) && "Incorrect meta generation comparison");

  const auto *IM2 = I2->getMetadata(MKind);
  if (!IM2)
    // if the second instruction has no metadata it means
    // that it's less than or equal to the first one
    return false;

  const auto *IM1 = I1->getMetadata(MKind);
  if (!IM1)
    // if the first instruction has no metadata assume that it's less
    return true;

  assert(IM1->getNumOperands() == 1 &&
         IM1->getNumOperands() == IM2->getNumOperands());

  const auto *IMGen1 = mdconst::extract<ConstantInt>(IM1->getOperand(0));
  const auto *IMGen2 = mdconst::extract<ConstantInt>(IM2->getOperand(0));

  return IMGen1->getZExtValue() < IMGen2->getZExtValue();
}

bool operator>=(const MetaGeneration &MG1, const MetaGeneration &MG2) {
  return !(MG1 < MG2);
}

bool operator==(const MetaGeneration &MG1, const MetaGeneration &MG2) {
  const auto *I1 = MG1.I;
  const auto *I2 = MG2.I;
  unsigned MKind = MG1.MKind;

  assert((MG1.MKind == MG2.MKind) && "Incorrect meta generation comparison");

  const auto *IM1 = I1->getMetadata(MKind);
  const auto *IM2 = I2->getMetadata(MKind);

  if (!IM1 && !IM2)
    // both have no metadata
    return true;

  if (!IM1 || !IM2)
    // only one has no no metadata
    return false;

  assert(IM1->getNumOperands() == 1 &&
         IM1->getNumOperands() == IM2->getNumOperands());

  const auto *IMGen1 = mdconst::extract<ConstantInt>(IM1->getOperand(0));
  const auto *IMGen2 = mdconst::extract<ConstantInt>(IM2->getOperand(0));

  return IMGen1->getZExtValue() == IMGen2->getZExtValue();
}

bool operator!=(const MetaGeneration &MG1, const MetaGeneration &MG2) {
  return !(MG1 == MG2);
}

/// @brief functor to check whether the instruction is a copy/cleanup.start or
///        copy/cleanup.end intrinsic with given metadata kind for the given
///        objects
template <Intrinsic::IndependentIntrinsics IntType, unsigned MDKind>
struct IsIntrStartOrEnd {
  explicit IsIntrStartOrEnd(SmallVector<const Instruction *, 2> Objs)
      : Objects(Objs) {}

  bool operator()(const Value *V) const { return operator()(*V); }

  bool operator()(const Value &V) const {
    const auto *II = dyn_cast<IntrinsicInst>(&V);
    if (!II || II->getIntrinsicID() != IntType)
      return false;

    unsigned MK = MDKind;

    // TODO: adapt
#ifndef NDEBUG
    // for (size_t I = 1; I < Objects.size(); ++I)
    //  assert(getMetaGen(Objects[I - 1], MK) == getMetaGen(Objects[I], MK));
#endif

    int OpNum = 0;
    for (const auto *Obj : Objects) {
      if (getMetaGen(Obj, MK) != getMetaGen(II, MK))
        return false;

      if (Obj != II->getOperand(OpNum))
        return false;

      ++OpNum;
    }

    return true;
  }

private:
  SmallVector<const Instruction *, 2> Objects;
};

/// @brief functor to check whether the instruction
///        is an init, copy_init or cleanup for the given objects
template <unsigned MDKind> struct IsInsWithCopyMD {
  explicit IsInsWithCopyMD(SmallVector<const Instruction *, 2> Objs)
      : Objects(Objs) {}

  bool operator()(const Value &V) const {
    const auto *I = dyn_cast<Instruction>(&V);
    if (!I)
      return false;

    unsigned MK = MDKind;

    // TODO: adapt
#ifndef NDEBUG
    // for (size_t J = 1; J < Objects.size(); ++J)
    //   assert(getMetaGen(Objects[J - 1], MK) == getMetaGen(Objects[J], MK));
#endif

    for (const auto *Obj : Objects)
      if (getMetaGen(Obj, MK) >= getMetaGen(I, MK))
        return false;

    return true;
  }

private:
  SmallVector<const Instruction *, 2> Objects;
};

struct IsCleanupStart
    : IsIntrStartOrEnd<Intrinsic::cleanup_start, LLVMContext::MD_cleanup> {
  explicit IsCleanupStart(const Instruction *I) : IsIntrStartOrEnd({I}) {}
};

struct IsCleanupEnd
    : IsIntrStartOrEnd<Intrinsic::cleanup_end, LLVMContext::MD_cleanup> {
  explicit IsCleanupEnd(const Instruction *I) : IsIntrStartOrEnd({I}) {}
};

struct IsCopyStart
    : IsIntrStartOrEnd<Intrinsic::copy_start, LLVMContext::MD_copy_init> {
  explicit IsCopyStart(const Instruction *Dst, const Instruction *Src)
      : IsIntrStartOrEnd({Dst, Src}) {}
};

struct IsCopyEnd
    : IsIntrStartOrEnd<Intrinsic::copy_end, LLVMContext::MD_copy_init> {
  explicit IsCopyEnd(Instruction *Dst, Instruction *Src)
      : IsIntrStartOrEnd({Dst, Src}) {}
};

struct IsInit : IsInsWithCopyMD<LLVMContext::MD_init> {
  explicit IsInit(const Instruction *I) : IsInsWithCopyMD({I}) {}
};

struct IsCopyInit : IsInsWithCopyMD<LLVMContext::MD_copy_init> {
  explicit IsCopyInit(const Instruction *Dst, const Instruction *Src)
      : IsInsWithCopyMD({Dst, Src}) {}
};

struct IsCleanup : IsInsWithCopyMD<LLVMContext::MD_cleanup> {
  explicit IsCleanup(const Instruction *I) : IsInsWithCopyMD({I}) {}
};

/// whether the instruction is any cleanup for the given object
struct IsAnyCleanup {
  explicit IsAnyCleanup(const Instruction *I)
      : CleanupStart(I), Cleanup(I), CleanupEnd(I) {}

  bool operator()(const Value &V) const {
    return CleanupStart(V) || Cleanup(V) || CleanupEnd(V);
  }

private:
  IsCleanupStart CleanupStart;
  IsCleanup Cleanup;
  IsCleanupEnd CleanupEnd;
};
///
/// END: helper classes to detect begin/end of copy or cleanup of the object
///

///
/// BEGIN: helper functions to collect users of the given
///        instruction if they satisfy predicate condition
///
template <typename PredicateT>
bool applyDisjunction(const Value &V, PredicateT &&P) {
  return P(V);
}

template <typename PredicateT, typename... PredicatesT>
bool applyDisjunction(const Value &V, PredicateT &&P, PredicatesT &&... Ps) {
  return P(V) || applyDisjunction(V, std::forward<PredicatesT>(Ps)...);
}

template <typename... PredicatesT>
bool applyPredicateToInstrOrItsUsers(const Instruction &I,
                                     PredicatesT &&... Ps) {
  if (applyDisjunction(I, std::forward<PredicatesT>(Ps)...))
    return true;

  if (!I.getNumUses())
    // Instruction has no users
    return false;

  for (const auto *U : I.users())
    if (!applyDisjunction(*U, Ps...))
      // At least one user doesn't satisfy predicate function
      return false;

  return true;
}

template <typename Cont, typename... PredicatesT>
void collectInstUsers(Instruction &OriginI, Instruction &I, Cont &C,
                      SmallPtrSet<Instruction *, 16> &Visited,
                      PredicatesT &&... Ps) {

  Visited.insert(&I);

  for (auto *U : I.users()) {
    if (auto *UI = dyn_cast<Instruction>(U))
      if (!Visited.count(UI))
        collectInstUsers(OriginI, *UI, C, Visited,
                         std::forward<PredicatesT>(Ps)...);
  }

  if (&OriginI != &I &&
      applyPredicateToInstrOrItsUsers(I, std::forward<PredicatesT>(Ps)...)) {
    LLVM_DEBUG(dbgs() << "  Collect Inst : " << I << "\n");
    C.push_back(&I);
  }
}

template <typename Cont, typename... PredicatesT>
void collectInstUsers(Instruction &I, Cont &C, PredicatesT &&... Ps) {
  SmallPtrSet<Instruction *, 16> Visited;

  collectInstUsers(I, I, C, Visited, std::forward<PredicatesT>(Ps)...);
}
///
/// END: helper functions to collect users of the given
///      instruction if they satisfy predicate condition
///

///
/// BEGIN: helper functions to check that Copy instruction and its arguments
///        or its users satisfy requirements for Copy Elision optimization
///
// NOTE: double-check
bool areSomeInstBeforeOtherInst(const SmallVectorImpl<Instruction *> &Ins1,
                                const SmallVectorImpl<Instruction *> &Ins2,
                                const DominatorTree *DT) {
  for (auto *I1 : Ins1) {
    // Firstly check dominators
    auto I1DominatesI2 = [DT, I1](const Instruction *I2) {
      bool Dominates = DT->dominates(I1, I2); // FIXME: consider loops

      LLVM_DEBUG(dbgs() << "  Inst1 : " << *I1 << "  BB1 : "
                        << I1->getParent()->getName() << "\n  Inst2 : " << *I2
                        << "  BB2 : " << I2->getParent()->getName()
                        << "\n  dominates(I1, I2) : " << Dominates << "\n");
      return Dominates;
    };

    if (any_of(Ins2, I1DominatesI2))
      return true;

    // Then check reachability
    auto I1ReachesI2 = [DT, I1](const Instruction *I2) {
      // TODO: consider loops
      bool Reachable = isPotentiallyReachable(I1, I2, nullptr, DT) &&
                       !isPotentiallyReachable(I2, I1, nullptr, DT);

      LLVM_DEBUG(dbgs() << "  Inst1 : " << *I1 << "  BB1 : "
                        << I1->getParent()->getName() << "\n  Inst2 : " << *I2
                        << "  BB2 : " << I2->getParent()->getName()
                        << "\n  isPotentiallyReachable(I1, I2) : " << Reachable
                        << "\n");
      return Reachable;
    };

    if (any_of(Ins2, I1ReachesI2))
      return true;
  }

  return false;
}

// NOTE: double-check
bool areAllTrivialInsInBetween(Instruction &IBegin, Instruction &IEnd) {
  SmallPtrSet<Instruction *, 32> InBetweenInsts;
  collectInstructionsInBetween(IBegin, IEnd, InBetweenInsts);

  LLVM_DEBUG(dbgs() << "Range begin : " << IBegin << "\n");

  for (const auto *I : InBetweenInsts) {
    LLVM_DEBUG(dbgs() << "  Ins between life ends : " << *I << "\n");

    if (!applyPredicateToInstrOrItsUsers(*I, isTrivial)) {
      LLVM_DEBUG(dbgs() << "  Non trivial ins : " << *I << "\n");
      LLVM_DEBUG(dbgs() << "Range end : " << IEnd << "\n");
      return false;
    }
  }

  LLVM_DEBUG(dbgs() << "Range end : " << IEnd << "\n");
  return true;
}

// NOTE: double-check
bool areAllTrivialInsInBetween(const SmallVectorImpl<Instruction *> &InsBegin,
                               const SmallVectorImpl<Instruction *> &InsEnd,
                               const DominatorTree *DT) {
  for (auto *I1 : InsBegin) {
    for (auto *I2 : InsEnd) {
      if (isPotentiallyReachable(I1, I2, nullptr, DT)) {
        if (!areAllTrivialInsInBetween(*I1, *I2))
          return false;
      }
    }
  }

  return true;
}
///
/// END: helper functions to check that Copy instruction and its arguments
///      or its users satisfy requirements for Copy Elision optimization
///

/// @brief copy.start intrinsic collector
class CopyInfoVisitor : public InstVisitor<CopyInfoVisitor> {
public:
  explicit CopyInfoVisitor(Function &F) : F(F) {}

  void collectCopyInfo() {
    CopyVec.clear();

    for (auto &BB : F)
      visit(BB);
  }

  CopyList &getCopies() { return CopyVec; }

  void visitIntrinsicInst(IntrinsicInst &II) {
    if (!isCopyStart(II))
      return;

    CopyVec.push_back(&II);

    // auto *DstOpnd = getCopyOperand(II, 0);
    // auto *SrcOpnd = getCopyOperand(II, 1);

    // if (CopyOps.count({DstOpnd, SrcOpnd})) {
    // // copy with the same operands is present
    //  CopyVec[CopyOps[{DstOpnd, SrcOpnd}]].push_back(&II);
    //  return;
    //}

    // CopyVec.push_back({&II});
    // CopyOps[{DstOpnd, SrcOpnd}] = CopyVec.size() - 1;
  }

private:
  Function &F;
  CopyList CopyVec;
  // SmallDenseMap<std::pair<Value*, Value*>, size_t> CopyOps;
};

class RedundantCopyEliminationLegacyPass : public FunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid

  RedundantCopyEliminationLegacyPass() : FunctionPass(ID) {
    initializeRedundantCopyEliminationLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F) || ClDisableCopyElimination)
      return false;

    Fn = &F;
    DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    PDT = &getAnalysis<PostDominatorTreeWrapperPass>().getPostDomTree();
    DL = &F.getParent()->getDataLayout();
    // LoopHeaders = findLoopHeaders(F);

    CopyInfoVisitor CV(F);
    CV.collectCopyInfo();

    bool IsApplied = performUltimateCopyElimination(F, CV.getCopies());

    return IsApplied;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<PostDominatorTreeWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();

    AU.addPreserved<PostDominatorTreeWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.addPreserved<GlobalsAAWrapperPass>();
  }

private:
  friend struct CapturesBeforeCopy;

  /// do main work
  bool performUltimateCopyElimination(Function &F, CopyList &Copies);

  /// returns whether copy_init can be deleted or not.
  /// Collects instructions for removing and other info
  ///
  /// SameCopies contains all copies with the same operands,
  /// this case can happen after Jump Threading pass
  bool analyzeCopyAndCollectInfo(Instruction &Copy);

  bool analyzeSrcAndDstUsers();
  bool areAllNonTrivialUsersAfterCopy(const Instruction &I);
  bool areAllNonTrivialUsersBeforeCopy(const Instruction &I);

  /// whether V is CopyStart/CopyEnd instruction or is in between them
  bool isCopyOrInCopy(const Value &V) const;

  /// collects all instructions that must be removed after applying pass
  /// SrcOrDst points to source or destination object for which cleanups
  /// and lifetime ends must be removed (may be nullptr)
  /// returns false if collection is not possible
  bool collectInstructionsToErase();

  /// collects all instructions to DeadInstList in range (start, end)
  /// returns false if collection is not possible
  template <typename StartPredicateT, typename InBetweenPredicateT,
            typename EndPredicateT>
  bool collectInBetweenInsToErase(Instruction &Ins, StartPredicateT IsStart,
                                  InBetweenPredicateT IsInBetween,
                                  EndPredicateT IsEnd);

  /// First return value indicates that analysis was success
  /// and optimization can proceed. The second return value holds a
  /// pointer to instruction for which cleanups need to be removed
  /// or nullptr if nothing to remove
  std::pair<bool, Instruction *> analyzeSrcAndDstLifetimes();

  void replaceDstObjWithSrcObj();

  void removeDeadInstructions(DomTreeUpdater &DTU, CopyList &Copies);

  /// Wrapper for operator<  overloading for convenient comparing sizes of
  /// objects of Alloca instructions
  struct AllocSize {
    explicit AllocSize(const AllocaInst &AI, const DataLayout &DL)
        : AI(AI), DL(DL) {}

    const AllocaInst &AI;
    const DataLayout &DL;
  };

  /// returns special object that holds information about alloca size.
  /// To this object can be applied comparison operators (<,>,== etc)
  /// for comparison of allocated objects sizes
  auto sizeOf(const AllocaInst &AI) const { return AllocSize(AI, *DL); }

  // friend bool operator<(AllocSize ASize1, AllocSize ASize2);
  friend bool operator>(AllocSize ASize1, AllocSize ASize2);
  // friend bool operator<=(AllocSize ASize1, AllocSize ASize2);
  // friend bool operator>=(AllocSize ASize1, AllocSize ASize2);
  // friend bool operator==(AllocSize ASize1, AllocSize ASize2);
  // friend bool operator!=(AllocSize ASize1, AllocSize ASize2);

  /// These functions are simple loggers to print failure/success statistics
  bool failure(const char *Msg = "") const {
    LLVM_DEBUG(dbgs() << ">>> FAILURE: " << Msg << "\n");
    LLVM_DEBUG(dbgs() << "    Module : " << Fn->getParent()->getName() << "\n");
    LLVM_DEBUG(dbgs() << "    Function : " << Fn->getName() << "\n");
    return false;
  }
  bool success(const char *Msg = "") const {
    LLVM_DEBUG(dbgs() << ">>> SUCCESS: " << Msg << "\n");
    LLVM_DEBUG(dbgs() << "    Module : " << Fn->getParent()->getName() << "\n");
    LLVM_DEBUG(dbgs() << "    Function : " << Fn->getName() << "\n");
    return true;
  }

  /// This class is responsible for automatic data cleanup of pass
  struct CtorStateRAII {
    explicit CtorStateRAII(RedundantCopyEliminationLegacyPass &P) : Pass(P) {}

    ~CtorStateRAII() {
      Pass.DeadInstList.clear();
      Pass.DstObj = Pass.SrcObj = nullptr;
      Pass.DstAlloc = Pass.SrcAlloc = nullptr;
      Pass.SrcOrDst = nullptr;
      Pass.CopyStart = Pass.CopyEnd = nullptr;
      // Pass.CopyStarts.clear();
      // Pass.CopyEnds.clear();
    }

  private:
    RedundantCopyEliminationLegacyPass &Pass;
  };

  Function *Fn = nullptr;
  DominatorTree *DT = nullptr;
  PostDominatorTree *PDT = nullptr;
  const DataLayout *DL = nullptr;
  // SmallPtrSet<BasicBlock *, 16> LoopHeaders;

  SmallVector<Instruction *, 128> DeadInstList;
  // Src And Dst objects of copy instruction
  Instruction *SrcObj = nullptr;
  Instruction *DstObj = nullptr;
  AllocaInst *SrcAlloc = nullptr;
  AllocaInst *DstAlloc = nullptr;
  // instruction for which cleanups must be removed
  Instruction *SrcOrDst = nullptr;
  // begin and end of copy instruction for Dst and Src objects
  Instruction *CopyStart = nullptr;
  Instruction *CopyEnd = nullptr;
  // SmallVector<Instruction *, 2> CopyStarts;
  // SmallVector<Instruction *, 2> CopyEnds;
};

/// @brief Only find pointer captures which happen before the given
///        copy.start instruction. Uses the dominator tree to determine
///        whether one instruction is before another.
struct CapturesBeforeCopy : public CaptureTracker {

  explicit CapturesBeforeCopy(const RedundantCopyEliminationLegacyPass &CEP)
      : Captured(false), CEP(CEP) {}

  void tooManyUses() override { Captured = true; }

  bool shouldExplore(const Use *U) override {
    const auto *I = cast<Instruction>(U->getUser());

    IsAnyCleanup AnyCleanup(CEP.SrcAlloc);

    auto IsCopyOrInCopy = [this](const Value &V) {
      return CEP.isCopyOrInCopy(V);
    };

    // TODO: remove ClNocaptureCtors after introducing nocapture attribute
    auto NoCaptureCtors = [this](const Value &V) {
      return ClNocaptureCtors &&
             (IsInit(CEP.SrcAlloc)(V) || IsCopyInit(CEP.DstObj, CEP.SrcObj)(V));
    };

    if (applyPredicateToInstrOrItsUsers(*I, AnyCleanup, isLifetimeStartOrEnd,
                                        IsCopyOrInCopy, NoCaptureCtors))
      return false;

    // if CopyStart dominates I we no need to consider I
    // FIXME: incorrect in loops
    if (CEP.DT->dominates(CEP.CopyStart, I))
      return false;

    // consider Ins only if CopyStart is reachable from it
    return isPotentiallyReachable(I, CEP.CopyStart, nullptr, CEP.DT);

    // if (dominatesAny(CEP.CopyStarts, I, CEP.DT))
    //  return false;

    // return isReachedAny(I, CEP.CopyStarts, CEP.DT);
  }

  bool captured(const Use *U) override {
    LLVM_DEBUG(dbgs() << "  Test for capturing : " << *U->getUser() << "\n");

    if (isa<ReturnInst>(U->getUser()))
      return false;

    // NOTE: double-check
    assert(!IsCleanupStart(CEP.SrcAlloc)(*U) &&
           !IsCleanupEnd(CEP.SrcAlloc)(*U) &&
           !IsCopyStart(CEP.DstObj, CEP.SrcObj)(*U) &&
           !IsCopyEnd(CEP.DstObj, CEP.SrcObj)(*U));

    if (!shouldExplore(U))
      return false;

    Captured = true;
    return true;
  }

  bool Captured;

private:
  const RedundantCopyEliminationLegacyPass &CEP;
};

// bool operator<(RedundantCopyEliminationLegacyPass::AllocSize ASize1,
//               RedundantCopyEliminationLegacyPass::AllocSize ASize2) {
//  auto Size1 = ASize1.AI.getAllocationSizeInBits(ASize1.DL);
//  auto Size2 = ASize2.AI.getAllocationSizeInBits(ASize2.DL);
//  assert((Size1.hasValue() && Size2.hasValue()) && "Types must be sized");
//
//  return Size1.getValue() < Size2.getValue();
//}

bool operator>(RedundantCopyEliminationLegacyPass::AllocSize ASize1,
               RedundantCopyEliminationLegacyPass::AllocSize ASize2) {
  auto Size1 = ASize1.AI.getAllocationSizeInBits(ASize1.DL);
  auto Size2 = ASize2.AI.getAllocationSizeInBits(ASize2.DL);
  assert((Size1.hasValue() && Size2.hasValue()) && "Types must be sized");

  return Size1.getValue() > Size2.getValue();
}

// bool operator<=(RedundantCopyEliminationLegacyPass::AllocSize ASize1,
//                RedundantCopyEliminationLegacyPass::AllocSize ASize2) {
//  return !(ASize1 > ASize2);
//}
//
// bool operator>=(RedundantCopyEliminationLegacyPass::AllocSize ASize1,
//                RedundantCopyEliminationLegacyPass::AllocSize ASize2) {
//  return !(ASize1 < ASize2);
//}

// bool operator==(RedundantCopyEliminationLegacyPass::AllocSize ASize1,
//                RedundantCopyEliminationLegacyPass::AllocSize ASize2) {
//  auto Size1 = ASize1.AI.getAllocationSizeInBits(ASize1.DL);
//  auto Size2 = ASize2.AI.getAllocationSizeInBits(ASize2.DL);
//  assert((Size1.hasValue() && Size2.hasValue()) && "Types must be sized");
//
//  return Size1.getValue() == Size2.getValue();
//}

// bool operator!=(RedundantCopyEliminationLegacyPass::AllocSize ASize1,
//                RedundantCopyEliminationLegacyPass::AllocSize ASize2) {
//  return !(ASize1 == ASize2);
//}

bool RedundantCopyEliminationLegacyPass::analyzeCopyAndCollectInfo(
    /*const SmallVectorImpl<Instruction *>& SameCopies*/ Instruction &Copy) {
  // auto* Copy = SameCopies.front();

  auto *DstOpnd = getCopyOperand(Copy, 0);
  auto *SrcOpnd = getCopyOperand(Copy, 1);

  DstAlloc = dyn_cast<AllocaInst>(getUnderlyingObject(DstOpnd));
  SrcAlloc = dyn_cast<AllocaInst>(getUnderlyingObject(SrcOpnd));

  // consider only automatic variables
  if (!SrcAlloc || !DstAlloc)
    return false;

  // if (DstAlloc->getType() != DstOpnd->getType() &&
  //    SrcAlloc->getType() != SrcOpnd->getType())
  //  return failure("Dst and Src are not full objects");

  if (DstAlloc->getType() != DstOpnd->getType())
    return failure("Dst is not a full object");

  if (sizeOf(*DstAlloc) > sizeOf(*SrcAlloc))
    return failure("Dst is bigger than Src");

  assert(DstOpnd->getType() == SrcOpnd->getType() &&
         "copy operands must have the same types!");

  SrcObj = SrcAlloc->getType() == SrcOpnd->getType()
               ? SrcAlloc
               : cast<Instruction>(SrcOpnd);
  DstObj = DstAlloc->getType() == DstOpnd->getType()
               ? DstAlloc
               : cast<Instruction>(DstOpnd);

  LLVM_DEBUG(dbgs() << "\n------------------------------------\n"
                    << "Function : " << Fn->getName() << "\nCtor : " << Copy
                    << "\nDstAlloc : " << *DstAlloc
                    << "\nSrcAlloc : " << *SrcAlloc << "\nDst: " << *DstObj
                    << "\nSrc : " << *SrcObj << "\n");

  // Such case can take place after applying Jump Threading pass
  // For example we can have the following code:
  //     %src = alloca %SomeType
  //     %dst = alloca %SomeType
  //     ...
  //     some_cond1:
  //       call @copy_ctor(%dst, %src)
  //     ...
  //     some_cond2:
  //       call @copy_ctor(%dst, %src)
  //     ...
  // When this pass applies to the first call it will look like:
  //     %dst = alloca %SomeType
  //     ...
  //     some_cond1:
  //     ...
  //     some_cond2:
  //       call @copy_ctor(%dst, %dst) // It will be after RAUW
  //     ...
  // Therefore this situation is not correct
  assert((DstAlloc != SrcAlloc) && "Invalid copies!");

  // find all copy.start/copy.end for the given Dst and Src
  LLVM_DEBUG(dbgs() << "Check copy.start\n");
  IntrinsicVector CopyStarts;
  collectInstUsers(*SrcAlloc, CopyStarts, IsCopyStart(DstObj, SrcObj));

  LLVM_DEBUG(dbgs() << "Find copy.end\n");
  IntrinsicVector CopyEnds;
  collectInstUsers(*SrcAlloc, CopyEnds, IsCopyEnd(DstObj, SrcObj));

  // TODO: adapt for Jump Threading
  if (CopyEnds.empty() || CopyStarts.size() > 1 || CopyEnds.size() > 1)
    return failure("No copy.end intrinsics or too many!");

  assert(CopyStarts.front() == &Copy);
  CopyStart = &Copy;
  CopyEnd = CopyEnds.front();

#ifndef NDEBUG
  LLVM_DEBUG(dbgs() << "Check all copies\n");
  SmallVector<Instruction *, 4> Copies;
  collectInstUsers(*DstAlloc, Copies, IsCopyStart(DstObj, SrcObj),
                   IsCopyEnd(DstObj, SrcObj));
  assert((Copies.size() == (CopyStarts.size() + CopyEnds.size())) &&
         "Incorrect copies!");
#endif

  if (!analyzeSrcAndDstUsers())
    return false;

  bool Succeed;
  std::tie(Succeed, SrcOrDst) = analyzeSrcAndDstLifetimes();

  if (!Succeed)
    return false;

  if (!collectInstructionsToErase())
    return false;

  return success();
}

bool RedundantCopyEliminationLegacyPass::collectInstructionsToErase() {
  assert(SrcOrDst == SrcAlloc || SrcOrDst == DstAlloc || SrcOrDst == nullptr);

  // TODO: If sub-obj && SrcEndsBeforeDst then need to create new
  //       life ends in Dst life ends location (see positive/h.cc)
  //       (SrcOrDst == SrcAlloc && SrcAlloc != SrcObj)

  // life start of Dst must be removed in all cases
  LLVM_DEBUG(dbgs() << "DstAlloc life starts to erase\n");
  collectInstUsers(*DstAlloc, DeadInstList, isLifetimeStart);

  if (SrcOrDst) {
    // collect all cleanup and lifetime instructions that will be deleted
    IsCleanupStart IsCleanStart(SrcOrDst);
    IsCleanupEnd IsCleanEnd(SrcOrDst);

    LLVM_DEBUG(
        dbgs() << "Dead lifetime.end and start/end.cleanup collection\n");
    collectInstUsers(*SrcOrDst, DeadInstList, isLifetimeEnd,
                     IsCleanupStart(SrcOrDst), IsCleanupEnd(SrcOrDst));

    LLVM_DEBUG(dbgs() << "In between CLEANUP collection for : " << *SrcOrDst
                      << "\n");
    if (!collectInBetweenInsToErase(*SrcOrDst, IsCleanStart,
                                    IsCleanup(SrcOrDst), IsCleanEnd))
      return failure("Impossible cleanup dead inst collection!");
  }

  // collect all copy_init instructions that will be deleted
  LLVM_DEBUG(dbgs() << "In between COPY collection for\n");

  if (!collectInBetweenInsToErase(*SrcAlloc, IsCopyStart(DstObj, SrcObj),
                                  IsCopyInit(DstObj, SrcObj),
                                  IsCopyEnd(DstObj, SrcObj)))
    return failure("Impossible copy dead inst collection!");

#ifndef NDEBUG
  DenseSet<Instruction *> Visited;
  for (auto *DI : DeadInstList) {
    LLVM_DEBUG(dbgs() << "Dead Inst : " << *DI << "\n");
    assert(Visited.insert(DI).second && "DeadInstList has the same inst!");
  }
#endif

  return true;
}

template <typename StartPredicateT, typename InBetweenPredicateT,
          typename EndPredicateT>
bool RedundantCopyEliminationLegacyPass::collectInBetweenInsToErase(
    Instruction &Ins, StartPredicateT IsStart, InBetweenPredicateT IsInBetween,
    EndPredicateT IsEnd) {

  LLVM_DEBUG(dbgs() << " Dead start instr collection\n");
  IntrinsicVector Starts;
  collectInstUsers(Ins, Starts, IsStart);

  LLVM_DEBUG(dbgs() << " Dead end instr collection\n");
  IntrinsicVector Ends;
  collectInstUsers(Ins, Ends, IsEnd);

  auto AddInsToDeadList =
      [this, &IsInBetween](BasicBlock::reverse_iterator StartPoint,
                           BasicBlock::reverse_iterator StopPoint) -> bool {
    for (auto &I : make_range(StartPoint, StopPoint)) {
      DeadInstList.push_back(&I);
      LLVM_DEBUG(dbgs() << "  Ins between : " << I << "\n");
      // assert(IsInBetween(I) && "Ins must be satisfied InBetweenPredicate!");
      // TODO: adapt
      if (!IsInBetween(I))
        return false;
    }

    return true;
  };

  // add all instructions from BB of 'ends' to DeadList.
  // Note that 'end' itself is not added to list because
  // it must be already added
  for (auto *EndI : Ends) {
    auto *EndBB = EndI->getParent();
    LLVM_DEBUG(dbgs() << "  BB : " << EndBB->getName() << "\n");

    auto It = find_if(Starts, [EndBB](const Instruction *I) {
      return I->getParent() == EndBB;
    });

    auto StopPoint =
        (It == Starts.end()) ? EndBB->rend() : (*It)->getReverseIterator();

    if (!AddInsToDeadList(std::next(EndI->getReverseIterator()), StopPoint))
      return false;
  }

  // Mark all 'ends' BB as visited to not process while PO traversal
  // NOTE: double-check
  SmallPtrSet<BasicBlock *, 8> VisitedBB;
  for (auto *EndI : Ends)
    VisitedBB.insert(EndI->getParent());

  // exclude all Starts successors if they are not in copy (this case can
  // be possible if copy ctor is invoke and wasn't inlined)
  for (auto *StartI : Starts)
    for (auto *SuccBB : successors(StartI->getParent()))
      if (!IsInBetween(SuccBB->front()))
        VisitedBB.insert(SuccBB);

  LLVM_DEBUG(
      dbgs()
      << " Start collection between 'start' and 'end' in PO order for ins : "
      << Ins << "\n");

  // Traverse through sub-graph that contains all nodes
  // in [StartsBB, EndsBB) range in PO order and collect all instructions
  // to DeadList. Note that 'start' itself is not added to list because
  // it must be already added
  for (auto *StartI : Starts) {
    auto *StartBB = StartI->getParent();

    for (auto *BB : post_order_ext(StartBB, VisitedBB)) {
      LLVM_DEBUG(dbgs() << "  BB : " << BB->getName() << "\n");
      if (!AddInsToDeadList(BB->rbegin(), (BB == StartBB)
                                              ? StartI->getReverseIterator()
                                              : BB->rend()))
        return false;
    }
  }

  return true;
}

bool RedundantCopyEliminationLegacyPass::isCopyOrInCopy(const Value &V) const {

  if (&V == CopyStart || &V == CopyEnd)
    return true;

  if (const auto *I = dyn_cast<Instruction>(&V)) {
    if (DT->dominates(CopyStart, I) &&
        (DT->dominates(I, CopyEnd) || PDT->dominates(CopyEnd, I))) {
      // FIXME: GVN strips metadata from GEP and bitast. It should be fixed!
      // TODO: adapt
      if (IsCopyInit(DstObj, SrcObj)(*I))
        return true;
    }
  }

  return false;

  // IsCopyInit IsCopy(DstObj, SrcObj);
  // for (const auto *CopyStart : CopyStarts) {
  //  if (&V == CopyStart)
  //    return true;

  //  for (const auto *CopyEnd : CopyEnds) {
  //    if (&V == CopyEnd)
  //      return true;

  //    if (const auto *I = dyn_cast<Instruction>(&V)) {
  //      if (DT->dominates(CopyStart, I) &&
  //          (DT->dominates(I, CopyEnd) || PDT->dominates(CopyEnd, I))) {
  //        if (IsCopy(*I))
  //          return true;
  //      }
  //    }
  //  }
  //}
}

bool RedundantCopyEliminationLegacyPass::analyzeSrcAndDstUsers() {
  // assert(areAllNonTrivialUsersAfterCopy(*DstAlloc) &&
  //       "Dst has users before copy!");
  // TODO: adapt
  if (!areAllNonTrivialUsersAfterCopy(*DstAlloc))
    return failure("Dst has users before copy!");

  // check that Src is not captured before copy
  LLVM_DEBUG(dbgs() << "SrcAlloc capture analysis\n");
  CapturesBeforeCopy CBC(*this);
  PointerMayBeCaptured(SrcAlloc, &CBC, 128); // TODO: limit

  if (CBC.Captured)
    return failure("Src can be captured before copy/move ctor");

  // check that Src or Dst has all non-trivial users before copy
  LLVM_DEBUG(dbgs() << "SrcAlloc and DstAlloc users analysis\n");
  if (!areAllNonTrivialUsersBeforeCopy(*SrcAlloc) &&
      !areAllNonTrivialUsersBeforeCopy(*DstAlloc))
    return failure("Src and Dst have users after copy/move ctor");

  return true;
}

/// whether all users of Inst are after copy_init
bool RedundantCopyEliminationLegacyPass::areAllNonTrivialUsersAfterCopy(
    const Instruction &I) {
  IsAnyCleanup AnyCleanup(&I);

  SmallVector<const User *, 8> WorkList;
  SmallPtrSet<const Instruction *, 8> Visited;

  WorkList.push_back(&I);
  Visited.insert(&I);

  auto IsCopyOrInCopy = [this](const Value &V) { return isCopyOrInCopy(V); };

  while (!WorkList.empty()) {
    const auto *CurI = WorkList.pop_back_val();

    for (const auto *U : CurI->users()) {
      const auto *UI = dyn_cast<Instruction>(U);
      if (!UI)
        continue;

      if (!Visited.insert(UI).second)
        continue;

      if (applyPredicateToInstrOrItsUsers(*UI, IsCopyOrInCopy,
                                          isLifetimeStartOrEnd, AnyCleanup))
        // skip user if it's Copy itself or in case
        // of lifetime/cleanup intrinsics and cleanups
        continue;

      // if (dominatesAny(CopyStarts, UI, DT))
      //  continue;
      if (DT->dominates(CopyStart, UI) || PDT->dominates(UI, CopyStart))
        continue;

      if (isa<BitCastInst>(UI) || isa<GetElementPtrInst>(UI)) {
        // these instructions can be located before Copy but this case
        // is appropriate if users of this instruction are after copy
        WorkList.append(UI->user_begin(), UI->user_end());
        continue;
      }

      // if (isReachedAny(UI, CopyStarts, DT) && !reachesAny(CopyStarts, UI,
      // DT))
      if (isPotentiallyReachable(UI, CopyStart, nullptr, DT) &&
          !isPotentiallyReachable(CopyStart, UI, nullptr, DT)) {
        // NOTE: double-check
        LLVM_DEBUG(dbgs() << "Ins has user before Copy : " << *UI << "\n");
        assert(!IsCleanup(&I)(*UI) && "cleanup must be dominated by copy_init");

        return false;
      }
    }
  }

  return true;
}

/// whether all users of Inst are before copy_init
bool RedundantCopyEliminationLegacyPass::areAllNonTrivialUsersBeforeCopy(
    const Instruction &I) {
  SmallVector<const Instruction *, 4> Worklist;
  SmallPtrSet<const Instruction *, 8> VisitedIns;

  IsAnyCleanup AnyCleanup(&I);

  auto IsCopyOrInCopy = [this](const Value &V) { return isCopyOrInCopy(V); };

  Worklist.push_back(&I);
  VisitedIns.insert(&I);

  LLVM_DEBUG(dbgs() << "Start users analysis for inst : " << I << "\n");

  while (!Worklist.empty()) {
    const auto *CurrI = Worklist.pop_back_val();
    LLVM_DEBUG(dbgs() << "  Cur ins : " << *CurrI << "\n");

    for (const auto *U : CurrI->users()) {
      const auto *UI = dyn_cast<Instruction>(U);
      if (!UI)
        return false;

      if (!VisitedIns.insert(UI).second)
        continue;

      LLVM_DEBUG(dbgs() << "  Cur ins user : " << *UI << "\n");

      if (applyPredicateToInstrOrItsUsers(*UI, isLifetimeStartOrEnd, AnyCleanup,
                                          IsCopyOrInCopy))
        continue;

      Worklist.push_back(UI);

      if (DT->dominates(UI, CopyStart))
        continue;
      // if (isDominatedAny(UI, CopyStarts, DT))
      //  continue;

      bool UserReachesCopy = isPotentiallyReachable(UI, CopyStart, nullptr, DT);
      bool CopyReachesUser = isPotentiallyReachable(CopyStart, UI, nullptr, DT);
      // bool UserReachesCopy = isReachedAny(UI, CopyStarts, DT);
      // bool CopyReachesUser = reachesAny(CopyStarts, UI, DT);

      // the following cases are appropriate to skip the user:
      //  1. the user reaches Copy but Copy don't reach the user (i.e. no loops)
      //  2. the user and Copy aren't reachable from each other
      if ((UserReachesCopy && !CopyReachesUser) ||
          (!UserReachesCopy && !CopyReachesUser))
        // TODO: consider loops in future
        continue;

      assert((UserReachesCopy && CopyReachesUser) ||
             (!UserReachesCopy && CopyReachesUser));
      LLVM_DEBUG(dbgs() << "  User after copy : " << *UI << "\n");

      if (!applyPredicateToInstrOrItsUsers(*UI, isTrivial))
        return false;
    }
  }

  return true;
}

std::pair<bool, Instruction *>
RedundantCopyEliminationLegacyPass::analyzeSrcAndDstLifetimes() {
  LLVM_DEBUG(dbgs() << "SrcAlloc life ends collection\n");
  IntrinsicVector SrcAllEnds;
  IsCleanupEnd IsSrcEnd(SrcAlloc);
  collectInstUsers(*SrcAlloc, SrcAllEnds, isLifetimeEnd, IsSrcEnd);

  LLVM_DEBUG(dbgs() << "DstAlloc life ends collection\n");
  IntrinsicVector DstAllEnds;
  IsCleanupEnd IsDstEnd(DstAlloc);
  collectInstUsers(*DstAlloc, DstAllEnds, isLifetimeEnd, IsDstEnd);

  bool SrcLifeEndsEmpty = SrcAllEnds.empty();
  bool DstLifeEndsEmpty = DstAllEnds.empty();

  if (SrcLifeEndsEmpty && DstLifeEndsEmpty)
    // there are no cleanups and life ends, so we can apply elision safely
    return {true, nullptr};

  if (SrcLifeEndsEmpty || DstLifeEndsEmpty) {
    auto IsSrcOrDstEnd = SrcLifeEndsEmpty ? IsDstEnd : IsSrcEnd;
    auto SrcOrDstLifeEnds = SrcLifeEndsEmpty ? DstAllEnds : SrcAllEnds;

    // check that if Src has no any life ends than Dst
    // has no cleanup.end or vice versa
    if (count_if(SrcOrDstLifeEnds, IsSrcOrDstEnd))
      return {failure("Mismatch cleanup.end intrinsics for Src and Dst objs "
                      "(frontend optimization maybe for no return functions)"),
              nullptr};

    return {true, SrcLifeEndsEmpty ? DstAlloc : SrcAlloc};
  }

  // select only Src cleanup.end or lifetime.end from all life ends
  IntrinsicVector SrcLifeEnds;
  copy_if(SrcAllEnds, std::back_inserter(SrcLifeEnds), IsSrcEnd);
  if (SrcLifeEnds.empty())
    copy_if(SrcAllEnds, std::back_inserter(SrcLifeEnds),
            [](const Instruction *I) { return isLifetimeEnd(*I); });

  // select only Dst cleanup.end or lifetime.end from all life ends
  IntrinsicVector DstLifeEnds;
  copy_if(DstAllEnds, std::back_inserter(DstLifeEnds), IsDstEnd);
  if (DstLifeEnds.empty())
    copy_if(DstAllEnds, std::back_inserter(DstLifeEnds),
            [](const Instruction *I) { return isLifetimeEnd(*I); });

  assert(!DstAllEnds.empty() && !SrcLifeEnds.empty() && "life ends are empty!");

  // TODO: minimize number of checks
  LLVM_DEBUG(dbgs() << "SrcLifeEnds --> DstLifeEnds analysis\n");
  bool SrcEndsBeforeDst =
      areSomeInstBeforeOtherInst(SrcLifeEnds, DstLifeEnds, DT);

  LLVM_DEBUG(dbgs() << "DstLifeEnds --> SrcLifeEnds analysis\n");
  bool DstEndsBeforeSrc =
      areSomeInstBeforeOtherInst(DstLifeEnds, SrcLifeEnds, DT);

  // TODO: adapt
  if (!SrcEndsBeforeDst && !DstEndsBeforeSrc)
    return {failure("Src life ends and Dst life ends are incorrect"), nullptr};
  // assert((SrcEndsBeforeDst || DstEndsBeforeSrc) &&
  //       "Src life ends and Dst life ends are incorrect");

  // TODO: adapt
  if (SrcEndsBeforeDst == DstEndsBeforeSrc)
    return {failure("Src and Dst life ends are reachable from each other"),
            nullptr};
  // NOTE: double-check
  // assert((SrcEndsBeforeDst != DstEndsBeforeSrc) &&
  //       "Src life ends and Dst life ends are reachable from each other");

  // NOTE: double-check
  if (DstEndsBeforeSrc) {
    // if Src lives longer than Dst we must check that there are
    // only trivial instruction between their cleanups
    IntrinsicVector SrcLifeStarts;
    collectInstUsers(*SrcAlloc, SrcLifeStarts, IsCleanupStart(SrcAlloc));

    if (SrcLifeStarts.empty())
      collectInstUsers(*SrcAlloc, SrcLifeStarts, isLifetimeStart);

    assert(!SrcLifeStarts.empty() && "Src life starts are empty!");

    if (!areAllTrivialInsInBetween(DstLifeEnds, SrcLifeStarts, DT))
      return {failure("There are non-trivial instructions"
                      " between Dst life ends and Src life ends"),
              nullptr};
  }

  return {true, SrcEndsBeforeDst ? SrcAlloc : DstAlloc};
}

void RedundantCopyEliminationLegacyPass::replaceDstObjWithSrcObj() {
  auto *SrcType = SrcObj->getType();
  auto *DstType = DstObj->getType();

  LLVM_DEBUG(dbgs() << "Replace Inst (DstObj) : " << *DstObj
                    << "\nWith (SrcObj) : " << *SrcObj << "\n");

  if (SrcType != DstType) {
    // Insert after all Alloca instructions
    auto *InsertBefore = SrcObj->getNextNode();
    while (isa<AllocaInst>(InsertBefore))
      InsertBefore = InsertBefore->getNextNode();

    SrcObj = CastInst::CreateBitOrPointerCast(SrcObj, SrcType, "elision.cast",
                                              InsertBefore);

    LLVM_DEBUG(dbgs() << "SrcType : " << *SrcType << "\nDstType : " << *DstType
                      << "\nCreate Cast (new SrcObj) : " << *SrcObj << "\n");
  }

  // Dst object can have some users like bitcast or GEP,
  // that can be located before Src object
  LLVM_DEBUG(dbgs() << "DstObj users:\n");
  for (auto *U : DstObj->users()) {
    LLVM_DEBUG(dbgs() << "  User : " << *U << "\n");
    if (auto *UI = dyn_cast<Instruction>(U)) {
      if (!isa<AllocaInst>(SrcObj) && !DT->dominates(SrcObj, UI) &&
          isPotentiallyReachable(UI, SrcObj, nullptr, DT)) {
        // assert(!isPotentiallyReachable(SrcObj, UI, nullptr, DT)); FIXME
        assert(isa<BitCastInst>(UI) || isa<GetElementPtrInst>(UI));
        LLVM_DEBUG(dbgs() << "  Move user : " << *U << "\n");

        UI->moveAfter(SrcObj);
      }
    }
  }

  NumErasedIns++;
  DstObj->replaceAllUsesWith(SrcObj);
  DstObj->eraseFromParent();
}

void RedundantCopyEliminationLegacyPass::removeDeadInstructions(
    DomTreeUpdater &DTU, CopyList &Copies) {
  // for (auto *Copy : CopyStarts) {
  //  NumErasedCMCtors++;
  //  LLVM_DEBUG(dbgs() << "Erase copy Ctor : " << *Copy << "\n");
  //  Copy->eraseFromParent();
  //}

  // for (auto *CopyEnd : CopyEnds)
  //  CopyEnd->eraseFromParent();

  for (auto *DeadI : DeadInstList) {
#ifndef NDEBUG
    LLVM_DEBUG(dbgs() << "Erase Inst : " << *DeadI << "\n");
    for (auto *U : DeadI->users())
      LLVM_DEBUG(dbgs() << "  Impossible User : " << *U << "\n");
#endif

    // if (isCopyStart(*DeadI)) {
    //  auto SameCopiesIt = find_if(
    //      Copies, [DeadI](const SmallVectorImpl<Instruction *> &SameCopies) {
    //        return find(SameCopies, DeadI) != SameCopies.end();
    //      });
    //  if (SameCopiesIt != Copies.end())
    //    std::fill(SameCopiesIt->begin(), SameCopiesIt->end(), nullptr);
    //}

    // assert(!DeadI->getNumUses() && "Erased instruction has uses");
    if (DeadI->getNumUses()) {
      // this case can happen after CSE
      LLVM_DEBUG(dbgs() << "  Skip : " << *DeadI << "\n");
      continue;
    }

    if (DeadI->isTerminator() && !DeadI->getNumSuccessors()) {
      // terminator without any successors can't be replaced
      // by unconditional branch, just ignore it.
      LLVM_DEBUG(dbgs() << "  Skip : " << *DeadI << "\n");
      continue;
    }

    // if removed instruction is in Copies vector reset it
    // (don't remove from vector to avoid iterator invalidation)
    if (isCopyStart(*DeadI)) {
      auto CopyIt = find(Copies, DeadI);
      if (CopyIt != Copies.end())
        *CopyIt = nullptr;
    }

    ArrayRef<DominatorTree::UpdateType> Updates;

    if (auto *DeadInvoke = dyn_cast<InvokeInst>(DeadI)) {
      NumErasedLpads++;
      DeadI = changeToCall(DeadInvoke, &DTU);
    } else if (DeadI->isTerminator()) {
      Updates = makeUnconditionalBranch(DeadI);
    }

    DeadI->eraseFromParent();
    NumErasedIns++;

    if (!Updates.empty())
      DTU.applyUpdatesPermissive(Updates);
  }
}

bool RedundantCopyEliminationLegacyPass::performUltimateCopyElimination(
    Function &F, CopyList &Copies) {
  DomTreeUpdater DTU(DT, PDT, DomTreeUpdater::UpdateStrategy::Eager);
  bool Changed = false;

  for (auto *Copy : Copies) {
    if (!Copy) {
      // it was removed already
      LLVM_DEBUG(dbgs() << "SKIP ERASED COPY\n");
      continue;
    }

    CtorStateRAII CleanupObj(*this);

    if (!analyzeCopyAndCollectInfo(*Copy))
      continue;

    if (isApplyingLimitReached())
      return false;

    // TODO remove
    // if (auto *Invoke = dyn_cast<InvokeInst>(Copy)) {
    //  NumErasedLpads++;
    //  Copy = changeToCall(Invoke, &DTU);
    //}

    NumErasedCMCtors++;
    LLVM_DEBUG(dbgs() << "Erase copy Ctor : " << *Copy << "\n");
    assert(Copy == CopyStart);
    CopyStart->eraseFromParent();
    CopyEnd->eraseFromParent();

    // remove cleanup instructions
    removeDeadInstructions(DTU, Copies);

    if (SrcObj != DstObj)
      replaceDstObjWithSrcObj();

    Changed = true;
  }

  if (Changed) {
    removeUnreachableBlocks(F, &DTU);

    if (ClVerify) {
      assert(!verifyFunction(F, &errs()));
      assert(DT->verify());
      assert(PDT->verify());
    }
  }

  return Changed;
}

} // end anonymous namespace

char RedundantCopyEliminationLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(RedundantCopyEliminationLegacyPass, "rce",
                      "Redundant Copy Elimination", false, false)
INITIALIZE_PASS_DEPENDENCY(PostDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(RedundantCopyEliminationLegacyPass, "rce",
                    "Redundant Copy Elimination", false, false)

FunctionPass *llvm::createRCEPass() {
  return new RedundantCopyEliminationLegacyPass();
}

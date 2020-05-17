//===- CopyElision.cpp - Eliminate calls of c++ copy/move ctors ------===//
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
// major:
//   1. Don't remove lifetime intrinsics. Need to select the biggest
//      lifetime or something else
// minor:
//   1. Add pass class for new pass manager
//   2. Add option to disable subobjects
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/LoopInfo.h"
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
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

#define DEBUG_TYPE "copy_elision"

STATISTIC(NumErasedCMCtors, "Number of erased copy/move constructors");
STATISTIC(NumErasedCleanup, "Number of erased cleanups");
STATISTIC(NumErasedLpads, "Number of erased landing pads");
STATISTIC(NumErasedOtherIns, "Number of erased unnecessary instructions");

static cl::opt<bool>
    ClForceCopyElision("force-copy-elision", cl::init(false),
                       cl::desc("apply ultimate copy elision even if source "
                                "object can be written to memory"));

// debugging option to select limit for removing copy/move constructors
static cl::opt<int> ClCopyElisionLimit(
    "copy-elision-limit", cl::init(-1),
    cl::desc("limit per translation unit for ultimate copy elision applying"),
    cl::Hidden);
static int CopyElisionCnt = 0;

namespace {

class CopyElisionLegacyPass;

using CtorVector = SmallVector<CallBase *, 32>;
using DtorVector = SmallVector<Instruction *, 2>;

bool isApplyingLimitReached() {
  if (ClCopyElisionLimit > -1) {
    if (CopyElisionCnt >= ClCopyElisionLimit)
      // applying limit was reached
      return true;
    ++CopyElisionCnt;
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

bool isInit(const Value &V) {
  if (const auto *I = dyn_cast<Instruction>(&V))
    if (I->getMetadata(LLVMContext::MD_init))
      return true;

  return false;
}

bool isCopyInit(const Value &V) {
  if (const auto *CB = dyn_cast<CallBase>(&V)) {
    if (CB->getMetadata(LLVMContext::MD_copy_init))
      // FIXME: don't rely on src argument number
      if (CB->getNumArgOperands() == 2)
        return true;
  }

  return false;
}

bool isCleanup(const Value &V) {
  if (const auto *I = dyn_cast<Instruction>(&V))
    if (I->getMetadata(LLVMContext::MD_cleanup))
      return true;

  return false;
}

bool isLifeTime(const Value &I) {
  if (const auto *II = dyn_cast<IntrinsicInst>(&I))
    if (II->isLifetimeStartOrEnd())
      return true;

  if (isa<GetElementPtrInst>(&I) && onlyUsedByLifetimeMarkers(&I))
    return true;

  if (isa<BitCastInst>(I) && onlyUsedByLifetimeMarkers(&I))
    return true;

  return false;
}

template <typename PredicateT>
bool applyPredicateToInstrOrItsUsers(const Instruction &I, PredicateT p) {
  if (p(I))
    return true;

  if (!I.getNumUses())
    // Instruction has no users
    return false;

  for (const auto *U : I.users())
    if (!p(*U))
      // At least one user doesn't satisfy predicate function
      return false;

  return true;
}

// copy/move constructors collector
class CtorVisitor : public InstVisitor<CtorVisitor> {
public:
  explicit CtorVisitor(Function &F, CtorVector &CtorVec)
      : F(F), CtorVec(CtorVec) {}

  void collectCtors() {
    for (auto &BB : F) {
      visit(BB);
    }
  }

  void visitCallBase(CallBase &CB) {
    // collect all calls/invokes without uses
    if (isCopyInit(CB))
      CtorVec.push_back(&CB);
  }

private:
  Function &F;
  CtorVector &CtorVec;
};

class CopyElisionLegacyPass : public FunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid

  CopyElisionLegacyPass() : FunctionPass(ID) {
    initializeCopyElisionLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;

    DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    DL = &F.getParent()->getDataLayout();

    CtorVector CtorVec;
    CtorVisitor CV(F, CtorVec);

    CV.collectCtors();

    return performUltimateCopyElision(F, CtorVec);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
  }

private:
  /// do main work
  bool performUltimateCopyElision(Function &F, CtorVector &Ctors);

  /// whether copy/move constructor can be deleted or not
  bool canCtorBeElided(CallBase &Ctor);

  /// whether all users of Src object are before copy/move constructor
  bool areAllUsersOfSourceBeforeCtor(const CallBase &Ctor,
                                     const Instruction &Src,
                                     bool ApplyForSubobj = false) const;

  /// whether all users of Dst object are after copy/move constructor
  bool areAllUsersOfDestinationAfterCtor(const CallBase &Ctor,
                                         const Instruction &DestAlloc) const;

  void replaceSrcObjWithDstObj();

  void removeDeadInstructions(DomTreeUpdater &DTU);

  void collectInstructionsToErase(Instruction &I);
  void collectInstructionsToErase(Instruction &I,
                                  SmallPtrSet<Instruction *, 16> &Visited);

  bool areSizesOfTypesTheSame(const Type *TyL, const Type *TyR) const;

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

  friend bool operator<(AllocSize ASize1, AllocSize ASize2);
  friend bool operator>(AllocSize ASize1, AllocSize ASize2);
  friend bool operator<=(AllocSize ASize1, AllocSize ASize2);
  friend bool operator>=(AllocSize ASize1, AllocSize ASize2);
  friend bool operator==(AllocSize ASize1, AllocSize ASize2);
  friend bool operator!=(AllocSize ASize1, AllocSize ASize2);

  /// These functions are simple loggers to print failure/success statistics
  static bool failure(const char *msg = "") {
    LLVM_DEBUG(dbgs() << ">>> FAILURE: " << msg << "\n");
    return false;
  }
  static bool success(const char *msg = "") {
    LLVM_DEBUG(dbgs() << ">>> SUCCESS: " << msg << "\n");
    return false;
  }

  /// This class is responsible for automatic data cleanup of pass
  struct CtorStateRAII {
    explicit CtorStateRAII(CopyElisionLegacyPass &P) : Pass(P) {}

    ~CtorStateRAII() {
      Pass.DeadInstList.clear();
      Pass.DstObj = Pass.SrcObj = nullptr;
    }

  private:
    CopyElisionLegacyPass &Pass;
  };

  DominatorTree *DT = nullptr;
  const DataLayout *DL = nullptr;
  SmallVector<Instruction *, 128> DeadInstList;
  Instruction *SrcObj = nullptr;
  Instruction *DstObj = nullptr;
};

bool operator<(CopyElisionLegacyPass::AllocSize ASize1,
               CopyElisionLegacyPass::AllocSize ASize2) {
  auto Size1 = ASize1.AI.getAllocationSizeInBits(ASize1.DL);
  auto Size2 = ASize2.AI.getAllocationSizeInBits(ASize2.DL);
  assert((Size1.hasValue() && Size2.hasValue()) && "Types must be sized");

  return Size1.getValue() < Size2.getValue();
}

bool operator>(CopyElisionLegacyPass::AllocSize ASize1,
               CopyElisionLegacyPass::AllocSize ASize2) {
  auto Size1 = ASize1.AI.getAllocationSizeInBits(ASize1.DL);
  auto Size2 = ASize2.AI.getAllocationSizeInBits(ASize2.DL);
  assert((Size1.hasValue() && Size2.hasValue()) && "Types must be sized");

  return Size1.getValue() > Size2.getValue();
}

bool operator<=(CopyElisionLegacyPass::AllocSize ASize1,
                CopyElisionLegacyPass::AllocSize ASize2) {
  return !(ASize1 > ASize2);
}

bool operator>=(CopyElisionLegacyPass::AllocSize ASize1,
                CopyElisionLegacyPass::AllocSize ASize2) {
  return !(ASize1 < ASize2);
}

bool operator==(CopyElisionLegacyPass::AllocSize ASize1,
                CopyElisionLegacyPass::AllocSize ASize2) {
  auto Size1 = ASize1.AI.getAllocationSizeInBits(ASize1.DL);
  auto Size2 = ASize2.AI.getAllocationSizeInBits(ASize2.DL);
  assert((Size1.hasValue() && Size2.hasValue()) && "Types must be sized");

  return Size1.getValue() == Size2.getValue();
}

bool operator!=(CopyElisionLegacyPass::AllocSize ASize1,
                CopyElisionLegacyPass::AllocSize ASize2) {
  return !(ASize1 == ASize2);
}

bool CopyElisionLegacyPass::canCtorBeElided(CallBase &Ctor) {
  auto *DstOpnd = Ctor.getOperand(0);
  auto *SrcOpnd = Ctor.getOperand(1);

  auto *DstAlloc = dyn_cast<AllocaInst>(GetUnderlyingObject(DstOpnd, *DL));
  auto *SrcAlloc = dyn_cast<AllocaInst>(GetUnderlyingObject(SrcOpnd, *DL));

  // consider only automatic variables
  if (!SrcAlloc || !DstAlloc)
    return false;

  if (!areSizesOfTypesTheSame(DstOpnd->getType(), DstAlloc->getType()))
    // We can copy from subobject but not to subobject
    return failure("Dst is not full object");

  if (sizeOf(*DstAlloc) > sizeOf(*SrcAlloc))
    return failure("Dst is bigger than Src");

  // Source can be full object or subobject
  auto *Source = sizeOf(*DstAlloc) == sizeOf(*SrcAlloc)
                     ? SrcAlloc
                     : cast<Instruction>(SrcOpnd);

  assert(areSizesOfTypesTheSame(Source->getType(), DstAlloc->getType()) &&
         "ctor operands must have the same types!");

  LLVM_DEBUG(dbgs() << "\n------------------------------------\n"
                    << "Function : " << Ctor.getFunction()->getName()
                    << "\nCtor : " << Ctor << "\nDstAlloc : " << *DstAlloc
                    << "\nSource : " << *Source << "\nSrcAlloc : " << *SrcAlloc
                    << "\n");

  if (DstAlloc == SrcAlloc) {
    // Such case can take place after applying this pass
    // For example we have the following code:
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
    // and we must eliminate second the call
    DstObj = SrcObj = DstAlloc;
    return true;
  }

  if (!areAllUsersOfDestinationAfterCtor(Ctor, *DstAlloc))
    // this case can take place after some optimizations
    // TODO: consider in future in more details
    return failure("Dst has users before copy/move ctor");

  if (!areAllUsersOfSourceBeforeCtor(Ctor, *SrcAlloc, SrcAlloc != Source))
    // Src has users after ctor
    return failure("Src has users after copy/move ctor");

  // collect all cleanup and lifetime instructions that will be deleted
  collectInstructionsToErase(*Source);

  DstObj = DstAlloc;
  SrcObj = Source;

  // TODO replace Dst start lifetime with Src start lifetime
  //  and make assert that Src start lifetime dominates Dst start lifetime
  //
  // We need to remove all lifetime instructions of value because
  // lifetime of value can be increased after replacement
  // TODO: for To we need to figure out the biggest lifetime of From and To
  //       but don't just remove intrinsics
  for (auto *U : DstObj->users())
    if (auto *UI = dyn_cast<Instruction>(U))
      if (isLifeTime(*UI)) {
        for (auto *UU : UI->users())
          DeadInstList.push_back(cast<Instruction>(UU));
        DeadInstList.push_back(UI);
      }

#ifndef NDEBUG
  for (const auto *DI : DeadInstList)
    LLVM_DEBUG(dbgs() << "Dead Inst : " << *DI << "\n");
#endif

  return true;
}

bool CopyElisionLegacyPass::areAllUsersOfSourceBeforeCtor(
    const CallBase &Ctor, const Instruction &Src, bool ApplyForSubobj) const {

  SmallVector<const Instruction *, 4> Worklist;
  SmallPtrSet<const Instruction *, 8> VisitedIns;

  Worklist.push_back(&Src);
  do {
    const auto *CurrI = Worklist.pop_back_val();
    LLVM_DEBUG(dbgs() << "Source : " << *CurrI << "\n");

    for (const auto *U : CurrI->users()) {
      const auto *I = dyn_cast<Instruction>(U);
      if (!I)
        return failure("Src user is not instruction");

      if (I == &Ctor)
        continue;

      LLVM_DEBUG(dbgs() << "  User : " << *I << "\n");

      if (applyPredicateToInstrOrItsUsers(*I, isCleanup) ||
          applyPredicateToInstrOrItsUsers(*I, isLifeTime))
        continue;

      // if instruction dominates ctor then it's always before it
      if (!DT->dominates(I, &Ctor) &&
          isPotentiallyReachable(&Ctor, I, nullptr, DT))
        // Src has users after ctor
        return failure("Src user is reachable from copy/move ctor");

      if (!ApplyForSubobj &&
          (applyPredicateToInstrOrItsUsers(*I, isInit) ||
           applyPredicateToInstrOrItsUsers(*I, isCopyInit)))
        // for non subobject these instructions are safe
        continue;

      if (I->mayWriteToMemory() && !ClForceCopyElision)
        // Src object can be saved somewhere and then
        // restored so we can't apply optimization in this case
        return failure("Src user writes to memory");

      if (VisitedIns.insert(I).second)
        Worklist.push_back(I);
    }

  } while (!Worklist.empty());

  return true;
}

void CopyElisionLegacyPass::replaceSrcObjWithDstObj() {
  if (SrcObj == DstObj)
    return;

  auto *FromType = SrcObj->getType();
  auto *ToType = DstObj->getType();

  LLVM_DEBUG(dbgs() << "Replace Inst (SrcObj) : " << *SrcObj
                    << "\nWith (DstObj) : " << *DstObj << "\n");

  if (FromType != ToType) {
    // Insert after all Alloca instructions
    auto *InsertBefore = DstObj->getNextNode();
    while (isa<AllocaInst>(InsertBefore))
      InsertBefore = InsertBefore->getNextNode();

    DstObj = CastInst::CreateBitOrPointerCast(DstObj, FromType,
                                              "elision.cast", InsertBefore);

    LLVM_DEBUG(dbgs() << "FromType : " << *FromType << "\nToType : " << *ToType
                      << "\nCreate Cast (new DstObj) : " << *DstObj << "\n");
  }

#ifndef NDEBUG
  LLVM_DEBUG(dbgs() << "SrcObj users:\n");
  for (const auto *U : SrcObj->users())
    LLVM_DEBUG(dbgs() << "  User : " << *U << "\n");
#endif

  NumErasedOtherIns++;
  SrcObj->replaceAllUsesWith(DstObj);
  SrcObj->eraseFromParent();
}

bool CopyElisionLegacyPass::areSizesOfTypesTheSame(const Type *TyL,
                                                   const Type *TyR) const {
  const auto *PTyL = dyn_cast<PointerType>(TyL);
  const auto *PTyR = dyn_cast<PointerType>(TyR);

  if (!PTyL || !PTyR)
    return false;

  auto *ETyL = PTyL->getElementType();
  auto *ETyR = PTyR->getElementType();

  return DL->getTypeAllocSize(ETyL) == DL->getTypeAllocSize(ETyR);
}

void CopyElisionLegacyPass::collectInstructionsToErase(Instruction &I) {
  SmallPtrSet<Instruction *, 16> Visited;

  I.setMetadata(LLVMContext::MD_cleanup, nullptr);
  collectInstructionsToErase(I, Visited);
}

void CopyElisionLegacyPass::collectInstructionsToErase(
    Instruction &I, SmallPtrSet<Instruction *, 16> &Visited) {

  Visited.insert(&I);

  for (auto *U : I.users()) {
    if (auto *UI = dyn_cast<Instruction>(U))
      if (!Visited.count(UI))
        collectInstructionsToErase(*UI, Visited);
  }

  if (applyPredicateToInstrOrItsUsers(I, isCleanup)) {
    LLVM_DEBUG(dbgs() << "Cleanup : " << I << "\n");
    DeadInstList.push_back(&I);
  } else if (applyPredicateToInstrOrItsUsers(I, isLifeTime)) {
    LLVM_DEBUG(dbgs() << "LifeTime : " << I << "\n");
    DeadInstList.push_back(&I);
  }
}

bool CopyElisionLegacyPass::areAllUsersOfDestinationAfterCtor(
    const CallBase &Ctor, const Instruction &Dst) const {
  auto *DstOpnd = Ctor.getOperand(0);

  for (const auto *U : Dst.users()) {
    const auto *UI = dyn_cast<Instruction>(U);
    if (!UI)
      continue;

    if (UI == &Ctor || UI == DstOpnd)
      // skip user if it's ctor itself or its dst object
      continue;

    if (applyPredicateToInstrOrItsUsers(*UI, isLifeTime))
      // skip lifetime intrinsic
      continue;

    if (isPotentiallyReachable(&Ctor, UI, nullptr, DT))
      // We can reach some user of Dest from Ctor. Note that
      // isPotentiallyReachable(&Ctor, UI) can return the same result that
      // isPotentiallyReachable(UI, &Ctor). For this reason we need to check
      // the first condition before the second one
      continue;

    if (isPotentiallyReachable(UI, &Ctor, nullptr, DT)) {
      LLVM_DEBUG(dbgs() << "Dest object has copy/move ctor that is reachable"
                           " from user : "
                        << *UI << "\n");
      assert(!isCleanup(*UI) && "cleanup must be dominated by ctor");
      return false;
    }
  }

  return true;
}

void CopyElisionLegacyPass::removeDeadInstructions(DomTreeUpdater &DTU) {
  for (auto *DeadI : DeadInstList) {
    if (isCleanup(*DeadI)) {
      NumErasedCleanup++;
      LLVM_DEBUG(dbgs() << "Erase Cleanup : " << *DeadI << "\n");
    } else {
      NumErasedOtherIns++;
      LLVM_DEBUG(dbgs() << "Erase Inst : " << *DeadI << "\n");
    }

    assert(!DeadI->getNumUses() && "Erased instruction has uses");

    ArrayRef<DominatorTree::UpdateType> Updates;

    if (auto *DeadInvoke = dyn_cast<InvokeInst>(DeadI)) {
      NumErasedLpads++;
      DeadI = changeToCall(DeadInvoke, &DTU);
    } else if (DeadI->isTerminator()) {
      Updates = makeUnconditionalBranch(DeadI);
    }

    DeadI->eraseFromParent();

    if (!Updates.empty())
      DTU.applyUpdatesPermissive(Updates);
  }
}

bool CopyElisionLegacyPass::performUltimateCopyElision(Function &F,
                                                       CtorVector &Ctors) {
  DomTreeUpdater DTU(DT, DomTreeUpdater::UpdateStrategy::Eager);
  bool Changed = false;

  for (auto *Call : Ctors) {
    CtorStateRAII CleanupObj(*this);

    if (!canCtorBeElided(*Call))
      continue;

    if (isApplyingLimitReached())
      return failure("applying limit is reached");

    LLVM_DEBUG(dbgs() << "Erase Ctor : " << *Call << "\n");

    if (auto *Invoke = dyn_cast<InvokeInst>(Call)) {
      NumErasedLpads++;
      Call = changeToCall(Invoke, &DTU);
    }

    // remove copy/move constructor
    NumErasedCMCtors++;
    Call->eraseFromParent();

    // remove cleanup instructions
    removeDeadInstructions(DTU);

    // replace all uses of Src with Dst
    replaceSrcObjWithDstObj();

    Changed = true;
  }

  if (Changed) {
    removeUnreachableBlocks(F, &DTU);
#ifndef NDEBUG
    assert(!verifyFunction(F, &errs()));
    assert(DT->verify(DominatorTree::VerificationLevel::Fast));
#endif
  }

  return Changed;
}

} // end anonymous namespace

char CopyElisionLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(CopyElisionLegacyPass, "copy_elision", "Copy Elision",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(CopyElisionLegacyPass, "copy_elision", "Copy Elision",
                    false, false)

FunctionPass *llvm::createCopyElisionPass() {
  return new CopyElisionLegacyPass();
}

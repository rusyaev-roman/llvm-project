//===- CXXCopyElision.cpp - Eliminate calls of c++ copy/move ctors ------===//
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
// 1. Don't apply for volatile objects
// 2. Add comments and make code cleanup
// 3. Don't remove lifetime intrinsics. Need to select the biggest
//    lifetime or something else
// 4. Add pass class for new pass manager
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

#define DEBUG_TYPE "cxx_copy_elision"

STATISTIC(NumErasedCMCtors, "Number of erased copy/move constructors");
STATISTIC(NumErasedCleanup, "Number of erased cxx cleanups");
STATISTIC(NumErasedLpads, "Number of erased landing pads");
STATISTIC(NumErasedOtherIns, "Number of erased unnecessary instructions");

// debugging option to select limit for removing copy/move constructors
static cl::opt<int> ClCxxCopyElisionLimit(
    "cxx-copy-elision-limit", cl::init(-1),
    cl::desc("limit per translation unit for ultimate copy elision applying"),
    cl::Hidden);
static int CxxCopyElisionCnt = 0;

namespace {

class CXXCopyElisionLegacyPass;

using CtorVector = SmallVector<CallBase *, 32>;
using DtorVector = SmallVector<Instruction *, 2>;

bool isApplyingLimitReached() {
  if (ClCxxCopyElisionLimit > -1) {
    if (CxxCopyElisionCnt >= ClCxxCopyElisionLimit)
      // applying limit was reached
      return true;
    ++CxxCopyElisionCnt;
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

bool isCxxCMCtor(const CallBase &CB) {
  if (CB.getMetadata(LLVMContext::MD_cxx_cm_ctor))
    // FIXME: don't rely on src argument number
    if (CB.getNumArgOperands() == 2)
      return true;

  return false;
}

bool isCxxCleanup(const Value &V) {
  if (const auto *I = dyn_cast<Instruction>(&V))
    if (I->getMetadata(LLVMContext::MD_cxx_cleanup))
      return true;

  return false;
}

bool isLifeTimeInstruction(const Instruction &I) {
  if (const auto *II = dyn_cast<IntrinsicInst>(&I))
    if (II->isLifetimeStartOrEnd())
      return true;

  if (isa<GetElementPtrInst>(&I) && onlyUsedByLifetimeMarkers(&I))
    return true;

  if (isa<BitCastInst>(I) && onlyUsedByLifetimeMarkers(&I))
    return true;

  return false;
}

bool isInstrCxxCleanupOrItsUsersAre(const Instruction &I) {
  if (isCxxCleanup(I))
    return true;

  if (!I.getNumUses())
    // Instruction is not cleanup and has no users
    return false;

  for (const auto *U : I.users())
    if (!isCxxCleanup(*U))
      // At least one user is not cxx cleanup
      return false;

  return true;
}

bool isInstrLifeTimeOrItsUsersAre(const Instruction &I) {
  if (isLifeTimeInstruction(I))
    return true;

  if (!I.getNumUses())
    // Instruction is not lifetime and has no users
    return false;

  for (const auto *U : I.users()) {
    const auto *UI = dyn_cast<Instruction>(U);
    if (!UI)
      return false;

    if (!isLifeTimeInstruction(*UI))
      return false;
  }

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
    if (isCxxCMCtor(CB))
      CtorVec.push_back(&CB);
  }

private:
  Function &F;
  CtorVector &CtorVec;
};

class CXXCopyElisionLegacyPass : public FunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid

  CXXCopyElisionLegacyPass() : FunctionPass(ID) {
    initializeCXXCopyElisionLegacyPassPass(*PassRegistry::getPassRegistry());
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
  bool areAllUsersOfSrcAllocaBeforeCtor(const CallBase &Ctor,
                                        const AllocaInst &SrcAlloc) const;

  /// whether all users of Dst object are after copy/move constructor
  bool areAllUsersOfDstAllocaAfterCtor(const CallBase &Ctor,
                                       const AllocaInst &DestAlloc) const;

  void replaceSrcObjWithDstObj();

  void removeDeadInstructions(DomTreeUpdater& DTU);

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

  //friend bool operator<(AllocSize ASize1, AllocSize ASize2);
  //friend bool operator>(AllocSize ASize1, AllocSize ASize2);
  friend bool operator==(AllocSize ASize1, AllocSize ASize2);
  friend bool operator!=(AllocSize ASize1, AllocSize ASize2);

  /// This class is responsible for automatic data cleanup of pass
  struct CtorStateRAII {
    explicit CtorStateRAII(CXXCopyElisionLegacyPass &P) : Pass(P) {}

    ~CtorStateRAII() {
      Pass.DeadInstList.clear();
      Pass.DstObj = Pass.SrcObj = nullptr;
    }

  private:
    CXXCopyElisionLegacyPass &Pass;
  };

  DominatorTree *DT = nullptr;
  const DataLayout *DL = nullptr;
  SmallVector<Instruction *, 128> DeadInstList;
  Instruction *SrcObj = nullptr;
  Instruction *DstObj = nullptr;
};

//bool operator<(CXXCopyElisionLegacyPass::AllocSize ASize1,
//               CXXCopyElisionLegacyPass::AllocSize ASize2) {
//  auto Size1 = ASize1.AI.getAllocationSizeInBits(ASize1.DL);
//  auto Size2 = ASize2.AI.getAllocationSizeInBits(ASize2.DL);
//  assert((Size1.hasValue() && Size2.hasValue()) && "Types must be sized");
//
//  return Size1.getValue() < Size2.getValue();
//}

//bool operator>(CXXCopyElisionLegacyPass::AllocSize ASize1,
//               CXXCopyElisionLegacyPass::AllocSize ASize2) {
//  auto Size1 = ASize1.AI.getAllocationSizeInBits(ASize1.DL);
//  auto Size2 = ASize2.AI.getAllocationSizeInBits(ASize2.DL);
//  assert((Size1.hasValue() && Size2.hasValue()) && "Types must be sized");
//
//  return Size1.getValue() > Size2.getValue();
//}

bool operator==(CXXCopyElisionLegacyPass::AllocSize ASize1,
                CXXCopyElisionLegacyPass::AllocSize ASize2) {
  auto Size1 = ASize1.AI.getAllocationSizeInBits(ASize1.DL);
  auto Size2 = ASize2.AI.getAllocationSizeInBits(ASize2.DL);
  assert((Size1.hasValue() && Size2.hasValue()) && "Types must be sized");

  return Size1.getValue() == Size2.getValue();
}

bool operator!=(CXXCopyElisionLegacyPass::AllocSize ASize1,
                CXXCopyElisionLegacyPass::AllocSize ASize2) {
  return !(ASize1 == ASize2);
}

bool CXXCopyElisionLegacyPass::canCtorBeElided(CallBase &Ctor) {
  auto *DstOpnd = Ctor.getOperand(0);
  auto *SrcOpnd = Ctor.getOperand(1);

  auto *DstAlloc = dyn_cast<AllocaInst>(GetUnderlyingObject(DstOpnd, *DL));
  auto *SrcAlloc = dyn_cast<AllocaInst>(GetUnderlyingObject(SrcOpnd, *DL));

  // consider only automatic variables
  if (!SrcAlloc || !DstAlloc)
    return false;

  // TODO: consider sub-object cases in future
  if (sizeOf(*SrcAlloc) != sizeOf(*DstAlloc))
    return false;

  if (!areSizesOfTypesTheSame(DstOpnd->getType(), DstAlloc->getType()))
    // construct sub-object of object. TODO: consider this in future
    return false;

  assert(areSizesOfTypesTheSame(SrcOpnd->getType(), SrcAlloc->getType()) &&
         "Dst and Src sizes must be the same");
  LLVM_DEBUG(dbgs() << "\n------------------------------------\n"
                    << "Function : " << Ctor.getFunction()->getName()
                    << "\nCtor : " << Ctor << "\nDstAlloc : " << *DstAlloc
                    << "\nSrcAlloc : " << *SrcAlloc << "\n");

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

  if (!areAllUsersOfDstAllocaAfterCtor(Ctor, *DstAlloc))
    // this case can take place after some optimizations
    // TODO: consider in future in more details
    return false;

  if (!areAllUsersOfSrcAllocaBeforeCtor(Ctor, *SrcAlloc))
    // Src has users after ctor
    return false;

  // collect all cleanup and lifetime instructions that will be deleted
  collectInstructionsToErase(*SrcAlloc);

  DstObj = DstAlloc;
  SrcObj = SrcAlloc;

  // We need to remove all lifetime instructions of value because
  // lifetime of value can be increased after replacement
  // TODO: for To we need to figure out the biggest lifetime of From and To
  //       but don't just remove intrinsics
  for (auto *U : DstObj->users())
    if (auto *UI = dyn_cast<Instruction>(U))
      if (isLifeTimeInstruction(*UI)) {
        for (auto *UU : UI->users())
          DeadInstList.push_back(cast<Instruction>(UU));
        DeadInstList.push_back(UI);
      }

#ifndef NDEBUG
  for (const auto *DI : DeadInstList) {
    LLVM_DEBUG(dbgs() << "*** Dead Inst *** : " << *DI << "\n");
  }
#endif

  return true;
}

bool CXXCopyElisionLegacyPass::areAllUsersOfSrcAllocaBeforeCtor(
    const CallBase &Ctor, const AllocaInst &SrcAlloc) const {
  for (auto *U : SrcAlloc.users()) {
    auto *I = dyn_cast<Instruction>(U);
    if (!I)
      return false;

    if (I == &Ctor)
      continue;

    LLVM_DEBUG(dbgs() << "User : " << *I << "\n");

    if (isInstrCxxCleanupOrItsUsersAre(*I) || isInstrLifeTimeOrItsUsersAre(*I))
      continue;

    if (DT->dominates(I, &Ctor))
      // if instruction dominates ctor then it's always before it
      continue;

    if (isPotentiallyReachable(&Ctor, I, nullptr, DT))
      // Src has users after ctor
      return false;
  }

  return true;
}

void CXXCopyElisionLegacyPass::replaceSrcObjWithDstObj() {
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
                                              "cxx.elision.cast", InsertBefore);

    LLVM_DEBUG(dbgs() << "FromType : " << *FromType
                      << "\nToType : " << *ToType
                      << "\nCreate Cast (new DstObj) : " << *DstObj << "\n");
  }

  NumErasedOtherIns++;
  SrcObj->replaceAllUsesWith(DstObj);
  SrcObj->eraseFromParent();
}

bool CXXCopyElisionLegacyPass::areSizesOfTypesTheSame(const Type *TyL,
                                                      const Type *TyR) const {
  const auto *PTyL = dyn_cast<PointerType>(TyL);
  const auto *PTyR = dyn_cast<PointerType>(TyR);

  if (!PTyL || !PTyR)
    return false;

  auto *ETyL = PTyL->getElementType();
  auto *ETyR = PTyR->getElementType();

  return DL->getTypeAllocSize(ETyL) == DL->getTypeAllocSize(ETyR);
}

void CXXCopyElisionLegacyPass::collectInstructionsToErase(Instruction &I) {
  SmallPtrSet<Instruction *, 16> Visited;

  I.setMetadata(LLVMContext::MD_cxx_cleanup, nullptr);
  collectInstructionsToErase(I, Visited);
}

void CXXCopyElisionLegacyPass::collectInstructionsToErase(
    Instruction &I, SmallPtrSet<Instruction *, 16> &Visited) {

  Visited.insert(&I);

  for (auto *U : I.users()) {
    if (auto *UI = dyn_cast<Instruction>(U))
      if (!Visited.count(UI))
        collectInstructionsToErase(*UI, Visited);
  }

  if (isInstrCxxCleanupOrItsUsersAre(I)) {
    LLVM_DEBUG(dbgs() << "CXX Cleanup : " << I << "\n");
    DeadInstList.push_back(&I);
  } else if (isInstrLifeTimeOrItsUsersAre(I)) {
    LLVM_DEBUG(dbgs() << "LIFETIME : " << I << "\n");
    DeadInstList.push_back(&I);
  }
}

bool CXXCopyElisionLegacyPass::areAllUsersOfDstAllocaAfterCtor(
    const CallBase &Ctor, const AllocaInst &DstAlloc) const {
  auto *DstOpnd = Ctor.getOperand(0);

  for (const auto *U : DstAlloc.users()) {
    const auto *UI = dyn_cast<Instruction>(U);
    if (!UI)
      continue;

    if (UI == &Ctor || UI == DstOpnd)
      // skip user if it's ctor itself or its dst object
      continue;

    if (isInstrLifeTimeOrItsUsersAre(*UI))
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
      assert(!isCxxCleanup(*UI) && "cleanup must be dominated by ctor");
      return false;
    }
  }

  return true;
}

void CXXCopyElisionLegacyPass::removeDeadInstructions(DomTreeUpdater &DTU) {
  for (auto *DeadI : DeadInstList) {
    if (isCxxCleanup(*DeadI)) {
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

bool CXXCopyElisionLegacyPass::performUltimateCopyElision(Function &F,
                                                          CtorVector &Ctors) {
  DomTreeUpdater DTU(DT, DomTreeUpdater::UpdateStrategy::Eager);
  bool Changed = false;

  for (auto *Call : Ctors) {
    CtorStateRAII CleanupObj(*this);

    if (!canCtorBeElided(*Call))
      continue;

    if (isApplyingLimitReached())
      return false;

    LLVM_DEBUG(dbgs() << "*** Erase Ctor *** : " << *Call << "\n");

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

char CXXCopyElisionLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(CXXCopyElisionLegacyPass, "cxx_copy_elision",
                      "CXX Copy Elision", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(CXXCopyElisionLegacyPass, "cxx_copy_elision",
                    "CXX Copy Elision", false, false)

FunctionPass *llvm::createCXXCopyElisionPass() {
  return new CXXCopyElisionLegacyPass();
}

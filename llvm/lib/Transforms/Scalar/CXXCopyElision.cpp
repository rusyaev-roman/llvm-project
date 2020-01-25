//===- CXXCopyElision.cpp - Eliminate calls of c++ copy/move ctors ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TODO
// 1. Add description of pass
// 2. Use DOM tree for speedup?
// 3. Don't apply for volatile objects
// 4. Add comments and make code cleanup
// 5. Don't remove lifetime intrinsics. Need to select the biggest lifetime
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

static cl::opt<int> ClCxxCopyElisionLimit(
    "cxx-copy-elision-limit", cl::init(-1),
    cl::desc("limit per translation unit for ultimate copy elision applying"),
    cl::Hidden);
static int CxxCopyElisionCnt = 0;

namespace {

class CXXCopyElisionLegacyPass;

using CtorVector = SmallVector<CallBase *, 32>;
using DtorVector = SmallVector<Instruction*, 2>;

auto makeUnconditionalBranch(Instruction *I) {
  assert(I->isTerminator() && "must be terminator");
  auto *BB = I->getParent();
  auto *Target = *successors(BB).begin();

  SmallVector<DominatorTree::UpdateType, 4> Updates;
  //(unsigned i = 0, e = I->getNumSuccessors(); i != e; ++i)
  //BasicBlock *Succ = I->getSuccessor(i);
  for (auto *Succ : successors(BB)) {
    if (Target == Succ)
      continue;
    Succ->removePredecessor(BB, true);
    Updates.push_back({DominatorTree::Delete, BB, Succ});
  }

  BranchInst::Create(Target, I);

  return Updates;
}

bool isCxxCMCtor(const CallBase& CB) {
  if (CB.getMetadata(LLVMContext::MD_cxx_cm_ctor))
    if (CB.getNumArgOperands() == 2)
      return true;

  return false;
}

bool isCxxCleanup(const Value& V) {
  if (auto* I = dyn_cast<Instruction>(&V))
    if (I->getMetadata(LLVMContext::MD_cxx_cleanup))
      return true;

  return false;
}

bool isLifeTimeInstruction(const Instruction &I) {
  if (auto *II = dyn_cast<IntrinsicInst>(&I))
    if (II->isLifetimeStartOrEnd())
      return true;

  if (isa<GetElementPtrInst>(&I) && onlyUsedByLifetimeMarkers(&I))
    return true;

  if (isa<BitCastInst>(I) && onlyUsedByLifetimeMarkers(&I))
    return true;

  return false;
}

//DtorVector findImmediateDtors(Instruction &I) {
//  DtorVector Dtors;
//
//  for (auto *U : I.users()) {
//    auto *UI = dyn_cast<Instruction>(U);
//
//    if (!UI)
//      continue;
//
//    if (isCxxCleanup(*UI)) {
//      Dtors.push_back(UI);
//    } else if (auto *UII = dyn_cast<IntrinsicInst>(UI)) {
//      if (UII->getIntrinsicID() == Intrinsic::lifetime_end)
//        Dtors.push_back(UI);
//    } else if ((isa<BitCastInst>(UI) || isa<GetElementPtrInst>(UI))
//               && UI->hasOneUse()) {
//      Value *UU = *UI->user_begin();
//      if (auto* UUI = dyn_cast<IntrinsicInst>(UU)) {
//        if (UUI->getIntrinsicID() == Intrinsic::lifetime_end)
//          Dtors.push_back(UI);
//      } else if (isCxxCleanup(*UU)) {
//          Dtors.push_back(UI);
//      }
//    }
//  }
//
//  return Dtors;
//}

bool isInstrCxxCleanupOrItsUsersAre(const Instruction& I) {
  if (isCxxCleanup(I))
    return true;

  if (!I.getNumUses())
    // Instruction is not cleanup and has no users
    return false;

  for (const auto* U : I.users())
    if (!isCxxCleanup(*U))
      // At least one user is not cxx cleanup
      return false;

  return true;
}

//bool areGEPsEqual(const GetElementPtrInst& GEP1,
//                  const GetElementPtrInst& GEP2) {
//
//  if (GEP1.getNumOperands() != GEP2.getNumOperands())
//    return false;
//
//  for (unsigned i = 1, e = GEP1.getNumOperands(); i != e; ++i) {
//    ConstantInt *CI1 = dyn_cast<ConstantInt>(GEP1.getOperand(i));
//    ConstantInt *CI2 = dyn_cast<ConstantInt>(GEP2.getOperand(i));
//
//    if (!CI1 || !CI2)
//      return false;
//
//    if (CI1->getValue() != CI2->getValue())
//      return false;
//  }
//
//  return true;
//}

bool isInstrLifeTimeOrItsUsersAre(const Instruction &I) {
  if (isLifeTimeInstruction(I))
    return true;

  if (!I.getNumUses())
    // Instruction is not cleanup and has no users
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

//bool isInstructionEqualToSubObject(const Instruction *I,
//                                   const Instruction *SubObj) {
//
//  auto *BI = dyn_cast<BitCastInst>(I);
//  auto *BSub = dyn_cast<BitCastInst>(SubObj);
//
//  if (BI && BSub) {
//    if ((BI->getSrcTy() == BSub->getSrcTy()) &&
//        (BI->getDestTy() == BSub->getDestTy()))
//      return true;
//  }
//
//  auto *IGep = dyn_cast<GetElementPtrInst>(I);
//  auto *SubGep = dyn_cast<GetElementPtrInst>(SubObj);
//
//  if (IGep && SubGep && areGEPsEqual(*IGep, *SubGep))
//    return true;
//
//  return false;
//}

class CtorVisitor : public InstVisitor<CtorVisitor> {
public:
  explicit CtorVisitor(Function& F, CtorVector& CtorVec)
      : F(F), CtorVec(CtorVec) {}

  void collectCtors() {
    for (auto &BB : F) {
      visit(BB);
    }
  }

  void visitCallBase(CallBase& CB) {
    // collect all calls/invokes without uses
    if (isCxxCMCtor(CB))
      CtorVec.push_back(&CB);
  }

private:
  Function& F;
  CtorVector& CtorVec;
};

class CXXCopyElisionLegacyPass : public FunctionPass {
public:
  static char ID;
  CXXCopyElisionLegacyPass() : FunctionPass(ID) {
    initializeCXXCopyElisionLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;

    DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    DL = &F.getParent()->getDataLayout();

    DomTreeUpdater DTU(DT, DomTreeUpdater::UpdateStrategy::Eager);

    CtorVector CtorVec;
    CtorVisitor CV(F, CtorVec);

    CV.collectCtors();

    bool Changed = false;
    for (auto &Call : CtorVec) {
      CtorStateRAII CleanupObj(this);

      if (canCtorBeElided(*Call)) {
        if (ClCxxCopyElisionLimit > -1) {
          if (CxxCopyElisionCnt >= ClCxxCopyElisionLimit)
            return false;
          ++CxxCopyElisionCnt;
        }

        LLVM_DEBUG(dbgs() << "*** Erase Ctor *** : " << *Call << "\n");
        if (auto* Invoke = dyn_cast<InvokeInst>(Call)) {
          NumErasedLpads++;
          Call = changeToCall(Invoke, &DTU);
        }
        NumErasedCMCtors++;
        Call->eraseFromParent();

        for (auto* EI : DeadInstList) {
          if (isCxxCleanup(*EI)) {
            NumErasedCleanup++;
            LLVM_DEBUG(dbgs() << "*** Erase Cleanup *** : " << *EI << "\n");
          } else {
            NumErasedOtherIns++;
            LLVM_DEBUG(dbgs() << "*** Erase Inst *** : " << *EI << "\n");
          }

          assert(!EI->getNumUses() && "Erased instruction has uses");

          if (auto* EII = dyn_cast<InvokeInst>(EI)) {
            NumErasedLpads++;
            EI = changeToCall(EII, &DTU);
            EI->eraseFromParent();
          } else if (EI->isTerminator()) {
            auto Updates = makeUnconditionalBranch(EI);
            EI->eraseFromParent();
            DTU.applyUpdatesPermissive(Updates);
          } else {
            EI->eraseFromParent();
          }
        }

        replaceCopiedObject();
        Changed = true;
      }
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

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
  }

private:
  bool canCtorBeElided(CallBase &Ctor) {
    auto *ObjTo = GetUnderlyingObject(Ctor.getOperand(0), *DL);
    auto *ObjFrom = GetUnderlyingObject(Ctor.getOperand(1), *DL);

    auto* AllocFrom = dyn_cast<AllocaInst>(ObjFrom);
    auto* AllocTo = dyn_cast<AllocaInst>(ObjTo);

    // consider only automatic variables
    if (!AllocTo || !AllocFrom)
      return false;

    // TODO: consider this cases in future
    if (isValueSubObjectOf(*AllocFrom, *AllocTo) ||
        isValueSubObjectOf(*AllocTo, *AllocFrom))
      return false;

    LLVM_DEBUG(dbgs() << "\n------------------------------------\n"
                      << "*** Function *** : " << Ctor.getFunction()->getName()
                      << "\n*** Ctor *** : " << Ctor
                      << "\n*** AllocTo *** : " << *AllocTo
                      << "\n*** AllocFrom *** : " << *AllocFrom << "\n");

    if (AllocTo == AllocFrom) {
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
      To = From = AllocTo;
      return true;
    }

    if (!areUsersOfDestObjectBeforeCtor(Ctor, *AllocTo))
      return false;

    if (!processUsersOfCopiedObject(Ctor, *AllocTo, *AllocFrom))
      return false;

    To = AllocTo;
    From = AllocFrom;

    // We need to remove all lifetime instructions of value because
    // lifetime of value can be increased after replacement
    // TODO: for To we need to figure out the biggest lifetime of From and To
    //       but don't just remove intrinsics
    for (auto *U : To->users())
      if (auto *UI = dyn_cast<Instruction>(U))
        if (isLifeTimeInstruction(*UI)) {
          for (auto *UU : UI->users())
            DeadInstList.push_back(cast<Instruction>(UU));
          DeadInstList.push_back(UI);
        }

#ifndef NDEBUG
    for (const auto* DI : DeadInstList) {
      LLVM_DEBUG(dbgs() << "*** Dead Inst *** : " << *DI << "\n");
    }
#endif

    return true;
  }

  bool processUsersOfCopiedObject(const CallBase &Ctor,
                                  const AllocaInst &AllocTo,
                                  AllocaInst &AllocFrom) {
    // collect all cleanup and lifetime instructions that will be deleted
    collectInstructionsToErase(AllocFrom);

    for (auto *U : AllocFrom.users()) {
      auto *I = dyn_cast<Instruction>(U);
      if (!I)
        return false;

      if (I == &Ctor)
        continue;

      LLVM_DEBUG(dbgs() << "*** User *** : " << *I << "\n");

      if (isInstrCxxCleanupOrItsUsersAre(*I) ||
          isInstrLifeTimeOrItsUsersAre(*I))
        continue;

      if (DT->dominates(I, &Ctor))
        continue;

      if (isPotentiallyReachable(&Ctor, I, nullptr, DT))
        return false;
    }

    return true;
  }

  void replaceCopiedObject() {
    if (To == From)
      return;

    auto* FromType = From->getType();
    auto* ToType = To->getType();

    LLVM_DEBUG(dbgs() << "*** Replace Inst (From) *** : " << *From
                      << "\n*** With (To) *** : " << *To << "\n");

    if (FromType != ToType) {
      To = CastInst::CreateBitOrPointerCast(
          To, FromType, "cxx.elision.cast",
          To->getNextNode());

      LLVM_DEBUG(dbgs() << "*** FromType *** : " << *FromType
                        << "\n*** ToType *** : " << *ToType
                        << "\n*** Create Cast (new To) *** : " << *To << "\n");
    }

    NumErasedOtherIns++;
    From->replaceAllUsesWith(To);
    From->eraseFromParent();
  }

  bool isValueSubObjectOf(const AllocaInst& V, const AllocaInst& Obj) const {
    auto VSize = V.getAllocationSizeInBits(*DL);
    auto ObjSize = Obj.getAllocationSizeInBits(*DL);

    assert((VSize.hasValue() && ObjSize.hasValue()) && "Types must be sized");

    return VSize.getValue() < ObjSize.getValue();
  }

  void collectInstructionsToErase(Instruction &I) {
    SmallPtrSet<Instruction*, 16> Visited;

    I.setMetadata(LLVMContext::MD_cxx_cleanup, nullptr);
    collectCxxCleanupsAndLifetimes(I, Visited);
  }

  void collectCxxCleanupsAndLifetimes(Instruction &I,
                                      SmallPtrSet<Instruction*, 16>& Visited) {
    Visited.insert(&I);

    for (auto *U : I.users()) {
      if (auto *UI = dyn_cast<Instruction>(U))
        if (!Visited.count(UI))
          collectCxxCleanupsAndLifetimes(*UI, Visited);
    }

    if (isInstrCxxCleanupOrItsUsersAre(I)) {
      LLVM_DEBUG(dbgs() << "*** CXX Cleanup *** : " << I << "\n");
      DeadInstList.push_back(&I);
    } else if (isInstrLifeTimeOrItsUsersAre(I)) {
      LLVM_DEBUG(dbgs() << "*** LIFETIME *** : " << I << "\n");
      DeadInstList.push_back(&I);
    }
  }

  bool areUsersOfDestObjectBeforeCtor(const CallBase &Ctor,
                                      const Instruction &Dest) {
    for (auto *U : Dest.users()) {
      if (auto *UI = dyn_cast<Instruction>(U)) {
        if ((UI == &Ctor) || isInstrLifeTimeOrItsUsersAre(*UI))
          continue;

        //DT->dominates(&Ctor, UI)
        if (isPotentiallyReachable(&Ctor, UI, nullptr, DT))
          continue;

        if (isPotentiallyReachable(UI, &Ctor, nullptr, DT)) {
          LLVM_DEBUG(dbgs() << "*** AllocTo has copy ctor that is reachable"
                               " from user *** : " << *UI << "\n");
          assert(!isCxxCleanup(*UI) && "cleanup must be dominated by ctor");
          return false;
        }
      }
    }

    return true;
  }

  /// This is responsible for automatic data cleanup of pass
  struct CtorStateRAII {
    explicit CtorStateRAII(CXXCopyElisionLegacyPass *P) : Pass(P) {}

    ~CtorStateRAII() {
      Pass->DeadInstList.clear();
      Pass->To = Pass->From = nullptr;
    }

  private:
    CXXCopyElisionLegacyPass *Pass;
  };

  DominatorTree *DT = nullptr;
  const DataLayout* DL = nullptr;
  SmallVector<Instruction *, 128> DeadInstList;
  Instruction *From = nullptr;
  Instruction *To = nullptr;
};

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

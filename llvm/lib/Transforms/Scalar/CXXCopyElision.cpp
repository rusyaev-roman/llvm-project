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
// 5. Add additional attribute for clean-ups of replaced object
//    (instead of dtor). Propagate this attribute at inline pass
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/InitializePasses.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

#define DEBUG_TYPE "cxx_copy_elision"

STATISTIC(NumErasedCMCtors, "Number of erased copy/move constructors");
STATISTIC(NumErasedCleanup, "Number of erased cleanups");
STATISTIC(NumErasedIns, "Number of all erased unnecessary instructions");

namespace {

class CXXCopyElisionLegacyPass;

using CtorVector = SmallVector<CallBase *, 32>;
using DtorVector = SmallVector<Instruction*, 2>;


void makeUnreachable(Instruction& I) {
  assert(I.isTerminator() && "must be terminator");
  BasicBlock *BB = I.getParent();

  for (auto *Succ : successors(BB))
    Succ->removePredecessor(BB);

  changeToUnreachable(&I, false);
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

DtorVector findImmediateDtors(Instruction &I) {
  DtorVector Dtors;

  for (auto *U : I.users()) {
    auto *UI = dyn_cast<Instruction>(U);

    if (!UI)
      continue;

    if (isCxxCleanup(*UI)) {
      Dtors.push_back(UI);
    } else if (auto *UII = dyn_cast<IntrinsicInst>(UI)) {
      if (UII->getIntrinsicID() == Intrinsic::lifetime_end)
        Dtors.push_back(UI);
    } else if ((isa<BitCastInst>(UI) || isa<GetElementPtrInst>(UI))
               && UI->hasOneUse()) {
      Value *UU = *UI->user_begin();
      if (auto* UUI = dyn_cast<IntrinsicInst>(UU)) {
        if (UUI->getIntrinsicID() == Intrinsic::lifetime_end)
          Dtors.push_back(UI);
      } else if (isCxxCleanup(*UU)) {
          Dtors.push_back(UI);
      }
    }
  }

  return Dtors;
}

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

bool isCtorEliminationSafe(const Instruction& Ctor,
                           const Instruction& ObjUse,
                           const DtorVector& SubObjDtors) {
  if (isPotentiallyReachable(&Ctor, &ObjUse)) {
    if (SubObjDtors.empty())
      return false;

    for (const auto* Dtor : SubObjDtors) {
      assert(isPotentiallyReachable(&Ctor, Dtor) &&
             "dtor is not reached by copy/move ctor");

      // check that this instruction is not sub-object destructor
      auto *GEP = dyn_cast<GetElementPtrInst>(&ObjUse);
      if (GEP && GEP->hasOneUse() && isCxxCleanup(*(*GEP->users().begin())))
        continue;

      if (!isPotentiallyReachable(Dtor, &ObjUse))
        return false;
    }
  }

  return true;
}

bool areGEPsEqual(const GetElementPtrInst& GEP1,
                  const GetElementPtrInst& GEP2) {

  if (GEP1.getNumOperands() != GEP2.getNumOperands())
    return false;

  for (unsigned i = 1, e = GEP1.getNumOperands(); i != e; ++i) {
    ConstantInt *CI1 = dyn_cast<ConstantInt>(GEP1.getOperand(i));
    ConstantInt *CI2 = dyn_cast<ConstantInt>(GEP2.getOperand(i));

    if (!CI1 || !CI2)
      return false;

    if (CI1->getValue() != CI2->getValue())
      return false;
  }

  return true;
}

void collectCxxCleanups(Instruction *I,
                        SmallPtrSet<Instruction*, 4>& Visited,
                        SmallVector<Instruction*, 32>& Result) {
  if (!Visited.insert(I).second)
    return;

  for (auto *U : I->users())
    if (auto *UI = dyn_cast<Instruction>(U))
      collectCxxCleanups(UI, Visited, Result);

  if (isInstrCxxCleanupOrItsUsersAre(*I)) {
    LLVM_DEBUG(dbgs() << "*** CXX Cleanup *** : " << *I << "\n");
    Result.push_back(I);
  }
}

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
    // TODO: need additional checks for invoke
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

    DL = &F.getParent()->getDataLayout();

    CtorVector CtorVec;
    CtorVisitor CV(F, CtorVec);

    CV.collectCtors();

    bool Changed = false;
    for (auto &Call : CtorVec) {
      if (canCtorBeElided(*Call)) {
        DeadInstList.push_back(Call);
        NumErasedCMCtors++;

        while (!DeadInstList.empty()) {
          auto *EI = DeadInstList.pop_back_val();

          if (isCxxCleanup(*EI))
            NumErasedCleanup++;
          else
            NumErasedIns++;

          LLVM_DEBUG(dbgs() << "*** Erase Inst *** : " << *EI << "\n");
          assert(!EI->getNumUses() && "Erased instruction has uses");

          if (auto* EII = dyn_cast<InvokeInst>(EI))
            EI = changeToCall(EII);

          if (EI->isTerminator())
            makeUnreachable(*EI);
          else
            EI->eraseFromParent();
        }

        replaceCopiedObject();
        Changed = true;
      }
    }

    if (Changed) {
      removeUnreachableBlocks(F);
      LLVM_DEBUG(dbgs() << "====================================\n"
                        << "*** Function was *** : " << F.getName()
                        << "\n====================================\n\n");
    }

    return Changed;
  }

private:
  bool canCtorBeElided(CallBase &Ctor) {
    auto *ObjTo = GetUnderlyingObject(Ctor.getOperand(0), *DL);
    auto *ObjFrom = GetUnderlyingObject(Ctor.getOperand(1), *DL);
    auto *ImmFrom = Ctor.getOperand(1);

    auto* AllocFrom = dyn_cast<AllocaInst>(ObjFrom);
    auto* AllocTo = dyn_cast<AllocaInst>(ObjTo);

    // consider only automatic variables
    if (!AllocTo || !AllocFrom)
      return false;

    // TODO: consider this case in future
    if (isValueSubObjectOf(*AllocFrom, *AllocTo))
      return false;

    LLVM_DEBUG(dbgs() << "*** Ctor *** : " << Ctor
                      << "\n*** AllocTo *** : " << *AllocTo
                      << "\n*** AllocFrom *** : " << *AllocFrom
                      << "\n*** ImmFrom *** : " << *ImmFrom << "\n");

    auto *ImmFromIns = cast<Instruction>(ImmFrom);

    DtorVector ImmDtors;
    if (ImmFrom != AllocFrom)
      ImmDtors = findImmediateDtors(*ImmFromIns);

    if (!processUsersOfCopiedObject(Ctor, *AllocTo, *AllocFrom,
                                    *ImmFromIns, ImmDtors)) {
      DeadInstList.clear();
      return false;
    }

    To = AllocTo;
    if (isValueSubObjectOf(*AllocTo, *AllocFrom)) {
      LLVM_DEBUG(dbgs() << "*** AllocTo is sub-object of AllocFrom ***\n");
      From = cast<Instruction>(ImmFrom);
    } else {
      assert((AllocFrom->getAllocationSizeInBits(*DL) ==
              AllocTo->getAllocationSizeInBits(*DL)) &&
             "Size of types is different");
      From = AllocFrom;
    }

    if (!ImmDtors.empty())
      DeadInstList.append(ImmDtors.begin(), ImmDtors.end());

    // We need to remove all lifetime instructions of value because
    // lifetime of value can be increased after replacement
    addLifeTimesToDeadList(*To);

#ifndef NDEBUG
    for (const auto* DI : DeadInstList) {
      LLVM_DEBUG(dbgs() << "*** Dead Inst *** : " << *DI << "\n");
    }
#endif

    return true;
  }

  bool processUsersOfCopiedObject(const CallBase &Ctor,
                                  const AllocaInst &AllocTo,
                                  AllocaInst &AllocFrom,
                                  const Instruction& ImmFrom,
                                  const DtorVector &ImmDtors) {
    // collect all cleanup instructions
    collectCxxCleanupsToDeadList(AllocFrom);

    for (auto *U : AllocFrom.users()) {
      auto *I = dyn_cast<Instruction>(U);
      if (!I)
        return false;

      if (I == &Ctor)
        continue;

      LLVM_DEBUG(dbgs() << "*** User *** : " << *U << "\n");

      if (isInstrCxxCleanupOrItsUsersAre(*I))
        continue;

      if (isLifeTimeInstruction(*I) &&
          isa<GetElementPtrInst>(I) && isa<GetElementPtrInst>(ImmFrom) &&
          !areGEPsEqual(cast<GetElementPtrInst>(*I),
                        cast<GetElementPtrInst>(ImmFrom)))
        // erase lifetime markers only for current sub-object
        continue;

      // collect all lifetime instructions
      if (addLifeTimesToDeadList(*I))
        continue;

      if (!isCtorEliminationSafe(Ctor, *I, ImmDtors))
        return false;
    }

    return true;
  }

  void replaceCopiedObject() {
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

    NumErasedIns++;
    From->replaceAllUsesWith(To);
    From->eraseFromParent();
  }

  bool isValueSubObjectOf(const AllocaInst& V, const AllocaInst& Obj) const {
    auto VSize = V.getAllocationSizeInBits(*DL);
    auto ObjSize = Obj.getAllocationSizeInBits(*DL);

    assert((VSize.hasValue() && ObjSize.hasValue()) && "Types must be sized");

    return VSize.getValue() < ObjSize.getValue();
  }

  bool addLifeTimesToDeadList(Instruction& I) {
    bool IsAdded = false;

    if (isLifeTimeInstruction(I)) {
      DeadInstList.push_back(&I);
      IsAdded = true;
    }

    for (auto *U : I.users()) {
      auto *UI = dyn_cast<Instruction>(U);
      if (!UI)
        continue;

      if (isLifeTimeInstruction(*UI)) {
        IsAdded = true;
        DeadInstList.push_back(UI);
        for (auto *UU : UI->users())
          DeadInstList.push_back(cast<Instruction>(UU));
      }
    }

    return IsAdded;
  }

  void collectCxxCleanupsToDeadList(Instruction &I) {
    SmallPtrSet<Instruction*, 4> Visited;
    SmallVector<Instruction*, 32> WorkList;

    I.setMetadata(LLVMContext::MD_cxx_cleanup, nullptr);
    collectCxxCleanups(&I, Visited, WorkList);

    while (!WorkList.empty())
      DeadInstList.push_back(WorkList.pop_back_val());
  }

  const DataLayout* DL = nullptr;

  SmallVector<Instruction *, 32> DeadInstList;
  Instruction *From = nullptr;
  Instruction *To = nullptr;
};

} // end anonymous namespace

char CXXCopyElisionLegacyPass::ID = 0;

INITIALIZE_PASS(CXXCopyElisionLegacyPass, "cxx_copy_elision",
                "CXX Copy Elision", false, false)

FunctionPass *llvm::createCXXCopyElisionPass() {
  return new CXXCopyElisionLegacyPass();
}

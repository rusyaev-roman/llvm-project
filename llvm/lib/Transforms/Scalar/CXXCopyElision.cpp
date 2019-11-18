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
STATISTIC(NumErasedDtors, "Number of erased destructors");
STATISTIC(NumErasedIns, "Number of all erased unnecessary instructions");

namespace {

class CXXCopyElisionLegacyPass;

using CtorVector = SmallVector<CallBase *, 32>;
using DtorVector = SmallVector<Instruction*, 2>;

bool isCxxCMCtor(const CallBase& CB) {
  return CB.isCxxCMCtorOrDtor() && (CB.getNumArgOperands() == 2);
}

bool isCxxDtor(const CallBase& CB) {
  return CB.isCxxCMCtorOrDtor() && (CB.getNumArgOperands() == 1);
}

bool isCxxDtor(const Instruction& I) {
  if (auto* CB = dyn_cast<CallBase>(&I)) {
    return isCxxDtor(*CB);
  }
  return false;
}

bool isCxxDtor(const Value& V) {
  if (auto* I = dyn_cast<Instruction>(&V)) {
    return isCxxDtor(*I);
  }
  return false;
}

//bool isCxxCMCtorOrDtor(const CallBase& CB) {
//  return isCxxCMCtor(CB) || isCxxDtor(CB);
//}

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

DtorVector findImmediateCopiedDtors(Instruction& I) {
  DtorVector Dtors;

  for (auto* IU : I.users()) {
    auto* DI = dyn_cast<Instruction>(IU);

    if (!DI)
      continue;

    if (isCxxDtor(*DI)) {
      Dtors.push_back(DI);
    } else if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
      if (II->getIntrinsicID() == Intrinsic::lifetime_end)
        Dtors.push_back(DI);
    } else if ((isa<BitCastInst>(DI) || isa<GetElementPtrInst>(DI))
               && DI->hasOneUse()) {
      auto U = DI->users().begin();
      if (auto* UII = dyn_cast<IntrinsicInst>(*U)) {
        if (UII->getIntrinsicID() == Intrinsic::lifetime_end)
          Dtors.push_back(DI);
      } else if (auto* UI = dyn_cast<Instruction>(*U)) {
        if (isCxxDtor(*UI))
          Dtors.push_back(DI);
      }
    }
  }

  return Dtors;
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
      if (GEP && GEP->hasOneUse() && isCxxDtor(*(*GEP->users().begin())))
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

          NumErasedIns++;
          if (isCxxDtor(*EI))
            NumErasedDtors++;

          LLVM_DEBUG(dbgs() << "*** Erase Inst *** : " << *EI << "\n");
          assert(!EI->getNumUses() && "Erased instruction has uses");

          if (auto* EII = dyn_cast<InvokeInst>(EI))
            EI = changeToCall(EII);

          EI->eraseFromParent();
        }

        replaceCopiedObject();

        Changed = true;
      }
    }

    if (Changed)
      LLVM_DEBUG(dbgs() << "====================================\n"
                        << "*** Function was *** : " << F.getName()
                        << "\n====================================\n\n");

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
                      << "\n*** AllocaFrom *** : " << *AllocFrom
                      << "\n*** ImmFrom *** : " << *ImmFrom << "\n");

    DtorVector ImmDtors;
    if (ImmFrom != AllocFrom)
      ImmDtors = findImmediateCopiedDtors(cast<Instruction>(*ImmFrom));

    SmallVector<Instruction *, 32> InstList;
    for (auto* U : AllocFrom->users()) {
      auto *I = dyn_cast<Instruction>(U);
      LLVM_DEBUG(dbgs() << "*** User *** : " << *U << "\n");

      if (!I)
        return false;

      if (I == &Ctor)
        continue;

      // remove all lifetime instructions later
      if (isCxxDtor(*I) || isLifeTimeInstruction(*I)) {
        if (isa<GetElementPtrInst>(I) && isa<GetElementPtrInst>(ImmFrom)
            && !areGEPsEqual(cast<GetElementPtrInst>(*I),
                             cast<GetElementPtrInst>(*ImmFrom)))
          // erase lifetime markers only for current sub-object
          continue;

        InstList.push_back(I);
        for (auto* UI : I->users())
          InstList.push_back(cast<Instruction>(UI));
        continue;
      }

      if (!isCtorEliminationSafe(Ctor, *I, ImmDtors))
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

    DeadInstList = std::move(InstList);
    if (!ImmDtors.empty())
      DeadInstList.insert(DeadInstList.end(), ImmDtors.begin(),
                          ImmDtors.end());

    // We need to remove all lifetime instructions of value because
    // lifetime of value can be increased after replacement
    addLifeTimeUsersToDeadList(*To);

#ifndef NDEBUG
    for (const auto* DI : DeadInstList)
      LLVM_DEBUG(dbgs() << "*** Dead Inst *** : " << *DI << "\n");
#endif

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

  void addLifeTimeUsersToDeadList(Instruction& I) {
    for (auto *U : I.users()) {
      auto *IU = dyn_cast<Instruction>(U);
      if (!IU)
        continue;

      if (isLifeTimeInstruction(*IU)) {
        DeadInstList.push_back(IU);
        for (auto *UU : IU->users())
          DeadInstList.push_back(cast<Instruction>(UU));
      }
    }
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

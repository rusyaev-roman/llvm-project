//===- FunctionCallsElision.cpp - Eliminate function calls ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TODO
//
//===----------------------------------------------------------------------===//

#include "llvm/InitializePasses.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include <llvm/IR/InstrTypes.h>
#include "llvm/IR/InstVisitor.h"

using namespace llvm;

namespace {

class FunctionCallsEliminationLegacyPass;

struct CallsEliminationVisitor : public InstVisitor<CallsEliminationVisitor> {
  explicit CallsEliminationVisitor(Function& F) : F(F) {}

  void collectCalls() {
    for (auto &BB : F) {
      visit(BB);
    }
  }

  void visitCallBase(CallBase& CB) {
    // collect all calls/invokes without uses
    // TODO: need additional checks for invoke
    if (CB.canBeElided() && !CB.getNumUses())
      CallsVec.push_back(&CB);
  }

  Function& F;
  SmallVector<CallBase *, 32> CallsVec;
};

class FunctionCallsEliminationLegacyPass : public FunctionPass {
public:
  static char ID;
  FunctionCallsEliminationLegacyPass() : FunctionPass(ID) {
    initializeFunctionCallsEliminationLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;

    CallsEliminationVisitor CEV(F);

    CEV.collectCalls();

    bool changed = false;

    for (auto &Call : CEV.CallsVec) {
      if (canCallBeElided(*Call)) {
        ErasedList.push_back(Call);

        while (!ErasedList.empty()) {
          auto *EI = ErasedList.pop_back_val();

          if (!EI->getNumUses())
            EI->eraseFromParent();
        }
      }
    }

    return changed;
  }

private:
  bool canCallBeElided(Instruction &I) {
    for (auto &O : I.operands()) {
      auto *OI = dyn_cast<Instruction>(O);

      if (!OI)
        return true;

      if (!isValidCallOperand(*OI))
        return false;

      if (canCallBeElided(*OI))
        ErasedList.push_back(OI);
    }

    return true;
  }

  bool isValidCallOperand(const Instruction &I) {
    if (auto *II = dyn_cast<IntrinsicInst>(&I))
      if (!II->isLifetimeStartOrEnd())
        return false;

    if (auto* CB = dyn_cast<CallBase>(&I))
      if (!CB->canBeElided())
        return false;

    return isa<BitCastInst>(I) || isa<GetElementPtrInst>(I) ||
           isa<AllocaInst>(I);
  }

  SmallVector<Instruction *, 128> ErasedList;
};

} // end anonymous namespace

char FunctionCallsEliminationLegacyPass::ID = 0;

INITIALIZE_PASS(FunctionCallsEliminationLegacyPass, "calls_elimination",
                "Function Calls Elimination", false, false)

FunctionPass *llvm::createFunctionCallsEliminationPass() {
  return new FunctionCallsEliminationLegacyPass();
}
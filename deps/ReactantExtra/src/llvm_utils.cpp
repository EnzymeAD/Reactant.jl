#include "llvm/Support/ExtensibleRTTI.h"

extern "C" void reactant_generic_llvm_rtti_root_dtor(llvm::RTTIRoot* root) {
    delete root;
}

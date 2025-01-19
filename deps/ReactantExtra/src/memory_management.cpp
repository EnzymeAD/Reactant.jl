#include "llvm/Support/ExtensibleRTTI.h"
#include "src/memory_management.hpp"
#include <map>
#include <memory>

std::map<void*, std::shared_ptr<void>> captured_shared_ptr;

extern "C" void reactant_release_shared(void* ptr) {
    captured_shared_ptr.erase(ptr);
}

extern "C" void reactant_capture_shared(std::shared_ptr<void>& ptr) {
    captured_shared_ptr[ptr.get()] = ptr;
}

// `map::contains` was introduced in C++20
extern "C" bool reactant_contains_shared(void* ptr) {
    return captured_shared_ptr.find(ptr) != captured_shared_ptr.end();
}

extern "C" void reactant_generic_llvm_rtti_root_dtor(llvm::RTTIRoot* root) {
    reactant::destruct_or_release_if_shared(root);
}

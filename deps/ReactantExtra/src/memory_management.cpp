#include "llvm/Support/ExtensibleRTTI.h"
#include "src/memory_management.hpp"
#include <map>
#include <memory>

std::map<void*, std::shared_ptr<void>> captured_shared_objects;

extern "C" void reactant_release_shared(void* ptr) {
    captured_shared_objects.erase(ptr);
}

extern "C" void reactant_capture_shared(std::shared_ptr<void>& ptr) {
    captured_shared_objects[ptr.get()] = ptr;
}

extern "C" bool reactant_contains_shared(void* ptr) {
    return captured_shared_objects.contains(ptr);
}

extern "C" void reactant_generic_llvm_rtti_root_dtor(llvm::RTTIRoot* root) {
    if (captured_shared_objects.contains(root))
        reactant_release_shared(root);
    else
        delete root;
}

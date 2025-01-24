#include <map>
#include <memory>
#include "src/memory_management.hpp"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/tsl/concurrency/ref_count.h"

std::map<void*, std::shared_ptr<void>> captured_shared_ptr;
std::map<void*, tsl::RCReference<void>> captured_rcreference;

extern "C" void reactant_release_shared(void* ptr) {
    captured_shared_ptr.erase(ptr);
}

extern "C" void reactant_release_rcreference(void* ptr) {
    captured_rcreference.erase(ptr);
}

template<>
void* reactant::capture_shared(std::shared_ptr<void> ptr) {
    captured_shared_ptr[ptr.get()] = ptr;
    return ptr.get();
}

template<>
void* reactant::capture_rcreference(tsl::RCReference<void> ptr) {
    captured_rcreference[ptr.get()] = ptr;
    return ptr.get();
}

// `map::contains` was introduced in C++20
extern "C" bool reactant_contains_shared(void* ptr) {
    return captured_shared_ptr.find(ptr) != captured_shared_ptr.end();
}

extern "C" bool reactant_contains_rcreference(void* ptr) {
    return captured_rcreference.find(ptr) != captured_rcreference.end();
}

std::shared_ptr<void> reactant::get_shared(void* ptr) {
    return captured_shared_ptr[ptr];
}

tsl::RCReference<void> reactant::get_rcreference(void* ptr) {
    return captured_rcreference[ptr];
}

extern "C" void reactant_generic_llvm_rtti_root_dtor(llvm::RTTIRoot* root) {
    reactant::destruct_or_release_if_shared(root);
}

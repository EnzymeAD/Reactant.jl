#include <map>
#include <memory>
#include "src/memory_management.hpp"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/value.h"
#include "xla/python/ifrt/device_list.h"

std::map<void*, std::shared_ptr<void>> captured_shared_ptr;
std::map<void*, reactant::RCRef> captured_rcreference;

extern "C" {
void reactant_release_shared(void* ptr) {
    captured_shared_ptr.erase(ptr);
}

void reactant_release_rcreference(void* ptr) {
    captured_rcreference.erase(ptr);
}

// `map::contains` was introduced in C++20
bool reactant_contains_shared(void* ptr) {
    return captured_shared_ptr.find(ptr) != captured_shared_ptr.end();
}

bool reactant_contains_rcreference(void* ptr) {
    return captured_rcreference.find(ptr) != captured_rcreference.end();
}

void reactant_generic_llvm_rtti_root_dtor(llvm::RTTIRoot* root) {
    reactant::destruct_or_release_if_shared(root);
}
} // extern "C"

namespace reactant {

RCRef::RCRef() noexcept = default;
RCRef::~RCRef() noexcept = default;

void* RCRef::get() const noexcept {
    return std::visit([](auto&& obj) -> void* {
        using T = std::decay_t<decltype(obj)>;
        if constexpr(std::is_same_v<T, std::monostate>) 
            return nullptr;
        else
            return obj.get();
    }, storage);
}

void RCRef::destroy() noexcept {
    storage = std::monostate{};
}

template<>
void* capture_shared(std::shared_ptr<void> ptr) {
    captured_shared_ptr[ptr.get()] = ptr;
    return ptr.get();
}

void* capture_rcreference(RCRef ptr) {
    captured_rcreference[ptr.get()] = ptr;
    return ptr.get();
}

std::shared_ptr<void> get_shared(void* ptr) {
    return captured_shared_ptr[ptr];
}

RCRef get_rcreference(void* ptr) {
    return captured_rcreference[ptr];
}

} // namespace reactant

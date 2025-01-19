#include <memory>

extern "C" void reactant_release_shared(void* ptr);
extern "C" void reactant_capture_shared(std::shared_ptr<void>& ptr);
extern "C" bool reactant_contains_shared(void* ptr);
extern "C" void reactant_generic_llvm_rtti_root_dtor(llvm::RTTIRoot* root);

namespace reactant {
template<typename  T>
inline void destruct_or_release_if_shared(T* ptr) {
    if (reactant_contains_shared(ptr))
        reactant_release_shared(ptr);
    else
        delete ptr;
}
}

#include <memory>

extern "C" void reactant_release_shared(void* ptr);
extern "C" bool reactant_contains_shared(void* ptr);

namespace reactant {
template <typename T, typename G = std::remove_cv_t<T>>
inline G* capture_shared(std::shared_ptr<T> ptr) {
    return reinterpret_cast<G*>(
        capture_shared(
            std::const_pointer_cast<G>(ptr)
        )
    );
}

template<>
void* capture_shared(std::shared_ptr<void> ptr);

template<typename  T>
inline void destruct_or_release_if_shared(T* ptr) {
    if (reactant_contains_shared(ptr))
        reactant_release_shared(ptr);
    else
        delete ptr;
}

template<typename T>
std::shared_ptr<T> get_or_insert_shared(T* ptr) {
    if (!reactant_contains_shared(ptr))
        reactant::capture_shared(std::shared_ptr<T>(ptr));
    return std::reinterpret_pointer_cast<T>(get_shared(ptr));
}

std::shared_ptr<void> get_shared(void* ptr);

}

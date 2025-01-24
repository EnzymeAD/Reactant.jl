#include <memory>
#include <type_traits>
#include "xla/tsl/concurrency/ref_count.h"

extern "C" void reactant_release_shared(void* ptr);
extern "C" bool reactant_contains_shared(void* ptr);
extern "C" void reactant_release_rcreference(void* ptr);
extern "C" bool reactant_contains_rcreference(void* ptr);

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

template <typename T, typename G = std::remove_cv_t<T>>
inline G* capture_rcreference(tsl::RCReference<T> ptr) {
    return reinterpret_cast<G*>(
        capture_rcreference(
            std::const_pointer_cast<G>(ptr)
        )
    );
}

template<>
void* capture_rcreference(tsl::RCReference<void> ptr);

template<typename  T>
inline void destruct_or_release_if_shared(T* ptr) {
    if (reactant_contains_shared(ptr))
        reactant_release_shared(ptr);
    else
        delete ptr;
}

template<typename  T>
inline void destruct_or_release_if_rcreference(T* ptr) {
    if (reactant_contains_rcreference(ptr))
        reactant_release_rcreference(ptr);
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

template<typename T>
std::shared_ptr<T> get_or_insert_rcreference(T* ptr) {
    if (!reactant_contains_rcreference(ptr))
        reactant::capture_rcreference(tsl::RCReference<T>(ptr));
    return std::reinterpret_pointer_cast<T>(get_rcreference(ptr));
}

std::shared_ptr<void> get_rcreference(void* ptr);

// TODO here we might have `std::shared_ptr` but also `tsl::RCReference`. what do we put?
// do we use polymorphism here and return a `Holder*` (pointer to abstract base class) or do we explicitly return a `StdSharedHolder*`/`TslRCReferenceHolder*` (pointer to derived class)?
// - if we use the derived class pointer, the we must make sure that (1) all methods using that class always use the same type of ref count manager, and (2) all of our C-API pass/return the appropiate derived types.
// - if we go for the polymorphism (which i'm not sure about), we should probably want a `HolderFactory`.
// template<typename T>
// struct Holder<T> {

//     virtual ~Holder();
//     virtual T* get() const;
// };

// template<typename T>
// struct StdSharedHolder : public Holder<T> {
//     std::shared_ptr<T> holded;

//     StdSharedHolder(std::shared_ptr<T>&) = default;

//     T* get() const override {
//         return this->holded.get();
//     }
// };

// template<typename T>
// struct TslRCReferenceHolder : public Holder<T> {
//     tsl::RCReference<T> holded;

//     TslRCReferenceHolder(tsl::RCReference<T>&) = default;
// };

}

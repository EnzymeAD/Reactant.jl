#pragma once

#include <memory>
#include <type_traits>
#include <variant>
#include "xla/tsl/concurrency/ref_count.h"

extern "C" {
void reactant_release_shared(void* ptr);
bool reactant_contains_shared(void* ptr);
void reactant_release_rcreference(void* ptr);
bool reactant_contains_rcreference(void* ptr);
}

namespace xla {
namespace ifrt {
class Array;
class Value;
class DeviceList;
class LoadedHostCallback;
} // namespace ifrt
} // namespace xla

namespace reactant {

// type erased tsl::RCReference
class RCRef {
public:
    RCRef() noexcept;
    ~RCRef() noexcept;

    template<typename T>
    explicit RCRef(tsl::RCReference<T> obj);

    void* get() const noexcept;
    void destroy() noexcept;

    template<typename T> T* get() const noexcept;
    template<typename T> tsl::RCReference<T> get_rcref() const noexcept;
    template<typename T> bool is() const noexcept; 

private:
    // TODO can we avoid using `Array` here and cast from/to `Value`?
    using storage_t = std::variant<
        std::monostate,
        tsl::RCReference<xla::ifrt::Value>, 
        tsl::RCReference<xla::ifrt::Array>, 
        // tsl::RCReference<xla::ifrt::DeviceList>, // we explicitly convert to span<Device*>
        tsl::RCReference<xla::ifrt::LoadedHostCallback>
        >;
    storage_t storage;
};

template <typename T, typename G = std::remove_cv_t<T>>
G* capture_shared(std::shared_ptr<T> ptr);

template<>
void* capture_shared(std::shared_ptr<void> ptr);

void* capture_rcreference(RCRef ptr);

template <typename T, typename G = std::remove_cv_t<T>>
G* capture_rcreference(tsl::RCReference<T> rcref);

template<typename  T>
void destruct_or_release_if_shared(T* ptr);

template<typename  T>
void destruct_or_release_if_rcreference(T* ptr);

std::shared_ptr<void> get_shared(void* ptr);
RCRef get_rcreference(void* ptr);

template<typename T>
std::shared_ptr<T> get_or_insert_shared(T* ptr);

template<typename T>
RCRef get_or_insert_rcreference(T* ptr);


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

/*
 * Definitions
 */
template<typename T>
T* RCRef::get() const noexcept {
    if (auto ptr = std::get_if<tsl::RCReference<T>>(&storage)) 
        return ptr->get();
    return nullptr;
}
template<typename T>
tsl::RCReference<T> RCRef::get_rcref() const noexcept {
    if (auto ptr = std::get_if<tsl::RCReference<T>>(&storage)) 
        return *ptr;
    return {};
}

template<typename T>
bool RCRef::is() const noexcept {
    return get<T>() != nullptr;
}

template <typename T, typename G>
G* capture_shared(std::shared_ptr<T> ptr) {
    return reinterpret_cast<G*>(
        capture_shared(std::reinterpret_pointer_cast<void>(std::const_pointer_cast<G>(ptr)))
    );
}

template <typename T, typename G>
G* capture_rcreference(tsl::RCReference<T> rcref) {
    return reinterpret_cast<G*>(
        capture_rcreference(RCRef{rcref})
    );
}

template<typename  T>
void destruct_or_release_if_shared(T* ptr) {
    if (reactant_contains_shared(ptr))
        reactant_release_shared(ptr);
    else
        delete ptr;
}

template<typename  T>
void destruct_or_release_if_rcreference(T* ptr) {
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


template<typename T>
RCRef get_or_insert_rcreference(T* ptr) {
    if (!reactant_contains_rcreference(ptr))
        reactant::capture_rcreference(tsl::FormRef(ptr));
    return get_rcreference(ptr);
}

} // namespace reactant

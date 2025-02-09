#pragma once

#include <memory>
#include <type_traits>
#include <variant>
#include "xla/tsl/concurrency/ref_count.h"

#define REACTANT_C_HOLDED_DESTRUCTOR(NAME, TYPE) \
extern "C" void reactant_holded_##NAME##_dtor(reactant::Holded<TYPE>* holded) { \
    delete holded; \
}

namespace reactant {

template <typename T> struct unwrap_type { typedef T type; };
template <typename T> struct unwrap_type<std::shared_ptr<T>> { typedef T type; };
template <typename T> struct unwrap_type<tsl::RCReference<T>> { typedef T type; };

template <typename T> using unwrap_type_t = typename unwrap_type<T>::type;

template<typename T>
struct Holded {
 public:
    Holded(T& obj) : holded(obj) {}
    ~Holded() = default;

    unwrap_type_t<T>* ptr() const {
        return holded.get();
    }

    // T operator[]() const {
    //     return holded;
    // }
    T obj() const {
        return holded;
    }

    T value() const {
        return holded;
    }

    unwrap_type_t<T>* operator->() const {
        return ptr();
    }

 private:
    T holded;
};

template <typename T>
Holded<T>* capture(T obj) {
    return new Holded<T>(obj);
}

// template <typename T>
// T* Holded<std::shared_ptr<T>>::ptr() const {
//     return this->holded.get();
// }

// template <typename T>
// T* Holded<tsl::RCReference<T>>::ptr() const {
//     return this->holded.get();
// }

} // namespace reactant

#pragma once

#include "absl/base/nullability.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/ref_count.h"
#include <type_traits>

namespace reactant {
template <typename T>
struct Type { };

template <typename T>
struct span {
    size_t size;
    T* ptr;

    T& operator[](size_t i) { return ptr[i]; }
};

template <typename T>
struct opaque {
    size_t size;
    T* ptr;

    opaque(T* ptr) : size(sizeof(T)), ptr(ptr) {}

    opaque(T&& ptr) : size(sizeof(T)), ptr(nullptr) {
        this->ptr = (T*)::operator new(sizeof(T));
    }

    ~opaque() {
        delete this->ptr;
    }
};

template <typename T>
auto convert(Type<const char*>, T text) -> const char* {
    char *cstr = (char *)malloc(text.size() + 1);
    memcpy(cstr, text.data(), text.size());
    cstr[text.size()] = '\0';
    return cstr;
}

template <typename T>
auto convert(Type<span<T>>, std::vector<T> vec) -> span<T>
{
    T* ptr = new T[vec.size()];
    for (int i = 0; i < vec.size(); i++) {
        ptr[i] = vec[i];
    }
    return span<T> { vec.size(), ptr };
}

template <typename U, typename T = std::remove_const_t<U>>
auto convert(Type<span<T>>, absl::Span<U> _span) -> span<T>
{
    T* ptr = new T[_span.size()];
    for (int i = 0; i < _span.size(); i++) {
        ptr[i] = _span[i];
    }
    return span<T>(_span.size(), ptr);
}

template <typename T>
auto convert(Type<absl::Span<T>>, span<T> span) -> absl::Span<T>
{
    return absl::Span<T>(span.ptr, span.size);
}

template <typename U, typename T = std::remove_pointer_t<U>>
auto convert(Type<absl::Span<tsl::RCReference<T>>>, span<U> span) -> absl::Span<tsl::RCReference<T>>
{
    auto values_ptr = new tsl::RCReference<T>[span.size];
    for (int i = 0; i < span.size; i++) {
        values_ptr = tsl::RCReference<T>();
        values_ptr[i].reset(&span[i]);
    }
    return absl::Span<tsl::RCReference<T>>(values_ptr, span.size);
}
} // namespace reactant

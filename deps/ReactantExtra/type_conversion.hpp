#pragma once

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/ref_count.h"

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
auto convert(Type<span<T>>, std::vector<T> vec) -> span<T>
{
    T* ptr = new T[vec.size()];
    for (int i = 0; i < vec.size(); i++) {
        ptr[i] = vec[i];
    }
    return span<T> { vec.size(), ptr };
}

template <typename T>
auto convert(Type<span<T>>, absl::Span<T> span) -> span<T>
{
    T* ptr = new T[span.size()];
    for (int i = 0; i < span.size(); i++) {
        ptr[i] = span[i];
    }
    return span<T> { span.size(), ptr };
}

template <typename T>
auto convert(Type<absl::Span<T>>, span<T> span) -> absl::Span<T>
{
    return absl::Span<T>(span.ptr, span.size);
}

template <typename T>
auto convert(Type<absl::Span<tsl::RCReference<T>>>, span<T> span) -> absl::Span<tsl::RCReference<T>>
{
    auto values_ptr = new tsl::RCReference<T>[span.size];
    for (int i = 0; i < span.size; i++) {
        values_ptr = tsl::RCReference<T>();
        values_ptr[i].reset(&span[i]);
    }
    return absl::Span<tsl::RCReference<T>>(values_ptr, span.size);
}
} // namespace reactant

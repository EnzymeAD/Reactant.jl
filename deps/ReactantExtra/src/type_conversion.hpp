#pragma once

#include "absl/base/nullability.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "memory_management.hpp"
#include <type_traits>
#include <optional>
#include <vector>
#include <memory>

namespace reactant {
template <typename T>
struct Type { };

template <typename T>
struct span {
    size_t _size;
    T* ptr;

    span() : span(0, nullptr) {}
    span(size_t size, T* ptr) : _size(size), ptr(ptr) {}

    T& operator[](size_t i) { return ptr[i]; }

    bool empty() const { return _size == 0; }
    size_t size() const { return _size; }
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

template <typename T, typename = std::enable_if_t<std::is_fundamental_v<T>>>
auto convert(Type<span<T>>, std::vector<T> vec) -> span<T>
{
    T* ptr = new T[vec.size()];
    for (int i = 0; i < vec.size(); i++) {
        ptr[i] = vec[i];
    }
    return span<T> { vec.size(), ptr };
}

template <typename T, 
    typename = std::enable_if_t<!std::is_fundamental_v<T> && std::is_copy_constructible_v<T>>
    >
auto convert(Type<span<T*>>, std::vector<T> vec) -> span<T*>
{
    T** ptr = new T*[vec.size()];
    for (int i = 0; i < vec.size(); i++) {
        ptr[i] = new T(vec[i]);
    }
    return span<T*> { vec.size(), ptr };
}

template <typename T>
auto convert(Type<span<T*>>, std::vector<std::unique_ptr<T>> vec) -> span<T*> {
    T** ptr = new T*[vec.size()];
    for (int i = 0; i < vec.size(); i++) {
        ptr[i] = vec[i].release();
    }
    return span<T*>(vec.size(), ptr);
}

template <typename T>
auto convert(Type<span<T*>>, std::vector<std::shared_ptr<T>> vec) -> span<T*> {
    T** ptr = new T*[vec.size()];
    for (int i = 0; i < vec.size(); i++) {
        ptr[i] = reactant::capture_shared(vec[i]);
    }
    return span<T*>(vec.size(), ptr);
}

template <typename T>
auto convert(Type<span<T*>>, std::vector<tsl::RCReference<T>> vec) -> span<T*> {
    T** ptr = new T*[vec.size()];
    for (int i = 0; i < vec.size(); i++) {
        ptr[i] = reactant::capture_rcreference(vec[i]);
    }
    return span<T*>(vec.size(), ptr);
}

template <typename T, typename U, typename = std::enable_if_t<std::is_convertible_v<T, U>>>
auto convert(Type<span<T>>, absl::Span<U> _span) -> span<T>
{
    using G = std::remove_const_t<T>;
    G* ptr = new G[_span.size()];
    for (int i = 0; i < _span.size(); i++) {
        ptr[i] = _span[i];
    }
    return span<T>(_span.size(), ptr);
}

template <typename T>
auto convert(Type<absl::Span<T>>, span<T> span) -> absl::Span<T>
{
    return absl::Span<T>(span.ptr, span.size());
}

template <typename T>
auto convert(Type<absl::Span<tsl::RCReference<T>>>, span<T*> span) -> absl::Span<tsl::RCReference<T>>
{
    auto values_ptr = new tsl::RCReference<T>[span.size()];
    for (int i = 0; i < span.size(); i++) {
        values_ptr[i] = reactant::get_or_insert_rcreference(span[i]).template get_rcref<T>();
    }
    return absl::Span<tsl::RCReference<T>>(values_ptr, span.size());
}

template <typename T>
auto convert(Type<std::optional<T>>, T* ptr) -> std::optional<T>
{
    if (ptr == nullptr) {
        return std::nullopt;
    }
    return *ptr;
}

template <typename T>
auto convert(Type<std::optional<T>>, T value) -> std::optional<T>
{
    return value;
}

// special case for DeviceList
auto convert(Type<span<xla::ifrt::Device*>>, xla::ifrt::DeviceList* dev_list) -> span<xla::ifrt::Device*>;
auto convert(Type<xla::ifrt::DeviceList*>, span<xla::ifrt::Device*> dev_list) -> xla::ifrt::DeviceList*;

} // namespace reactant

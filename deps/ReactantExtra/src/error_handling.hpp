#pragma once

#include "absl/status/statusor.h"
#include "xla/pjrt/exceptions.h"
#include "xla/pjrt/status_casters.h"

extern "C" void (*ReactantThrowError)(const char *);

// Utilities for `StatusOr`.
template <typename T> T MyValueOrThrow(absl::StatusOr<T> v) {
    if (ReactantThrowError) {
        if (!v.ok()) {
            ReactantThrowError(v.status().ToString().c_str());
            throw xla::XlaRuntimeError(v.status().ToString().c_str());
        }
        return std::move(v).value();
    } else {
        return xla::ValueOrThrow(std::move(v));
    }
}

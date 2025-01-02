#include "../type_conversion.hpp"
#include "xla/python/ifrt/dtype.h"

using namespace xla::ifrt;
using namespace reactant;

extern "C" ifrt::DType* ifrt_dtype_ctor(ifrt::DType::Kind kind)
{
    return new ifrt::DType(kind);
}

extern "C" void ifrt_dtype_free(ifrt::DType* dtype) { delete dtype; }

extern "C" ifrt::DType::Kind ifrt_dtype_kind(ifrt::DType* dtype)
{
    return dtype->kind();
}

extern "C" bool ifrt_dtype_eq(ifrt::DType* dtype1, ifrt::DType* dtype2)
{
    return *dtype1 == *dtype2;
}

extern "C" bool ifrt_dtype_ne(ifrt::DType* dtype1, ifrt::DType* dtype2)
{
    return *dtype1 != *dtype2;
}

// Returns -1 if not aligned to a byte boundary or there is no fixed size
extern "C" int ifrt_dtype_byte_size(ifrt::DType* dtype)
{
    auto byte_size = dtype->byte_size();
    if (byte_size.has_value()) {
        return byte_size.value();
    }
    return -1;
}

// Returns -1 if there is no fixed size
extern "C" int ifrt_dtype_bit_size(ifrt::DType* dtype)
{
    auto bit_size = dtype->bit_size();
    if (bit_size.has_value()) {
        return bit_size.value();
    }
    return -1;
}

extern "C" const char* ifrt_dtype_debug_string(ifrt::DType* dtype)
{
    return cstr_from_string(dtype->DebugString());
}

// xla::PrimitiveType is a enum, so we use int to represent it on Julia side
extern "C" xla::PrimitiveType ifrt_to_primitive_type(ifrt::DType* dtype)
{
    return MyValueOrThrow(ifrt::ToPrimitiveType(*dtype));
}

// xla::PrimitiveType is a enum, so we use int to represent it on Julia side
extern "C" ifrt::DType* ifrt_to_dtype(xla::PrimitiveType primitive_type)
{
    auto dtype = MyValueOrThrow(ifrt::ToDType(primitive_type));
    return new ifrt::DType(dtype.kind());
}

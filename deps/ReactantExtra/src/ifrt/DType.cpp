#include "src/type_conversion.hpp"
#include "xla/python/ifrt/dtype.h"

using namespace xla::ifrt;
using namespace reactant;

extern "C" DType* ifrt_dtype_ctor(DType::Kind kind)
{
    return new DType(kind);
}

extern "C" void ifrt_dtype_free(DType* dtype) { delete dtype; }

extern "C" DType::Kind ifrt_dtype_kind(DType* dtype)
{
    return dtype->kind();
}

extern "C" bool ifrt_dtype_eq(DType* dtype1, DType* dtype2)
{
    return *dtype1 == *dtype2;
}

extern "C" bool ifrt_dtype_ne(DType* dtype1, DType* dtype2)
{
    return *dtype1 != *dtype2;
}

// Returns -1 if not aligned to a byte boundary or there is no fixed size
extern "C" int ifrt_dtype_byte_size(DType* dtype)
{
    auto byte_size = dtype->byte_size();
    if (byte_size.has_value()) {
        return byte_size.value();
    }
    return -1;
}

// Returns -1 if there is no fixed size
extern "C" int ifrt_dtype_bit_size(DType* dtype)
{
    auto bit_size = dtype->bit_size();
    if (bit_size.has_value()) {
        return bit_size.value();
    }
    return -1;
}

extern "C" const char* ifrt_dtype_debug_string(DType* dtype)
{
    return convert(Type<const char*>(), dtype->DebugString());
}

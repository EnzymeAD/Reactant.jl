#include "src/type_conversion.hpp"
#include "src/error_handling.hpp"
#include "xla/python/ifrt/dtype.h"
#include "xla/xla_data.pb.h"

using namespace xla::ifrt;
using namespace reactant;

// xla::PrimitiveType is a enum, so we use int to represent it on Julia side
extern "C" xla::PrimitiveType ifrt_to_primitive_type(DType* dtype)
{
    return MyValueOrThrow(ToPrimitiveType(*dtype));
}

// xla::PrimitiveType is a enum, so we use int to represent it on Julia side
extern "C" DType* ifrt_to_dtype(xla::PrimitiveType primitive_type)
{
    auto dtype = MyValueOrThrow(ToDType(primitive_type));
    return new DType(dtype.kind());
}

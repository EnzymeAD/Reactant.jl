#include "../type_conversion.hpp"
#include "xla/python/ifrt/shape.h"

using namespace xla::ifrt;
using namespace reactant;

extern "C" ifrt::Shape* ifrt_shape_ctor(const int64_t* dims, size_t dims_size)
{
    return new ifrt::Shape(absl::Span<const int64_t>(dims, dims_size));
}

extern "C" void ifrt_shape_free(ifrt::Shape* shape) { delete shape; }

extern "C" span<const int64_t> ifrt_shape_dims(ifrt::Shape* shape)
{
    return reactant::convert(Type<span<const int64_t>>(), shape->dims());
}

extern "C" bool ifrt_shape_eq(ifrt::Shape* shape1, ifrt::Shape* shape2)
{
    return *shape1 == *shape2;
}

extern "C" bool ifrt_shape_ne(ifrt::Shape* shape1, ifrt::Shape* shape2)
{
    return *shape1 != *shape2;
}

extern "C" int64_t ifrt_shape_dims_num_elements(ifrt::Shape* shape)
{
    return shape->num_elements();
}

extern "C" const char* ifrt_shape_debug_string(ifrt::Shape* shape)
{
    return cstr_from_string(shape->DebugString());
}

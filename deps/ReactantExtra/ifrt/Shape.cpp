#include "src/type_conversion.hpp"
#include "xla/python/ifrt/shape.h"

using namespace xla::ifrt;
using namespace reactant;

extern "C" Shape* ifrt_shape_ctor(const int64_t* dims, size_t dims_size)
{
    return new Shape(absl::Span<const int64_t>(dims, dims_size));
}

extern "C" void ifrt_shape_free(Shape* shape) { delete shape; }

extern "C" span<int64_t> ifrt_shape_dims(Shape* shape)
{
    return reactant::convert(Type<span<int64_t>>(), shape->dims());
}

extern "C" bool ifrt_shape_eq(Shape* shape1, Shape* shape2)
{
    return *shape1 == *shape2;
}

extern "C" bool ifrt_shape_ne(Shape* shape1, Shape* shape2)
{
    return *shape1 != *shape2;
}

extern "C" int64_t ifrt_shape_num_elements(Shape* shape)
{
    return shape->num_elements();
}

extern "C" const char* ifrt_shape_debug_string(Shape* shape)
{
    return convert(Type<const char*>(), shape->DebugString());
}

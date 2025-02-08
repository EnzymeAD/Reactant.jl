#include "src/type_conversion.hpp"
#include "src/error_handling.hpp"
#include "xla/python/ifrt/shape.h"

using namespace xla::ifrt;
using namespace reactant;

extern "C" DynamicShape* ifrt_dynamicshape_ctor(Shape* shape, const bool* dynamic_dims_mask)
{
    auto tag = BoundedDynamicShapeTag(
        absl::Span<const bool>(dynamic_dims_mask, shape->dims().size()));
    auto dynshape = MyValueOrThrow(DynamicShape::Create(*shape, tag));
    return new DynamicShape(dynshape);
}

extern "C" void ifrt_dynamicshape_free(DynamicShape* shape) { delete shape; }

// TODO DynamicShape::GetTag

extern "C" bool ifrt_dynamicshape_eq(DynamicShape* shape1, DynamicShape* shape2)
{
    return *shape1 == *shape2;
}

extern "C" bool ifrt_dynamicshape_ne(DynamicShape* shape1, DynamicShape* shape2)
{
    return *shape1 != *shape2;
}

extern "C" Shape* ifrt_dynamicshape_get_padded_shape(DynamicShape* shape)
{
    auto padshape = MyValueOrThrow(shape->GetPaddedShape());
    return new Shape(padshape);
}

extern "C" bool ifrt_dynamicshape_is_dynamic_dim(DynamicShape* shape, int dimension)
{
    return shape->IsDynamicDim(dimension);
}

extern "C" const char* ifrt_dynamicshape_debug_string(DynamicShape* shape)
{
    return convert(Type<const char*>(), shape->DebugString());
}

#include "../type_conversion.hpp"
#include "xla/python/ifrt/shape.h"

using namespace xla::ifrt;
using namespace reactant;

extern "C" ifrt::DynamicShape*
ifrt_dynamicshape_ctor(ifrt::Shape* shape, const bool* dynamic_dims_mask)
{
    auto tag = ifrt::BoundedDynamicShapeTag(
        absl::Span<const bool>(dynamic_dims_mask, shape->dims().size()));
    auto dynshape = MyValueOrThrow(ifrt::DynamicShape::Create(*shape, tag));
    return new ifrt::DynamicShape(dynshape);
}

extern "C" void ifrt_dynamicshape_free(ifrt::DynamicShape* shape)
{
    delete shape;
}

// TODO ifrt::DynamicShape::GetTag

extern "C" bool ifrt_dynamicshape_eq(ifrt::DynamicShape* shape1,
    ifrt::DynamicShape* shape2)
{
    return *shape1 == *shape2;
}

extern "C" bool ifrt_dynamicshape_ne(ifrt::DynamicShape* shape1,
    ifrt::DynamicShape* shape2)
{
    return *shape1 != *shape2;
}

extern "C" ifrt::Shape*
ifrt_dynamicshape_get_padded_shape(ifrt::DynamicShape* shape)
{
    auto padshape = MyValueOrThrow(shape->GetPaddedShape());
    return new ifrt::Shape(padshape);
}

extern "C" bool ifrt_dynamicshape_is_dynamic_dim(ifrt::DynamicShape* shape,
    int dimension)
{
    return shape->IsDynamicDim(dimension);
}

extern "C" const char*
ifrt_dynamicshape_debug_string(ifrt::DynamicShape* shape)
{
    return cstr_from_string(shape->DebugString());
}

#include "src/type_conversion.hpp"
#include "xla/python/ifrt/index_domain.h"

using namespace xla::ifrt;
using namespace reactant;

extern "C" IndexDomain* ifrt_indexdomain_ctor(Shape* shape)
{
    return new IndexDomain(*shape);
}

extern "C" IndexDomain* ifrt_indexdomain_ctor_with_origin(Index* origin, Shape* shape)
{
    return new IndexDomain(*origin, *shape);
}

extern "C" void ifrt_indexdomain_free(IndexDomain* index_domain)
{
    delete index_domain;
}

extern "C" const Index* ifrt_indexdomain_origin(IndexDomain* index_domain)
{
    return new Index(index_domain->origin());
}

extern "C" const Shape* ifrt_indexdomain_shape(IndexDomain* index_domain)
{
    return new Shape(index_domain->shape());
}

extern "C" bool ifrt_indexdomain_eq(IndexDomain* index_domain1, IndexDomain* index_domain2)
{
    return *index_domain1 == *index_domain2;
}

extern "C" bool ifrt_indexdomain_ne(IndexDomain* index_domain1,
    IndexDomain* index_domain2)
{
    return *index_domain1 != *index_domain2;
}

extern "C" IndexDomain* ifrt_indexdomain_add(IndexDomain* index_domain, Index* offset)
{
    return new IndexDomain(*index_domain + *offset);
}

extern "C" IndexDomain* ifrt_indexdomain_sub(IndexDomain* index_domain, Index* offset)
{
    return new IndexDomain(*index_domain - *offset);
}

extern "C" void ifrt_indexdomain_add_inplace(IndexDomain* index_domain,
    Index* offset)
{
    *index_domain += *offset;
}

extern "C" void ifrt_indexdomain_sub_inplace(IndexDomain* index_domain,
    Index* offset)
{
    *index_domain -= *offset;
}

extern "C" const char* ifrt_indexdomain_debug_string(IndexDomain* index_domain)
{
    return convert(Type<const char*>(), index_domain->DebugString());
}
#pragma endregion

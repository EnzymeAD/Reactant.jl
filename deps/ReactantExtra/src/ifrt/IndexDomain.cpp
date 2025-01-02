#include "../type_conversion.hpp"
#include "xla/python/ifrt/index_domain.h"

using namespace xla::ifrt;
using namespace reactant;

#pragma region xla::ifrt::IndexDomain
extern "C" ifrt::IndexDomain* ifrt_indexdomain_ctor(ifrt::Shape* shape)
{
    return new ifrt::IndexDomain(*shape);
}

extern "C" ifrt::IndexDomain* ifrt_indexdomain_ctor_with_origin(ifrt::Index* origin, ifrt::Shape* shape)
{
    return new ifrt::IndexDomain(*origin, *shape);
}

extern "C" void ifrt_indexdomain_free(ifrt::IndexDomain* index_domain)
{
    delete index_domain;
}

extern "C" const ifrt::Index* ifrt_indexdomain_origin(ifrt::IndexDomain* index_domain)
{
    return new ifrt::Index(index_domain->origin());
}

extern "C" const ifrt::Shape* ifrt_indexdomain_shape(ifrt::IndexDomain* index_domain)
{
    return new ifrt::Shape(index_domain->shape());
}

extern "C" bool ifrt_indexdomain_eq(ifrt::IndexDomain* index_domain1, ifrt::IndexDomain* index_domain2)
{
    return *index_domain1 == *index_domain2;
}

extern "C" bool ifrt_indexdomain_ne(ifrt::IndexDomain* index_domain1,
    ifrt::IndexDomain* index_domain2)
{
    return *index_domain1 != *index_domain2;
}

extern "C" ifrt::IndexDomain* ifrt_indexdomain_add(ifrt::IndexDomain* index_domain, ifrt::Index* offset)
{
    return new ifrt::IndexDomain(*index_domain + *offset);
}

extern "C" ifrt::IndexDomain* ifrt_indexdomain_sub(ifrt::IndexDomain* index_domain, ifrt::Index* offset)
{
    return new ifrt::IndexDomain(*index_domain - *offset);
}

extern "C" void ifrt_indexdomain_add_inplace(ifrt::IndexDomain* index_domain,
    ifrt::Index* offset)
{
    *index_domain += *offset;
}

extern "C" void ifrt_indexdomain_sub_inplace(ifrt::IndexDomain* index_domain,
    ifrt::Index* offset)
{
    *index_domain -= *offset;
}

extern "C" const char* ifrt_indexdomain_debug_string(ifrt::IndexDomain* index_domain)
{
    return cstr_from_string(index_domain->DebugString());
}
#pragma endregion

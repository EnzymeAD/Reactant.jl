#include "../type_conversion.hpp"
#include "xla/python/ifrt/index.h"

using namespace xla::ifrt;
using namespace reactant;

extern "C" ifrt::Index* ifrt_index_ctor(const int64_t* elements, size_t elements_size)
{
    return new ifrt::Index(absl::Span<const int64_t>(elements, elements_size));
}

extern "C" void ifrt_index_free(ifrt::Index* index) { delete index; }

extern "C" ifrt::Index* ifrt_index_zeros(int num_elements)
{
    return new ifrt::Index(ifrt::Index::Zeros(num_elements));
}

extern "C" const int64_t* ifrt_index_elements(ifrt::Index* index)
{
    return index->elements().data();
}

extern "C" int ifrt_index_count(ifrt::Index* index)
{
    return index->elements().size();
}

extern "C" bool ifrt_index_eq(ifrt::Index* index1, ifrt::Index* index2)
{
    return *index1 == *index2;
}

extern "C" bool ifrt_index_ne(ifrt::Index* index1, ifrt::Index* index2)
{
    return *index1 != *index2;
}

extern "C" ifrt::Index* ifrt_index_add(ifrt::Index* index,
    ifrt::Index* offset)
{
    return new ifrt::Index(*index + *offset);
}

extern "C" ifrt::Index* ifrt_index_sub(ifrt::Index* index,
    ifrt::Index* offset)
{
    return new ifrt::Index(*index - *offset);
}

// WARN we're not checking if the multiplier has the same size as the index
extern "C" ifrt::Index* ifrt_index_mul(ifrt::Index* index, const int64_t* multiplier)
{
    return new ifrt::Index(*index * absl::Span<const int64_t>(multiplier, ifrt_index_count(index)));
}

extern "C" void ifrt_index_add_inplace(ifrt::Index* index,
    ifrt::Index* offset)
{
    *index += *offset;
}

extern "C" void ifrt_index_sub_inplace(ifrt::Index* index,
    ifrt::Index* offset)
{
    *index -= *offset;
}

extern "C" void ifrt_index_mul_inplace(ifrt::Index* index,
    const int64_t* multiplier)
{
    *index *= absl::Span<const int64_t>(multiplier, ifrt_index_count(index));
}

extern "C" const char* ifrt_index_debug_string(ifrt::Index* index)
{
    return cstr_from_string(index->DebugString());
}

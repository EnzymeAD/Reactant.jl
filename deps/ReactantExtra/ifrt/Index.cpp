#include "src/type_conversion.hpp"
#include "xla/python/ifrt/index.h"

using namespace xla::ifrt;
using namespace reactant;

extern "C" Index* ifrt_index_ctor(const int64_t* elements, size_t elements_size)
{
    return new Index(absl::Span<const int64_t>(elements, elements_size));
}

extern "C" void ifrt_index_free(Index* index) { delete index; }

extern "C" Index* ifrt_index_zeros(int num_elements)
{
    return new Index(Index::Zeros(num_elements));
}

extern "C" span<const int64_t> ifrt_index_elements(Index* index)
{
    return convert(Type<span<const int64_t>>(), index->elements());
}

extern "C" int ifrt_index_count(Index* index)
{
    return index->elements().size();
}

extern "C" bool ifrt_index_eq(Index* index1, Index* index2)
{
    return *index1 == *index2;
}

extern "C" bool ifrt_index_ne(Index* index1, Index* index2)
{
    return *index1 != *index2;
}

extern "C" Index* ifrt_index_add(Index* index, Index* offset)
{
    return new Index(*index + *offset);
}

extern "C" Index* ifrt_index_sub(Index* index, Index* offset)
{
    return new Index(*index - *offset);
}

// WARN we're not checking if the multiplier has the same size as the index -> check in Julia
extern "C" Index* ifrt_index_mul(Index* index, span<const int64_t> c_multiplier)
{
    auto multiplier = convert(Type<absl::Span<const int64_t>>(), c_multiplier);
    return new Index(*index * multiplier);
}

extern "C" void ifrt_index_add_inplace(Index* index, Index* offset)
{
    *index += *offset;
}

extern "C" void ifrt_index_sub_inplace(Index* index, Index* offset)
{
    *index -= *offset;
}

// WARN we're not checking if the multiplier has the same size as the index -> check in Julia
extern "C" void ifrt_index_mul_inplace(Index* index, span<const int64_t> c_multiplier)
{
    auto multiplier = convert(Type<absl::Span<const int64_t>>(), c_multiplier);
    *index *= multiplier;
}

extern "C" const char* ifrt_index_debug_string(Index* index)
{
    return convert(Type<const char*>(), index->DebugString());
}

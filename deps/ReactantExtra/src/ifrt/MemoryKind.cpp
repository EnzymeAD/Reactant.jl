#include "../type_conversion.hpp"
#include "xla/python/ifrt/memory.h"

using namespace xla::ifrt;
using namespace reactant;

// Pass a nullptr to create a `MemoryKind` with no memory chosen.
extern "C" ifrt::MemoryKind* ifrt_memorykind_ctor(const char* memory_kind)
{
    if (memory_kind == nullptr)
        return new ifrt::MemoryKind();
    return new ifrt::MemoryKind(std::string(memory_kind));
}

extern "C" void ifrt_memorykind_free(ifrt::MemoryKind* memory_kind)
{
    delete memory_kind;
}

extern "C" bool ifrt_memorykind_eq(ifrt::MemoryKind* mk1,
    ifrt::MemoryKind* mk2)
{
    return *mk1 == *mk2;
}

extern "C" bool ifrt_memorykind_ne(ifrt::MemoryKind* mk1,
    ifrt::MemoryKind* mk2)
{
    return *mk1 != *mk2;
}

extern "C" const char* ifrt_memorykind_string(ifrt::MemoryKind* memory_kind)
{
    if (memory_kind->memory_kind().has_value())
        return cstr_from_string(memory_kind->memory_kind().value());
    else
        return nullptr;
}

extern "C" ifrt::MemoryKind* ifrt_memorykind_canonicalize(ifrt::MemoryKind* memory_kind, ifrt::Device* device)
{
    return new ifrt::MemoryKind(CanonicalizeMemoryKind(*memory_kind, device));
}

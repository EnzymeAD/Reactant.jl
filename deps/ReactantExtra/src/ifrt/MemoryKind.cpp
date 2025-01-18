#include "src/type_conversion.hpp"
#include "xla/python/ifrt/memory.h"

using namespace xla::ifrt;
using namespace reactant;

// Pass a nullptr to create a `MemoryKind` with no memory chosen.
extern "C" MemoryKind* ifrt_memorykind_ctor(const char* memory_kind)
{
    if (memory_kind == nullptr)
        return new MemoryKind();
    return new MemoryKind(std::string(memory_kind));
}

extern "C" void ifrt_memorykind_free(MemoryKind* memory_kind)
{
    delete memory_kind;
}

extern "C" bool ifrt_memorykind_eq(MemoryKind* mk1,
    MemoryKind* mk2)
{
    return *mk1 == *mk2;
}

extern "C" bool ifrt_memorykind_ne(MemoryKind* mk1,
    MemoryKind* mk2)
{
    return *mk1 != *mk2;
}

extern "C" const char* ifrt_memorykind_string(MemoryKind* memory_kind)
{
    if (memory_kind->memory_kind().has_value())
        return convert(Type<const char*>(), memory_kind->memory_kind().value());
    else
        return nullptr;
}

extern "C" MemoryKind* ifrt_memorykind_canonicalize(MemoryKind* memory_kind, Device* device)
{
    return new MemoryKind(CanonicalizeMemoryKind(*memory_kind, device));
}

#include "../type_conversion.hpp"
#include "xla/python/ifrt/compiler.h"

using namespace xla::ifrt;
using namespace reactant;

extern "C" ifrt::LoadedExecutable* ifrt_compiler_compile(ifrt::Compiler* compiler, ifrt::Program* program)
{
    // apparently ifrt::CompileOptions is a legacy artifact so we don't use it and
    // set directly to the default
    auto program_ptr = std::make_unique<ifrt::Program>(*program);
    auto options = std::make_unique<ifrt::CompileOptions>();
    return MyValueOrThrow(
        compiler->Compile(std::move(program_ptr), std::move(options)))
        .release();
}

extern "C" ifrt::Executable* ifrt_compiler_compile_with_topology(ifrt::Compiler* compiler, ifrt::Program* program, const ifrt::Topology* topology)
{
    // apparently ifrt::CompileOptions is a legacy artifact so we don't use it and
    // set directly to the default
    auto options = std::make_unique<ifrt::CompileOptions>();
    auto program_ptr = std::make_unique<ifrt::Program>(*program);
    auto exec_ptr = MyValueOrThrow(compiler->Compile(std::move(program_ptr), *topology,
                                       std::move(options)))
                        .release();
    return exec_ptr;
}

extern "C" ifrt::LoadedExecutable* ifrt_compiler_deserialize_loadedexecutable(ifrt::Compiler* compiler, const char* data)
{
    // apparently ifrt::DeserializeExecutableOptions is a legacy artifact so we
    // don't use it and set directly to the default
    auto options = std::make_unique<ifrt::DeserializeExecutableOptions>();
    return MyValueOrThrow(compiler->DeserializeLoadedExecutable(
                              std::string(data), std::move(options)))
        .release();
}

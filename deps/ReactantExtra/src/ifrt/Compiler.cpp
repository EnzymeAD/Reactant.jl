#include "src/type_conversion.hpp"
#include "src/error_handling.hpp"
#include "xla/python/ifrt/compiler.h"

using namespace xla::ifrt;
using namespace reactant;

extern "C" LoadedExecutable* ifrt_compiler_compile(Compiler* compiler, Program* program)
{
    // apparently CompileOptions is a legacy artifact so we don't use it and
    // set directly to the default
    auto program_ptr = std::make_unique<Program>(*program);
    auto options = std::make_unique<CompileOptions>();
    return MyValueOrThrow(
        compiler->Compile(std::move(program_ptr), std::move(options)))
        .release();
}

extern "C" Executable* ifrt_compiler_compile_with_topology(Compiler* compiler, Program* program, const Topology* topology)
{
    // apparently CompileOptions is a legacy artifact so we don't use it and
    // set directly to the default
    auto options = std::make_unique<CompileOptions>();
    auto program_ptr = std::make_unique<Program>(*program);
    auto exec_ptr = MyValueOrThrow(compiler->Compile(std::move(program_ptr), *topology,
                                       std::move(options)))
                        .release();
    return exec_ptr;
}

extern "C" LoadedExecutable* ifrt_compiler_deserialize_loadedexecutable(Compiler* compiler, const char* data)
{
    // apparently DeserializeExecutableOptions is a legacy artifact so we
    // don't use it and set directly to the default
    auto options = std::make_unique<DeserializeExecutableOptions>();
    return MyValueOrThrow(compiler->DeserializeLoadedExecutable(
                              std::string(data), std::move(options)))
        .release();
}

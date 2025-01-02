#include "src/type_conversion.hpp"
#include "src/error_handling.hpp"
#include "xla/python/pjrt_ifrt/pjrt_executable.h"

using namespace xla::ifrt;
using namespace reactant;

// TODO there are problems with using `make_shared
// extern "C" Executable* ifrt_pjrt_executable_ctor(xla::PjRtExecutable*
// pjrt_executable, XlaCompileOptions* compile_options) {
//     auto pjrt_executable_shared =
//     std::make_shared<xla::PjRtExecutable>(*pjrt_executable); auto options =
//     std::make_unique<XlaCompileOptions>(*compile_options); return
//     MyValueOrThrow(PjRtExecutable::Create(pjrt_executable_shared,
//     std::move(options))).release();
// }

extern "C" void ifrt_pjrt_executable_free(PjRtExecutable* executable) { delete executable; }

extern "C" xla::PjRtExecutable* ifrt_pjrt_executable_pjrt_executable(PjRtExecutable* executable)
{
    return executable->pjrt_executable();
}

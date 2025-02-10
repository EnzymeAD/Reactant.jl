#include "src/type_conversion.hpp"
#include "src/error_handling.hpp"
#include "src/memory_management.hpp"
#include "xla/python/pjrt_ifrt/pjrt_executable.h"

using namespace xla::ifrt;
using namespace reactant;

// TODO is there any problem with ownership by using `std::shared_ptr` here?
extern "C" Executable* ifrt_pjrt_executable_ctor(Holded<std::shared_ptr<xla::PjRtExecutable>>* out_pjrt_executable, xla::PjRtExecutable* c_pjrt_executable) {
    auto pjrt_executable = std::shared_ptr<xla::PjRtExecutable>(c_pjrt_executable);
    (*out_pjrt_executable) = *capture(pjrt_executable);
    return MyValueOrThrow(PjRtExecutable::Create(pjrt_executable)).release();
}

extern "C" void ifrt_pjrt_executable_dtor(PjRtExecutable* executable) { delete executable; }

extern "C" xla::PjRtExecutable* ifrt_pjrt_executable_pjrt_executable(PjRtExecutable* executable)
{
    return executable->pjrt_executable();
}

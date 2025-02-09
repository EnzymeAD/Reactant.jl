#include "src/type_conversion.hpp"
#include "src/error_handling.hpp"
#include "xla/python/pjrt_ifrt/pjrt_executable.h"

using namespace xla::ifrt;
using namespace reactant;

// TODO add support for LoadedHostCallback
extern "C" LoadedExecutable* ifrt_pjrt_loadedexecutable_ctor(Holded<std::shared_ptr<xla::PjRtLoadedExecutable>>* out_pjrt_loaded_executable, PjRtCompatibleClient* client, xla::PjRtLoadedExecutable* c_pjrt_loaded_executable)
{
    auto pjrt_loaded_executable = std::shared_ptr<xla::PjRtLoadedExecutable>(c_pjrt_loaded_executable);
    (*out_pjrt_loaded_executable) = *capture(pjrt_loaded_executable);
    return MyValueOrThrow(PjRtLoadedExecutable::Create(client, pjrt_loaded_executable, std::vector<tsl::RCReference<LoadedHostCallback>>())).release();
}

// TODO add support for LoadedHostCallback
extern "C" LoadedExecutable* ifrt_pjrt_loadedexecutable_ctor_from_mlir_module(
    PjRtCompatibleClient* client,
    mlir::ModuleOp* module,
    xla::CompileOptions* compile_options)
{
    return MyValueOrThrow(
        PjRtLoadedExecutable::Create(
            client, *module, *compile_options,
            std::vector<tsl::RCReference<LoadedHostCallback>>()))
        .release();
}

extern "C" void ifrt_pjrt_loadedexecutable_dtor(PjRtLoadedExecutable* executable) { delete executable; }

extern "C" xla::PjRtLoadedExecutable* ifrt_pjrt_loadedexecutable_pjrt_loadedexecutable(PjRtLoadedExecutable* executable)
{
    return executable->pjrt_loaded_executable();
}

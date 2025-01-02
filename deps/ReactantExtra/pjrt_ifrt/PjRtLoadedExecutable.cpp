#include "type_conversion.hpp"
#include "xla/python/pjrt_ifrt/pjrt_executable.h"

using namespace xla::ifrt;
using namespace reactant;

// TODO add support for LoadedHostCallback
// TODO there are problems with using `make_shared
// extern "C" LoadedExecutable*
// ifrt_pjrt_loadedexecutable_ctor(PjRtCompatibleClient* client,
// xla::PjRtLoadedExecutable* pjrt_loaded_executable) {
//     auto pjrt_loaded_executable_ptr =
//     std::make_shared<xla::PjRtLoadedExecutable>(*pjrt_loaded_executable);
//     return MyValueOrThrow(PjRtLoadedExecutable::Create(client,
//     pjrt_loaded_executable_ptr,
//     std::vector<tsl::RCReference<LoadedHostCallback>>())).release();
// }

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

extern "C" void ifrt_pjrt_loadedexecutable_free(PjRtLoadedExecutable* executable) { delete executable; }

extern "C" xla::PjRtLoadedExecutable* ifrt_pjrt_loadedexecutable_pjrt_loadedexecutable(PjRtLoadedExecutable* executable)
{
    return executable->pjrt_loaded_executable();
}

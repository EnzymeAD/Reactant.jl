#include "src/type_conversion.hpp"
#include "src/error_handling.hpp"
#include "xla/python/pjrt_ifrt/pjrt_client.h"

using namespace xla::ifrt;
using namespace reactant;

// PjRtClient::CreateOptions
// TODO support more parameters of `PjRtClient::CreateOptions`
extern "C" PjRtClient::CreateOptions* ifrt_pjrt_client_create_options_ctor(xla::PjRtClient* c_pjrt_client)
{
    std::shared_ptr<xla::PjRtClient> pjrt_client = reactant::get_or_insert_shared(c_pjrt_client);
    return new PjRtClient::CreateOptions{pjrt_client};
}

extern "C" PjRtClient::CreateOptions* ifrt_pjrt_client_create_options_dtor(PjRtClient::CreateOptions* options)
{
    delete options;
}

// PjRtClient
extern "C" PjRtClient* ifrt_pjrt_client_create(PjRtClient::CreateOptions* options)
{
    return MyValueOrThrow(PjRtClient::Create(*options)).release();
}

extern "C" void ifrt_pjrt_client_dtor(PjRtClient* client)
{
    reactant::destruct_or_release_if_shared(client);
}

// NOTE we use `shared_ptr_pjrt_client` instead of `pjrt_client` because latter uses just the `shared_ptr::get` and we could delete it accidentally
extern "C" xla::PjRtClient* ifrt_pjrt_client_pjrt_client(PjRtClient* client)
{
    return reactant::capture_shared(client->shared_ptr_pjrt_client());
}

// NOTE we don't implement `CreatePjRtArray` because there are already equivalent methods in `PjRtArray` class

extern "C" PjRtCompatibleDevice* ifrt_pjrt_client_lookup_pjrt_device(PjRtClient* client, xla::PjRtDevice* pjrt_device)
{
    return MyValueOrThrow(client->LookupPjRtDevice(pjrt_device));
}

extern "C" PjRtCompatibleMemory* ifrt_pjrt_client_lookup_pjrt_memory(PjRtClient* client,
    xla::PjRtMemorySpace* pjrt_memory_space)
{
    return MyValueOrThrow(client->LookupPjRtMemory(pjrt_memory_space));
}

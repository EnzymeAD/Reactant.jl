#include "type_conversion.hpp"
#include "xla/python/pjrt_ifrt/pjrt_client.h"

using namespace xla::ifrt;
using namespace reactant;

// TODO support more parameters of `PjRtClient::CreateOptions`
extern "C" PjRtClient* ifrt_pjrt_client_ctor(xla::PjRtClient* pjrt_client)
{
    return MyValueOrThrow(
        PjRtClient::Create(PjRtClient::CreateOptions {
            std::shared_ptr<xla::PjRtClient> { pjrt_client } }))
        .release();
}

extern "C" void ifrt_pjrt_client_free(PjRtClient* client) { delete client; }

extern "C" xla::PjRtClient* ifrt_pjrt_client_pjrt_client(PjRtClient* client)
{
    return client->pjrt_client();
}

// TODO there are problems with using `make_shared
// extern "C" PjRtCompatibleArray*
// ifrt_pjrt_client_create_pjrt_array(PjRtClient* client, xla::PjRtBuffer*
// pjrt_buffer) {
//     auto buffer_ptr = std::make_shared<xla::PjRtBuffer>(*pjrt_buffer);
//     return MyValueOrThrow(client->CreatePjRtArray(buffer_ptr)).release();
// }

// TODO extern "C" PjRtCompatibleArray*
// ifrt_pjrt_client_create_pjrt_array_from_buffers(Shape* shape,
// PjRtBuffer** pjrt_buffers, int num_buffers) {}

extern "C" PjRtCompatibleDevice* ifrt_pjrt_client_lookup_pjrt_device(PjRtClient* client, xla::PjRtDevice* pjrt_device)
{
    return MyValueOrThrow(client->LookupPjRtDevice(pjrt_device));
}

extern "C" PjRtCompatibleMemory* ifrt_pjrt_client_lookup_pjrt_memory(PjRtClient* client,
    xla::PjRtMemorySpace* pjrt_memory_space)
{
    return MyValueOrThrow(client->LookupPjRtMemory(pjrt_memory_space));
}

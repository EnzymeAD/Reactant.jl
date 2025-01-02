#include "src/type_conversion.hpp"
#include "xla/python/pjrt_ifrt/pjrt_host_callback.h"

using namespace xla::ifrt;
using namespace reactant;

extern "C" PjRtHostSendAndRecvLoadedHostCallback* ifrt_pjrt_hostsendandrecv_loadhostcallback_ctor(PjRtClient* client, xla::HostCallback* host_callback)
{
    auto xla_callback_ptr = std::make_unique<xla::HostCallback>(*host_callback);
    return new PjRtHostSendAndRecvLoadedHostCallback(client, std::move(xla_callback_ptr));
}

extern "C" void ifrt_pjrt_hostsendandrecv_loadhostcallback_free(PjRtHostSendAndRecvLoadedHostCallback* host_callback) { delete host_callback; }

extern "C" xla::HostCallback* ifrt_pjrt_hostsendandrecv_loadhostcallback_host_callback(PjRtHostSendAndRecvLoadedHostCallback* host_callback)
{
    return new xla::HostCallback(host_callback->host_callback());
}

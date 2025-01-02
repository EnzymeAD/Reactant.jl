#include "../type_conversion.hpp"
#include "xla/python/ifrt/sharding.h"

using namespace xla::ifrt;
using namespace reactant;

// TODO ifrt_sharding_devices
// TODO ifrt_sharding_memory_kind

// extern "C" void ifrt_sharding_disassemble(ifrt::Sharding* sharding,
// ifrt::Shape* shape, char** error) {
//     auto status = sharding->Disassemble(*shape);
//     if (!status.ok()) {
//         auto str = status.message();
//         char* err = (char*)malloc(str.size()+1);
//         memcpy(err, str.data(), str.size()+1);
//         *error = err;
//     }
// }

// TODO ifrt_sharding_disassemble_dynamic_shape
// TODO ifrt_sharding_index_domains

extern "C" const char* ifrt_sharding_debug_string(ifrt::Sharding* sharding)
{
    return cstr_from_string(sharding->DebugString());
}

#include "src/type_conversion.hpp"
#include "xla/python/pjrt_ifrt/pjrt_compiler.h"

using namespace xla::ifrt;
using namespace reactant;

extern "C" PjRtCompiler* ifrt_pjrt_compiler_ctor(PjRtClient* client)
{
    return new PjRtCompiler(client);
}

extern "C" void ifrt_pjrt_compiler_free(PjRtCompiler* compiler) { delete compiler; }
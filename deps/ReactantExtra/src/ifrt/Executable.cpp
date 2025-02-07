#include "src/type_conversion.hpp"
#include "src/error_handling.hpp"
#include "xla/python/ifrt/executable.h"
#include "xla/xla_data.pb.h"

using namespace xla::ifrt;
using namespace reactant;

extern "C" const char* ifrt_executable_name(Executable* executable)
{
    return convert(Type<const char*>(), executable->name());
}

extern "C" const char* ifrt_executable_fingerprint(Executable* executable)
{
    auto result = MyValueOrThrow(executable->Fingerprint());
    if (!result.has_value())
        return "";
    return convert(Type<const char*>(), result.value());
}

extern "C" const char* ifrt_executable_serialize(Executable* executable)
{
    return convert(Type<const char*>(), MyValueOrThrow(executable->Serialize()));
}

extern "C" int ifrt_executable_num_devices(Executable* executable)
{
    return executable->num_devices();
}

extern "C" int64_t ifrt_executable_byte_size(Executable* executable)
{
    return executable->SizeOfGeneratedCodeInBytes();
}

// TODO xla::Executable::GetCompiledMemoryStats

extern "C" span<xla::OpSharding*> ifrt_executable_parameter_shardings(Executable* executable)
{
    auto shardings = executable->GetParameterShardings();
    if (!shardings.has_value())
        return {};

    return convert(Type<span<xla::OpSharding*>>(), shardings.value());
}

extern "C" span<xla::OpSharding*> ifrt_executable_output_shardings(Executable* executable)
{
    auto shardings = executable->GetOutputShardings();
    if (!shardings.has_value())
        return {};
    return convert(Type<span<xla::OpSharding*>>{}, shardings.value());
}

extern "C" span<xla::PjRtLayout*> ifrt_executable_parameter_layouts(Executable* executable)
{
    auto layouts = MyValueOrThrow(executable->GetParameterLayouts());
    return convert(Type<span<xla::PjRtLayout*>>(), layouts);
}

extern "C" span<xla::PjRtLayout*> ifrt_executable_output_layouts(Executable* executable)
{
    auto layouts = MyValueOrThrow(executable->GetOutputLayouts());
    return convert(Type<span<xla::PjRtLayout*>>(), layouts);
}

extern "C" span<xla::HloModule*> ifrt_executable_hlo_modules(Executable* executable)
{
    auto modules = MyValueOrThrow(executable->GetHloModules());
    return convert(Type<span<xla::HloModule*>>(), modules);
}

extern "C" span<span<const char*>> ifrt_executable_output_memory_kinds(Executable* executable)
{
    auto memory_kinds = MyValueOrThrow(executable->GetOutputMemoryKinds());
    span<span<const char*>> text_matrix = span<span<const char*>>(memory_kinds.size(), new span<const char*>[memory_kinds.size()]);
    for (int i = 0; i < memory_kinds.size(); i++) {
        text_matrix[i] = span<const char*>(memory_kinds[i].size(), new const char*[memory_kinds[i].size()]);
        for (int j = 0; j < memory_kinds.size(); j++) {
            text_matrix[i][j] = convert(Type<const char*>(), memory_kinds[i][j]);
        }
    }
    return text_matrix;
}

// TODO xla::Executable::GetCostAnalysis

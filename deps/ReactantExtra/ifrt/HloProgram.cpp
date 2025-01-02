#include "src/type_conversion.hpp"
#include "xla/python/ifrt/hlo/hlo_program.h"

using namespace xla::ifrt;
using namespace reactant;

extern "C" HloProgram* ifrt_hloprogram_ctor()
{
    return new HloProgram();
}

extern "C" HloProgram*
ifrt_hloprogram_ctor_with_module(mlir::ModuleOp* module)
{
    return new HloProgram(*module);
}

// extern "C" HloProgram*
// ifrt_hloprogram_ctor_with_context_and_module(mlir::MLIRContext* context,
// mlir::ModuleOp* module) {
//     auto context_ptr = std::make_unique<mlir::MLIRContext>(*context);
//     return new HloProgram(std::move(context_ptr), *module);
// }

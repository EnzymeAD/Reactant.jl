#include "src/type_conversion.hpp"
#include "xla/pjrt/pjrt_future.h"

// using namespace xla::ifrt;
using namespace reactant;

// extern "C" bool pjrt_futurebase_isvalid(xla::PjRtFuture* future) {
//     return future->IsValid();
// }

// extern "C" bool pjrt_futurebase_isready(xla::PjRtFuture* future) {
//     return future->IsReady();
// }

// extern "C" bool pjrt_futurebase_isknownready(xla::PjRtFuture* future) {
//     return future->IsKnownReady();
// }

// TODO AssertHappensBefore
// TODO OnBlockStart, OnBlockEnd

// extern "C" void pjrt_futurebase_block_until_ready(xla::PjRtFuture* future) {
//     future->BlockUntilReady();
// }

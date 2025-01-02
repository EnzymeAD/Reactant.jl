#include "../type_conversion.hpp"
#include "xla/python/ifrt/tuple.h"

using namespace xla::ifrt;
using namespace reactant;

extern "C" int ifrt_tuple_arity(ifrt::Tuple* tuple) { return tuple->Arity(); }

// TODO ifrt::Tuple::Unpack

module IFRT

using CxxWrap
using Reactant_jll

@wrapmodule(() -> Reactant_jll.libReactantExtra, :reactant_module_ifrt)

end

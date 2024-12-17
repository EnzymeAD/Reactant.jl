module IFRT

using CxxWrap
using Reactant_jll

@wrapmodule(() -> joinpath(Reactant_jll.libdir, :reactant_module_ifrt))

end

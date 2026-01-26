module ReactantOMEinsumExt

using Reactant: @skip_rewrite_func
using OMEinsum

function __init__()
    @skip_rewrite_func OMEinsum.analyze_binary
end

end

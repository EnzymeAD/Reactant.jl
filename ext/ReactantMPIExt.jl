module ReactantMPIExt

using Reactant
using MPI

function __init__()
    for name in (
        "MPI_Init",
        "MPI_Finalize",
        "MPI_Comm_rank",
        "MPI_Comm_size",
        "MPI_Send",
        "MPI_Recv",
        "MPI_Isend",
        "MPI_Irecv",
        "MPI_Wait",
        "MPI_Request_free",
    )
        sym = Libdl.dlsym(MPI.API.libmpi, name)
        @ccall MLIR.API.mlir_c.EnzymeJaXMapSymbol(name::Cstring, sym::Ptr{Cvoid})::Cvoid
    end
end

end

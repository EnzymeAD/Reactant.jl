module IFRT

struct Future
    ptr::Ptr{Cvoid}
end

struct Value
    ptr::Ptr{Cvoid}
end

struct Tuple
    ptr::Ptr{Cvoid}
end

struct DType
    ptr::Ptr{Cvoid}
end

struct Shape
    ptr::Ptr{Cvoid}
end

struct DynamicShape
    ptr::Ptr{Cvoid}
end

struct Index
    ptr::Ptr{Cvoid}
end

struct IndexDomain
    ptr::Ptr{Cvoid}
end

struct MemoryKind
    ptr::Ptr{Cvoid}
end

struct Memory
    ptr::Ptr{Cvoid}
end

struct Device
    ptr::Ptr{Cvoid}
end

struct Sharding
    ptr::Ptr{Cvoid}
end

struct Array
    ptr::Ptr{Cvoid}
end

struct Topology
    ptr::Ptr{Cvoid}
end

struct Client
    ptr::Ptr{Cvoid}
end

struct HostCallback
    ptr::Ptr{Cvoid}
end

struct LoadedHostCallback
    ptr::Ptr{Cvoid}
end

struct Executable
    ptr::Ptr{Cvoid}
end

struct LoadedExecutable
    ptr::Ptr{Cvoid}
end

struct Compiler
    ptr::Ptr{Cvoid}
end

end

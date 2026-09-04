using CEnum: CEnum, @cenum

const IS_LIBC_MUSL = occursin("musl", Base.MACHINE)

if Sys.islinux() && Sys.ARCH === :aarch64 && !IS_LIBC_MUSL
    const __off_t = Clong
    const off_t = __off_t
elseif Sys.islinux() && Sys.ARCH === :aarch64 && IS_LIBC_MUSL
    const off_t = Clong
elseif Sys.islinux() && startswith(string(Sys.ARCH), "arm") && !IS_LIBC_MUSL
    const __off_t = Clong
    const off_t = __off_t
elseif Sys.islinux() && startswith(string(Sys.ARCH), "arm") && IS_LIBC_MUSL
    const off_t = Clonglong
elseif Sys.islinux() && Sys.ARCH === :i686 && !IS_LIBC_MUSL
    const __off_t = Clong
    const off_t = __off_t
elseif Sys.islinux() && Sys.ARCH === :i686 && IS_LIBC_MUSL
    const off_t = Clonglong
elseif Sys.iswindows() && Sys.ARCH === :i686
    const off32_t = Clong
    const off_t = off32_t
elseif Sys.islinux() && Sys.ARCH === :powerpc64le
    const __off_t = Clong
    const off_t = __off_t
elseif Sys.isapple()
    const __darwin_off_t = Int64
    const off_t = __darwin_off_t
elseif Sys.islinux() && Sys.ARCH === :x86_64 && !IS_LIBC_MUSL
    const __off_t = Clong
    const off_t = __off_t
elseif Sys.islinux() && Sys.ARCH === :x86_64 && IS_LIBC_MUSL
    const off_t = Clong
elseif Sys.isbsd() && !Sys.isapple()
    const __off_t = Int64
    const off_t = __off_t
elseif Sys.iswindows() && Sys.ARCH === :x86_64
    const off32_t = Clong
    const off_t = off32_t
end

struct MlirDialectHandle
    ptr::Ptr{Cvoid}
end

"""
    MlirLlvmThreadPool

Re-export llvm::ThreadPool so as to avoid including the LLVM C API directly.
"""
struct MlirLlvmThreadPool
    ptr::Ptr{Cvoid}
end

"""
    MlirLlvmRawFdOStream

Re-export llvm::raw\\_fd\\_ostream so as to avoid including the LLVM C API directly.
"""
struct MlirLlvmRawFdOStream
    ptr::Ptr{Cvoid}
end

struct MlirTypeID
    ptr::Ptr{Cvoid}
end

struct MlirTypeIDAllocator
    ptr::Ptr{Cvoid}
end

"""
    MlirStringRef

A pointer to a sized fragment of a string, not necessarily null-terminated. Does not own the underlying string. This is equivalent to llvm::StringRef.

| Field  | Note                          |
| :----- | :---------------------------- |
| data   | Pointer to the first symbol.  |
| length | Length of the fragment.       |
"""
struct MlirStringRef
    data::Cstring
    length::Csize_t
end

"""
    mlirStringRefCreate(str, length)

Constructs a string reference from the pointer and length. The pointer need not reference to a null-terminated string.
"""
function mlirStringRefCreate(str, length)
    @ccall Reactant_jll.libReactantExtra.mlirStringRefCreate(
        str::Cstring, length::Csize_t
    )::MlirStringRef
end

"""
    mlirStringRefCreateFromCString(str)

Constructs a string reference from a null-terminated C string. Prefer [`mlirStringRefCreate`](@ref) if the length of the string is known.
"""
function mlirStringRefCreateFromCString(str)
    @ccall Reactant_jll.libReactantExtra.mlirStringRefCreateFromCString(
        str::Cstring
    )::MlirStringRef
end

"""
    mlirStringRefEqual(string, other)

Returns true if two string references are equal, false otherwise.
"""
function mlirStringRefEqual(string, other)
    @ccall Reactant_jll.libReactantExtra.mlirStringRefEqual(
        string::MlirStringRef, other::MlirStringRef
    )::Bool
end

# typedef void ( * MlirStringCallback ) ( MlirStringRef , void * )
"""
A callback for returning string references.

This function is called back by the functions that need to return a reference to the portion of the string with the following arguments: - an [`MlirStringRef`](@ref) representing the current portion of the string - a pointer to user data forwarded from the printing call.
"""
const MlirStringCallback = Ptr{Cvoid}

"""
    MlirLogicalResult

A logical result value, essentially a boolean with named states. LLVM convention for using boolean values to designate success or failure of an operation is a moving target, so MLIR opted for an explicit class. Instances of [`MlirLogicalResult`](@ref) must only be inspected using the associated functions.
"""
struct MlirLogicalResult
    value::Int8
end

"""
    mlirLogicalResultIsSuccess(res)

Checks if the given logical result represents a success.
"""
function mlirLogicalResultIsSuccess(res)
    @ccall Reactant_jll.libReactantExtra.mlirLogicalResultIsSuccess(
        res::MlirLogicalResult
    )::Bool
end

"""
    mlirLogicalResultIsFailure(res)

Checks if the given logical result represents a failure.
"""
function mlirLogicalResultIsFailure(res)
    @ccall Reactant_jll.libReactantExtra.mlirLogicalResultIsFailure(
        res::MlirLogicalResult
    )::Bool
end

"""
    mlirLogicalResultSuccess()

Creates a logical result representing a success.
"""
function mlirLogicalResultSuccess()
    @ccall Reactant_jll.libReactantExtra.mlirLogicalResultSuccess()::MlirLogicalResult
end

"""
    mlirLogicalResultFailure()

Creates a logical result representing a failure.
"""
function mlirLogicalResultFailure()
    @ccall Reactant_jll.libReactantExtra.mlirLogicalResultFailure()::MlirLogicalResult
end

"""
    mlirLlvmThreadPoolCreate()

Create an LLVM thread pool. This is reexported here to avoid directly pulling in the LLVM headers directly.
"""
function mlirLlvmThreadPoolCreate()
    @ccall Reactant_jll.libReactantExtra.mlirLlvmThreadPoolCreate()::MlirLlvmThreadPool
end

"""
    mlirLlvmThreadPoolDestroy(pool)

Destroy an LLVM thread pool.
"""
function mlirLlvmThreadPoolDestroy(pool)
    @ccall Reactant_jll.libReactantExtra.mlirLlvmThreadPoolDestroy(
        pool::MlirLlvmThreadPool
    )::Cvoid
end

"""
    mlirLlvmThreadPoolGetMaxConcurrency(pool)

Returns the maximum number of threads in the thread pool.
"""
function mlirLlvmThreadPoolGetMaxConcurrency(pool)
    @ccall Reactant_jll.libReactantExtra.mlirLlvmThreadPoolGetMaxConcurrency(
        pool::MlirLlvmThreadPool
    )::Cint
end

"""
    mlirLlvmRawFdOStreamCreate(path, binary, errorCallback, userData)

Create a raw\\_fd\\_ostream for the given path. This wrapper is needed because std::ostream does not provide the file sharing semantics required on Windows. - `path`: output file path. - `binary`: controls text vs binary mode. - `errorCallback`: called with an error message on failure (optional). - `userData`: forwarded to `errorCallback` so it can copy the error message into caller-owned storage (e.g., a `std::string`). On failure, returns a null stream and invokes the optional error callback with the error message.
"""
function mlirLlvmRawFdOStreamCreate(path, binary, errorCallback, userData)
    @ccall Reactant_jll.libReactantExtra.mlirLlvmRawFdOStreamCreate(
        path::Cstring, binary::Bool, errorCallback::MlirStringCallback, userData::Ptr{Cvoid}
    )::MlirLlvmRawFdOStream
end

"""
    mlirLlvmRawFdOStreamWrite(stream, string)

Write a string to a raw\\_fd\\_ostream created with [`mlirLlvmRawFdOStreamCreate`](@ref).
"""
function mlirLlvmRawFdOStreamWrite(stream, string)
    @ccall Reactant_jll.libReactantExtra.mlirLlvmRawFdOStreamWrite(
        stream::MlirLlvmRawFdOStream, string::MlirStringRef
    )::Cvoid
end

"""
    mlirLlvmRawFdOStreamIsNull(stream)

Checks if a raw\\_fd\\_ostream is null.
"""
function mlirLlvmRawFdOStreamIsNull(stream)
    @ccall Reactant_jll.libReactantExtra.mlirLlvmRawFdOStreamIsNull(
        stream::MlirLlvmRawFdOStream
    )::Bool
end

"""
    mlirLlvmRawFdOStreamDestroy(stream)

Destroy a raw\\_fd\\_ostream created with [`mlirLlvmRawFdOStreamCreate`](@ref).
"""
function mlirLlvmRawFdOStreamDestroy(stream)
    @ccall Reactant_jll.libReactantExtra.mlirLlvmRawFdOStreamDestroy(
        stream::MlirLlvmRawFdOStream
    )::Cvoid
end

"""
    mlirTypeIDCreate(ptr)

`ptr` must be 8 byte aligned and unique to a type valid for the duration of the returned type id's usage
"""
function mlirTypeIDCreate(ptr)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIDCreate(ptr::Ptr{Cvoid})::MlirTypeID
end

"""
    mlirTypeIDIsNull(typeID)

Checks whether a type id is null.
"""
function mlirTypeIDIsNull(typeID)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIDIsNull(typeID::MlirTypeID)::Bool
end

"""
    mlirTypeIDEqual(typeID1, typeID2)

Checks if two type ids are equal.
"""
function mlirTypeIDEqual(typeID1, typeID2)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIDEqual(
        typeID1::MlirTypeID, typeID2::MlirTypeID
    )::Bool
end

"""
    mlirTypeIDHashValue(typeID)

Returns the hash value of the type id.
"""
function mlirTypeIDHashValue(typeID)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIDHashValue(typeID::MlirTypeID)::Csize_t
end

"""
    mlirTypeIDAllocatorCreate()

Creates a type id allocator for dynamic type id creation
"""
function mlirTypeIDAllocatorCreate()
    @ccall Reactant_jll.libReactantExtra.mlirTypeIDAllocatorCreate()::MlirTypeIDAllocator
end

"""
    mlirTypeIDAllocatorDestroy(allocator)

Deallocates the allocator and all allocated type ids
"""
function mlirTypeIDAllocatorDestroy(allocator)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIDAllocatorDestroy(
        allocator::MlirTypeIDAllocator
    )::Cvoid
end

"""
    mlirTypeIDAllocatorAllocateTypeID(allocator)

Allocates a type id that is valid for the lifetime of the allocator
"""
function mlirTypeIDAllocatorAllocateTypeID(allocator)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIDAllocatorAllocateTypeID(
        allocator::MlirTypeIDAllocator
    )::MlirTypeID
end

struct MlirAsmState
    ptr::Ptr{Cvoid}
end

struct MlirBytecodeWriterConfig
    ptr::Ptr{Cvoid}
end

struct MlirContext
    ptr::Ptr{Cvoid}
end

struct MlirDialect
    ptr::Ptr{Cvoid}
end

struct MlirDialectRegistry
    ptr::Ptr{Cvoid}
end

struct MlirOperation
    ptr::Ptr{Cvoid}
end

struct MlirOpOperand
    ptr::Ptr{Cvoid}
end

struct MlirOpPrintingFlags
    ptr::Ptr{Cvoid}
end

struct MlirBlock
    ptr::Ptr{Cvoid}
end

struct MlirRegion
    ptr::Ptr{Cvoid}
end

struct MlirSymbolTable
    ptr::Ptr{Cvoid}
end

struct MlirIRMapping
    ptr::Ptr{Cvoid}
end

struct MlirAttribute
    ptr::Ptr{Cvoid}
end

struct MlirIdentifier
    ptr::Ptr{Cvoid}
end

struct MlirLocation
    ptr::Ptr{Cvoid}
end

struct MlirModule
    ptr::Ptr{Cvoid}
end

struct MlirType
    ptr::Ptr{Cvoid}
end

struct MlirValue
    ptr::Ptr{Cvoid}
end

"""
    MlirNamedAttribute

Named MLIR attribute.

A named attribute is essentially a (name, attribute) pair where the name is a string.
"""
struct MlirNamedAttribute
    name::MlirIdentifier
    attribute::MlirAttribute
end

"""
    mlirContextCreate()

Creates an MLIR context and transfers its ownership to the caller. This sets the default multithreading option (enabled).
"""
function mlirContextCreate()
    @ccall Reactant_jll.libReactantExtra.mlirContextCreate()::MlirContext
end

"""
    mlirContextCreateWithThreading(threadingEnabled)

Creates an MLIR context with an explicit setting of the multithreading setting and transfers its ownership to the caller.
"""
function mlirContextCreateWithThreading(threadingEnabled)
    @ccall Reactant_jll.libReactantExtra.mlirContextCreateWithThreading(
        threadingEnabled::Bool
    )::MlirContext
end

"""
    mlirContextCreateWithRegistry(registry, threadingEnabled)

Creates an MLIR context, setting the multithreading setting explicitly and pre-loading the dialects from the provided DialectRegistry.
"""
function mlirContextCreateWithRegistry(registry, threadingEnabled)
    @ccall Reactant_jll.libReactantExtra.mlirContextCreateWithRegistry(
        registry::MlirDialectRegistry, threadingEnabled::Bool
    )::MlirContext
end

"""
    mlirContextEqual(ctx1, ctx2)

Checks if two contexts are equal.
"""
function mlirContextEqual(ctx1, ctx2)
    @ccall Reactant_jll.libReactantExtra.mlirContextEqual(
        ctx1::MlirContext, ctx2::MlirContext
    )::Bool
end

"""
    mlirContextIsNull(context)

Checks whether a context is null.
"""
function mlirContextIsNull(context)
    @ccall Reactant_jll.libReactantExtra.mlirContextIsNull(context::MlirContext)::Bool
end

"""
    mlirContextDestroy(context)

Takes an MLIR context owned by the caller and destroys it.
"""
function mlirContextDestroy(context)
    @ccall Reactant_jll.libReactantExtra.mlirContextDestroy(context::MlirContext)::Cvoid
end

"""
    mlirContextSetAllowUnregisteredDialects(context, allow)

Sets whether unregistered dialects are allowed in this context.
"""
function mlirContextSetAllowUnregisteredDialects(context, allow)
    @ccall Reactant_jll.libReactantExtra.mlirContextSetAllowUnregisteredDialects(
        context::MlirContext, allow::Bool
    )::Cvoid
end

"""
    mlirContextGetAllowUnregisteredDialects(context)

Returns whether the context allows unregistered dialects.
"""
function mlirContextGetAllowUnregisteredDialects(context)
    @ccall Reactant_jll.libReactantExtra.mlirContextGetAllowUnregisteredDialects(
        context::MlirContext
    )::Bool
end

"""
    mlirContextGetNumRegisteredDialects(context)

Returns the number of dialects registered with the given context. A registered dialect will be loaded if needed by the parser.
"""
function mlirContextGetNumRegisteredDialects(context)
    @ccall Reactant_jll.libReactantExtra.mlirContextGetNumRegisteredDialects(
        context::MlirContext
    )::Cptrdiff_t
end

"""
    mlirContextAppendDialectRegistry(ctx, registry)

Append the contents of the given dialect registry to the registry associated with the context.
"""
function mlirContextAppendDialectRegistry(ctx, registry)
    @ccall Reactant_jll.libReactantExtra.mlirContextAppendDialectRegistry(
        ctx::MlirContext, registry::MlirDialectRegistry
    )::Cvoid
end

"""
    mlirContextGetNumLoadedDialects(context)

Returns the number of dialects loaded by the context.
"""
function mlirContextGetNumLoadedDialects(context)
    @ccall Reactant_jll.libReactantExtra.mlirContextGetNumLoadedDialects(
        context::MlirContext
    )::Cptrdiff_t
end

"""
    mlirContextGetOrLoadDialect(context, name)

Gets the dialect instance owned by the given context using the dialect namespace to identify it, loads (i.e., constructs the instance of) the dialect if necessary. If the dialect is not registered with the context, returns null. Use mlirContextLoad<Name>Dialect to load an unregistered dialect.
"""
function mlirContextGetOrLoadDialect(context, name)
    @ccall Reactant_jll.libReactantExtra.mlirContextGetOrLoadDialect(
        context::MlirContext, name::MlirStringRef
    )::MlirDialect
end

"""
    mlirContextGetLoadedDialect(context, name)

Gets the dialect instance owned by the given context using the dialect namespace to identify it. If the dialect is not loaded by the context, returns null. Use [`mlirContextGetOrLoadDialect`](@ref) to load a dialect if it is registered with the context.
"""
function mlirContextGetLoadedDialect(context, name)
    @ccall Reactant_jll.libReactantExtra.mlirContextGetLoadedDialect(
        context::MlirContext, name::MlirStringRef
    )::MlirDialect
end

"""
    mlirContextEnableMultithreading(context, enable)

Set threading mode (must be set to false to mlir-print-ir-after-all).
"""
function mlirContextEnableMultithreading(context, enable)
    @ccall Reactant_jll.libReactantExtra.mlirContextEnableMultithreading(
        context::MlirContext, enable::Bool
    )::Cvoid
end

"""
    mlirContextLoadAllAvailableDialects(context)

Eagerly loads all available dialects registered with a context, making them available for use for IR construction.
"""
function mlirContextLoadAllAvailableDialects(context)
    @ccall Reactant_jll.libReactantExtra.mlirContextLoadAllAvailableDialects(
        context::MlirContext
    )::Cvoid
end

"""
    mlirContextIsRegisteredOperation(context, name)

Returns whether the given fully-qualified operation (i.e. 'dialect.operation') is registered with the context. This will return true if the dialect is loaded and the operation is registered within the dialect.
"""
function mlirContextIsRegisteredOperation(context, name)
    @ccall Reactant_jll.libReactantExtra.mlirContextIsRegisteredOperation(
        context::MlirContext, name::MlirStringRef
    )::Bool
end

"""
    mlirContextSetThreadPool(context, threadPool)

Sets the thread pool of the context explicitly, enabling multithreading in the process. This API should be used to avoid re-creating thread pools in long-running applications that perform multiple compilations, see the C++ documentation for MLIRContext for details.
"""
function mlirContextSetThreadPool(context, threadPool)
    @ccall Reactant_jll.libReactantExtra.mlirContextSetThreadPool(
        context::MlirContext, threadPool::MlirLlvmThreadPool
    )::Cvoid
end

"""
    mlirContextGetNumThreads(context)

Gets the number of threads of the thread pool of the context when multithreading is enabled. Returns 1 if no multithreading.
"""
function mlirContextGetNumThreads(context)
    @ccall Reactant_jll.libReactantExtra.mlirContextGetNumThreads(
        context::MlirContext
    )::Cuint
end

"""
    mlirContextGetThreadPool(context)

Gets the thread pool of the context when enabled multithreading, otherwise an assertion is raised.
"""
function mlirContextGetThreadPool(context)
    @ccall Reactant_jll.libReactantExtra.mlirContextGetThreadPool(
        context::MlirContext
    )::MlirLlvmThreadPool
end

"""
    mlirDialectGetContext(dialect)

Returns the context that owns the dialect.
"""
function mlirDialectGetContext(dialect)
    @ccall Reactant_jll.libReactantExtra.mlirDialectGetContext(
        dialect::MlirDialect
    )::MlirContext
end

"""
    mlirDialectIsNull(dialect)

Checks if the dialect is null.
"""
function mlirDialectIsNull(dialect)
    @ccall Reactant_jll.libReactantExtra.mlirDialectIsNull(dialect::MlirDialect)::Bool
end

"""
    mlirDialectEqual(dialect1, dialect2)

Checks if two dialects that belong to the same context are equal. Dialects from different contexts will not compare equal.
"""
function mlirDialectEqual(dialect1, dialect2)
    @ccall Reactant_jll.libReactantExtra.mlirDialectEqual(
        dialect1::MlirDialect, dialect2::MlirDialect
    )::Bool
end

"""
    mlirDialectGetNamespace(dialect)

Returns the namespace of the given dialect.
"""
function mlirDialectGetNamespace(dialect)
    @ccall Reactant_jll.libReactantExtra.mlirDialectGetNamespace(
        dialect::MlirDialect
    )::MlirStringRef
end

"""
    mlirDialectHandleGetNamespace(arg1)

Returns the namespace associated with the provided dialect handle.
"""
function mlirDialectHandleGetNamespace(arg1)
    @ccall Reactant_jll.libReactantExtra.mlirDialectHandleGetNamespace(
        arg1::MlirDialectHandle
    )::MlirStringRef
end

"""
    mlirDialectHandleInsertDialect(arg1, arg2)

Inserts the dialect associated with the provided dialect handle into the provided dialect registry
"""
function mlirDialectHandleInsertDialect(arg1, arg2)
    @ccall Reactant_jll.libReactantExtra.mlirDialectHandleInsertDialect(
        arg1::MlirDialectHandle, arg2::MlirDialectRegistry
    )::Cvoid
end

"""
    mlirDialectHandleRegisterDialect(arg1, arg2)

Registers the dialect associated with the provided dialect handle.
"""
function mlirDialectHandleRegisterDialect(arg1, arg2)
    @ccall Reactant_jll.libReactantExtra.mlirDialectHandleRegisterDialect(
        arg1::MlirDialectHandle, arg2::MlirContext
    )::Cvoid
end

"""
    mlirDialectHandleLoadDialect(arg1, arg2)

Loads the dialect associated with the provided dialect handle.
"""
function mlirDialectHandleLoadDialect(arg1, arg2)
    @ccall Reactant_jll.libReactantExtra.mlirDialectHandleLoadDialect(
        arg1::MlirDialectHandle, arg2::MlirContext
    )::MlirDialect
end

"""
    mlirDialectRegistryCreate()

Creates a dialect registry and transfers its ownership to the caller.
"""
function mlirDialectRegistryCreate()
    @ccall Reactant_jll.libReactantExtra.mlirDialectRegistryCreate()::MlirDialectRegistry
end

"""
    mlirDialectRegistryIsNull(registry)

Checks if the dialect registry is null.
"""
function mlirDialectRegistryIsNull(registry)
    @ccall Reactant_jll.libReactantExtra.mlirDialectRegistryIsNull(
        registry::MlirDialectRegistry
    )::Bool
end

"""
    mlirDialectRegistryDestroy(registry)

Takes a dialect registry owned by the caller and destroys it.
"""
function mlirDialectRegistryDestroy(registry)
    @ccall Reactant_jll.libReactantExtra.mlirDialectRegistryDestroy(
        registry::MlirDialectRegistry
    )::Cvoid
end

"""
    mlirLocationGetAttribute(location)

Returns the underlying location attribute of this location.
"""
function mlirLocationGetAttribute(location)
    @ccall Reactant_jll.libReactantExtra.mlirLocationGetAttribute(
        location::MlirLocation
    )::MlirAttribute
end

"""
    mlirLocationFromAttribute(attribute)

Creates a location from a location attribute.
"""
function mlirLocationFromAttribute(attribute)
    @ccall Reactant_jll.libReactantExtra.mlirLocationFromAttribute(
        attribute::MlirAttribute
    )::MlirLocation
end

"""
    mlirLocationFileLineColGet(context, filename, line, col)

Creates an File/Line/Column location owned by the given context.
"""
function mlirLocationFileLineColGet(context, filename, line, col)
    @ccall Reactant_jll.libReactantExtra.mlirLocationFileLineColGet(
        context::MlirContext, filename::MlirStringRef, line::Cuint, col::Cuint
    )::MlirLocation
end

"""
    mlirLocationFileLineColRangeGet(context, filename, start_line, start_col, end_line, end_col)

Creates an File/Line/Column range location owned by the given context.
"""
function mlirLocationFileLineColRangeGet(
    context, filename, start_line, start_col, end_line, end_col
)
    @ccall Reactant_jll.libReactantExtra.mlirLocationFileLineColRangeGet(
        context::MlirContext,
        filename::MlirStringRef,
        start_line::Cuint,
        start_col::Cuint,
        end_line::Cuint,
        end_col::Cuint,
    )::MlirLocation
end

"""
    mlirLocationFileLineColRangeGetFilename(location)

Getter for filename of FileLineColRange.
"""
function mlirLocationFileLineColRangeGetFilename(location)
    @ccall Reactant_jll.libReactantExtra.mlirLocationFileLineColRangeGetFilename(
        location::MlirLocation
    )::MlirIdentifier
end

"""
    mlirLocationFileLineColRangeGetStartLine(location)

Getter for start\\_line of FileLineColRange.
"""
function mlirLocationFileLineColRangeGetStartLine(location)
    @ccall Reactant_jll.libReactantExtra.mlirLocationFileLineColRangeGetStartLine(
        location::MlirLocation
    )::Cint
end

"""
    mlirLocationFileLineColRangeGetStartColumn(location)

Getter for start\\_column of FileLineColRange.
"""
function mlirLocationFileLineColRangeGetStartColumn(location)
    @ccall Reactant_jll.libReactantExtra.mlirLocationFileLineColRangeGetStartColumn(
        location::MlirLocation
    )::Cint
end

"""
    mlirLocationFileLineColRangeGetEndLine(location)

Getter for end\\_line of FileLineColRange.
"""
function mlirLocationFileLineColRangeGetEndLine(location)
    @ccall Reactant_jll.libReactantExtra.mlirLocationFileLineColRangeGetEndLine(
        location::MlirLocation
    )::Cint
end

"""
    mlirLocationFileLineColRangeGetEndColumn(location)

Getter for end\\_column of FileLineColRange.
"""
function mlirLocationFileLineColRangeGetEndColumn(location)
    @ccall Reactant_jll.libReactantExtra.mlirLocationFileLineColRangeGetEndColumn(
        location::MlirLocation
    )::Cint
end

"""
    mlirLocationFileLineColRangeGetTypeID()

TypeID Getter for FileLineColRange.
"""
function mlirLocationFileLineColRangeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirLocationFileLineColRangeGetTypeID()::MlirTypeID
end

"""
    mlirLocationIsAFileLineColRange(location)

Checks whether the given location is an FileLineColRange.
"""
function mlirLocationIsAFileLineColRange(location)
    @ccall Reactant_jll.libReactantExtra.mlirLocationIsAFileLineColRange(
        location::MlirLocation
    )::Bool
end

"""
    mlirLocationCallSiteGet(callee, caller)

Creates a call site location with a callee and a caller.
"""
function mlirLocationCallSiteGet(callee, caller)
    @ccall Reactant_jll.libReactantExtra.mlirLocationCallSiteGet(
        callee::MlirLocation, caller::MlirLocation
    )::MlirLocation
end

"""
    mlirLocationCallSiteGetCallee(location)

Getter for callee of CallSite.
"""
function mlirLocationCallSiteGetCallee(location)
    @ccall Reactant_jll.libReactantExtra.mlirLocationCallSiteGetCallee(
        location::MlirLocation
    )::MlirLocation
end

"""
    mlirLocationCallSiteGetCaller(location)

Getter for caller of CallSite.
"""
function mlirLocationCallSiteGetCaller(location)
    @ccall Reactant_jll.libReactantExtra.mlirLocationCallSiteGetCaller(
        location::MlirLocation
    )::MlirLocation
end

"""
    mlirLocationCallSiteGetTypeID()

TypeID Getter for CallSite.
"""
function mlirLocationCallSiteGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirLocationCallSiteGetTypeID()::MlirTypeID
end

"""
    mlirLocationIsACallSite(location)

Checks whether the given location is an CallSite.
"""
function mlirLocationIsACallSite(location)
    @ccall Reactant_jll.libReactantExtra.mlirLocationIsACallSite(
        location::MlirLocation
    )::Bool
end

"""
    mlirLocationFusedGet(ctx, nLocations, locations, metadata)

Creates a fused location with an array of locations and metadata.
"""
function mlirLocationFusedGet(ctx, nLocations, locations, metadata)
    @ccall Reactant_jll.libReactantExtra.mlirLocationFusedGet(
        ctx::MlirContext,
        nLocations::Cptrdiff_t,
        locations::Ptr{MlirLocation},
        metadata::MlirAttribute,
    )::MlirLocation
end

"""
    mlirLocationFusedGetNumLocations(location)

Getter for number of locations fused together.
"""
function mlirLocationFusedGetNumLocations(location)
    @ccall Reactant_jll.libReactantExtra.mlirLocationFusedGetNumLocations(
        location::MlirLocation
    )::Cuint
end

"""
    mlirLocationFusedGetLocations(location, locationsCPtr)

Getter for locations of Fused. Requires pre-allocated memory of #fusedLocations X sizeof([`MlirLocation`](@ref)).
"""
function mlirLocationFusedGetLocations(location, locationsCPtr)
    @ccall Reactant_jll.libReactantExtra.mlirLocationFusedGetLocations(
        location::MlirLocation, locationsCPtr::Ptr{MlirLocation}
    )::Cvoid
end

"""
    mlirLocationFusedGetMetadata(location)

Getter for metadata of Fused.
"""
function mlirLocationFusedGetMetadata(location)
    @ccall Reactant_jll.libReactantExtra.mlirLocationFusedGetMetadata(
        location::MlirLocation
    )::MlirAttribute
end

"""
    mlirLocationFusedGetTypeID()

TypeID Getter for Fused.
"""
function mlirLocationFusedGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirLocationFusedGetTypeID()::MlirTypeID
end

"""
    mlirLocationIsAFused(location)

Checks whether the given location is an Fused.
"""
function mlirLocationIsAFused(location)
    @ccall Reactant_jll.libReactantExtra.mlirLocationIsAFused(location::MlirLocation)::Bool
end

"""
    mlirLocationNameGet(context, name, childLoc)

Creates a name location owned by the given context. Providing null location for childLoc is allowed and if childLoc is null location, then the behavior is the same as having unknown child location.
"""
function mlirLocationNameGet(context, name, childLoc)
    @ccall Reactant_jll.libReactantExtra.mlirLocationNameGet(
        context::MlirContext, name::MlirStringRef, childLoc::MlirLocation
    )::MlirLocation
end

"""
    mlirLocationNameGetName(location)

Getter for name of Name.
"""
function mlirLocationNameGetName(location)
    @ccall Reactant_jll.libReactantExtra.mlirLocationNameGetName(
        location::MlirLocation
    )::MlirIdentifier
end

"""
    mlirLocationNameGetChildLoc(location)

Getter for childLoc of Name.
"""
function mlirLocationNameGetChildLoc(location)
    @ccall Reactant_jll.libReactantExtra.mlirLocationNameGetChildLoc(
        location::MlirLocation
    )::MlirLocation
end

"""
    mlirLocationNameGetTypeID()

TypeID Getter for Name.
"""
function mlirLocationNameGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirLocationNameGetTypeID()::MlirTypeID
end

"""
    mlirLocationIsAName(location)

Checks whether the given location is an Name.
"""
function mlirLocationIsAName(location)
    @ccall Reactant_jll.libReactantExtra.mlirLocationIsAName(location::MlirLocation)::Bool
end

"""
    mlirLocationUnknownGet(context)

Creates a location with unknown position owned by the given context.
"""
function mlirLocationUnknownGet(context)
    @ccall Reactant_jll.libReactantExtra.mlirLocationUnknownGet(
        context::MlirContext
    )::MlirLocation
end

"""
    mlirLocationUnknownGetTypeID()

TypeID Getter for Unknown.
"""
function mlirLocationUnknownGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirLocationUnknownGetTypeID()::MlirTypeID
end

"""
    mlirLocationIsAUnknown(location)

Checks whether the given location is an Unknown.
"""
function mlirLocationIsAUnknown(location)
    @ccall Reactant_jll.libReactantExtra.mlirLocationIsAUnknown(
        location::MlirLocation
    )::Bool
end

"""
    mlirLocationGetContext(location)

Gets the context that a location was created with.
"""
function mlirLocationGetContext(location)
    @ccall Reactant_jll.libReactantExtra.mlirLocationGetContext(
        location::MlirLocation
    )::MlirContext
end

"""
    mlirLocationIsNull(location)

Checks if the location is null.
"""
function mlirLocationIsNull(location)
    @ccall Reactant_jll.libReactantExtra.mlirLocationIsNull(location::MlirLocation)::Bool
end

"""
    mlirLocationEqual(l1, l2)

Checks if two locations are equal.
"""
function mlirLocationEqual(l1, l2)
    @ccall Reactant_jll.libReactantExtra.mlirLocationEqual(
        l1::MlirLocation, l2::MlirLocation
    )::Bool
end

"""
    mlirLocationPrint(location, callback, userData)

Prints a location by sending chunks of the string representation and forwarding `userData to `callback`. Note that the callback may be called several times with consecutive chunks of the string.
"""
function mlirLocationPrint(location, callback, userData)
    @ccall Reactant_jll.libReactantExtra.mlirLocationPrint(
        location::MlirLocation, callback::MlirStringCallback, userData::Ptr{Cvoid}
    )::Cvoid
end

"""
    mlirModuleCreateEmpty(location)

Creates a new, empty module and transfers ownership to the caller.
"""
function mlirModuleCreateEmpty(location)
    @ccall Reactant_jll.libReactantExtra.mlirModuleCreateEmpty(
        location::MlirLocation
    )::MlirModule
end

"""
    mlirModuleCreateParse(context, _module)

Parses a module from the string and transfers ownership to the caller.
"""
function mlirModuleCreateParse(context, _module)
    @ccall Reactant_jll.libReactantExtra.mlirModuleCreateParse(
        context::MlirContext, _module::MlirStringRef
    )::MlirModule
end

"""
    mlirModuleCreateParseFromFile(context, fileName)

Parses a module from file and transfers ownership to the caller.
"""
function mlirModuleCreateParseFromFile(context, fileName)
    @ccall Reactant_jll.libReactantExtra.mlirModuleCreateParseFromFile(
        context::MlirContext, fileName::MlirStringRef
    )::MlirModule
end

"""
    mlirModuleGetContext(_module)

Gets the context that a module was created with.
"""
function mlirModuleGetContext(_module)
    @ccall Reactant_jll.libReactantExtra.mlirModuleGetContext(
        _module::MlirModule
    )::MlirContext
end

"""
    mlirModuleGetBody(_module)

Gets the body of the module, i.e. the only block it contains.
"""
function mlirModuleGetBody(_module)
    @ccall Reactant_jll.libReactantExtra.mlirModuleGetBody(_module::MlirModule)::MlirBlock
end

"""
    mlirModuleIsNull(_module)

Checks whether a module is null.
"""
function mlirModuleIsNull(_module)
    @ccall Reactant_jll.libReactantExtra.mlirModuleIsNull(_module::MlirModule)::Bool
end

"""
    mlirModuleDestroy(_module)

Takes a module owned by the caller and deletes it.
"""
function mlirModuleDestroy(_module)
    @ccall Reactant_jll.libReactantExtra.mlirModuleDestroy(_module::MlirModule)::Cvoid
end

"""
    mlirModuleGetOperation(_module)

Views the module as a generic operation.
"""
function mlirModuleGetOperation(_module)
    @ccall Reactant_jll.libReactantExtra.mlirModuleGetOperation(
        _module::MlirModule
    )::MlirOperation
end

"""
    mlirModuleFromOperation(op)

Views the generic operation as a module. The returned module is null when the input operation was not a ModuleOp.
"""
function mlirModuleFromOperation(op)
    @ccall Reactant_jll.libReactantExtra.mlirModuleFromOperation(
        op::MlirOperation
    )::MlirModule
end

"""
    mlirModuleEqual(lhs, rhs)

Checks if two modules are equal.
"""
function mlirModuleEqual(lhs, rhs)
    @ccall Reactant_jll.libReactantExtra.mlirModuleEqual(
        lhs::MlirModule, rhs::MlirModule
    )::Bool
end

"""
    mlirModuleHashValue(mod)

Compute a hash for the given module.
"""
function mlirModuleHashValue(mod)
    @ccall Reactant_jll.libReactantExtra.mlirModuleHashValue(mod::MlirModule)::Csize_t
end

"""
    MlirOperationState

An auxiliary class for constructing operations.

This class contains all the information necessary to construct the operation. It owns the MlirRegions it has pointers to and does not own anything else. By default, the state can be constructed from a name and location, the latter being also used to access the context, and has no other components. These components can be added progressively until the operation is constructed. Users are not expected to rely on the internals of this class and should use mlirOperationState* functions instead.
"""
struct MlirOperationState
    name::MlirStringRef
    location::MlirLocation
    nResults::Cptrdiff_t
    results::Ptr{MlirType}
    nOperands::Cptrdiff_t
    operands::Ptr{MlirValue}
    nRegions::Cptrdiff_t
    regions::Ptr{MlirRegion}
    nSuccessors::Cptrdiff_t
    successors::Ptr{MlirBlock}
    nAttributes::Cptrdiff_t
    attributes::Ptr{MlirNamedAttribute}
    enableResultTypeInference::Bool
end

"""
    mlirOperationStateGet(name, loc)

Constructs an operation state from a name and a location.
"""
function mlirOperationStateGet(name, loc)
    @ccall Reactant_jll.libReactantExtra.mlirOperationStateGet(
        name::MlirStringRef, loc::MlirLocation
    )::MlirOperationState
end

"""
    mlirOperationStateAddResults(state, n, results)

Adds a list of components to the operation state.
"""
function mlirOperationStateAddResults(state, n, results)
    @ccall Reactant_jll.libReactantExtra.mlirOperationStateAddResults(
        state::Ptr{MlirOperationState}, n::Cptrdiff_t, results::Ptr{MlirType}
    )::Cvoid
end

function mlirOperationStateAddOperands(state, n, operands)
    @ccall Reactant_jll.libReactantExtra.mlirOperationStateAddOperands(
        state::Ptr{MlirOperationState}, n::Cptrdiff_t, operands::Ptr{MlirValue}
    )::Cvoid
end

function mlirOperationStateAddOwnedRegions(state, n, regions)
    @ccall Reactant_jll.libReactantExtra.mlirOperationStateAddOwnedRegions(
        state::Ptr{MlirOperationState}, n::Cptrdiff_t, regions::Ptr{MlirRegion}
    )::Cvoid
end

function mlirOperationStateAddSuccessors(state, n, successors)
    @ccall Reactant_jll.libReactantExtra.mlirOperationStateAddSuccessors(
        state::Ptr{MlirOperationState}, n::Cptrdiff_t, successors::Ptr{MlirBlock}
    )::Cvoid
end

function mlirOperationStateAddAttributes(state, n, attributes)
    @ccall Reactant_jll.libReactantExtra.mlirOperationStateAddAttributes(
        state::Ptr{MlirOperationState}, n::Cptrdiff_t, attributes::Ptr{MlirNamedAttribute}
    )::Cvoid
end

"""
    mlirOperationStateEnableResultTypeInference(state)

Enables result type inference for the operation under construction. If enabled, then the caller must not have called [`mlirOperationStateAddResults`](@ref)(). Note that if enabled, the [`mlirOperationCreate`](@ref)() call is failable: it will return a null operation on inference failure and will emit diagnostics.
"""
function mlirOperationStateEnableResultTypeInference(state)
    @ccall Reactant_jll.libReactantExtra.mlirOperationStateEnableResultTypeInference(
        state::Ptr{MlirOperationState}
    )::Cvoid
end

"""
    mlirAsmStateCreateForOperation(op, flags)

Creates new AsmState, as with AsmState the IR should not be mutated in-between using this state. Must be freed with a call to [`mlirAsmStateDestroy`](@ref)().
"""
function mlirAsmStateCreateForOperation(op, flags)
    @ccall Reactant_jll.libReactantExtra.mlirAsmStateCreateForOperation(
        op::MlirOperation, flags::MlirOpPrintingFlags
    )::MlirAsmState
end

"""
    mlirAsmStateCreateForValue(value, flags)

Creates new AsmState from value. Must be freed with a call to [`mlirAsmStateDestroy`](@ref)().
"""
function mlirAsmStateCreateForValue(value, flags)
    @ccall Reactant_jll.libReactantExtra.mlirAsmStateCreateForValue(
        value::MlirValue, flags::MlirOpPrintingFlags
    )::MlirAsmState
end

"""
    mlirAsmStateDestroy(state)

Destroys printing flags created with mlirAsmStateCreate.
"""
function mlirAsmStateDestroy(state)
    @ccall Reactant_jll.libReactantExtra.mlirAsmStateDestroy(state::MlirAsmState)::Cvoid
end

"""
    mlirOpPrintingFlagsCreate()

Creates new printing flags with defaults, intended for customization. Must be freed with a call to [`mlirOpPrintingFlagsDestroy`](@ref)().
"""
function mlirOpPrintingFlagsCreate()
    @ccall Reactant_jll.libReactantExtra.mlirOpPrintingFlagsCreate()::MlirOpPrintingFlags
end

"""
    mlirOpPrintingFlagsDestroy(flags)

Destroys printing flags created with [`mlirOpPrintingFlagsCreate`](@ref).
"""
function mlirOpPrintingFlagsDestroy(flags)
    @ccall Reactant_jll.libReactantExtra.mlirOpPrintingFlagsDestroy(
        flags::MlirOpPrintingFlags
    )::Cvoid
end

"""
    mlirOpPrintingFlagsElideLargeElementsAttrs(flags, largeElementLimit)

Enables the elision of large elements attributes by printing a lexically valid but otherwise meaningless form instead of the element data. The `largeElementLimit` is used to configure what is considered to be a "large" ElementsAttr by providing an upper limit to the number of elements.
"""
function mlirOpPrintingFlagsElideLargeElementsAttrs(flags, largeElementLimit)
    @ccall Reactant_jll.libReactantExtra.mlirOpPrintingFlagsElideLargeElementsAttrs(
        flags::MlirOpPrintingFlags, largeElementLimit::Cptrdiff_t
    )::Cvoid
end

"""
    mlirOpPrintingFlagsElideLargeResourceString(flags, largeResourceLimit)

Enables the elision of large resources strings by omitting them from the `dialect_resources` section. The `largeResourceLimit` is used to configure what is considered to be a "large" resource by providing an upper limit to the string size.
"""
function mlirOpPrintingFlagsElideLargeResourceString(flags, largeResourceLimit)
    @ccall Reactant_jll.libReactantExtra.mlirOpPrintingFlagsElideLargeResourceString(
        flags::MlirOpPrintingFlags, largeResourceLimit::Cptrdiff_t
    )::Cvoid
end

"""
    mlirOpPrintingFlagsEnableDebugInfo(flags, enable, prettyForm)

Enable or disable printing of debug information (based on `enable`). If 'prettyForm' is set to true, debug information is printed in a more readable 'pretty' form. Note: The IR generated with 'prettyForm' is not parsable.
"""
function mlirOpPrintingFlagsEnableDebugInfo(flags, enable, prettyForm)
    @ccall Reactant_jll.libReactantExtra.mlirOpPrintingFlagsEnableDebugInfo(
        flags::MlirOpPrintingFlags, enable::Bool, prettyForm::Bool
    )::Cvoid
end

"""
    mlirOpPrintingFlagsPrintGenericOpForm(flags)

Always print operations in the generic form.
"""
function mlirOpPrintingFlagsPrintGenericOpForm(flags)
    @ccall Reactant_jll.libReactantExtra.mlirOpPrintingFlagsPrintGenericOpForm(
        flags::MlirOpPrintingFlags
    )::Cvoid
end

"""
    mlirOpPrintingFlagsPrintNameLocAsPrefix(flags)

Print the name and location, if NamedLoc, as a prefix to the SSA ID.
"""
function mlirOpPrintingFlagsPrintNameLocAsPrefix(flags)
    @ccall Reactant_jll.libReactantExtra.mlirOpPrintingFlagsPrintNameLocAsPrefix(
        flags::MlirOpPrintingFlags
    )::Cvoid
end

"""
    mlirOpPrintingFlagsUseLocalScope(flags)

Use local scope when printing the operation. This allows for using the printer in a more localized and thread-safe setting, but may not necessarily be identical to what the IR will look like when dumping the full module.
"""
function mlirOpPrintingFlagsUseLocalScope(flags)
    @ccall Reactant_jll.libReactantExtra.mlirOpPrintingFlagsUseLocalScope(
        flags::MlirOpPrintingFlags
    )::Cvoid
end

"""
    mlirOpPrintingFlagsAssumeVerified(flags)

Do not verify the operation when using custom operation printers.
"""
function mlirOpPrintingFlagsAssumeVerified(flags)
    @ccall Reactant_jll.libReactantExtra.mlirOpPrintingFlagsAssumeVerified(
        flags::MlirOpPrintingFlags
    )::Cvoid
end

"""
    mlirOpPrintingFlagsSkipRegions(flags)

Skip printing regions.
"""
function mlirOpPrintingFlagsSkipRegions(flags)
    @ccall Reactant_jll.libReactantExtra.mlirOpPrintingFlagsSkipRegions(
        flags::MlirOpPrintingFlags
    )::Cvoid
end

"""
    mlirBytecodeWriterConfigCreate()

Creates new printing flags with defaults, intended for customization. Must be freed with a call to [`mlirBytecodeWriterConfigDestroy`](@ref)().
"""
function mlirBytecodeWriterConfigCreate()
    @ccall Reactant_jll.libReactantExtra.mlirBytecodeWriterConfigCreate()::MlirBytecodeWriterConfig
end

"""
    mlirBytecodeWriterConfigDestroy(config)

Destroys printing flags created with [`mlirBytecodeWriterConfigCreate`](@ref).
"""
function mlirBytecodeWriterConfigDestroy(config)
    @ccall Reactant_jll.libReactantExtra.mlirBytecodeWriterConfigDestroy(
        config::MlirBytecodeWriterConfig
    )::Cvoid
end

"""
    mlirBytecodeWriterConfigDesiredEmitVersion(flags, version)

Sets the version to emit in the writer config.
"""
function mlirBytecodeWriterConfigDesiredEmitVersion(flags, version)
    @ccall Reactant_jll.libReactantExtra.mlirBytecodeWriterConfigDesiredEmitVersion(
        flags::MlirBytecodeWriterConfig, version::Int64
    )::Cvoid
end

"""
    mlirOperationCreate(state)

Creates an operation and transfers ownership to the caller. Note that caller owned child objects are transferred in this call and must not be further used. Particularly, this applies to any regions added to the state (the implementation may invalidate any such pointers).

This call can fail under the following conditions, in which case, it will return a null operation and emit diagnostics: - Result type inference is enabled and cannot be performed.
"""
function mlirOperationCreate(state)
    @ccall Reactant_jll.libReactantExtra.mlirOperationCreate(
        state::Ptr{MlirOperationState}
    )::MlirOperation
end

"""
    mlirOperationCreateParse(context, sourceStr, sourceName)

Parses an operation, giving ownership to the caller. If parsing fails a null operation will be returned, and an error diagnostic emitted.

`sourceStr` may be either the text assembly format, or binary bytecode format. `sourceName` is used as the file name of the source; any IR without locations will get a `FileLineColLoc` location with `sourceName` as the file name.
"""
function mlirOperationCreateParse(context, sourceStr, sourceName)
    @ccall Reactant_jll.libReactantExtra.mlirOperationCreateParse(
        context::MlirContext, sourceStr::MlirStringRef, sourceName::MlirStringRef
    )::MlirOperation
end

"""
    mlirOperationClone(op)

Creates a deep copy of an operation. The operation is not inserted and ownership is transferred to the caller.
"""
function mlirOperationClone(op)
    @ccall Reactant_jll.libReactantExtra.mlirOperationClone(
        op::MlirOperation
    )::MlirOperation
end

"""
    mlirOperationDestroy(op)

Takes an operation owned by the caller and destroys it.
"""
function mlirOperationDestroy(op)
    @ccall Reactant_jll.libReactantExtra.mlirOperationDestroy(op::MlirOperation)::Cvoid
end

"""
    mlirOperationRemoveFromParent(op)

Removes the given operation from its parent block. The operation is not destroyed. The ownership of the operation is transferred to the caller.
"""
function mlirOperationRemoveFromParent(op)
    @ccall Reactant_jll.libReactantExtra.mlirOperationRemoveFromParent(
        op::MlirOperation
    )::Cvoid
end

"""
    mlirOperationIsNull(op)

Checks whether the underlying operation is null.
"""
function mlirOperationIsNull(op)
    @ccall Reactant_jll.libReactantExtra.mlirOperationIsNull(op::MlirOperation)::Bool
end

"""
    mlirOperationEqual(op, other)

Checks whether two operation handles point to the same operation. This does not perform deep comparison.
"""
function mlirOperationEqual(op, other)
    @ccall Reactant_jll.libReactantExtra.mlirOperationEqual(
        op::MlirOperation, other::MlirOperation
    )::Bool
end

"""
    mlirOperationHashValue(op)

Compute a hash for the given operation. Operand and result SSA values are hashed by identity and locations are significant, so equivalent-but-distinct operations hash differently; use [`mlirOperationStructuralHashValue`](@ref) for a hash that pairs with [`mlirOperationIsStructurallyEquivalent`](@ref).
"""
function mlirOperationHashValue(op)
    @ccall Reactant_jll.libReactantExtra.mlirOperationHashValue(op::MlirOperation)::Csize_t
end

"""
    MlirOperationEquivalenceFlags

Flags controlling structural operation equivalence and hashing. These mirror `mlir::OperationEquivalence::Flags` and may be combined with bitwise OR.
"""
@cenum MlirOperationEquivalenceFlags::UInt32 begin
    MLIR_OPERATION_EQUIVALENCE_NONE = 0x0000000000000000
    MLIR_OPERATION_EQUIVALENCE_IGNORE_LOCATIONS = 0x0000000000000001
    MLIR_OPERATION_EQUIVALENCE_IGNORE_DISCARDABLE_ATTRS = 0x0000000000000002
    MLIR_OPERATION_EQUIVALENCE_IGNORE_PROPERTIES = 0x0000000000000004
    MLIR_OPERATION_EQUIVALENCE_IGNORE_COMMUTATIVITY = 0x0000000000000008
end

"""
    mlirOperationIsStructurallyEquivalent(lhs, rhs, flags)

Checks whether two operations are structurally equivalent, i.e. they have the same name, attributes, operand and result types, and recursively equivalent regions. Operand equivalence is tracked structurally while recursing into regions, so operands defined inside the compared regions need not be the exact same SSA values; operands defined outside must be. `flags` is a bitwise OR of [`MlirOperationEquivalenceFlags`](@ref) values.
"""
function mlirOperationIsStructurallyEquivalent(lhs, rhs, flags)
    @ccall Reactant_jll.libReactantExtra.mlirOperationIsStructurallyEquivalent(
        lhs::MlirOperation, rhs::MlirOperation, flags::UInt32
    )::Bool
end

"""
    mlirOperationStructuralHashValue(op, flags)

Computes a hash for the given operation that pairs with [`mlirOperationIsStructurallyEquivalent`](@ref): two operations that are structurally equivalent under the same `flags` hash equally. Operands are hashed by identity, results are not hashed at all, and regions do not participate in the hash. `flags` is a bitwise OR of [`MlirOperationEquivalenceFlags`](@ref) values.
"""
function mlirOperationStructuralHashValue(op, flags)
    @ccall Reactant_jll.libReactantExtra.mlirOperationStructuralHashValue(
        op::MlirOperation, flags::UInt32
    )::Csize_t
end

"""
    mlirOperationGetContext(op)

Gets the context this operation is associated with
"""
function mlirOperationGetContext(op)
    @ccall Reactant_jll.libReactantExtra.mlirOperationGetContext(
        op::MlirOperation
    )::MlirContext
end

"""
    mlirOperationNameHasTrait(opName, traitTypeID, context)

Checks if the operation name has a trait identified by the given type id.
"""
function mlirOperationNameHasTrait(opName, traitTypeID, context)
    @ccall Reactant_jll.libReactantExtra.mlirOperationNameHasTrait(
        opName::MlirStringRef, traitTypeID::MlirTypeID, context::MlirContext
    )::Bool
end

"""
    mlirOperationGetLocation(op)

Gets the location of the operation.
"""
function mlirOperationGetLocation(op)
    @ccall Reactant_jll.libReactantExtra.mlirOperationGetLocation(
        op::MlirOperation
    )::MlirLocation
end

"""
    mlirOperationSetLocation(op, loc)

Sets the location of the operation.
"""
function mlirOperationSetLocation(op, loc)
    @ccall Reactant_jll.libReactantExtra.mlirOperationSetLocation(
        op::MlirOperation, loc::MlirLocation
    )::Cvoid
end

"""
    mlirOperationGetTypeID(op)

Gets the type id of the operation. Returns null if the operation does not have a registered operation description.
"""
function mlirOperationGetTypeID(op)
    @ccall Reactant_jll.libReactantExtra.mlirOperationGetTypeID(
        op::MlirOperation
    )::MlirTypeID
end

"""
    mlirOperationGetName(op)

Gets the name of the operation as an identifier.
"""
function mlirOperationGetName(op)
    @ccall Reactant_jll.libReactantExtra.mlirOperationGetName(
        op::MlirOperation
    )::MlirIdentifier
end

"""
    mlirOperationGetBlock(op)

Gets the block that owns this operation, returning null if the operation is not owned.
"""
function mlirOperationGetBlock(op)
    @ccall Reactant_jll.libReactantExtra.mlirOperationGetBlock(op::MlirOperation)::MlirBlock
end

"""
    mlirOperationGetParentOperation(op)

Gets the operation that owns this operation, returning null if the operation is not owned.
"""
function mlirOperationGetParentOperation(op)
    @ccall Reactant_jll.libReactantExtra.mlirOperationGetParentOperation(
        op::MlirOperation
    )::MlirOperation
end

"""
    mlirOperationGetNumRegions(op)

Returns the number of regions attached to the given operation.
"""
function mlirOperationGetNumRegions(op)
    @ccall Reactant_jll.libReactantExtra.mlirOperationGetNumRegions(
        op::MlirOperation
    )::Cptrdiff_t
end

"""
    mlirOperationGetRegion(op, pos)

Returns `pos`-th region attached to the operation.
"""
function mlirOperationGetRegion(op, pos)
    @ccall Reactant_jll.libReactantExtra.mlirOperationGetRegion(
        op::MlirOperation, pos::Cptrdiff_t
    )::MlirRegion
end

"""
    mlirOperationGetNextInBlock(op)

Returns an operation immediately following the given operation it its enclosing block.
"""
function mlirOperationGetNextInBlock(op)
    @ccall Reactant_jll.libReactantExtra.mlirOperationGetNextInBlock(
        op::MlirOperation
    )::MlirOperation
end

"""
    mlirOperationGetNumOperands(op)

Returns the number of operands of the operation.
"""
function mlirOperationGetNumOperands(op)
    @ccall Reactant_jll.libReactantExtra.mlirOperationGetNumOperands(
        op::MlirOperation
    )::Cptrdiff_t
end

"""
    mlirOperationGetOperand(op, pos)

Returns `pos`-th operand of the operation.
"""
function mlirOperationGetOperand(op, pos)
    @ccall Reactant_jll.libReactantExtra.mlirOperationGetOperand(
        op::MlirOperation, pos::Cptrdiff_t
    )::MlirValue
end

"""
    mlirOperationGetOpOperand(op, pos)

Returns `pos`-th OpOperand of the operation.
"""
function mlirOperationGetOpOperand(op, pos)
    @ccall Reactant_jll.libReactantExtra.mlirOperationGetOpOperand(
        op::MlirOperation, pos::Cptrdiff_t
    )::MlirOpOperand
end

"""
    mlirOperationSetOperand(op, pos, newValue)

Sets the `pos`-th operand of the operation.
"""
function mlirOperationSetOperand(op, pos, newValue)
    @ccall Reactant_jll.libReactantExtra.mlirOperationSetOperand(
        op::MlirOperation, pos::Cptrdiff_t, newValue::MlirValue
    )::Cvoid
end

"""
    mlirOperationSetOperands(op, nOperands, operands)

Replaces the operands of the operation.
"""
function mlirOperationSetOperands(op, nOperands, operands)
    @ccall Reactant_jll.libReactantExtra.mlirOperationSetOperands(
        op::MlirOperation, nOperands::Cptrdiff_t, operands::Ptr{MlirValue}
    )::Cvoid
end

"""
    mlirOperationGetNumResults(op)

Returns the number of results of the operation.
"""
function mlirOperationGetNumResults(op)
    @ccall Reactant_jll.libReactantExtra.mlirOperationGetNumResults(
        op::MlirOperation
    )::Cptrdiff_t
end

"""
    mlirOperationGetResult(op, pos)

Returns `pos`-th result of the operation.
"""
function mlirOperationGetResult(op, pos)
    @ccall Reactant_jll.libReactantExtra.mlirOperationGetResult(
        op::MlirOperation, pos::Cptrdiff_t
    )::MlirValue
end

"""
    mlirOperationGetNumSuccessors(op)

Returns the number of successor blocks of the operation.
"""
function mlirOperationGetNumSuccessors(op)
    @ccall Reactant_jll.libReactantExtra.mlirOperationGetNumSuccessors(
        op::MlirOperation
    )::Cptrdiff_t
end

"""
    mlirOperationGetSuccessor(op, pos)

Returns `pos`-th successor of the operation.
"""
function mlirOperationGetSuccessor(op, pos)
    @ccall Reactant_jll.libReactantExtra.mlirOperationGetSuccessor(
        op::MlirOperation, pos::Cptrdiff_t
    )::MlirBlock
end

"""
    mlirOperationSetSuccessor(op, pos, block)

Set `pos`-th successor of the operation.
"""
function mlirOperationSetSuccessor(op, pos, block)
    @ccall Reactant_jll.libReactantExtra.mlirOperationSetSuccessor(
        op::MlirOperation, pos::Cptrdiff_t, block::MlirBlock
    )::Cvoid
end

"""
    mlirOperationHasInherentAttributeByName(op, name)

Returns true if this operation defines an inherent attribute with this name. Note: the attribute can be optional, so [`mlirOperationGetInherentAttributeByName`](@ref) can still return a null attribute.
"""
function mlirOperationHasInherentAttributeByName(op, name)
    @ccall Reactant_jll.libReactantExtra.mlirOperationHasInherentAttributeByName(
        op::MlirOperation, name::MlirStringRef
    )::Bool
end

"""
    mlirOperationGetInherentAttributeByName(op, name)

Returns an inherent attribute attached to the operation given its name.
"""
function mlirOperationGetInherentAttributeByName(op, name)
    @ccall Reactant_jll.libReactantExtra.mlirOperationGetInherentAttributeByName(
        op::MlirOperation, name::MlirStringRef
    )::MlirAttribute
end

"""
    mlirOperationSetInherentAttributeByName(op, name, attr)

Sets an inherent attribute by name, replacing the existing if it exists. This has no effect if "name" does not match an inherent attribute.
"""
function mlirOperationSetInherentAttributeByName(op, name, attr)
    @ccall Reactant_jll.libReactantExtra.mlirOperationSetInherentAttributeByName(
        op::MlirOperation, name::MlirStringRef, attr::MlirAttribute
    )::Cvoid
end

"""
    mlirOperationGetNumDiscardableAttributes(op)

Returns the number of discardable attributes attached to the operation.
"""
function mlirOperationGetNumDiscardableAttributes(op)
    @ccall Reactant_jll.libReactantExtra.mlirOperationGetNumDiscardableAttributes(
        op::MlirOperation
    )::Cptrdiff_t
end

"""
    mlirOperationGetDiscardableAttribute(op, pos)

Return `pos`-th discardable attribute of the operation.
"""
function mlirOperationGetDiscardableAttribute(op, pos)
    @ccall Reactant_jll.libReactantExtra.mlirOperationGetDiscardableAttribute(
        op::MlirOperation, pos::Cptrdiff_t
    )::MlirNamedAttribute
end

"""
    mlirOperationGetDiscardableAttributeByName(op, name)

Returns a discardable attribute attached to the operation given its name.
"""
function mlirOperationGetDiscardableAttributeByName(op, name)
    @ccall Reactant_jll.libReactantExtra.mlirOperationGetDiscardableAttributeByName(
        op::MlirOperation, name::MlirStringRef
    )::MlirAttribute
end

"""
    mlirOperationSetDiscardableAttributeByName(op, name, attr)

Sets a discardable attribute by name, replacing the existing if it exists or adding a new one otherwise. The new `attr` Attribute is not allowed to be null, use [`mlirOperationRemoveDiscardableAttributeByName`](@ref) to remove an Attribute instead.
"""
function mlirOperationSetDiscardableAttributeByName(op, name, attr)
    @ccall Reactant_jll.libReactantExtra.mlirOperationSetDiscardableAttributeByName(
        op::MlirOperation, name::MlirStringRef, attr::MlirAttribute
    )::Cvoid
end

"""
    mlirOperationRemoveDiscardableAttributeByName(op, name)

Removes a discardable attribute by name. Returns false if the attribute was not found and true if removed.
"""
function mlirOperationRemoveDiscardableAttributeByName(op, name)
    @ccall Reactant_jll.libReactantExtra.mlirOperationRemoveDiscardableAttributeByName(
        op::MlirOperation, name::MlirStringRef
    )::Bool
end

"""
    mlirOperationGetNumAttributes(op)

Returns the number of attributes attached to the operation. Deprecated, please use `mlirOperationGetNumInherentAttributes` or [`mlirOperationGetNumDiscardableAttributes`](@ref).
"""
function mlirOperationGetNumAttributes(op)
    @ccall Reactant_jll.libReactantExtra.mlirOperationGetNumAttributes(
        op::MlirOperation
    )::Cptrdiff_t
end

"""
    mlirOperationGetAttribute(op, pos)

Return `pos`-th attribute of the operation. Deprecated, please use `mlirOperationGetInherentAttribute` or [`mlirOperationGetDiscardableAttribute`](@ref).
"""
function mlirOperationGetAttribute(op, pos)
    @ccall Reactant_jll.libReactantExtra.mlirOperationGetAttribute(
        op::MlirOperation, pos::Cptrdiff_t
    )::MlirNamedAttribute
end

"""
    mlirOperationGetAttributeByName(op, name)

Returns an attribute attached to the operation given its name. Deprecated, please use [`mlirOperationGetInherentAttributeByName`](@ref) or [`mlirOperationGetDiscardableAttributeByName`](@ref).
"""
function mlirOperationGetAttributeByName(op, name)
    @ccall Reactant_jll.libReactantExtra.mlirOperationGetAttributeByName(
        op::MlirOperation, name::MlirStringRef
    )::MlirAttribute
end

"""
    mlirOperationSetAttributeByName(op, name, attr)

Sets an attribute by name, replacing the existing if it exists or adding a new one otherwise. Deprecated, please use [`mlirOperationSetInherentAttributeByName`](@ref) or [`mlirOperationSetDiscardableAttributeByName`](@ref).
"""
function mlirOperationSetAttributeByName(op, name, attr)
    @ccall Reactant_jll.libReactantExtra.mlirOperationSetAttributeByName(
        op::MlirOperation, name::MlirStringRef, attr::MlirAttribute
    )::Cvoid
end

"""
    mlirOperationRemoveAttributeByName(op, name)

Removes an attribute by name. Returns false if the attribute was not found and true if removed. Deprecated, please use `mlirOperationRemoveInherentAttributeByName` or [`mlirOperationRemoveDiscardableAttributeByName`](@ref).
"""
function mlirOperationRemoveAttributeByName(op, name)
    @ccall Reactant_jll.libReactantExtra.mlirOperationRemoveAttributeByName(
        op::MlirOperation, name::MlirStringRef
    )::Bool
end

"""
    mlirOperationPrint(op, callback, userData)

Prints an operation by sending chunks of the string representation and forwarding `userData to `callback`. Note that the callback may be called several times with consecutive chunks of the string.
"""
function mlirOperationPrint(op, callback, userData)
    @ccall Reactant_jll.libReactantExtra.mlirOperationPrint(
        op::MlirOperation, callback::MlirStringCallback, userData::Ptr{Cvoid}
    )::Cvoid
end

"""
    mlirOperationPrintWithFlags(op, flags, callback, userData)

Same as [`mlirOperationPrint`](@ref) but accepts flags controlling the printing behavior.
"""
function mlirOperationPrintWithFlags(op, flags, callback, userData)
    @ccall Reactant_jll.libReactantExtra.mlirOperationPrintWithFlags(
        op::MlirOperation,
        flags::MlirOpPrintingFlags,
        callback::MlirStringCallback,
        userData::Ptr{Cvoid},
    )::Cvoid
end

"""
    mlirOperationPrintWithState(op, state, callback, userData)

Same as [`mlirOperationPrint`](@ref) but accepts AsmState controlling the printing behavior as well as caching computed names.
"""
function mlirOperationPrintWithState(op, state, callback, userData)
    @ccall Reactant_jll.libReactantExtra.mlirOperationPrintWithState(
        op::MlirOperation,
        state::MlirAsmState,
        callback::MlirStringCallback,
        userData::Ptr{Cvoid},
    )::Cvoid
end

"""
    mlirOperationWriteBytecode(op, callback, userData)

Same as [`mlirOperationPrint`](@ref) but writing the bytecode format.
"""
function mlirOperationWriteBytecode(op, callback, userData)
    @ccall Reactant_jll.libReactantExtra.mlirOperationWriteBytecode(
        op::MlirOperation, callback::MlirStringCallback, userData::Ptr{Cvoid}
    )::Cvoid
end

"""
    mlirOperationWriteBytecodeWithConfig(op, config, callback, userData)

Same as [`mlirOperationWriteBytecode`](@ref) but with writer config and returns failure only if desired bytecode could not be honored.
"""
function mlirOperationWriteBytecodeWithConfig(op, config, callback, userData)
    @ccall Reactant_jll.libReactantExtra.mlirOperationWriteBytecodeWithConfig(
        op::MlirOperation,
        config::MlirBytecodeWriterConfig,
        callback::MlirStringCallback,
        userData::Ptr{Cvoid},
    )::MlirLogicalResult
end

"""
    mlirOperationDump(op)

Prints an operation to stderr.
"""
function mlirOperationDump(op)
    @ccall Reactant_jll.libReactantExtra.mlirOperationDump(op::MlirOperation)::Cvoid
end

"""
    mlirOperationVerify(op)

Verify the operation and return true if it passes, false if it fails.
"""
function mlirOperationVerify(op)
    @ccall Reactant_jll.libReactantExtra.mlirOperationVerify(op::MlirOperation)::Bool
end

"""
    mlirOperationMoveAfter(op, other)

Moves the given operation immediately after the other operation in its parent block. The given operation may be owned by the caller or by its current block. The other operation must belong to a block. In any case, the ownership is transferred to the block of the other operation.
"""
function mlirOperationMoveAfter(op, other)
    @ccall Reactant_jll.libReactantExtra.mlirOperationMoveAfter(
        op::MlirOperation, other::MlirOperation
    )::Cvoid
end

"""
    mlirOperationMoveBefore(op, other)

Moves the given operation immediately before the other operation in its parent block. The given operation may be owner by the caller or by its current block. The other operation must belong to a block. In any case, the ownership is transferred to the block of the other operation.
"""
function mlirOperationMoveBefore(op, other)
    @ccall Reactant_jll.libReactantExtra.mlirOperationMoveBefore(
        op::MlirOperation, other::MlirOperation
    )::Cvoid
end

"""
    mlirOperationIsBeforeInBlock(op, other)

Given an operation 'other' that is within the same parent block, return whether the current operation is before 'other' in the operation list of the parent block. Note: This function has an average complexity of O(1), but worst case may take O(N) where N is the number of operations within the parent block.
"""
function mlirOperationIsBeforeInBlock(op, other)
    @ccall Reactant_jll.libReactantExtra.mlirOperationIsBeforeInBlock(
        op::MlirOperation, other::MlirOperation
    )::Bool
end

"""
    MlirWalkResult

[`Operation`](@ref) walk result.
"""
@cenum MlirWalkResult::UInt32 begin
    MlirWalkResultAdvance = 0x0000000000000000
    MlirWalkResultInterrupt = 0x0000000000000001
    MlirWalkResultSkip = 0x0000000000000002
end

"""
    MlirWalkOrder

Traversal order for operation walk.
"""
@cenum MlirWalkOrder::UInt32 begin
    MlirWalkPreOrder = 0x0000000000000000
    MlirWalkPostOrder = 0x0000000000000001
end

# typedef MlirWalkResult ( * MlirOperationWalkCallback ) ( MlirOperation , void * userData )
"""
[`Operation`](@ref) walker type. The handler is passed an (opaque) reference to an operation and a pointer to a `userData`.
"""
const MlirOperationWalkCallback = Ptr{Cvoid}

"""
    mlirOperationWalk(op, callback, userData, walkOrder)

Walks operation `op` in `walkOrder` and calls `callback` on that operation. `*userData` is passed to the callback as well and can be used to tunnel some context or other data into the callback.
"""
function mlirOperationWalk(op, callback, userData, walkOrder)
    @ccall Reactant_jll.libReactantExtra.mlirOperationWalk(
        op::MlirOperation,
        callback::MlirOperationWalkCallback,
        userData::Ptr{Cvoid},
        walkOrder::MlirWalkOrder,
    )::Cvoid
end

"""
    mlirOperationReplaceUsesOfWith(op, of, with)

Replace uses of 'of' value with the 'with' value inside the 'op' operation.
"""
function mlirOperationReplaceUsesOfWith(op, of, with)
    @ccall Reactant_jll.libReactantExtra.mlirOperationReplaceUsesOfWith(
        op::MlirOperation, of::MlirValue, with::MlirValue
    )::Cvoid
end

"""
    mlirRegionCreate()

Creates a new empty region and transfers ownership to the caller.
"""
function mlirRegionCreate()
    @ccall Reactant_jll.libReactantExtra.mlirRegionCreate()::MlirRegion
end

"""
    mlirRegionDestroy(region)

Takes a region owned by the caller and destroys it.
"""
function mlirRegionDestroy(region)
    @ccall Reactant_jll.libReactantExtra.mlirRegionDestroy(region::MlirRegion)::Cvoid
end

"""
    mlirRegionIsNull(region)

Checks whether a region is null.
"""
function mlirRegionIsNull(region)
    @ccall Reactant_jll.libReactantExtra.mlirRegionIsNull(region::MlirRegion)::Bool
end

"""
    mlirRegionEqual(region, other)

Checks whether two region handles point to the same region. This does not perform deep comparison.
"""
function mlirRegionEqual(region, other)
    @ccall Reactant_jll.libReactantExtra.mlirRegionEqual(
        region::MlirRegion, other::MlirRegion
    )::Bool
end

"""
    mlirRegionGetFirstBlock(region)

Gets the first block in the region.
"""
function mlirRegionGetFirstBlock(region)
    @ccall Reactant_jll.libReactantExtra.mlirRegionGetFirstBlock(
        region::MlirRegion
    )::MlirBlock
end

"""
    mlirRegionAppendOwnedBlock(region, block)

Takes a block owned by the caller and appends it to the given region.
"""
function mlirRegionAppendOwnedBlock(region, block)
    @ccall Reactant_jll.libReactantExtra.mlirRegionAppendOwnedBlock(
        region::MlirRegion, block::MlirBlock
    )::Cvoid
end

"""
    mlirRegionInsertOwnedBlock(region, pos, block)

Takes a block owned by the caller and inserts it at `pos` to the given region. This is an expensive operation that linearly scans the region, prefer insertAfter/Before instead.
"""
function mlirRegionInsertOwnedBlock(region, pos, block)
    @ccall Reactant_jll.libReactantExtra.mlirRegionInsertOwnedBlock(
        region::MlirRegion, pos::Cptrdiff_t, block::MlirBlock
    )::Cvoid
end

"""
    mlirRegionInsertOwnedBlockAfter(region, reference, block)

Takes a block owned by the caller and inserts it after the (non-owned) reference block in the given region. The reference block must belong to the region. If the reference block is null, prepends the block to the region.
"""
function mlirRegionInsertOwnedBlockAfter(region, reference, block)
    @ccall Reactant_jll.libReactantExtra.mlirRegionInsertOwnedBlockAfter(
        region::MlirRegion, reference::MlirBlock, block::MlirBlock
    )::Cvoid
end

"""
    mlirRegionInsertOwnedBlockBefore(region, reference, block)

Takes a block owned by the caller and inserts it before the (non-owned) reference block in the given region. The reference block must belong to the region. If the reference block is null, appends the block to the region.
"""
function mlirRegionInsertOwnedBlockBefore(region, reference, block)
    @ccall Reactant_jll.libReactantExtra.mlirRegionInsertOwnedBlockBefore(
        region::MlirRegion, reference::MlirBlock, block::MlirBlock
    )::Cvoid
end

"""
    mlirOperationGetFirstRegion(op)

Returns first region attached to the operation.
"""
function mlirOperationGetFirstRegion(op)
    @ccall Reactant_jll.libReactantExtra.mlirOperationGetFirstRegion(
        op::MlirOperation
    )::MlirRegion
end

"""
    mlirRegionGetNextInOperation(region)

Returns the region immediately following the given region in its parent operation.
"""
function mlirRegionGetNextInOperation(region)
    @ccall Reactant_jll.libReactantExtra.mlirRegionGetNextInOperation(
        region::MlirRegion
    )::MlirRegion
end

"""
    mlirRegionTakeBody(target, source)

Moves the entire content of the source region to the target region.
"""
function mlirRegionTakeBody(target, source)
    @ccall Reactant_jll.libReactantExtra.mlirRegionTakeBody(
        target::MlirRegion, source::MlirRegion
    )::Cvoid
end

"""
    mlirBlockCreate(nArgs, args, locs)

Creates a new empty block with the given argument types and transfers ownership to the caller.
"""
function mlirBlockCreate(nArgs, args, locs)
    @ccall Reactant_jll.libReactantExtra.mlirBlockCreate(
        nArgs::Cptrdiff_t, args::Ptr{MlirType}, locs::Ptr{MlirLocation}
    )::MlirBlock
end

"""
    mlirBlockDestroy(block)

Takes a block owned by the caller and destroys it.
"""
function mlirBlockDestroy(block)
    @ccall Reactant_jll.libReactantExtra.mlirBlockDestroy(block::MlirBlock)::Cvoid
end

"""
    mlirBlockDetach(block)

Detach a block from the owning region and assume ownership.
"""
function mlirBlockDetach(block)
    @ccall Reactant_jll.libReactantExtra.mlirBlockDetach(block::MlirBlock)::Cvoid
end

"""
    mlirBlockIsNull(block)

Checks whether a block is null.
"""
function mlirBlockIsNull(block)
    @ccall Reactant_jll.libReactantExtra.mlirBlockIsNull(block::MlirBlock)::Bool
end

"""
    mlirBlockEqual(block, other)

Checks whether two blocks handles point to the same block. This does not perform deep comparison.
"""
function mlirBlockEqual(block, other)
    @ccall Reactant_jll.libReactantExtra.mlirBlockEqual(
        block::MlirBlock, other::MlirBlock
    )::Bool
end

"""
    mlirBlockGetParentOperation(arg1)

Returns the closest surrounding operation that contains this block.
"""
function mlirBlockGetParentOperation(arg1)
    @ccall Reactant_jll.libReactantExtra.mlirBlockGetParentOperation(
        arg1::MlirBlock
    )::MlirOperation
end

"""
    mlirBlockGetParentRegion(block)

Returns the region that contains this block.
"""
function mlirBlockGetParentRegion(block)
    @ccall Reactant_jll.libReactantExtra.mlirBlockGetParentRegion(
        block::MlirBlock
    )::MlirRegion
end

"""
    mlirBlockGetNextInRegion(block)

Returns the block immediately following the given block in its parent region.
"""
function mlirBlockGetNextInRegion(block)
    @ccall Reactant_jll.libReactantExtra.mlirBlockGetNextInRegion(
        block::MlirBlock
    )::MlirBlock
end

"""
    mlirBlockGetFirstOperation(block)

Returns the first operation in the block.
"""
function mlirBlockGetFirstOperation(block)
    @ccall Reactant_jll.libReactantExtra.mlirBlockGetFirstOperation(
        block::MlirBlock
    )::MlirOperation
end

"""
    mlirBlockGetTerminator(block)

Returns the terminator operation in the block or null if no terminator.
"""
function mlirBlockGetTerminator(block)
    @ccall Reactant_jll.libReactantExtra.mlirBlockGetTerminator(
        block::MlirBlock
    )::MlirOperation
end

"""
    mlirBlockAppendOwnedOperation(block, operation)

Takes an operation owned by the caller and appends it to the block.
"""
function mlirBlockAppendOwnedOperation(block, operation)
    @ccall Reactant_jll.libReactantExtra.mlirBlockAppendOwnedOperation(
        block::MlirBlock, operation::MlirOperation
    )::Cvoid
end

"""
    mlirBlockInsertOwnedOperation(block, pos, operation)

Takes an operation owned by the caller and inserts it as `pos` to the block. This is an expensive operation that scans the block linearly, prefer insertBefore/After instead.
"""
function mlirBlockInsertOwnedOperation(block, pos, operation)
    @ccall Reactant_jll.libReactantExtra.mlirBlockInsertOwnedOperation(
        block::MlirBlock, pos::Cptrdiff_t, operation::MlirOperation
    )::Cvoid
end

"""
    mlirBlockInsertOwnedOperationAfter(block, reference, operation)

Takes an operation owned by the caller and inserts it after the (non-owned) reference operation in the given block. If the reference is null, prepends the operation. Otherwise, the reference must belong to the block.
"""
function mlirBlockInsertOwnedOperationAfter(block, reference, operation)
    @ccall Reactant_jll.libReactantExtra.mlirBlockInsertOwnedOperationAfter(
        block::MlirBlock, reference::MlirOperation, operation::MlirOperation
    )::Cvoid
end

"""
    mlirBlockInsertOwnedOperationBefore(block, reference, operation)

Takes an operation owned by the caller and inserts it before the (non-owned) reference operation in the given block. If the reference is null, appends the operation. Otherwise, the reference must belong to the block.
"""
function mlirBlockInsertOwnedOperationBefore(block, reference, operation)
    @ccall Reactant_jll.libReactantExtra.mlirBlockInsertOwnedOperationBefore(
        block::MlirBlock, reference::MlirOperation, operation::MlirOperation
    )::Cvoid
end

"""
    mlirBlockGetNumArguments(block)

Returns the number of arguments of the block.
"""
function mlirBlockGetNumArguments(block)
    @ccall Reactant_jll.libReactantExtra.mlirBlockGetNumArguments(
        block::MlirBlock
    )::Cptrdiff_t
end

"""
    mlirBlockAddArgument(block, type, loc)

Appends an argument of the specified type to the block. Returns the newly added argument.
"""
function mlirBlockAddArgument(block, type, loc)
    @ccall Reactant_jll.libReactantExtra.mlirBlockAddArgument(
        block::MlirBlock, type::MlirType, loc::MlirLocation
    )::MlirValue
end

"""
    mlirBlockEraseArgument(block, index)

Erase the argument at 'index' and remove it from the argument list.
"""
function mlirBlockEraseArgument(block, index)
    @ccall Reactant_jll.libReactantExtra.mlirBlockEraseArgument(
        block::MlirBlock, index::Cuint
    )::Cvoid
end

"""
    mlirBlockInsertArgument(block, pos, type, loc)

Inserts an argument of the specified type at a specified index to the block. Returns the newly added argument.
"""
function mlirBlockInsertArgument(block, pos, type, loc)
    @ccall Reactant_jll.libReactantExtra.mlirBlockInsertArgument(
        block::MlirBlock, pos::Cptrdiff_t, type::MlirType, loc::MlirLocation
    )::MlirValue
end

"""
    mlirBlockGetArgument(block, pos)

Returns `pos`-th argument of the block.
"""
function mlirBlockGetArgument(block, pos)
    @ccall Reactant_jll.libReactantExtra.mlirBlockGetArgument(
        block::MlirBlock, pos::Cptrdiff_t
    )::MlirValue
end

"""
    mlirBlockPrint(block, callback, userData)

Prints a block by sending chunks of the string representation and forwarding `userData to `callback`. Note that the callback may be called several times with consecutive chunks of the string.
"""
function mlirBlockPrint(block, callback, userData)
    @ccall Reactant_jll.libReactantExtra.mlirBlockPrint(
        block::MlirBlock, callback::MlirStringCallback, userData::Ptr{Cvoid}
    )::Cvoid
end

"""
    mlirBlockGetNumSuccessors(block)

Returns the number of successor blocks of the block.
"""
function mlirBlockGetNumSuccessors(block)
    @ccall Reactant_jll.libReactantExtra.mlirBlockGetNumSuccessors(
        block::MlirBlock
    )::Cptrdiff_t
end

"""
    mlirBlockGetSuccessor(block, pos)

Returns `pos`-th successor of the block.
"""
function mlirBlockGetSuccessor(block, pos)
    @ccall Reactant_jll.libReactantExtra.mlirBlockGetSuccessor(
        block::MlirBlock, pos::Cptrdiff_t
    )::MlirBlock
end

"""
    mlirBlockGetNumPredecessors(block)

Returns the number of predecessor blocks of the block.
"""
function mlirBlockGetNumPredecessors(block)
    @ccall Reactant_jll.libReactantExtra.mlirBlockGetNumPredecessors(
        block::MlirBlock
    )::Cptrdiff_t
end

"""
    mlirBlockGetPredecessor(block, pos)

Returns `pos`-th predecessor of the block.

WARNING: This getter is more expensive than the others here because the impl actually iterates the use-def chain (of block operands) anew for each indexed access.
"""
function mlirBlockGetPredecessor(block, pos)
    @ccall Reactant_jll.libReactantExtra.mlirBlockGetPredecessor(
        block::MlirBlock, pos::Cptrdiff_t
    )::MlirBlock
end

"""
    mlirValueIsNull(value)

Returns whether the value is null.
"""
function mlirValueIsNull(value)
    @ccall Reactant_jll.libReactantExtra.mlirValueIsNull(value::MlirValue)::Bool
end

"""
    mlirValueEqual(value1, value2)

Returns 1 if two values are equal, 0 otherwise.
"""
function mlirValueEqual(value1, value2)
    @ccall Reactant_jll.libReactantExtra.mlirValueEqual(
        value1::MlirValue, value2::MlirValue
    )::Bool
end

"""
    mlirValueIsABlockArgument(value)

Returns 1 if the value is a block argument, 0 otherwise.
"""
function mlirValueIsABlockArgument(value)
    @ccall Reactant_jll.libReactantExtra.mlirValueIsABlockArgument(value::MlirValue)::Bool
end

"""
    mlirValueIsAOpResult(value)

Returns 1 if the value is an operation result, 0 otherwise.
"""
function mlirValueIsAOpResult(value)
    @ccall Reactant_jll.libReactantExtra.mlirValueIsAOpResult(value::MlirValue)::Bool
end

"""
    mlirBlockArgumentGetOwner(value)

Returns the block in which this value is defined as an argument. Asserts if the value is not a block argument.
"""
function mlirBlockArgumentGetOwner(value)
    @ccall Reactant_jll.libReactantExtra.mlirBlockArgumentGetOwner(
        value::MlirValue
    )::MlirBlock
end

"""
    mlirBlockArgumentGetArgNumber(value)

Returns the position of the value in the argument list of its block.
"""
function mlirBlockArgumentGetArgNumber(value)
    @ccall Reactant_jll.libReactantExtra.mlirBlockArgumentGetArgNumber(
        value::MlirValue
    )::Cptrdiff_t
end

"""
    mlirBlockArgumentSetType(value, type)

Sets the type of the block argument to the given type.
"""
function mlirBlockArgumentSetType(value, type)
    @ccall Reactant_jll.libReactantExtra.mlirBlockArgumentSetType(
        value::MlirValue, type::MlirType
    )::Cvoid
end

"""
    mlirBlockArgumentSetLocation(value, loc)

Sets the location of the block argument to the given location.
"""
function mlirBlockArgumentSetLocation(value, loc)
    @ccall Reactant_jll.libReactantExtra.mlirBlockArgumentSetLocation(
        value::MlirValue, loc::MlirLocation
    )::Cvoid
end

"""
    mlirOpResultGetOwner(value)

Returns an operation that produced this value as its result. Asserts if the value is not an op result.
"""
function mlirOpResultGetOwner(value)
    @ccall Reactant_jll.libReactantExtra.mlirOpResultGetOwner(
        value::MlirValue
    )::MlirOperation
end

"""
    mlirOpResultGetResultNumber(value)

Returns the position of the value in the list of results of the operation that produced it.
"""
function mlirOpResultGetResultNumber(value)
    @ccall Reactant_jll.libReactantExtra.mlirOpResultGetResultNumber(
        value::MlirValue
    )::Cptrdiff_t
end

"""
    mlirValueGetType(value)

Returns the type of the value.
"""
function mlirValueGetType(value)
    @ccall Reactant_jll.libReactantExtra.mlirValueGetType(value::MlirValue)::MlirType
end

"""
    mlirValueSetType(value, type)

Set the type of the value.
"""
function mlirValueSetType(value, type)
    @ccall Reactant_jll.libReactantExtra.mlirValueSetType(
        value::MlirValue, type::MlirType
    )::Cvoid
end

"""
    mlirValueDump(value)

Prints the value to the standard error stream.
"""
function mlirValueDump(value)
    @ccall Reactant_jll.libReactantExtra.mlirValueDump(value::MlirValue)::Cvoid
end

"""
    mlirValuePrint(value, callback, userData)

Prints a value by sending chunks of the string representation and forwarding `userData to `callback`. Note that the callback may be called several times with consecutive chunks of the string.
"""
function mlirValuePrint(value, callback, userData)
    @ccall Reactant_jll.libReactantExtra.mlirValuePrint(
        value::MlirValue, callback::MlirStringCallback, userData::Ptr{Cvoid}
    )::Cvoid
end

"""
    mlirValuePrintAsOperand(value, state, callback, userData)

Prints a value as an operand (i.e., the ValueID).
"""
function mlirValuePrintAsOperand(value, state, callback, userData)
    @ccall Reactant_jll.libReactantExtra.mlirValuePrintAsOperand(
        value::MlirValue,
        state::MlirAsmState,
        callback::MlirStringCallback,
        userData::Ptr{Cvoid},
    )::Cvoid
end

"""
    mlirValueGetFirstUse(value)

Returns an op operand representing the first use of the value, or a null op operand if there are no uses.
"""
function mlirValueGetFirstUse(value)
    @ccall Reactant_jll.libReactantExtra.mlirValueGetFirstUse(
        value::MlirValue
    )::MlirOpOperand
end

"""
    mlirValueReplaceAllUsesOfWith(of, with)

Replace all uses of 'of' value with the 'with' value, updating anything in the IR that uses 'of' to use the other value instead. When this returns there are zero uses of 'of'.
"""
function mlirValueReplaceAllUsesOfWith(of, with)
    @ccall Reactant_jll.libReactantExtra.mlirValueReplaceAllUsesOfWith(
        of::MlirValue, with::MlirValue
    )::Cvoid
end

"""
    mlirValueReplaceAllUsesExcept(of, with, numExceptions, exceptions)

Replace all uses of 'of' value with 'with' value, updating anything in the IR that uses 'of' to use 'with' instead, except if the user is listed in 'exceptions'. The 'exceptions' parameter is an array of [`MlirOperation`](@ref) pointers with a length of 'numExceptions'.
"""
function mlirValueReplaceAllUsesExcept(of, with, numExceptions, exceptions)
    @ccall Reactant_jll.libReactantExtra.mlirValueReplaceAllUsesExcept(
        of::MlirValue,
        with::MlirValue,
        numExceptions::Cptrdiff_t,
        exceptions::Ptr{MlirOperation},
    )::Cvoid
end

# typedef bool ( * MlirOpOperandReplaceFilterCallback ) ( MlirOpOperand opOperand , void * userData )
"""
Callback deciding whether a particular use should be replaced. It is passed the use as an [`MlirOpOperand`](@ref) (from which the owner operation, operand number and value can be queried) and the user-provided `userData`. Returns true to replace this use.
"""
const MlirOpOperandReplaceFilterCallback = Ptr{Cvoid}

"""
    mlirValueReplaceUsesWithIf(of, with, filter, userData)

Replace uses of 'of' value with 'with' value, but only for the uses for which the `filter` callback returns true. `filter` must not be NULL; this is only checked by an assertion, i.e. in builds with assertions enabled.
"""
function mlirValueReplaceUsesWithIf(of, with, filter, userData)
    @ccall Reactant_jll.libReactantExtra.mlirValueReplaceUsesWithIf(
        of::MlirValue,
        with::MlirValue,
        filter::MlirOpOperandReplaceFilterCallback,
        userData::Ptr{Cvoid},
    )::Cvoid
end

"""
    mlirValueGetLocation(v)

Gets the location of the value.
"""
function mlirValueGetLocation(v)
    @ccall Reactant_jll.libReactantExtra.mlirValueGetLocation(v::MlirValue)::MlirLocation
end

"""
    mlirValueGetContext(v)

Gets the context that a value was created with.
"""
function mlirValueGetContext(v)
    @ccall Reactant_jll.libReactantExtra.mlirValueGetContext(v::MlirValue)::MlirContext
end

"""
    mlirOpOperandIsNull(opOperand)

Returns whether the op operand is null.
"""
function mlirOpOperandIsNull(opOperand)
    @ccall Reactant_jll.libReactantExtra.mlirOpOperandIsNull(opOperand::MlirOpOperand)::Bool
end

"""
    mlirOpOperandGetValue(opOperand)

Returns the value of an op operand.
"""
function mlirOpOperandGetValue(opOperand)
    @ccall Reactant_jll.libReactantExtra.mlirOpOperandGetValue(
        opOperand::MlirOpOperand
    )::MlirValue
end

"""
    mlirOpOperandGetOwner(opOperand)

Returns the owner operation of an op operand.
"""
function mlirOpOperandGetOwner(opOperand)
    @ccall Reactant_jll.libReactantExtra.mlirOpOperandGetOwner(
        opOperand::MlirOpOperand
    )::MlirOperation
end

"""
    mlirOpOperandGetOperandNumber(opOperand)

Returns the operand number of an op operand.
"""
function mlirOpOperandGetOperandNumber(opOperand)
    @ccall Reactant_jll.libReactantExtra.mlirOpOperandGetOperandNumber(
        opOperand::MlirOpOperand
    )::Cuint
end

"""
    mlirOpOperandGetNextUse(opOperand)

Returns an op operand representing the next use of the value, or a null op operand if there is no next use.
"""
function mlirOpOperandGetNextUse(opOperand)
    @ccall Reactant_jll.libReactantExtra.mlirOpOperandGetNextUse(
        opOperand::MlirOpOperand
    )::MlirOpOperand
end

"""
    mlirTypeParseGet(context, type)

Parses a type. The type is owned by the context.
"""
function mlirTypeParseGet(context, type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeParseGet(
        context::MlirContext, type::MlirStringRef
    )::MlirType
end

"""
    mlirTypeGetContext(type)

Gets the context that a type was created with.
"""
function mlirTypeGetContext(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeGetContext(type::MlirType)::MlirContext
end

"""
    mlirTypeGetTypeID(type)

Gets the type ID of the type.
"""
function mlirTypeGetTypeID(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeGetTypeID(type::MlirType)::MlirTypeID
end

"""
    mlirTypeGetDialect(type)

Gets the dialect a type belongs to.
"""
function mlirTypeGetDialect(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeGetDialect(type::MlirType)::MlirDialect
end

"""
    mlirTypeIsNull(type)

Checks whether a type is null.
"""
function mlirTypeIsNull(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsNull(type::MlirType)::Bool
end

"""
    mlirTypeEqual(t1, t2)

Checks if two types are equal.
"""
function mlirTypeEqual(t1, t2)
    @ccall Reactant_jll.libReactantExtra.mlirTypeEqual(t1::MlirType, t2::MlirType)::Bool
end

"""
    mlirTypePrint(type, callback, userData)

Prints a location by sending chunks of the string representation and forwarding `userData to `callback`. Note that the callback may be called several times with consecutive chunks of the string.
"""
function mlirTypePrint(type, callback, userData)
    @ccall Reactant_jll.libReactantExtra.mlirTypePrint(
        type::MlirType, callback::MlirStringCallback, userData::Ptr{Cvoid}
    )::Cvoid
end

"""
    mlirTypeDump(type)

Prints the type to the standard error stream.
"""
function mlirTypeDump(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeDump(type::MlirType)::Cvoid
end

"""
    mlirAttributeParseGet(context, attr)

Parses an attribute. The attribute is owned by the context.
"""
function mlirAttributeParseGet(context, attr)
    @ccall Reactant_jll.libReactantExtra.mlirAttributeParseGet(
        context::MlirContext, attr::MlirStringRef
    )::MlirAttribute
end

"""
    mlirAttributeGetContext(attribute)

Gets the context that an attribute was created with.
"""
function mlirAttributeGetContext(attribute)
    @ccall Reactant_jll.libReactantExtra.mlirAttributeGetContext(
        attribute::MlirAttribute
    )::MlirContext
end

"""
    mlirAttributeGetType(attribute)

Gets the type of this attribute.
"""
function mlirAttributeGetType(attribute)
    @ccall Reactant_jll.libReactantExtra.mlirAttributeGetType(
        attribute::MlirAttribute
    )::MlirType
end

"""
    mlirAttributeGetTypeID(attribute)

Gets the type id of the attribute.
"""
function mlirAttributeGetTypeID(attribute)
    @ccall Reactant_jll.libReactantExtra.mlirAttributeGetTypeID(
        attribute::MlirAttribute
    )::MlirTypeID
end

"""
    mlirAttributeGetDialect(attribute)

Gets the dialect of the attribute.
"""
function mlirAttributeGetDialect(attribute)
    @ccall Reactant_jll.libReactantExtra.mlirAttributeGetDialect(
        attribute::MlirAttribute
    )::MlirDialect
end

"""
    mlirAttributeIsNull(attr)

Checks whether an attribute is null.
"""
function mlirAttributeIsNull(attr)
    @ccall Reactant_jll.libReactantExtra.mlirAttributeIsNull(attr::MlirAttribute)::Bool
end

"""
    mlirAttributeEqual(a1, a2)

Checks if two attributes are equal.
"""
function mlirAttributeEqual(a1, a2)
    @ccall Reactant_jll.libReactantExtra.mlirAttributeEqual(
        a1::MlirAttribute, a2::MlirAttribute
    )::Bool
end

"""
    mlirAttributePrint(attr, callback, userData)

Prints an attribute by sending chunks of the string representation and forwarding `userData to `callback`. Note that the callback may be called several times with consecutive chunks of the string.
"""
function mlirAttributePrint(attr, callback, userData)
    @ccall Reactant_jll.libReactantExtra.mlirAttributePrint(
        attr::MlirAttribute, callback::MlirStringCallback, userData::Ptr{Cvoid}
    )::Cvoid
end

"""
    mlirAttributeDump(attr)

Prints the attribute to the standard error stream.
"""
function mlirAttributeDump(attr)
    @ccall Reactant_jll.libReactantExtra.mlirAttributeDump(attr::MlirAttribute)::Cvoid
end

"""
    mlirNamedAttributeGet(name, attr)

Associates an attribute with the name. Takes ownership of neither.
"""
function mlirNamedAttributeGet(name, attr)
    @ccall Reactant_jll.libReactantExtra.mlirNamedAttributeGet(
        name::MlirIdentifier, attr::MlirAttribute
    )::MlirNamedAttribute
end

"""
    mlirIdentifierGet(context, str)

Gets an identifier with the given string value.
"""
function mlirIdentifierGet(context, str)
    @ccall Reactant_jll.libReactantExtra.mlirIdentifierGet(
        context::MlirContext, str::MlirStringRef
    )::MlirIdentifier
end

"""
    mlirIdentifierGetContext(arg1)

Returns the context associated with this identifier
"""
function mlirIdentifierGetContext(arg1)
    @ccall Reactant_jll.libReactantExtra.mlirIdentifierGetContext(
        arg1::MlirIdentifier
    )::MlirContext
end

"""
    mlirIdentifierEqual(ident, other)

Checks whether two identifiers are the same.
"""
function mlirIdentifierEqual(ident, other)
    @ccall Reactant_jll.libReactantExtra.mlirIdentifierEqual(
        ident::MlirIdentifier, other::MlirIdentifier
    )::Bool
end

"""
    mlirIdentifierStr(ident)

Gets the string value of the identifier.
"""
function mlirIdentifierStr(ident)
    @ccall Reactant_jll.libReactantExtra.mlirIdentifierStr(
        ident::MlirIdentifier
    )::MlirStringRef
end

"""
    mlirSymbolTableGetSymbolAttributeName()

Returns the name of the attribute used to store symbol names compatible with symbol tables.
"""
function mlirSymbolTableGetSymbolAttributeName()
    @ccall Reactant_jll.libReactantExtra.mlirSymbolTableGetSymbolAttributeName()::MlirStringRef
end

"""
    mlirSymbolTableGetVisibilityAttributeName()

Returns the name of the attribute used to store symbol visibility.
"""
function mlirSymbolTableGetVisibilityAttributeName()
    @ccall Reactant_jll.libReactantExtra.mlirSymbolTableGetVisibilityAttributeName()::MlirStringRef
end

"""
    mlirSymbolTableCreate(operation)

Creates a symbol table for the given operation. If the operation does not have the SymbolTable trait, returns a null symbol table.
"""
function mlirSymbolTableCreate(operation)
    @ccall Reactant_jll.libReactantExtra.mlirSymbolTableCreate(
        operation::MlirOperation
    )::MlirSymbolTable
end

"""
    mlirSymbolTableIsNull(symbolTable)

Returns true if the symbol table is null.
"""
function mlirSymbolTableIsNull(symbolTable)
    @ccall Reactant_jll.libReactantExtra.mlirSymbolTableIsNull(
        symbolTable::MlirSymbolTable
    )::Bool
end

"""
    mlirSymbolTableDestroy(symbolTable)

Destroys the symbol table created with [`mlirSymbolTableCreate`](@ref). This does not affect the operations in the table.
"""
function mlirSymbolTableDestroy(symbolTable)
    @ccall Reactant_jll.libReactantExtra.mlirSymbolTableDestroy(
        symbolTable::MlirSymbolTable
    )::Cvoid
end

"""
    mlirSymbolTableLookup(symbolTable, name)

Looks up a symbol with the given name in the given symbol table and returns the operation that corresponds to the symbol. If the symbol cannot be found, returns a null operation.
"""
function mlirSymbolTableLookup(symbolTable, name)
    @ccall Reactant_jll.libReactantExtra.mlirSymbolTableLookup(
        symbolTable::MlirSymbolTable, name::MlirStringRef
    )::MlirOperation
end

"""
    mlirSymbolTableInsert(symbolTable, operation)

Inserts the given operation into the given symbol table. The operation must have the symbol trait. If the symbol table already has a symbol with the same name, renames the symbol being inserted to ensure name uniqueness. Note that this does not move the operation itself into the block of the symbol table operation, this should be done separately. Returns the name of the symbol after insertion.
"""
function mlirSymbolTableInsert(symbolTable, operation)
    @ccall Reactant_jll.libReactantExtra.mlirSymbolTableInsert(
        symbolTable::MlirSymbolTable, operation::MlirOperation
    )::MlirAttribute
end

"""
    mlirSymbolTableErase(symbolTable, operation)

Removes the given operation from the symbol table and erases it.
"""
function mlirSymbolTableErase(symbolTable, operation)
    @ccall Reactant_jll.libReactantExtra.mlirSymbolTableErase(
        symbolTable::MlirSymbolTable, operation::MlirOperation
    )::Cvoid
end

"""
    mlirSymbolTableReplaceAllSymbolUses(oldSymbol, newSymbol, from)

Attempt to replace all uses that are nested within the given operation of the given symbol 'oldSymbol' with the provided 'newSymbol'. This does not traverse into nested symbol tables. Will fail atomically if there are any unknown operations that may be potential symbol tables.
"""
function mlirSymbolTableReplaceAllSymbolUses(oldSymbol, newSymbol, from)
    @ccall Reactant_jll.libReactantExtra.mlirSymbolTableReplaceAllSymbolUses(
        oldSymbol::MlirStringRef, newSymbol::MlirStringRef, from::MlirOperation
    )::MlirLogicalResult
end

"""
    mlirSymbolTableWalkSymbolTables(from, allSymUsesVisible, callback, userData)

Walks all symbol table operations nested within, and including, `op`. For each symbol table operation, the provided callback is invoked with the op and a boolean signifying if the symbols within that symbol table can be treated as if all uses within the IR are visible to the caller. `allSymUsesVisible` identifies whether all of the symbol uses of symbols within `op` are visible.
"""
function mlirSymbolTableWalkSymbolTables(from, allSymUsesVisible, callback, userData)
    @ccall Reactant_jll.libReactantExtra.mlirSymbolTableWalkSymbolTables(
        from::MlirOperation,
        allSymUsesVisible::Bool,
        callback::Ptr{Cvoid},
        userData::Ptr{Cvoid},
    )::Cvoid
end

"""
    mlirIRMappingCreate()

Creates a new empty IRMapping.
"""
function mlirIRMappingCreate()
    @ccall Reactant_jll.libReactantExtra.mlirIRMappingCreate()::MlirIRMapping
end

"""
    mlirIRMappingDestroy(mapping)

Destroys the given IRMapping.
"""
function mlirIRMappingDestroy(mapping)
    @ccall Reactant_jll.libReactantExtra.mlirIRMappingDestroy(mapping::MlirIRMapping)::Cvoid
end

"""
    mlirIRMappingIsNull(mapping)

Checks whether an IRMapping is null.
"""
function mlirIRMappingIsNull(mapping)
    @ccall Reactant_jll.libReactantExtra.mlirIRMappingIsNull(mapping::MlirIRMapping)::Bool
end

"""
    mlirIRMappingMapValue(mapping, from, to)

Maps a Value in the mapping.
"""
function mlirIRMappingMapValue(mapping, from, to)
    @ccall Reactant_jll.libReactantExtra.mlirIRMappingMapValue(
        mapping::MlirIRMapping, from::MlirValue, to::MlirValue
    )::Cvoid
end

"""
    mlirIRMappingMapBlock(mapping, from, to)

Maps a Block in the mapping.
"""
function mlirIRMappingMapBlock(mapping, from, to)
    @ccall Reactant_jll.libReactantExtra.mlirIRMappingMapBlock(
        mapping::MlirIRMapping, from::MlirBlock, to::MlirBlock
    )::Cvoid
end

"""
    mlirIRMappingMapOperation(mapping, from, to)

Maps an [`Operation`](@ref) in the mapping.
"""
function mlirIRMappingMapOperation(mapping, from, to)
    @ccall Reactant_jll.libReactantExtra.mlirIRMappingMapOperation(
        mapping::MlirIRMapping, from::MlirOperation, to::MlirOperation
    )::Cvoid
end

"""
    mlirIRMappingClear(mapping)

Clears all mappings.
"""
function mlirIRMappingClear(mapping)
    @ccall Reactant_jll.libReactantExtra.mlirIRMappingClear(mapping::MlirIRMapping)::Cvoid
end

"""
    mlirIRMappingLookupOrDefaultValue(mapping, from)

Looks up a mapped Value. Returns the mapped value, or the input value if no mapping exists.
"""
function mlirIRMappingLookupOrDefaultValue(mapping, from)
    @ccall Reactant_jll.libReactantExtra.mlirIRMappingLookupOrDefaultValue(
        mapping::MlirIRMapping, from::MlirValue
    )::MlirValue
end

"""
    mlirIRMappingLookupOrNullValue(mapping, from)

Looks up a mapped Value. Returns a null [`MlirValue`](@ref) if no mapping exists.
"""
function mlirIRMappingLookupOrNullValue(mapping, from)
    @ccall Reactant_jll.libReactantExtra.mlirIRMappingLookupOrNullValue(
        mapping::MlirIRMapping, from::MlirValue
    )::MlirValue
end

"""
    mlirIRMappingLookupOrDefaultBlock(mapping, from)

Looks up a mapped Block. Returns the mapped block, or the input block if no mapping exists.
"""
function mlirIRMappingLookupOrDefaultBlock(mapping, from)
    @ccall Reactant_jll.libReactantExtra.mlirIRMappingLookupOrDefaultBlock(
        mapping::MlirIRMapping, from::MlirBlock
    )::MlirBlock
end

"""
    mlirIRMappingLookupOrNullBlock(mapping, from)

Looks up a mapped Block. Returns a null [`MlirBlock`](@ref) if no mapping exists.
"""
function mlirIRMappingLookupOrNullBlock(mapping, from)
    @ccall Reactant_jll.libReactantExtra.mlirIRMappingLookupOrNullBlock(
        mapping::MlirIRMapping, from::MlirBlock
    )::MlirBlock
end

"""
    mlirIRMappingLookupOrDefaultOperation(mapping, from)

Looks up a mapped [`Operation`](@ref). Returns the mapped operation, or the input operation if no mapping exists.
"""
function mlirIRMappingLookupOrDefaultOperation(mapping, from)
    @ccall Reactant_jll.libReactantExtra.mlirIRMappingLookupOrDefaultOperation(
        mapping::MlirIRMapping, from::MlirOperation
    )::MlirOperation
end

"""
    mlirIRMappingLookupOrNullOperation(mapping, from)

Looks up a mapped [`Operation`](@ref). Returns a null [`MlirOperation`](@ref) if no mapping exists.
"""
function mlirIRMappingLookupOrNullOperation(mapping, from)
    @ccall Reactant_jll.libReactantExtra.mlirIRMappingLookupOrNullOperation(
        mapping::MlirIRMapping, from::MlirOperation
    )::MlirOperation
end

"""
    mlirIRMappingContainsValue(mapping, value)

Returns true if the mapping contains a mapping for the given value.
"""
function mlirIRMappingContainsValue(mapping, value)
    @ccall Reactant_jll.libReactantExtra.mlirIRMappingContainsValue(
        mapping::MlirIRMapping, value::MlirValue
    )::Bool
end

"""
    mlirIRMappingContainsBlock(mapping, block)

Returns true if the mapping contains a mapping for the given block.
"""
function mlirIRMappingContainsBlock(mapping, block)
    @ccall Reactant_jll.libReactantExtra.mlirIRMappingContainsBlock(
        mapping::MlirIRMapping, block::MlirBlock
    )::Bool
end

"""
    mlirIRMappingContainsOperation(mapping, op)

Returns true if the mapping contains a mapping for the given operation.
"""
function mlirIRMappingContainsOperation(mapping, op)
    @ccall Reactant_jll.libReactantExtra.mlirIRMappingContainsOperation(
        mapping::MlirIRMapping, op::MlirOperation
    )::Bool
end

"""
    mlirIRMappingEraseValue(mapping, value)

Erases a value mapping.
"""
function mlirIRMappingEraseValue(mapping, value)
    @ccall Reactant_jll.libReactantExtra.mlirIRMappingEraseValue(
        mapping::MlirIRMapping, value::MlirValue
    )::Cvoid
end

"""
    mlirIRMappingEraseBlock(mapping, block)

Erases a block mapping.
"""
function mlirIRMappingEraseBlock(mapping, block)
    @ccall Reactant_jll.libReactantExtra.mlirIRMappingEraseBlock(
        mapping::MlirIRMapping, block::MlirBlock
    )::Cvoid
end

"""
    mlirIRMappingEraseOperation(mapping, op)

Erases an operation mapping.
"""
function mlirIRMappingEraseOperation(mapping, op)
    @ccall Reactant_jll.libReactantExtra.mlirIRMappingEraseOperation(
        mapping::MlirIRMapping, op::MlirOperation
    )::Cvoid
end

"""
    mlirOperationCloneWithMapping(op, mapping)

Clones the operation with the given mapping. The mapping is updated with the cloned operation's results and regions.
"""
function mlirOperationCloneWithMapping(op, mapping)
    @ccall Reactant_jll.libReactantExtra.mlirOperationCloneWithMapping(
        op::MlirOperation, mapping::MlirIRMapping
    )::MlirOperation
end

struct MlirAffineExpr
    ptr::Ptr{Cvoid}
end

"""
    mlirAffineExprGetContext(affineExpr)

Gets the context that owns the affine expression.
"""
function mlirAffineExprGetContext(affineExpr)
    @ccall Reactant_jll.libReactantExtra.mlirAffineExprGetContext(
        affineExpr::MlirAffineExpr
    )::MlirContext
end

"""
    mlirAffineExprEqual(lhs, rhs)

Returns `true` if the two affine expressions are equal.
"""
function mlirAffineExprEqual(lhs, rhs)
    @ccall Reactant_jll.libReactantExtra.mlirAffineExprEqual(
        lhs::MlirAffineExpr, rhs::MlirAffineExpr
    )::Bool
end

"""
    mlirAffineExprIsNull(affineExpr)

Returns `true` if the given affine expression is a null expression. Note constant zero is not a null expression.
"""
function mlirAffineExprIsNull(affineExpr)
    @ccall Reactant_jll.libReactantExtra.mlirAffineExprIsNull(
        affineExpr::MlirAffineExpr
    )::Bool
end

"""
    mlirAffineExprPrint(affineExpr, callback, userData)

Prints an affine expression by sending chunks of the string representation and forwarding `userData to `callback`. Note that the callback may be called several times with consecutive chunks of the string.
"""
function mlirAffineExprPrint(affineExpr, callback, userData)
    @ccall Reactant_jll.libReactantExtra.mlirAffineExprPrint(
        affineExpr::MlirAffineExpr, callback::MlirStringCallback, userData::Ptr{Cvoid}
    )::Cvoid
end

"""
    mlirAffineExprDump(affineExpr)

Prints the affine expression to the standard error stream.
"""
function mlirAffineExprDump(affineExpr)
    @ccall Reactant_jll.libReactantExtra.mlirAffineExprDump(
        affineExpr::MlirAffineExpr
    )::Cvoid
end

"""
    mlirAffineExprIsSymbolicOrConstant(affineExpr)

Checks whether the given affine expression is made out of only symbols and constants.
"""
function mlirAffineExprIsSymbolicOrConstant(affineExpr)
    @ccall Reactant_jll.libReactantExtra.mlirAffineExprIsSymbolicOrConstant(
        affineExpr::MlirAffineExpr
    )::Bool
end

"""
    mlirAffineExprIsPureAffine(affineExpr)

Checks whether the given affine expression is a pure affine expression, i.e. mul, floordiv, ceildic, and mod is only allowed w.r.t constants.
"""
function mlirAffineExprIsPureAffine(affineExpr)
    @ccall Reactant_jll.libReactantExtra.mlirAffineExprIsPureAffine(
        affineExpr::MlirAffineExpr
    )::Bool
end

"""
    mlirAffineExprGetLargestKnownDivisor(affineExpr)

Returns the greatest known integral divisor of this affine expression. The result is always positive.
"""
function mlirAffineExprGetLargestKnownDivisor(affineExpr)
    @ccall Reactant_jll.libReactantExtra.mlirAffineExprGetLargestKnownDivisor(
        affineExpr::MlirAffineExpr
    )::Int64
end

"""
    mlirAffineExprIsMultipleOf(affineExpr, factor)

Checks whether the given affine expression is a multiple of 'factor'.
"""
function mlirAffineExprIsMultipleOf(affineExpr, factor)
    @ccall Reactant_jll.libReactantExtra.mlirAffineExprIsMultipleOf(
        affineExpr::MlirAffineExpr, factor::Int64
    )::Bool
end

"""
    mlirAffineExprIsFunctionOfDim(affineExpr, position)

Checks whether the given affine expression involves AffineDimExpr 'position'.
"""
function mlirAffineExprIsFunctionOfDim(affineExpr, position)
    @ccall Reactant_jll.libReactantExtra.mlirAffineExprIsFunctionOfDim(
        affineExpr::MlirAffineExpr, position::Cptrdiff_t
    )::Bool
end

struct MlirAffineMap
    ptr::Ptr{Cvoid}
end

"""
    mlirAffineExprCompose(affineExpr, affineMap)

Composes the given map with the given expression.
"""
function mlirAffineExprCompose(affineExpr, affineMap)
    @ccall Reactant_jll.libReactantExtra.mlirAffineExprCompose(
        affineExpr::MlirAffineExpr, affineMap::MlirAffineMap
    )::MlirAffineExpr
end

"""
    mlirAffineExprShiftDims(affineExpr, numDims, shift, offset)

Replace dims[offset ... numDims) by dims[offset + shift ... shift + numDims).
"""
function mlirAffineExprShiftDims(affineExpr, numDims, shift, offset)
    @ccall Reactant_jll.libReactantExtra.mlirAffineExprShiftDims(
        affineExpr::MlirAffineExpr, numDims::UInt32, shift::UInt32, offset::UInt32
    )::MlirAffineExpr
end

"""
    mlirAffineExprShiftSymbols(affineExpr, numSymbols, shift, offset)

Replace symbols[offset ... numSymbols) by symbols[offset + shift ... shift + numSymbols).
"""
function mlirAffineExprShiftSymbols(affineExpr, numSymbols, shift, offset)
    @ccall Reactant_jll.libReactantExtra.mlirAffineExprShiftSymbols(
        affineExpr::MlirAffineExpr, numSymbols::UInt32, shift::UInt32, offset::UInt32
    )::MlirAffineExpr
end

"""
    mlirSimplifyAffineExpr(expr, numDims, numSymbols)

Simplify an affine expression by flattening and some amount of simple analysis. This has complexity linear in the number of nodes in 'expr'. Returns the simplified expression, which is the same as the input expression if it can't be simplified. When `expr` is semi-affine, a simplified semi-affine expression is constructed in the sorted order of dimension and symbol positions.
"""
function mlirSimplifyAffineExpr(expr, numDims, numSymbols)
    @ccall Reactant_jll.libReactantExtra.mlirSimplifyAffineExpr(
        expr::MlirAffineExpr, numDims::UInt32, numSymbols::UInt32
    )::MlirAffineExpr
end

"""
    mlirAffineExprIsADim(affineExpr)

Checks whether the given affine expression is a dimension expression.
"""
function mlirAffineExprIsADim(affineExpr)
    @ccall Reactant_jll.libReactantExtra.mlirAffineExprIsADim(
        affineExpr::MlirAffineExpr
    )::Bool
end

"""
    mlirAffineDimExprGet(ctx, position)

Creates an affine dimension expression with 'position' in the context.
"""
function mlirAffineDimExprGet(ctx, position)
    @ccall Reactant_jll.libReactantExtra.mlirAffineDimExprGet(
        ctx::MlirContext, position::Cptrdiff_t
    )::MlirAffineExpr
end

"""
    mlirAffineDimExprGetPosition(affineExpr)

Returns the position of the given affine dimension expression.
"""
function mlirAffineDimExprGetPosition(affineExpr)
    @ccall Reactant_jll.libReactantExtra.mlirAffineDimExprGetPosition(
        affineExpr::MlirAffineExpr
    )::Cptrdiff_t
end

"""
    mlirAffineExprIsASymbol(affineExpr)

Checks whether the given affine expression is a symbol expression.
"""
function mlirAffineExprIsASymbol(affineExpr)
    @ccall Reactant_jll.libReactantExtra.mlirAffineExprIsASymbol(
        affineExpr::MlirAffineExpr
    )::Bool
end

"""
    mlirAffineSymbolExprGet(ctx, position)

Creates an affine symbol expression with 'position' in the context.
"""
function mlirAffineSymbolExprGet(ctx, position)
    @ccall Reactant_jll.libReactantExtra.mlirAffineSymbolExprGet(
        ctx::MlirContext, position::Cptrdiff_t
    )::MlirAffineExpr
end

"""
    mlirAffineSymbolExprGetPosition(affineExpr)

Returns the position of the given affine symbol expression.
"""
function mlirAffineSymbolExprGetPosition(affineExpr)
    @ccall Reactant_jll.libReactantExtra.mlirAffineSymbolExprGetPosition(
        affineExpr::MlirAffineExpr
    )::Cptrdiff_t
end

"""
    mlirAffineExprIsAConstant(affineExpr)

Checks whether the given affine expression is a constant expression.
"""
function mlirAffineExprIsAConstant(affineExpr)
    @ccall Reactant_jll.libReactantExtra.mlirAffineExprIsAConstant(
        affineExpr::MlirAffineExpr
    )::Bool
end

"""
    mlirAffineConstantExprGet(ctx, constant)

Creates an affine constant expression with 'constant' in the context.
"""
function mlirAffineConstantExprGet(ctx, constant)
    @ccall Reactant_jll.libReactantExtra.mlirAffineConstantExprGet(
        ctx::MlirContext, constant::Int64
    )::MlirAffineExpr
end

"""
    mlirAffineConstantExprGetValue(affineExpr)

Returns the value of the given affine constant expression.
"""
function mlirAffineConstantExprGetValue(affineExpr)
    @ccall Reactant_jll.libReactantExtra.mlirAffineConstantExprGetValue(
        affineExpr::MlirAffineExpr
    )::Int64
end

"""
    mlirAffineExprIsAAdd(affineExpr)

Checks whether the given affine expression is an add expression.
"""
function mlirAffineExprIsAAdd(affineExpr)
    @ccall Reactant_jll.libReactantExtra.mlirAffineExprIsAAdd(
        affineExpr::MlirAffineExpr
    )::Bool
end

"""
    mlirAffineAddExprGet(lhs, rhs)

Creates an affine add expression with 'lhs' and 'rhs'.
"""
function mlirAffineAddExprGet(lhs, rhs)
    @ccall Reactant_jll.libReactantExtra.mlirAffineAddExprGet(
        lhs::MlirAffineExpr, rhs::MlirAffineExpr
    )::MlirAffineExpr
end

"""
    mlirAffineExprIsAMul(affineExpr)

Checks whether the given affine expression is an mul expression.
"""
function mlirAffineExprIsAMul(affineExpr)
    @ccall Reactant_jll.libReactantExtra.mlirAffineExprIsAMul(
        affineExpr::MlirAffineExpr
    )::Bool
end

"""
    mlirAffineMulExprGet(lhs, rhs)

Creates an affine mul expression with 'lhs' and 'rhs'.
"""
function mlirAffineMulExprGet(lhs, rhs)
    @ccall Reactant_jll.libReactantExtra.mlirAffineMulExprGet(
        lhs::MlirAffineExpr, rhs::MlirAffineExpr
    )::MlirAffineExpr
end

"""
    mlirAffineExprIsAMod(affineExpr)

Checks whether the given affine expression is an mod expression.
"""
function mlirAffineExprIsAMod(affineExpr)
    @ccall Reactant_jll.libReactantExtra.mlirAffineExprIsAMod(
        affineExpr::MlirAffineExpr
    )::Bool
end

"""
    mlirAffineModExprGet(lhs, rhs)

Creates an affine mod expression with 'lhs' and 'rhs'.
"""
function mlirAffineModExprGet(lhs, rhs)
    @ccall Reactant_jll.libReactantExtra.mlirAffineModExprGet(
        lhs::MlirAffineExpr, rhs::MlirAffineExpr
    )::MlirAffineExpr
end

"""
    mlirAffineExprIsAFloorDiv(affineExpr)

Checks whether the given affine expression is an floordiv expression.
"""
function mlirAffineExprIsAFloorDiv(affineExpr)
    @ccall Reactant_jll.libReactantExtra.mlirAffineExprIsAFloorDiv(
        affineExpr::MlirAffineExpr
    )::Bool
end

"""
    mlirAffineFloorDivExprGet(lhs, rhs)

Creates an affine floordiv expression with 'lhs' and 'rhs'.
"""
function mlirAffineFloorDivExprGet(lhs, rhs)
    @ccall Reactant_jll.libReactantExtra.mlirAffineFloorDivExprGet(
        lhs::MlirAffineExpr, rhs::MlirAffineExpr
    )::MlirAffineExpr
end

"""
    mlirAffineExprIsACeilDiv(affineExpr)

Checks whether the given affine expression is an ceildiv expression.
"""
function mlirAffineExprIsACeilDiv(affineExpr)
    @ccall Reactant_jll.libReactantExtra.mlirAffineExprIsACeilDiv(
        affineExpr::MlirAffineExpr
    )::Bool
end

"""
    mlirAffineCeilDivExprGet(lhs, rhs)

Creates an affine ceildiv expression with 'lhs' and 'rhs'.
"""
function mlirAffineCeilDivExprGet(lhs, rhs)
    @ccall Reactant_jll.libReactantExtra.mlirAffineCeilDivExprGet(
        lhs::MlirAffineExpr, rhs::MlirAffineExpr
    )::MlirAffineExpr
end

"""
    mlirAffineExprIsABinary(affineExpr)

Checks whether the given affine expression is binary.
"""
function mlirAffineExprIsABinary(affineExpr)
    @ccall Reactant_jll.libReactantExtra.mlirAffineExprIsABinary(
        affineExpr::MlirAffineExpr
    )::Bool
end

"""
    mlirAffineBinaryOpExprGetLHS(affineExpr)

Returns the left hand side affine expression of the given affine binary operation expression.
"""
function mlirAffineBinaryOpExprGetLHS(affineExpr)
    @ccall Reactant_jll.libReactantExtra.mlirAffineBinaryOpExprGetLHS(
        affineExpr::MlirAffineExpr
    )::MlirAffineExpr
end

"""
    mlirAffineBinaryOpExprGetRHS(affineExpr)

Returns the right hand side affine expression of the given affine binary operation expression.
"""
function mlirAffineBinaryOpExprGetRHS(affineExpr)
    @ccall Reactant_jll.libReactantExtra.mlirAffineBinaryOpExprGetRHS(
        affineExpr::MlirAffineExpr
    )::MlirAffineExpr
end

"""
    mlirAffineMapGetContext(affineMap)

Gets the context that the given affine map was created with
"""
function mlirAffineMapGetContext(affineMap)
    @ccall Reactant_jll.libReactantExtra.mlirAffineMapGetContext(
        affineMap::MlirAffineMap
    )::MlirContext
end

"""
    mlirAffineMapIsNull(affineMap)

Checks whether an affine map is null.
"""
function mlirAffineMapIsNull(affineMap)
    @ccall Reactant_jll.libReactantExtra.mlirAffineMapIsNull(affineMap::MlirAffineMap)::Bool
end

"""
    mlirAffineMapEqual(a1, a2)

Checks if two affine maps are equal.
"""
function mlirAffineMapEqual(a1, a2)
    @ccall Reactant_jll.libReactantExtra.mlirAffineMapEqual(
        a1::MlirAffineMap, a2::MlirAffineMap
    )::Bool
end

"""
    mlirAffineMapPrint(affineMap, callback, userData)

Prints an affine map by sending chunks of the string representation and forwarding `userData to `callback`. Note that the callback may be called several times with consecutive chunks of the string.
"""
function mlirAffineMapPrint(affineMap, callback, userData)
    @ccall Reactant_jll.libReactantExtra.mlirAffineMapPrint(
        affineMap::MlirAffineMap, callback::MlirStringCallback, userData::Ptr{Cvoid}
    )::Cvoid
end

"""
    mlirAffineMapDump(affineMap)

Prints the affine map to the standard error stream.
"""
function mlirAffineMapDump(affineMap)
    @ccall Reactant_jll.libReactantExtra.mlirAffineMapDump(affineMap::MlirAffineMap)::Cvoid
end

"""
    mlirAffineMapEmptyGet(ctx)

Creates a zero result affine map with no dimensions or symbols in the context. The affine map is owned by the context.
"""
function mlirAffineMapEmptyGet(ctx)
    @ccall Reactant_jll.libReactantExtra.mlirAffineMapEmptyGet(
        ctx::MlirContext
    )::MlirAffineMap
end

"""
    mlirAffineMapZeroResultGet(ctx, dimCount, symbolCount)

Creates a zero result affine map of the given dimensions and symbols in the context. The affine map is owned by the context.
"""
function mlirAffineMapZeroResultGet(ctx, dimCount, symbolCount)
    @ccall Reactant_jll.libReactantExtra.mlirAffineMapZeroResultGet(
        ctx::MlirContext, dimCount::Cptrdiff_t, symbolCount::Cptrdiff_t
    )::MlirAffineMap
end

"""
    mlirAffineMapGet(ctx, dimCount, symbolCount, nAffineExprs, affineExprs)

Creates an affine map with results defined by the given list of affine expressions. The map resulting map also has the requested number of input dimensions and symbols, regardless of them being used in the results.
"""
function mlirAffineMapGet(ctx, dimCount, symbolCount, nAffineExprs, affineExprs)
    @ccall Reactant_jll.libReactantExtra.mlirAffineMapGet(
        ctx::MlirContext,
        dimCount::Cptrdiff_t,
        symbolCount::Cptrdiff_t,
        nAffineExprs::Cptrdiff_t,
        affineExprs::Ptr{MlirAffineExpr},
    )::MlirAffineMap
end

"""
    mlirAffineMapConstantGet(ctx, val)

Creates a single constant result affine map in the context. The affine map is owned by the context.
"""
function mlirAffineMapConstantGet(ctx, val)
    @ccall Reactant_jll.libReactantExtra.mlirAffineMapConstantGet(
        ctx::MlirContext, val::Int64
    )::MlirAffineMap
end

"""
    mlirAffineMapMultiDimIdentityGet(ctx, numDims)

Creates an affine map with 'numDims' identity in the context. The affine map is owned by the context.
"""
function mlirAffineMapMultiDimIdentityGet(ctx, numDims)
    @ccall Reactant_jll.libReactantExtra.mlirAffineMapMultiDimIdentityGet(
        ctx::MlirContext, numDims::Cptrdiff_t
    )::MlirAffineMap
end

"""
    mlirAffineMapMinorIdentityGet(ctx, dims, results)

Creates an identity affine map on the most minor dimensions in the context. The affine map is owned by the context. The function asserts that the number of dimensions is greater or equal to the number of results.
"""
function mlirAffineMapMinorIdentityGet(ctx, dims, results)
    @ccall Reactant_jll.libReactantExtra.mlirAffineMapMinorIdentityGet(
        ctx::MlirContext, dims::Cptrdiff_t, results::Cptrdiff_t
    )::MlirAffineMap
end

"""
    mlirAffineMapPermutationGet(ctx, size, permutation)

Creates an affine map with a permutation expression and its size in the context. The permutation expression is a non-empty vector of integers. The elements of the permutation vector must be continuous from 0 and cannot be repeated (i.e. `[1,2,0]` is a valid permutation. `[2,0]` or `[1,1,2]` is an invalid permutation.) The affine map is owned by the context.
"""
function mlirAffineMapPermutationGet(ctx, size, permutation)
    @ccall Reactant_jll.libReactantExtra.mlirAffineMapPermutationGet(
        ctx::MlirContext, size::Cptrdiff_t, permutation::Ptr{Cuint}
    )::MlirAffineMap
end

"""
    mlirAffineMapIsIdentity(affineMap)

Checks whether the given affine map is an identity affine map. The function asserts that the number of dimensions is greater or equal to the number of results.
"""
function mlirAffineMapIsIdentity(affineMap)
    @ccall Reactant_jll.libReactantExtra.mlirAffineMapIsIdentity(
        affineMap::MlirAffineMap
    )::Bool
end

"""
    mlirAffineMapIsMinorIdentity(affineMap)

Checks whether the given affine map is a minor identity affine map.
"""
function mlirAffineMapIsMinorIdentity(affineMap)
    @ccall Reactant_jll.libReactantExtra.mlirAffineMapIsMinorIdentity(
        affineMap::MlirAffineMap
    )::Bool
end

"""
    mlirAffineMapIsEmpty(affineMap)

Checks whether the given affine map is an empty affine map.
"""
function mlirAffineMapIsEmpty(affineMap)
    @ccall Reactant_jll.libReactantExtra.mlirAffineMapIsEmpty(
        affineMap::MlirAffineMap
    )::Bool
end

"""
    mlirAffineMapIsSingleConstant(affineMap)

Checks whether the given affine map is a single result constant affine map.
"""
function mlirAffineMapIsSingleConstant(affineMap)
    @ccall Reactant_jll.libReactantExtra.mlirAffineMapIsSingleConstant(
        affineMap::MlirAffineMap
    )::Bool
end

"""
    mlirAffineMapGetSingleConstantResult(affineMap)

Returns the constant result of the given affine map. The function asserts that the map has a single constant result.
"""
function mlirAffineMapGetSingleConstantResult(affineMap)
    @ccall Reactant_jll.libReactantExtra.mlirAffineMapGetSingleConstantResult(
        affineMap::MlirAffineMap
    )::Int64
end

"""
    mlirAffineMapGetNumDims(affineMap)

Returns the number of dimensions of the given affine map.
"""
function mlirAffineMapGetNumDims(affineMap)
    @ccall Reactant_jll.libReactantExtra.mlirAffineMapGetNumDims(
        affineMap::MlirAffineMap
    )::Cptrdiff_t
end

"""
    mlirAffineMapGetNumSymbols(affineMap)

Returns the number of symbols of the given affine map.
"""
function mlirAffineMapGetNumSymbols(affineMap)
    @ccall Reactant_jll.libReactantExtra.mlirAffineMapGetNumSymbols(
        affineMap::MlirAffineMap
    )::Cptrdiff_t
end

"""
    mlirAffineMapGetNumResults(affineMap)

Returns the number of results of the given affine map.
"""
function mlirAffineMapGetNumResults(affineMap)
    @ccall Reactant_jll.libReactantExtra.mlirAffineMapGetNumResults(
        affineMap::MlirAffineMap
    )::Cptrdiff_t
end

"""
    mlirAffineMapGetResult(affineMap, pos)

Returns the result at the given position.
"""
function mlirAffineMapGetResult(affineMap, pos)
    @ccall Reactant_jll.libReactantExtra.mlirAffineMapGetResult(
        affineMap::MlirAffineMap, pos::Cptrdiff_t
    )::MlirAffineExpr
end

"""
    mlirAffineMapGetNumInputs(affineMap)

Returns the number of inputs (dimensions + symbols) of the given affine map.
"""
function mlirAffineMapGetNumInputs(affineMap)
    @ccall Reactant_jll.libReactantExtra.mlirAffineMapGetNumInputs(
        affineMap::MlirAffineMap
    )::Cptrdiff_t
end

"""
    mlirAffineMapIsProjectedPermutation(affineMap)

Checks whether the given affine map represents a subset of a symbol-less permutation map.
"""
function mlirAffineMapIsProjectedPermutation(affineMap)
    @ccall Reactant_jll.libReactantExtra.mlirAffineMapIsProjectedPermutation(
        affineMap::MlirAffineMap
    )::Bool
end

"""
    mlirAffineMapIsPermutation(affineMap)

Checks whether the given affine map represents a symbol-less permutation map.
"""
function mlirAffineMapIsPermutation(affineMap)
    @ccall Reactant_jll.libReactantExtra.mlirAffineMapIsPermutation(
        affineMap::MlirAffineMap
    )::Bool
end

"""
    mlirAffineMapGetSubMap(affineMap, size, resultPos)

Returns the affine map consisting of the `resultPos` subset.
"""
function mlirAffineMapGetSubMap(affineMap, size, resultPos)
    @ccall Reactant_jll.libReactantExtra.mlirAffineMapGetSubMap(
        affineMap::MlirAffineMap, size::Cptrdiff_t, resultPos::Ptr{Cptrdiff_t}
    )::MlirAffineMap
end

"""
    mlirAffineMapGetMajorSubMap(affineMap, numResults)

Returns the affine map consisting of the most major `numResults` results. Returns the null AffineMap if the `numResults` is equal to zero. Returns the `affineMap` if `numResults` is greater or equals to number of results of the given affine map.
"""
function mlirAffineMapGetMajorSubMap(affineMap, numResults)
    @ccall Reactant_jll.libReactantExtra.mlirAffineMapGetMajorSubMap(
        affineMap::MlirAffineMap, numResults::Cptrdiff_t
    )::MlirAffineMap
end

"""
    mlirAffineMapGetMinorSubMap(affineMap, numResults)

Returns the affine map consisting of the most minor `numResults` results. Returns the null AffineMap if the `numResults` is equal to zero. Returns the `affineMap` if `numResults` is greater or equals to number of results of the given affine map.
"""
function mlirAffineMapGetMinorSubMap(affineMap, numResults)
    @ccall Reactant_jll.libReactantExtra.mlirAffineMapGetMinorSubMap(
        affineMap::MlirAffineMap, numResults::Cptrdiff_t
    )::MlirAffineMap
end

"""
    mlirAffineMapReplace(affineMap, expression, replacement, numResultDims, numResultSyms)

Apply AffineExpr::replace(`map`) to each of the results and return a new new AffineMap with the new results and the specified number of dims and symbols.
"""
function mlirAffineMapReplace(
    affineMap, expression, replacement, numResultDims, numResultSyms
)
    @ccall Reactant_jll.libReactantExtra.mlirAffineMapReplace(
        affineMap::MlirAffineMap,
        expression::MlirAffineExpr,
        replacement::MlirAffineExpr,
        numResultDims::Cptrdiff_t,
        numResultSyms::Cptrdiff_t,
    )::MlirAffineMap
end

"""
    mlirAffineMapCompressUnusedSymbols(affineMaps, size, result, populateResult)

Returns the simplified affine map resulting from dropping the symbols that do not appear in any of the individual maps in `affineMaps`. Asserts that all maps in `affineMaps` are normalized to the same number of dims and symbols. Takes a callback `populateResult` to fill the `res` container with value `m` at entry `idx`. This allows returning without worrying about ownership considerations.
"""
function mlirAffineMapCompressUnusedSymbols(affineMaps, size, result, populateResult)
    @ccall Reactant_jll.libReactantExtra.mlirAffineMapCompressUnusedSymbols(
        affineMaps::Ptr{MlirAffineMap},
        size::Cptrdiff_t,
        result::Ptr{Cvoid},
        populateResult::Ptr{Cvoid},
    )::Cvoid
end

struct MlirIntegerSet
    ptr::Ptr{Cvoid}
end

"""
    mlirIntegerSetGetContext(set)

Gets the context in which the given integer set lives.
"""
function mlirIntegerSetGetContext(set)
    @ccall Reactant_jll.libReactantExtra.mlirIntegerSetGetContext(
        set::MlirIntegerSet
    )::MlirContext
end

"""
    mlirIntegerSetIsNull(set)

Checks whether an integer set is a null object.
"""
function mlirIntegerSetIsNull(set)
    @ccall Reactant_jll.libReactantExtra.mlirIntegerSetIsNull(set::MlirIntegerSet)::Bool
end

"""
    mlirIntegerSetEqual(s1, s2)

Checks if two integer set objects are equal. This is a "shallow" comparison of two objects. Only the sets with some small number of constraints are uniqued and compare equal here. Set objects that represent the same integer set with different constraints may be considered non-equal by this check. Set difference followed by an (expensive) emptiness check should be used to check equivalence of the underlying integer sets.
"""
function mlirIntegerSetEqual(s1, s2)
    @ccall Reactant_jll.libReactantExtra.mlirIntegerSetEqual(
        s1::MlirIntegerSet, s2::MlirIntegerSet
    )::Bool
end

"""
    mlirIntegerSetPrint(set, callback, userData)

Prints an integer set by sending chunks of the string representation and forwarding `userData to `callback`. Note that the callback may be called several times with consecutive chunks of the string.
"""
function mlirIntegerSetPrint(set, callback, userData)
    @ccall Reactant_jll.libReactantExtra.mlirIntegerSetPrint(
        set::MlirIntegerSet, callback::MlirStringCallback, userData::Ptr{Cvoid}
    )::Cvoid
end

"""
    mlirIntegerSetDump(set)

Prints an integer set to the standard error stream.
"""
function mlirIntegerSetDump(set)
    @ccall Reactant_jll.libReactantExtra.mlirIntegerSetDump(set::MlirIntegerSet)::Cvoid
end

"""
    mlirIntegerSetEmptyGet(context, numDims, numSymbols)

Gets or creates a new canonically empty integer set with the give number of dimensions and symbols in the given context.
"""
function mlirIntegerSetEmptyGet(context, numDims, numSymbols)
    @ccall Reactant_jll.libReactantExtra.mlirIntegerSetEmptyGet(
        context::MlirContext, numDims::Cptrdiff_t, numSymbols::Cptrdiff_t
    )::MlirIntegerSet
end

"""
    mlirIntegerSetGet(context, numDims, numSymbols, numConstraints, constraints, eqFlags)

Gets or creates a new integer set in the given context. The set is defined by a list of affine constraints, with the given number of input dimensions and symbols, which are treated as either equalities (eqFlags is 1) or inequalities (eqFlags is 0). Both `constraints` and `eqFlags` are expected to point to at least `numConstraint` consecutive values.
"""
function mlirIntegerSetGet(
    context, numDims, numSymbols, numConstraints, constraints, eqFlags
)
    @ccall Reactant_jll.libReactantExtra.mlirIntegerSetGet(
        context::MlirContext,
        numDims::Cptrdiff_t,
        numSymbols::Cptrdiff_t,
        numConstraints::Cptrdiff_t,
        constraints::Ptr{MlirAffineExpr},
        eqFlags::Ptr{Bool},
    )::MlirIntegerSet
end

"""
    mlirIntegerSetReplaceGet(set, dimReplacements, symbolReplacements, numResultDims, numResultSymbols)

Gets or creates a new integer set in which the values and dimensions of the given set are replaced with the given affine expressions. `dimReplacements` and `symbolReplacements` are expected to point to at least as many consecutive expressions as the given set has dimensions and symbols, respectively. The new set will have `numResultDims` and `numResultSymbols` dimensions and symbols, respectively.
"""
function mlirIntegerSetReplaceGet(
    set, dimReplacements, symbolReplacements, numResultDims, numResultSymbols
)
    @ccall Reactant_jll.libReactantExtra.mlirIntegerSetReplaceGet(
        set::MlirIntegerSet,
        dimReplacements::Ptr{MlirAffineExpr},
        symbolReplacements::Ptr{MlirAffineExpr},
        numResultDims::Cptrdiff_t,
        numResultSymbols::Cptrdiff_t,
    )::MlirIntegerSet
end

"""
    mlirIntegerSetIsCanonicalEmpty(set)

Checks whether the given set is a canonical empty set, e.g., the set returned by [`mlirIntegerSetEmptyGet`](@ref).
"""
function mlirIntegerSetIsCanonicalEmpty(set)
    @ccall Reactant_jll.libReactantExtra.mlirIntegerSetIsCanonicalEmpty(
        set::MlirIntegerSet
    )::Bool
end

"""
    mlirIntegerSetGetNumDims(set)

Returns the number of dimensions in the given set.
"""
function mlirIntegerSetGetNumDims(set)
    @ccall Reactant_jll.libReactantExtra.mlirIntegerSetGetNumDims(
        set::MlirIntegerSet
    )::Cptrdiff_t
end

"""
    mlirIntegerSetGetNumSymbols(set)

Returns the number of symbols in the given set.
"""
function mlirIntegerSetGetNumSymbols(set)
    @ccall Reactant_jll.libReactantExtra.mlirIntegerSetGetNumSymbols(
        set::MlirIntegerSet
    )::Cptrdiff_t
end

"""
    mlirIntegerSetGetNumInputs(set)

Returns the number of inputs (dimensions + symbols) in the given set.
"""
function mlirIntegerSetGetNumInputs(set)
    @ccall Reactant_jll.libReactantExtra.mlirIntegerSetGetNumInputs(
        set::MlirIntegerSet
    )::Cptrdiff_t
end

"""
    mlirIntegerSetGetNumConstraints(set)

Returns the number of constraints (equalities + inequalities) in the given set.
"""
function mlirIntegerSetGetNumConstraints(set)
    @ccall Reactant_jll.libReactantExtra.mlirIntegerSetGetNumConstraints(
        set::MlirIntegerSet
    )::Cptrdiff_t
end

"""
    mlirIntegerSetGetNumEqualities(set)

Returns the number of equalities in the given set.
"""
function mlirIntegerSetGetNumEqualities(set)
    @ccall Reactant_jll.libReactantExtra.mlirIntegerSetGetNumEqualities(
        set::MlirIntegerSet
    )::Cptrdiff_t
end

"""
    mlirIntegerSetGetNumInequalities(set)

Returns the number of inequalities in the given set.
"""
function mlirIntegerSetGetNumInequalities(set)
    @ccall Reactant_jll.libReactantExtra.mlirIntegerSetGetNumInequalities(
        set::MlirIntegerSet
    )::Cptrdiff_t
end

"""
    mlirIntegerSetGetConstraint(set, pos)

Returns `pos`-th constraint of the set.
"""
function mlirIntegerSetGetConstraint(set, pos)
    @ccall Reactant_jll.libReactantExtra.mlirIntegerSetGetConstraint(
        set::MlirIntegerSet, pos::Cptrdiff_t
    )::MlirAffineExpr
end

"""
    mlirIntegerSetIsConstraintEq(set, pos)

Returns `true` of the `pos`-th constraint of the set is an equality constraint, `false` otherwise.
"""
function mlirIntegerSetIsConstraintEq(set, pos)
    @ccall Reactant_jll.libReactantExtra.mlirIntegerSetIsConstraintEq(
        set::MlirIntegerSet, pos::Cptrdiff_t
    )::Bool
end

"""
    mlirAttributeGetNull()

Returns an empty attribute.
"""
function mlirAttributeGetNull()
    @ccall Reactant_jll.libReactantExtra.mlirAttributeGetNull()::MlirAttribute
end

function mlirAttributeIsALocation(attr)
    @ccall Reactant_jll.libReactantExtra.mlirAttributeIsALocation(attr::MlirAttribute)::Bool
end

"""
    mlirAttributeIsAAffineMap(attr)

Checks whether the given attribute is an affine map attribute.
"""
function mlirAttributeIsAAffineMap(attr)
    @ccall Reactant_jll.libReactantExtra.mlirAttributeIsAAffineMap(
        attr::MlirAttribute
    )::Bool
end

"""
    mlirAffineMapAttrGet(map)

Creates an affine map attribute wrapping the given map. The attribute belongs to the same context as the affine map.
"""
function mlirAffineMapAttrGet(map)
    @ccall Reactant_jll.libReactantExtra.mlirAffineMapAttrGet(
        map::MlirAffineMap
    )::MlirAttribute
end

function mlirAffineMapAttrGetName()
    @ccall Reactant_jll.libReactantExtra.mlirAffineMapAttrGetName()::MlirStringRef
end

"""
    mlirAffineMapAttrGetValue(attr)

Returns the affine map wrapped in the given affine map attribute.
"""
function mlirAffineMapAttrGetValue(attr)
    @ccall Reactant_jll.libReactantExtra.mlirAffineMapAttrGetValue(
        attr::MlirAttribute
    )::MlirAffineMap
end

"""
    mlirAffineMapAttrGetTypeID()

Returns the typeID of an AffineMap attribute.
"""
function mlirAffineMapAttrGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirAffineMapAttrGetTypeID()::MlirTypeID
end

"""
    mlirAttributeIsAArray(attr)

Checks whether the given attribute is an array attribute.
"""
function mlirAttributeIsAArray(attr)
    @ccall Reactant_jll.libReactantExtra.mlirAttributeIsAArray(attr::MlirAttribute)::Bool
end

"""
    mlirArrayAttrGet(ctx, numElements, elements)

Creates an array element containing the given list of elements in the given context.
"""
function mlirArrayAttrGet(ctx, numElements, elements)
    @ccall Reactant_jll.libReactantExtra.mlirArrayAttrGet(
        ctx::MlirContext, numElements::Cptrdiff_t, elements::Ptr{MlirAttribute}
    )::MlirAttribute
end

function mlirArrayAttrGetName()
    @ccall Reactant_jll.libReactantExtra.mlirArrayAttrGetName()::MlirStringRef
end

"""
    mlirArrayAttrGetNumElements(attr)

Returns the number of elements stored in the given array attribute.
"""
function mlirArrayAttrGetNumElements(attr)
    @ccall Reactant_jll.libReactantExtra.mlirArrayAttrGetNumElements(
        attr::MlirAttribute
    )::Cptrdiff_t
end

"""
    mlirArrayAttrGetElement(attr, pos)

Returns pos-th element stored in the given array attribute.
"""
function mlirArrayAttrGetElement(attr, pos)
    @ccall Reactant_jll.libReactantExtra.mlirArrayAttrGetElement(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::MlirAttribute
end

"""
    mlirArrayAttrGetTypeID()

Returns the typeID of an Array attribute.
"""
function mlirArrayAttrGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirArrayAttrGetTypeID()::MlirTypeID
end

"""
    mlirAttributeIsADictionary(attr)

Checks whether the given attribute is a dictionary attribute.
"""
function mlirAttributeIsADictionary(attr)
    @ccall Reactant_jll.libReactantExtra.mlirAttributeIsADictionary(
        attr::MlirAttribute
    )::Bool
end

"""
    mlirDictionaryAttrGet(ctx, numElements, elements)

Creates a dictionary attribute containing the given list of elements in the provided context.
"""
function mlirDictionaryAttrGet(ctx, numElements, elements)
    @ccall Reactant_jll.libReactantExtra.mlirDictionaryAttrGet(
        ctx::MlirContext, numElements::Cptrdiff_t, elements::Ptr{MlirNamedAttribute}
    )::MlirAttribute
end

function mlirDictionaryAttrGetName()
    @ccall Reactant_jll.libReactantExtra.mlirDictionaryAttrGetName()::MlirStringRef
end

"""
    mlirDictionaryAttrGetNumElements(attr)

Returns the number of attributes contained in a dictionary attribute.
"""
function mlirDictionaryAttrGetNumElements(attr)
    @ccall Reactant_jll.libReactantExtra.mlirDictionaryAttrGetNumElements(
        attr::MlirAttribute
    )::Cptrdiff_t
end

"""
    mlirDictionaryAttrGetElement(attr, pos)

Returns pos-th element of the given dictionary attribute.
"""
function mlirDictionaryAttrGetElement(attr, pos)
    @ccall Reactant_jll.libReactantExtra.mlirDictionaryAttrGetElement(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::MlirNamedAttribute
end

"""
    mlirDictionaryAttrGetElementByName(attr, name)

Returns the dictionary attribute element with the given name or NULL if the given name does not exist in the dictionary.
"""
function mlirDictionaryAttrGetElementByName(attr, name)
    @ccall Reactant_jll.libReactantExtra.mlirDictionaryAttrGetElementByName(
        attr::MlirAttribute, name::MlirStringRef
    )::MlirAttribute
end

"""
    mlirDictionaryAttrGetTypeID()

Returns the typeID of a Dictionary attribute.
"""
function mlirDictionaryAttrGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirDictionaryAttrGetTypeID()::MlirTypeID
end

"""
    mlirAttributeIsAFloat(attr)

Checks whether the given attribute is a floating point attribute.
"""
function mlirAttributeIsAFloat(attr)
    @ccall Reactant_jll.libReactantExtra.mlirAttributeIsAFloat(attr::MlirAttribute)::Bool
end

function mlirFloatAttrGetName()
    @ccall Reactant_jll.libReactantExtra.mlirFloatAttrGetName()::MlirStringRef
end

"""
    mlirFloatAttrDoubleGet(ctx, type, value)

Creates a floating point attribute in the given context with the given double value and double-precision FP semantics.
"""
function mlirFloatAttrDoubleGet(ctx, type, value)
    @ccall Reactant_jll.libReactantExtra.mlirFloatAttrDoubleGet(
        ctx::MlirContext, type::MlirType, value::Cdouble
    )::MlirAttribute
end

"""
    mlirFloatAttrDoubleGetChecked(loc, type, value)

Same as "[`mlirFloatAttrDoubleGet`](@ref)", but if the type is not valid for a construction of a FloatAttr, returns a null [`MlirAttribute`](@ref).
"""
function mlirFloatAttrDoubleGetChecked(loc, type, value)
    @ccall Reactant_jll.libReactantExtra.mlirFloatAttrDoubleGetChecked(
        loc::MlirLocation, type::MlirType, value::Cdouble
    )::MlirAttribute
end

"""
    mlirFloatAttrGetValueDouble(attr)

Returns the value stored in the given floating point attribute, interpreting the value as double.
"""
function mlirFloatAttrGetValueDouble(attr)
    @ccall Reactant_jll.libReactantExtra.mlirFloatAttrGetValueDouble(
        attr::MlirAttribute
    )::Cdouble
end

"""
    mlirFloatAttrGetTypeID()

Returns the typeID of a Float attribute.
"""
function mlirFloatAttrGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirFloatAttrGetTypeID()::MlirTypeID
end

"""
    mlirAttributeIsAInteger(attr)

Checks whether the given attribute is an integer attribute.
"""
function mlirAttributeIsAInteger(attr)
    @ccall Reactant_jll.libReactantExtra.mlirAttributeIsAInteger(attr::MlirAttribute)::Bool
end

"""
    mlirIntegerAttrGet(type, value)

Creates an integer attribute of the given type with the given integer value.
"""
function mlirIntegerAttrGet(type, value)
    @ccall Reactant_jll.libReactantExtra.mlirIntegerAttrGet(
        type::MlirType, value::Int64
    )::MlirAttribute
end

function mlirIntegerAttrGetName()
    @ccall Reactant_jll.libReactantExtra.mlirIntegerAttrGetName()::MlirStringRef
end

"""
    mlirIntegerAttrGetValueInt(attr)

Returns the value stored in the given integer attribute, assuming the value is of signless type and fits into a signed 64-bit integer.
"""
function mlirIntegerAttrGetValueInt(attr)
    @ccall Reactant_jll.libReactantExtra.mlirIntegerAttrGetValueInt(
        attr::MlirAttribute
    )::Int64
end

"""
    mlirIntegerAttrGetValueSInt(attr)

Returns the value stored in the given integer attribute, assuming the value is of signed type and fits into a signed 64-bit integer.
"""
function mlirIntegerAttrGetValueSInt(attr)
    @ccall Reactant_jll.libReactantExtra.mlirIntegerAttrGetValueSInt(
        attr::MlirAttribute
    )::Int64
end

"""
    mlirIntegerAttrGetValueUInt(attr)

Returns the value stored in the given integer attribute, assuming the value is of unsigned type and fits into an unsigned 64-bit integer.
"""
function mlirIntegerAttrGetValueUInt(attr)
    @ccall Reactant_jll.libReactantExtra.mlirIntegerAttrGetValueUInt(
        attr::MlirAttribute
    )::UInt64
end

"""
    mlirIntegerAttrGetValueBitWidth(attr)

Returns the bit width of the integer attribute's underlying APInt value. This is useful for determining the size of the integer, especially for values larger than 64 bits.
"""
function mlirIntegerAttrGetValueBitWidth(attr)
    @ccall Reactant_jll.libReactantExtra.mlirIntegerAttrGetValueBitWidth(
        attr::MlirAttribute
    )::Cuint
end

"""
    mlirIntegerAttrGetValueNumWords(attr)

Returns the number of 64-bit words that make up the integer attribute's underlying APInt value. For integers <= 64 bits, this returns 1.
"""
function mlirIntegerAttrGetValueNumWords(attr)
    @ccall Reactant_jll.libReactantExtra.mlirIntegerAttrGetValueNumWords(
        attr::MlirAttribute
    )::Cuint
end

"""
    mlirIntegerAttrGetValueWords(attr, words)

Copies the 64-bit words making up the integer attribute's APInt value into the provided buffer. The buffer must have space for at least [`mlirIntegerAttrGetValueNumWords`](@ref)(attr) elements. Words are stored in little-endian order (least significant word first). The sign information is not encoded in the words themselves; use the type's signedness to interpret the value correctly.
"""
function mlirIntegerAttrGetValueWords(attr, words)
    @ccall Reactant_jll.libReactantExtra.mlirIntegerAttrGetValueWords(
        attr::MlirAttribute, words::Ptr{UInt64}
    )::Cvoid
end

"""
    mlirIntegerAttrGetFromWords(type, numWords, words)

Creates an integer attribute of the given type from an array of 64-bit words. This is useful for creating integer attributes with values with widths larger than 64 bits. Words are in little-endian order (least significant word first). The number of words must match the bit width of the type: numWords = ceil(bitWidth / 64).
"""
function mlirIntegerAttrGetFromWords(type, numWords, words)
    @ccall Reactant_jll.libReactantExtra.mlirIntegerAttrGetFromWords(
        type::MlirType, numWords::Cuint, words::Ptr{UInt64}
    )::MlirAttribute
end

"""
    mlirIntegerAttrGetTypeID()

Returns the typeID of an Integer attribute.
"""
function mlirIntegerAttrGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirIntegerAttrGetTypeID()::MlirTypeID
end

"""
    mlirAttributeIsABool(attr)

Checks whether the given attribute is a bool attribute.
"""
function mlirAttributeIsABool(attr)
    @ccall Reactant_jll.libReactantExtra.mlirAttributeIsABool(attr::MlirAttribute)::Bool
end

"""
    mlirBoolAttrGet(ctx, value)

Creates a bool attribute in the given context with the given value.
"""
function mlirBoolAttrGet(ctx, value)
    @ccall Reactant_jll.libReactantExtra.mlirBoolAttrGet(
        ctx::MlirContext, value::Cint
    )::MlirAttribute
end

"""
    mlirBoolAttrGetValue(attr)

Returns the value stored in the given bool attribute.
"""
function mlirBoolAttrGetValue(attr)
    @ccall Reactant_jll.libReactantExtra.mlirBoolAttrGetValue(attr::MlirAttribute)::Bool
end

"""
    mlirAttributeIsAIntegerSet(attr)

Checks whether the given attribute is an integer set attribute.
"""
function mlirAttributeIsAIntegerSet(attr)
    @ccall Reactant_jll.libReactantExtra.mlirAttributeIsAIntegerSet(
        attr::MlirAttribute
    )::Bool
end

"""
    mlirIntegerSetAttrGet(set)

Creates an integer set attribute wrapping the given set. The attribute belongs to the same context as the integer set.
"""
function mlirIntegerSetAttrGet(set)
    @ccall Reactant_jll.libReactantExtra.mlirIntegerSetAttrGet(
        set::MlirIntegerSet
    )::MlirAttribute
end

function mlirIntegerSetAttrGetName()
    @ccall Reactant_jll.libReactantExtra.mlirIntegerSetAttrGetName()::MlirStringRef
end

"""
    mlirIntegerSetAttrGetValue(attr)

Returns the integer set wrapped in the given integer set attribute.
"""
function mlirIntegerSetAttrGetValue(attr)
    @ccall Reactant_jll.libReactantExtra.mlirIntegerSetAttrGetValue(
        attr::MlirAttribute
    )::MlirIntegerSet
end

"""
    mlirIntegerSetAttrGetTypeID()

Returns the typeID of an IntegerSet attribute.
"""
function mlirIntegerSetAttrGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirIntegerSetAttrGetTypeID()::MlirTypeID
end

"""
    mlirAttributeIsAOpaque(attr)

Checks whether the given attribute is an opaque attribute.
"""
function mlirAttributeIsAOpaque(attr)
    @ccall Reactant_jll.libReactantExtra.mlirAttributeIsAOpaque(attr::MlirAttribute)::Bool
end

"""
    mlirOpaqueAttrGet(ctx, dialectNamespace, dataLength, data, type)

Creates an opaque attribute in the given context associated with the dialect identified by its namespace. The attribute contains opaque byte data of the specified length (data need not be null-terminated).
"""
function mlirOpaqueAttrGet(ctx, dialectNamespace, dataLength, data, type)
    @ccall Reactant_jll.libReactantExtra.mlirOpaqueAttrGet(
        ctx::MlirContext,
        dialectNamespace::MlirStringRef,
        dataLength::Cptrdiff_t,
        data::Cstring,
        type::MlirType,
    )::MlirAttribute
end

function mlirOpaqueAttrGetName()
    @ccall Reactant_jll.libReactantExtra.mlirOpaqueAttrGetName()::MlirStringRef
end

"""
    mlirOpaqueAttrGetDialectNamespace(attr)

Returns the namespace of the dialect with which the given opaque attribute is associated. The namespace string is owned by the context.
"""
function mlirOpaqueAttrGetDialectNamespace(attr)
    @ccall Reactant_jll.libReactantExtra.mlirOpaqueAttrGetDialectNamespace(
        attr::MlirAttribute
    )::MlirStringRef
end

"""
    mlirOpaqueAttrGetData(attr)

Returns the raw data as a string reference. The data remains live as long as the context in which the attribute lives.
"""
function mlirOpaqueAttrGetData(attr)
    @ccall Reactant_jll.libReactantExtra.mlirOpaqueAttrGetData(
        attr::MlirAttribute
    )::MlirStringRef
end

"""
    mlirOpaqueAttrGetTypeID()

Returns the typeID of an Opaque attribute.
"""
function mlirOpaqueAttrGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirOpaqueAttrGetTypeID()::MlirTypeID
end

"""
    mlirAttributeIsAString(attr)

Checks whether the given attribute is a string attribute.
"""
function mlirAttributeIsAString(attr)
    @ccall Reactant_jll.libReactantExtra.mlirAttributeIsAString(attr::MlirAttribute)::Bool
end

"""
    mlirStringAttrGet(ctx, str)

Creates a string attribute in the given context containing the given string.
"""
function mlirStringAttrGet(ctx, str)
    @ccall Reactant_jll.libReactantExtra.mlirStringAttrGet(
        ctx::MlirContext, str::MlirStringRef
    )::MlirAttribute
end

function mlirStringAttrGetName()
    @ccall Reactant_jll.libReactantExtra.mlirStringAttrGetName()::MlirStringRef
end

"""
    mlirStringAttrTypedGet(type, str)

Creates a string attribute in the given context containing the given string. Additionally, the attribute has the given type.
"""
function mlirStringAttrTypedGet(type, str)
    @ccall Reactant_jll.libReactantExtra.mlirStringAttrTypedGet(
        type::MlirType, str::MlirStringRef
    )::MlirAttribute
end

"""
    mlirStringAttrGetValue(attr)

Returns the attribute values as a string reference. The data remains live as long as the context in which the attribute lives.
"""
function mlirStringAttrGetValue(attr)
    @ccall Reactant_jll.libReactantExtra.mlirStringAttrGetValue(
        attr::MlirAttribute
    )::MlirStringRef
end

"""
    mlirStringAttrGetTypeID()

Returns the typeID of a String attribute.
"""
function mlirStringAttrGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirStringAttrGetTypeID()::MlirTypeID
end

"""
    mlirAttributeIsASymbolRef(attr)

Checks whether the given attribute is a symbol reference attribute.
"""
function mlirAttributeIsASymbolRef(attr)
    @ccall Reactant_jll.libReactantExtra.mlirAttributeIsASymbolRef(
        attr::MlirAttribute
    )::Bool
end

"""
    mlirSymbolRefAttrGet(ctx, symbol, numReferences, references)

Creates a symbol reference attribute in the given context referencing a symbol identified by the given string inside a list of nested references. Each of the references in the list must not be nested.
"""
function mlirSymbolRefAttrGet(ctx, symbol, numReferences, references)
    @ccall Reactant_jll.libReactantExtra.mlirSymbolRefAttrGet(
        ctx::MlirContext,
        symbol::MlirStringRef,
        numReferences::Cptrdiff_t,
        references::Ptr{MlirAttribute},
    )::MlirAttribute
end

function mlirSymbolRefAttrGetName()
    @ccall Reactant_jll.libReactantExtra.mlirSymbolRefAttrGetName()::MlirStringRef
end

"""
    mlirSymbolRefAttrGetRootReference(attr)

Returns the string reference to the root referenced symbol. The data remains live as long as the context in which the attribute lives.
"""
function mlirSymbolRefAttrGetRootReference(attr)
    @ccall Reactant_jll.libReactantExtra.mlirSymbolRefAttrGetRootReference(
        attr::MlirAttribute
    )::MlirStringRef
end

"""
    mlirSymbolRefAttrGetLeafReference(attr)

Returns the string reference to the leaf referenced symbol. The data remains live as long as the context in which the attribute lives.
"""
function mlirSymbolRefAttrGetLeafReference(attr)
    @ccall Reactant_jll.libReactantExtra.mlirSymbolRefAttrGetLeafReference(
        attr::MlirAttribute
    )::MlirStringRef
end

"""
    mlirSymbolRefAttrGetNumNestedReferences(attr)

Returns the number of references nested in the given symbol reference attribute.
"""
function mlirSymbolRefAttrGetNumNestedReferences(attr)
    @ccall Reactant_jll.libReactantExtra.mlirSymbolRefAttrGetNumNestedReferences(
        attr::MlirAttribute
    )::Cptrdiff_t
end

"""
    mlirSymbolRefAttrGetNestedReference(attr, pos)

Returns pos-th reference nested in the given symbol reference attribute.
"""
function mlirSymbolRefAttrGetNestedReference(attr, pos)
    @ccall Reactant_jll.libReactantExtra.mlirSymbolRefAttrGetNestedReference(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::MlirAttribute
end

"""
    mlirSymbolRefAttrGetTypeID()

Returns the typeID of an SymbolRef attribute.
"""
function mlirSymbolRefAttrGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirSymbolRefAttrGetTypeID()::MlirTypeID
end

"""
    mlirDistinctAttrCreate(referencedAttr)

Creates a DistinctAttr with the referenced attribute.
"""
function mlirDistinctAttrCreate(referencedAttr)
    @ccall Reactant_jll.libReactantExtra.mlirDistinctAttrCreate(
        referencedAttr::MlirAttribute
    )::MlirAttribute
end

"""
    mlirAttributeIsAFlatSymbolRef(attr)

Checks whether the given attribute is a flat symbol reference attribute.
"""
function mlirAttributeIsAFlatSymbolRef(attr)
    @ccall Reactant_jll.libReactantExtra.mlirAttributeIsAFlatSymbolRef(
        attr::MlirAttribute
    )::Bool
end

"""
    mlirFlatSymbolRefAttrGet(ctx, symbol)

Creates a flat symbol reference attribute in the given context referencing a symbol identified by the given string.
"""
function mlirFlatSymbolRefAttrGet(ctx, symbol)
    @ccall Reactant_jll.libReactantExtra.mlirFlatSymbolRefAttrGet(
        ctx::MlirContext, symbol::MlirStringRef
    )::MlirAttribute
end

function mlirFlatSymbolRefAttrGetName()
    @ccall Reactant_jll.libReactantExtra.mlirFlatSymbolRefAttrGetName()::MlirStringRef
end

"""
    mlirFlatSymbolRefAttrGetValue(attr)

Returns the referenced symbol as a string reference. The data remains live as long as the context in which the attribute lives.
"""
function mlirFlatSymbolRefAttrGetValue(attr)
    @ccall Reactant_jll.libReactantExtra.mlirFlatSymbolRefAttrGetValue(
        attr::MlirAttribute
    )::MlirStringRef
end

"""
    mlirAttributeIsAType(attr)

Checks whether the given attribute is a type attribute.
"""
function mlirAttributeIsAType(attr)
    @ccall Reactant_jll.libReactantExtra.mlirAttributeIsAType(attr::MlirAttribute)::Bool
end

"""
    mlirTypeAttrGet(type)

Creates a type attribute wrapping the given type in the same context as the type.
"""
function mlirTypeAttrGet(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeAttrGet(type::MlirType)::MlirAttribute
end

function mlirTypeAttrGetName()
    @ccall Reactant_jll.libReactantExtra.mlirTypeAttrGetName()::MlirStringRef
end

"""
    mlirTypeAttrGetValue(attr)

Returns the type stored in the given type attribute.
"""
function mlirTypeAttrGetValue(attr)
    @ccall Reactant_jll.libReactantExtra.mlirTypeAttrGetValue(attr::MlirAttribute)::MlirType
end

"""
    mlirTypeAttrGetTypeID()

Returns the typeID of a Type attribute.
"""
function mlirTypeAttrGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirTypeAttrGetTypeID()::MlirTypeID
end

"""
    mlirAttributeIsAUnit(attr)

Checks whether the given attribute is a unit attribute.
"""
function mlirAttributeIsAUnit(attr)
    @ccall Reactant_jll.libReactantExtra.mlirAttributeIsAUnit(attr::MlirAttribute)::Bool
end

"""
    mlirUnitAttrGet(ctx)

Creates a unit attribute in the given context.
"""
function mlirUnitAttrGet(ctx)
    @ccall Reactant_jll.libReactantExtra.mlirUnitAttrGet(ctx::MlirContext)::MlirAttribute
end

function mlirUnitAttrGetName()
    @ccall Reactant_jll.libReactantExtra.mlirUnitAttrGetName()::MlirStringRef
end

"""
    mlirUnitAttrGetTypeID()

Returns the typeID of a Unit attribute.
"""
function mlirUnitAttrGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirUnitAttrGetTypeID()::MlirTypeID
end

"""
    mlirAttributeIsAElements(attr)

Checks whether the given attribute is an elements attribute.
"""
function mlirAttributeIsAElements(attr)
    @ccall Reactant_jll.libReactantExtra.mlirAttributeIsAElements(attr::MlirAttribute)::Bool
end

"""
    mlirElementsAttrGetValue(attr, rank, idxs)

Returns the element at the given rank-dimensional index.
"""
function mlirElementsAttrGetValue(attr, rank, idxs)
    @ccall Reactant_jll.libReactantExtra.mlirElementsAttrGetValue(
        attr::MlirAttribute, rank::Cptrdiff_t, idxs::Ptr{UInt64}
    )::MlirAttribute
end

"""
    mlirElementsAttrIsValidIndex(attr, rank, idxs)

Checks whether the given rank-dimensional index is valid in the given elements attribute.
"""
function mlirElementsAttrIsValidIndex(attr, rank, idxs)
    @ccall Reactant_jll.libReactantExtra.mlirElementsAttrIsValidIndex(
        attr::MlirAttribute, rank::Cptrdiff_t, idxs::Ptr{UInt64}
    )::Bool
end

"""
    mlirElementsAttrGetNumElements(attr)

Gets the total number of elements in the given elements attribute. In order to iterate over the attribute, obtain its type, which must be a statically shaped type and use its sizes to build a multi-dimensional index.
"""
function mlirElementsAttrGetNumElements(attr)
    @ccall Reactant_jll.libReactantExtra.mlirElementsAttrGetNumElements(
        attr::MlirAttribute
    )::Int64
end

function mlirDenseArrayAttrGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirDenseArrayAttrGetTypeID()::MlirTypeID
end

"""
    mlirAttributeIsADenseBoolArray(attr)

Checks whether the given attribute is a dense array attribute.
"""
function mlirAttributeIsADenseBoolArray(attr)
    @ccall Reactant_jll.libReactantExtra.mlirAttributeIsADenseBoolArray(
        attr::MlirAttribute
    )::Bool
end

function mlirAttributeIsADenseI8Array(attr)
    @ccall Reactant_jll.libReactantExtra.mlirAttributeIsADenseI8Array(
        attr::MlirAttribute
    )::Bool
end

function mlirAttributeIsADenseI16Array(attr)
    @ccall Reactant_jll.libReactantExtra.mlirAttributeIsADenseI16Array(
        attr::MlirAttribute
    )::Bool
end

function mlirAttributeIsADenseI32Array(attr)
    @ccall Reactant_jll.libReactantExtra.mlirAttributeIsADenseI32Array(
        attr::MlirAttribute
    )::Bool
end

function mlirAttributeIsADenseI64Array(attr)
    @ccall Reactant_jll.libReactantExtra.mlirAttributeIsADenseI64Array(
        attr::MlirAttribute
    )::Bool
end

function mlirAttributeIsADenseF32Array(attr)
    @ccall Reactant_jll.libReactantExtra.mlirAttributeIsADenseF32Array(
        attr::MlirAttribute
    )::Bool
end

function mlirAttributeIsADenseF64Array(attr)
    @ccall Reactant_jll.libReactantExtra.mlirAttributeIsADenseF64Array(
        attr::MlirAttribute
    )::Bool
end

"""
    mlirDenseBoolArrayGet(ctx, size, values)

Create a dense array attribute with the given elements.
"""
function mlirDenseBoolArrayGet(ctx, size, values)
    @ccall Reactant_jll.libReactantExtra.mlirDenseBoolArrayGet(
        ctx::MlirContext, size::Cptrdiff_t, values::Ptr{Cint}
    )::MlirAttribute
end

function mlirDenseI8ArrayGet(ctx, size, values)
    @ccall Reactant_jll.libReactantExtra.mlirDenseI8ArrayGet(
        ctx::MlirContext, size::Cptrdiff_t, values::Ptr{Int8}
    )::MlirAttribute
end

function mlirDenseI16ArrayGet(ctx, size, values)
    @ccall Reactant_jll.libReactantExtra.mlirDenseI16ArrayGet(
        ctx::MlirContext, size::Cptrdiff_t, values::Ptr{Int16}
    )::MlirAttribute
end

function mlirDenseI32ArrayGet(ctx, size, values)
    @ccall Reactant_jll.libReactantExtra.mlirDenseI32ArrayGet(
        ctx::MlirContext, size::Cptrdiff_t, values::Ptr{Int32}
    )::MlirAttribute
end

function mlirDenseI64ArrayGet(ctx, size, values)
    @ccall Reactant_jll.libReactantExtra.mlirDenseI64ArrayGet(
        ctx::MlirContext, size::Cptrdiff_t, values::Ptr{Int64}
    )::MlirAttribute
end

function mlirDenseF32ArrayGet(ctx, size, values)
    @ccall Reactant_jll.libReactantExtra.mlirDenseF32ArrayGet(
        ctx::MlirContext, size::Cptrdiff_t, values::Ptr{Cfloat}
    )::MlirAttribute
end

function mlirDenseF64ArrayGet(ctx, size, values)
    @ccall Reactant_jll.libReactantExtra.mlirDenseF64ArrayGet(
        ctx::MlirContext, size::Cptrdiff_t, values::Ptr{Cdouble}
    )::MlirAttribute
end

"""
    mlirDenseArrayGetNumElements(attr)

Get the size of a dense array.
"""
function mlirDenseArrayGetNumElements(attr)
    @ccall Reactant_jll.libReactantExtra.mlirDenseArrayGetNumElements(
        attr::MlirAttribute
    )::Cptrdiff_t
end

"""
    mlirDenseBoolArrayGetElement(attr, pos)

Get an element of a dense array.
"""
function mlirDenseBoolArrayGetElement(attr, pos)
    @ccall Reactant_jll.libReactantExtra.mlirDenseBoolArrayGetElement(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Bool
end

function mlirDenseI8ArrayGetElement(attr, pos)
    @ccall Reactant_jll.libReactantExtra.mlirDenseI8ArrayGetElement(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int8
end

function mlirDenseI16ArrayGetElement(attr, pos)
    @ccall Reactant_jll.libReactantExtra.mlirDenseI16ArrayGetElement(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int16
end

function mlirDenseI32ArrayGetElement(attr, pos)
    @ccall Reactant_jll.libReactantExtra.mlirDenseI32ArrayGetElement(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int32
end

function mlirDenseI64ArrayGetElement(attr, pos)
    @ccall Reactant_jll.libReactantExtra.mlirDenseI64ArrayGetElement(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function mlirDenseF32ArrayGetElement(attr, pos)
    @ccall Reactant_jll.libReactantExtra.mlirDenseF32ArrayGetElement(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Cfloat
end

function mlirDenseF64ArrayGetElement(attr, pos)
    @ccall Reactant_jll.libReactantExtra.mlirDenseF64ArrayGetElement(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Cdouble
end

"""
    mlirAttributeIsADenseElements(attr)

Checks whether the given attribute is a dense elements attribute.
"""
function mlirAttributeIsADenseElements(attr)
    @ccall Reactant_jll.libReactantExtra.mlirAttributeIsADenseElements(
        attr::MlirAttribute
    )::Bool
end

function mlirAttributeIsADenseIntElements(attr)
    @ccall Reactant_jll.libReactantExtra.mlirAttributeIsADenseIntElements(
        attr::MlirAttribute
    )::Bool
end

function mlirAttributeIsADenseFPElements(attr)
    @ccall Reactant_jll.libReactantExtra.mlirAttributeIsADenseFPElements(
        attr::MlirAttribute
    )::Bool
end

"""
    mlirDenseTypedElementsAttrGetTypeID()

Returns the typeID of a DenseTypedElements attribute.
"""
function mlirDenseTypedElementsAttrGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirDenseTypedElementsAttrGetTypeID()::MlirTypeID
end

"""
    mlirDenseIntOrFPElementsAttrGetTypeID()

Deprecated API. Will be removed in the future.
"""
function mlirDenseIntOrFPElementsAttrGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirDenseIntOrFPElementsAttrGetTypeID()::MlirTypeID
end

"""
    mlirDenseElementsAttrGet(shapedType, numElements, elements)

Creates a dense elements attribute with the given Shaped type and elements in the same context as the type.
"""
function mlirDenseElementsAttrGet(shapedType, numElements, elements)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrGet(
        shapedType::MlirType, numElements::Cptrdiff_t, elements::Ptr{MlirAttribute}
    )::MlirAttribute
end

"""
    mlirDenseElementsAttrRawBufferGet(shapedType, rawBufferSize, rawBuffer)

Creates a dense elements attribute with the given Shaped type and elements populated from a packed, row-major opaque buffer of contents.

The format of the raw buffer is a densely packed array of values that can be bitcast to the storage format of the element type specified. Types that are not byte aligned will be: - For bitwidth > 1: Rounded up to the next byte. - For bitwidth = 1: Packed into 8bit bytes with bits corresponding to the linear order of the shape type from MSB to LSB, padded to on the right.

A raw buffer of a single element (or for 1-bit, a byte of value 0 or 255) will be interpreted as a splat. User code should be prepared for additional, conformant patterns to be identified as splats in the future.
"""
function mlirDenseElementsAttrRawBufferGet(shapedType, rawBufferSize, rawBuffer)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrRawBufferGet(
        shapedType::MlirType, rawBufferSize::Csize_t, rawBuffer::Ptr{Cvoid}
    )::MlirAttribute
end

"""
    mlirDenseElementsAttrSplatGet(shapedType, element)

Creates a dense elements attribute with the given Shaped type containing a single replicated element (splat).
"""
function mlirDenseElementsAttrSplatGet(shapedType, element)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrSplatGet(
        shapedType::MlirType, element::MlirAttribute
    )::MlirAttribute
end

function mlirDenseElementsAttrBoolSplatGet(shapedType, element)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrBoolSplatGet(
        shapedType::MlirType, element::Bool
    )::MlirAttribute
end

function mlirDenseElementsAttrUInt8SplatGet(shapedType, element)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrUInt8SplatGet(
        shapedType::MlirType, element::UInt8
    )::MlirAttribute
end

function mlirDenseElementsAttrInt8SplatGet(shapedType, element)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrInt8SplatGet(
        shapedType::MlirType, element::Int8
    )::MlirAttribute
end

function mlirDenseElementsAttrUInt32SplatGet(shapedType, element)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrUInt32SplatGet(
        shapedType::MlirType, element::UInt32
    )::MlirAttribute
end

function mlirDenseElementsAttrInt32SplatGet(shapedType, element)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrInt32SplatGet(
        shapedType::MlirType, element::Int32
    )::MlirAttribute
end

function mlirDenseElementsAttrUInt64SplatGet(shapedType, element)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrUInt64SplatGet(
        shapedType::MlirType, element::UInt64
    )::MlirAttribute
end

function mlirDenseElementsAttrInt64SplatGet(shapedType, element)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrInt64SplatGet(
        shapedType::MlirType, element::Int64
    )::MlirAttribute
end

function mlirDenseElementsAttrFloatSplatGet(shapedType, element)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrFloatSplatGet(
        shapedType::MlirType, element::Cfloat
    )::MlirAttribute
end

function mlirDenseElementsAttrDoubleSplatGet(shapedType, element)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrDoubleSplatGet(
        shapedType::MlirType, element::Cdouble
    )::MlirAttribute
end

"""
    mlirDenseElementsAttrBoolGet(shapedType, numElements, elements)

Creates a dense elements attribute with the given shaped type from elements of a specific type. Expects the element type of the shaped type to match the data element type.
"""
function mlirDenseElementsAttrBoolGet(shapedType, numElements, elements)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrBoolGet(
        shapedType::MlirType, numElements::Cptrdiff_t, elements::Ptr{Cint}
    )::MlirAttribute
end

function mlirDenseElementsAttrUInt8Get(shapedType, numElements, elements)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrUInt8Get(
        shapedType::MlirType, numElements::Cptrdiff_t, elements::Ptr{UInt8}
    )::MlirAttribute
end

function mlirDenseElementsAttrInt8Get(shapedType, numElements, elements)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrInt8Get(
        shapedType::MlirType, numElements::Cptrdiff_t, elements::Ptr{Int8}
    )::MlirAttribute
end

function mlirDenseElementsAttrUInt16Get(shapedType, numElements, elements)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrUInt16Get(
        shapedType::MlirType, numElements::Cptrdiff_t, elements::Ptr{UInt16}
    )::MlirAttribute
end

function mlirDenseElementsAttrInt16Get(shapedType, numElements, elements)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrInt16Get(
        shapedType::MlirType, numElements::Cptrdiff_t, elements::Ptr{Int16}
    )::MlirAttribute
end

function mlirDenseElementsAttrUInt32Get(shapedType, numElements, elements)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrUInt32Get(
        shapedType::MlirType, numElements::Cptrdiff_t, elements::Ptr{UInt32}
    )::MlirAttribute
end

function mlirDenseElementsAttrInt32Get(shapedType, numElements, elements)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrInt32Get(
        shapedType::MlirType, numElements::Cptrdiff_t, elements::Ptr{Int32}
    )::MlirAttribute
end

function mlirDenseElementsAttrUInt64Get(shapedType, numElements, elements)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrUInt64Get(
        shapedType::MlirType, numElements::Cptrdiff_t, elements::Ptr{UInt64}
    )::MlirAttribute
end

function mlirDenseElementsAttrInt64Get(shapedType, numElements, elements)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrInt64Get(
        shapedType::MlirType, numElements::Cptrdiff_t, elements::Ptr{Int64}
    )::MlirAttribute
end

function mlirDenseElementsAttrFloatGet(shapedType, numElements, elements)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrFloatGet(
        shapedType::MlirType, numElements::Cptrdiff_t, elements::Ptr{Cfloat}
    )::MlirAttribute
end

function mlirDenseElementsAttrDoubleGet(shapedType, numElements, elements)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrDoubleGet(
        shapedType::MlirType, numElements::Cptrdiff_t, elements::Ptr{Cdouble}
    )::MlirAttribute
end

function mlirDenseElementsAttrBFloat16Get(shapedType, numElements, elements)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrBFloat16Get(
        shapedType::MlirType, numElements::Cptrdiff_t, elements::Ptr{UInt16}
    )::MlirAttribute
end

function mlirDenseElementsAttrFloat16Get(shapedType, numElements, elements)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrFloat16Get(
        shapedType::MlirType, numElements::Cptrdiff_t, elements::Ptr{UInt16}
    )::MlirAttribute
end

"""
    mlirDenseElementsAttrStringGet(shapedType, numElements, strs)

Creates a dense elements attribute with the given shaped type from string elements.
"""
function mlirDenseElementsAttrStringGet(shapedType, numElements, strs)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrStringGet(
        shapedType::MlirType, numElements::Cptrdiff_t, strs::Ptr{MlirStringRef}
    )::MlirAttribute
end

"""
    mlirDenseElementsAttrReshapeGet(attr, shapedType)

Creates a dense elements attribute that has the same data as the given dense elements attribute and a different shaped type. The new type must have the same total number of elements.
"""
function mlirDenseElementsAttrReshapeGet(attr, shapedType)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrReshapeGet(
        attr::MlirAttribute, shapedType::MlirType
    )::MlirAttribute
end

"""
    mlirDenseElementsAttrIsSplat(attr)

Checks whether the given dense elements attribute contains a single replicated value (splat).
"""
function mlirDenseElementsAttrIsSplat(attr)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrIsSplat(
        attr::MlirAttribute
    )::Bool
end

"""
    mlirDenseElementsAttrGetSplatValue(attr)

Returns the single replicated value (splat) of a specific type contained by the given dense elements attribute.
"""
function mlirDenseElementsAttrGetSplatValue(attr)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrGetSplatValue(
        attr::MlirAttribute
    )::MlirAttribute
end

function mlirDenseElementsAttrGetBoolSplatValue(attr)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrGetBoolSplatValue(
        attr::MlirAttribute
    )::Cint
end

function mlirDenseElementsAttrGetInt8SplatValue(attr)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrGetInt8SplatValue(
        attr::MlirAttribute
    )::Int8
end

function mlirDenseElementsAttrGetUInt8SplatValue(attr)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrGetUInt8SplatValue(
        attr::MlirAttribute
    )::UInt8
end

function mlirDenseElementsAttrGetInt32SplatValue(attr)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrGetInt32SplatValue(
        attr::MlirAttribute
    )::Int32
end

function mlirDenseElementsAttrGetUInt32SplatValue(attr)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrGetUInt32SplatValue(
        attr::MlirAttribute
    )::UInt32
end

function mlirDenseElementsAttrGetInt64SplatValue(attr)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrGetInt64SplatValue(
        attr::MlirAttribute
    )::Int64
end

function mlirDenseElementsAttrGetUInt64SplatValue(attr)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrGetUInt64SplatValue(
        attr::MlirAttribute
    )::UInt64
end

function mlirDenseElementsAttrGetFloatSplatValue(attr)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrGetFloatSplatValue(
        attr::MlirAttribute
    )::Cfloat
end

function mlirDenseElementsAttrGetDoubleSplatValue(attr)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrGetDoubleSplatValue(
        attr::MlirAttribute
    )::Cdouble
end

function mlirDenseElementsAttrGetStringSplatValue(attr)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrGetStringSplatValue(
        attr::MlirAttribute
    )::MlirStringRef
end

"""
    mlirDenseElementsAttrGetBoolValue(attr, pos)

Returns the pos-th value (flat contiguous indexing) of a specific type contained by the given dense elements attribute.
"""
function mlirDenseElementsAttrGetBoolValue(attr, pos)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrGetBoolValue(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Bool
end

function mlirDenseElementsAttrGetInt8Value(attr, pos)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrGetInt8Value(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int8
end

function mlirDenseElementsAttrGetUInt8Value(attr, pos)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrGetUInt8Value(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::UInt8
end

function mlirDenseElementsAttrGetInt16Value(attr, pos)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrGetInt16Value(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int16
end

function mlirDenseElementsAttrGetUInt16Value(attr, pos)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrGetUInt16Value(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::UInt16
end

function mlirDenseElementsAttrGetInt32Value(attr, pos)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrGetInt32Value(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int32
end

function mlirDenseElementsAttrGetUInt32Value(attr, pos)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrGetUInt32Value(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::UInt32
end

function mlirDenseElementsAttrGetInt64Value(attr, pos)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrGetInt64Value(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function mlirDenseElementsAttrGetUInt64Value(attr, pos)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrGetUInt64Value(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::UInt64
end

function mlirDenseElementsAttrGetIndexValue(attr, pos)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrGetIndexValue(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::UInt64
end

function mlirDenseElementsAttrGetFloatValue(attr, pos)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrGetFloatValue(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Cfloat
end

function mlirDenseElementsAttrGetDoubleValue(attr, pos)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrGetDoubleValue(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Cdouble
end

function mlirDenseElementsAttrGetStringValue(attr, pos)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrGetStringValue(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::MlirStringRef
end

"""
    mlirDenseElementsAttrGetRawData(attr)

Returns the raw data of the given dense elements attribute.
"""
function mlirDenseElementsAttrGetRawData(attr)
    @ccall Reactant_jll.libReactantExtra.mlirDenseElementsAttrGetRawData(
        attr::MlirAttribute
    )::Ptr{Cvoid}
end

function mlirAttributeIsADenseResourceElements(attr)
    @ccall Reactant_jll.libReactantExtra.mlirAttributeIsADenseResourceElements(
        attr::MlirAttribute
    )::Bool
end

"""
    mlirUnmanagedDenseResourceElementsAttrGet(shapedType, name, data, dataLength, dataAlignment, dataIsMutable, deleter, userData)

Unlike the typed accessors below, constructs the attribute with a raw data buffer and no type/alignment checking. Use a more strongly typed accessor if possible. If dataIsMutable is false, then an immutable AsmResourceBlob will be created and that passed data contents will be treated as const. If the deleter is non NULL, then it will be called when the data buffer can no longer be accessed (passing userData to it).
"""
function mlirUnmanagedDenseResourceElementsAttrGet(
    shapedType, name, data, dataLength, dataAlignment, dataIsMutable, deleter, userData
)
    @ccall Reactant_jll.libReactantExtra.mlirUnmanagedDenseResourceElementsAttrGet(
        shapedType::MlirType,
        name::MlirStringRef,
        data::Ptr{Cvoid},
        dataLength::Csize_t,
        dataAlignment::Csize_t,
        dataIsMutable::Bool,
        deleter::Ptr{Cvoid},
        userData::Ptr{Cvoid},
    )::MlirAttribute
end

function mlirDenseResourceElementsAttrGetName()
    @ccall Reactant_jll.libReactantExtra.mlirDenseResourceElementsAttrGetName()::MlirStringRef
end

function mlirUnmanagedDenseBoolResourceElementsAttrGet(
    shapedType, name, numElements, elements
)
    @ccall Reactant_jll.libReactantExtra.mlirUnmanagedDenseBoolResourceElementsAttrGet(
        shapedType::MlirType,
        name::MlirStringRef,
        numElements::Cptrdiff_t,
        elements::Ptr{Cint},
    )::MlirAttribute
end

function mlirUnmanagedDenseUInt8ResourceElementsAttrGet(
    shapedType, name, numElements, elements
)
    @ccall Reactant_jll.libReactantExtra.mlirUnmanagedDenseUInt8ResourceElementsAttrGet(
        shapedType::MlirType,
        name::MlirStringRef,
        numElements::Cptrdiff_t,
        elements::Ptr{UInt8},
    )::MlirAttribute
end

function mlirUnmanagedDenseInt8ResourceElementsAttrGet(
    shapedType, name, numElements, elements
)
    @ccall Reactant_jll.libReactantExtra.mlirUnmanagedDenseInt8ResourceElementsAttrGet(
        shapedType::MlirType,
        name::MlirStringRef,
        numElements::Cptrdiff_t,
        elements::Ptr{Int8},
    )::MlirAttribute
end

function mlirUnmanagedDenseUInt16ResourceElementsAttrGet(
    shapedType, name, numElements, elements
)
    @ccall Reactant_jll.libReactantExtra.mlirUnmanagedDenseUInt16ResourceElementsAttrGet(
        shapedType::MlirType,
        name::MlirStringRef,
        numElements::Cptrdiff_t,
        elements::Ptr{UInt16},
    )::MlirAttribute
end

function mlirUnmanagedDenseInt16ResourceElementsAttrGet(
    shapedType, name, numElements, elements
)
    @ccall Reactant_jll.libReactantExtra.mlirUnmanagedDenseInt16ResourceElementsAttrGet(
        shapedType::MlirType,
        name::MlirStringRef,
        numElements::Cptrdiff_t,
        elements::Ptr{Int16},
    )::MlirAttribute
end

function mlirUnmanagedDenseUInt32ResourceElementsAttrGet(
    shapedType, name, numElements, elements
)
    @ccall Reactant_jll.libReactantExtra.mlirUnmanagedDenseUInt32ResourceElementsAttrGet(
        shapedType::MlirType,
        name::MlirStringRef,
        numElements::Cptrdiff_t,
        elements::Ptr{UInt32},
    )::MlirAttribute
end

function mlirUnmanagedDenseInt32ResourceElementsAttrGet(
    shapedType, name, numElements, elements
)
    @ccall Reactant_jll.libReactantExtra.mlirUnmanagedDenseInt32ResourceElementsAttrGet(
        shapedType::MlirType,
        name::MlirStringRef,
        numElements::Cptrdiff_t,
        elements::Ptr{Int32},
    )::MlirAttribute
end

function mlirUnmanagedDenseUInt64ResourceElementsAttrGet(
    shapedType, name, numElements, elements
)
    @ccall Reactant_jll.libReactantExtra.mlirUnmanagedDenseUInt64ResourceElementsAttrGet(
        shapedType::MlirType,
        name::MlirStringRef,
        numElements::Cptrdiff_t,
        elements::Ptr{UInt64},
    )::MlirAttribute
end

function mlirUnmanagedDenseInt64ResourceElementsAttrGet(
    shapedType, name, numElements, elements
)
    @ccall Reactant_jll.libReactantExtra.mlirUnmanagedDenseInt64ResourceElementsAttrGet(
        shapedType::MlirType,
        name::MlirStringRef,
        numElements::Cptrdiff_t,
        elements::Ptr{Int64},
    )::MlirAttribute
end

function mlirUnmanagedDenseFloatResourceElementsAttrGet(
    shapedType, name, numElements, elements
)
    @ccall Reactant_jll.libReactantExtra.mlirUnmanagedDenseFloatResourceElementsAttrGet(
        shapedType::MlirType,
        name::MlirStringRef,
        numElements::Cptrdiff_t,
        elements::Ptr{Cfloat},
    )::MlirAttribute
end

function mlirUnmanagedDenseDoubleResourceElementsAttrGet(
    shapedType, name, numElements, elements
)
    @ccall Reactant_jll.libReactantExtra.mlirUnmanagedDenseDoubleResourceElementsAttrGet(
        shapedType::MlirType,
        name::MlirStringRef,
        numElements::Cptrdiff_t,
        elements::Ptr{Cdouble},
    )::MlirAttribute
end

"""
    mlirDenseBoolResourceElementsAttrGetValue(attr, pos)

Returns the pos-th value (flat contiguous indexing) of a specific type contained by the given dense resource elements attribute.
"""
function mlirDenseBoolResourceElementsAttrGetValue(attr, pos)
    @ccall Reactant_jll.libReactantExtra.mlirDenseBoolResourceElementsAttrGetValue(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Bool
end

function mlirDenseInt8ResourceElementsAttrGetValue(attr, pos)
    @ccall Reactant_jll.libReactantExtra.mlirDenseInt8ResourceElementsAttrGetValue(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int8
end

function mlirDenseUInt8ResourceElementsAttrGetValue(attr, pos)
    @ccall Reactant_jll.libReactantExtra.mlirDenseUInt8ResourceElementsAttrGetValue(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::UInt8
end

function mlirDenseInt16ResourceElementsAttrGetValue(attr, pos)
    @ccall Reactant_jll.libReactantExtra.mlirDenseInt16ResourceElementsAttrGetValue(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int16
end

function mlirDenseUInt16ResourceElementsAttrGetValue(attr, pos)
    @ccall Reactant_jll.libReactantExtra.mlirDenseUInt16ResourceElementsAttrGetValue(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::UInt16
end

function mlirDenseInt32ResourceElementsAttrGetValue(attr, pos)
    @ccall Reactant_jll.libReactantExtra.mlirDenseInt32ResourceElementsAttrGetValue(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int32
end

function mlirDenseUInt32ResourceElementsAttrGetValue(attr, pos)
    @ccall Reactant_jll.libReactantExtra.mlirDenseUInt32ResourceElementsAttrGetValue(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::UInt32
end

function mlirDenseInt64ResourceElementsAttrGetValue(attr, pos)
    @ccall Reactant_jll.libReactantExtra.mlirDenseInt64ResourceElementsAttrGetValue(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function mlirDenseUInt64ResourceElementsAttrGetValue(attr, pos)
    @ccall Reactant_jll.libReactantExtra.mlirDenseUInt64ResourceElementsAttrGetValue(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::UInt64
end

function mlirDenseFloatResourceElementsAttrGetValue(attr, pos)
    @ccall Reactant_jll.libReactantExtra.mlirDenseFloatResourceElementsAttrGetValue(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Cfloat
end

function mlirDenseDoubleResourceElementsAttrGetValue(attr, pos)
    @ccall Reactant_jll.libReactantExtra.mlirDenseDoubleResourceElementsAttrGetValue(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Cdouble
end

"""
    mlirAttributeIsASparseElements(attr)

Checks whether the given attribute is a sparse elements attribute.
"""
function mlirAttributeIsASparseElements(attr)
    @ccall Reactant_jll.libReactantExtra.mlirAttributeIsASparseElements(
        attr::MlirAttribute
    )::Bool
end

"""
    mlirSparseElementsAttribute(shapedType, denseIndices, denseValues)

Creates a sparse elements attribute of the given shape from a list of indices and a list of associated values. Both lists are expected to be dense elements attributes with the same number of elements. The list of indices is expected to contain 64-bit integers. The attribute is created in the same context as the type.
"""
function mlirSparseElementsAttribute(shapedType, denseIndices, denseValues)
    @ccall Reactant_jll.libReactantExtra.mlirSparseElementsAttribute(
        shapedType::MlirType, denseIndices::MlirAttribute, denseValues::MlirAttribute
    )::MlirAttribute
end

"""
    mlirSparseElementsAttrGetIndices(attr)

Returns the dense elements attribute containing 64-bit integer indices of non-null elements in the given sparse elements attribute.
"""
function mlirSparseElementsAttrGetIndices(attr)
    @ccall Reactant_jll.libReactantExtra.mlirSparseElementsAttrGetIndices(
        attr::MlirAttribute
    )::MlirAttribute
end

"""
    mlirSparseElementsAttrGetValues(attr)

Returns the dense elements attribute containing the non-null elements in the given sparse elements attribute.
"""
function mlirSparseElementsAttrGetValues(attr)
    @ccall Reactant_jll.libReactantExtra.mlirSparseElementsAttrGetValues(
        attr::MlirAttribute
    )::MlirAttribute
end

"""
    mlirSparseElementsAttrGetTypeID()

Returns the typeID of a SparseElements attribute.
"""
function mlirSparseElementsAttrGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirSparseElementsAttrGetTypeID()::MlirTypeID
end

function mlirAttributeIsAStridedLayout(attr)
    @ccall Reactant_jll.libReactantExtra.mlirAttributeIsAStridedLayout(
        attr::MlirAttribute
    )::Bool
end

function mlirStridedLayoutAttrGet(ctx, offset, numStrides, strides)
    @ccall Reactant_jll.libReactantExtra.mlirStridedLayoutAttrGet(
        ctx::MlirContext, offset::Int64, numStrides::Cptrdiff_t, strides::Ptr{Int64}
    )::MlirAttribute
end

function mlirStridedLayoutAttrGetName()
    @ccall Reactant_jll.libReactantExtra.mlirStridedLayoutAttrGetName()::MlirStringRef
end

function mlirStridedLayoutAttrGetOffset(attr)
    @ccall Reactant_jll.libReactantExtra.mlirStridedLayoutAttrGetOffset(
        attr::MlirAttribute
    )::Int64
end

function mlirStridedLayoutAttrGetNumStrides(attr)
    @ccall Reactant_jll.libReactantExtra.mlirStridedLayoutAttrGetNumStrides(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function mlirStridedLayoutAttrGetStride(attr, pos)
    @ccall Reactant_jll.libReactantExtra.mlirStridedLayoutAttrGetStride(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

"""
    mlirStridedLayoutAttrGetTypeID()

Returns the typeID of a StridedLayout attribute.
"""
function mlirStridedLayoutAttrGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirStridedLayoutAttrGetTypeID()::MlirTypeID
end

"""
    mlirIntegerTypeGetTypeID()

Returns the typeID of an Integer type.
"""
function mlirIntegerTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirIntegerTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAInteger(type)

Checks whether the given type is an integer type.
"""
function mlirTypeIsAInteger(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAInteger(type::MlirType)::Bool
end

"""
    mlirIntegerTypeGet(ctx, bitwidth)

Creates a signless integer type of the given bitwidth in the context. The type is owned by the context.
"""
function mlirIntegerTypeGet(ctx, bitwidth)
    @ccall Reactant_jll.libReactantExtra.mlirIntegerTypeGet(
        ctx::MlirContext, bitwidth::Cuint
    )::MlirType
end

function mlirIntegerTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirIntegerTypeGetName()::MlirStringRef
end

"""
    mlirIntegerTypeSignedGet(ctx, bitwidth)

Creates a signed integer type of the given bitwidth in the context. The type is owned by the context.
"""
function mlirIntegerTypeSignedGet(ctx, bitwidth)
    @ccall Reactant_jll.libReactantExtra.mlirIntegerTypeSignedGet(
        ctx::MlirContext, bitwidth::Cuint
    )::MlirType
end

"""
    mlirIntegerTypeUnsignedGet(ctx, bitwidth)

Creates an unsigned integer type of the given bitwidth in the context. The type is owned by the context.
"""
function mlirIntegerTypeUnsignedGet(ctx, bitwidth)
    @ccall Reactant_jll.libReactantExtra.mlirIntegerTypeUnsignedGet(
        ctx::MlirContext, bitwidth::Cuint
    )::MlirType
end

"""
    mlirIntegerTypeGetWidth(type)

Returns the bitwidth of an integer type.
"""
function mlirIntegerTypeGetWidth(type)
    @ccall Reactant_jll.libReactantExtra.mlirIntegerTypeGetWidth(type::MlirType)::Cuint
end

"""
    mlirIntegerTypeIsSignless(type)

Checks whether the given integer type is signless.
"""
function mlirIntegerTypeIsSignless(type)
    @ccall Reactant_jll.libReactantExtra.mlirIntegerTypeIsSignless(type::MlirType)::Bool
end

"""
    mlirIntegerTypeIsSigned(type)

Checks whether the given integer type is signed.
"""
function mlirIntegerTypeIsSigned(type)
    @ccall Reactant_jll.libReactantExtra.mlirIntegerTypeIsSigned(type::MlirType)::Bool
end

"""
    mlirIntegerTypeIsUnsigned(type)

Checks whether the given integer type is unsigned.
"""
function mlirIntegerTypeIsUnsigned(type)
    @ccall Reactant_jll.libReactantExtra.mlirIntegerTypeIsUnsigned(type::MlirType)::Bool
end

"""
    mlirIndexTypeGetTypeID()

Returns the typeID of an Index type.
"""
function mlirIndexTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirIndexTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAIndex(type)

Checks whether the given type is an index type.
"""
function mlirTypeIsAIndex(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAIndex(type::MlirType)::Bool
end

"""
    mlirIndexTypeGet(ctx)

Creates an index type in the given context. The type is owned by the context.
"""
function mlirIndexTypeGet(ctx)
    @ccall Reactant_jll.libReactantExtra.mlirIndexTypeGet(ctx::MlirContext)::MlirType
end

function mlirIndexTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirIndexTypeGetName()::MlirStringRef
end

"""
    mlirTypeIsAFloat(type)

Checks whether the given type is a floating-point type.
"""
function mlirTypeIsAFloat(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAFloat(type::MlirType)::Bool
end

"""
    mlirFloatTypeGetWidth(type)

Returns the bitwidth of a floating-point type.
"""
function mlirFloatTypeGetWidth(type)
    @ccall Reactant_jll.libReactantExtra.mlirFloatTypeGetWidth(type::MlirType)::Cuint
end

"""
    mlirFloat4E2M1FNTypeGetTypeID()

Returns the typeID of an Float4E2M1FN type.
"""
function mlirFloat4E2M1FNTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirFloat4E2M1FNTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAFloat4E2M1FN(type)

Checks whether the given type is an f4E2M1FN type.
"""
function mlirTypeIsAFloat4E2M1FN(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAFloat4E2M1FN(type::MlirType)::Bool
end

"""
    mlirFloat4E2M1FNTypeGet(ctx)

Creates an f4E2M1FN type in the given context. The type is owned by the context.
"""
function mlirFloat4E2M1FNTypeGet(ctx)
    @ccall Reactant_jll.libReactantExtra.mlirFloat4E2M1FNTypeGet(ctx::MlirContext)::MlirType
end

function mlirFloat4E2M1FNTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirFloat4E2M1FNTypeGetName()::MlirStringRef
end

"""
    mlirFloat6E2M3FNTypeGetTypeID()

Returns the typeID of an Float6E2M3FN type.
"""
function mlirFloat6E2M3FNTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirFloat6E2M3FNTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAFloat6E2M3FN(type)

Checks whether the given type is an f6E2M3FN type.
"""
function mlirTypeIsAFloat6E2M3FN(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAFloat6E2M3FN(type::MlirType)::Bool
end

"""
    mlirFloat6E2M3FNTypeGet(ctx)

Creates an f6E2M3FN type in the given context. The type is owned by the context.
"""
function mlirFloat6E2M3FNTypeGet(ctx)
    @ccall Reactant_jll.libReactantExtra.mlirFloat6E2M3FNTypeGet(ctx::MlirContext)::MlirType
end

function mlirFloat6E2M3FNTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirFloat6E2M3FNTypeGetName()::MlirStringRef
end

"""
    mlirFloat6E3M2FNTypeGetTypeID()

Returns the typeID of an Float6E3M2FN type.
"""
function mlirFloat6E3M2FNTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirFloat6E3M2FNTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAFloat6E3M2FN(type)

Checks whether the given type is an f6E3M2FN type.
"""
function mlirTypeIsAFloat6E3M2FN(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAFloat6E3M2FN(type::MlirType)::Bool
end

"""
    mlirFloat6E3M2FNTypeGet(ctx)

Creates an f6E3M2FN type in the given context. The type is owned by the context.
"""
function mlirFloat6E3M2FNTypeGet(ctx)
    @ccall Reactant_jll.libReactantExtra.mlirFloat6E3M2FNTypeGet(ctx::MlirContext)::MlirType
end

function mlirFloat6E3M2FNTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirFloat6E3M2FNTypeGetName()::MlirStringRef
end

"""
    mlirFloat8E5M2TypeGetTypeID()

Returns the typeID of an Float8E5M2 type.
"""
function mlirFloat8E5M2TypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirFloat8E5M2TypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAFloat8E5M2(type)

Checks whether the given type is an f8E5M2 type.
"""
function mlirTypeIsAFloat8E5M2(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAFloat8E5M2(type::MlirType)::Bool
end

"""
    mlirFloat8E5M2TypeGet(ctx)

Creates an f8E5M2 type in the given context. The type is owned by the context.
"""
function mlirFloat8E5M2TypeGet(ctx)
    @ccall Reactant_jll.libReactantExtra.mlirFloat8E5M2TypeGet(ctx::MlirContext)::MlirType
end

function mlirFloat8E5M2TypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirFloat8E5M2TypeGetName()::MlirStringRef
end

"""
    mlirFloat8E4M3TypeGetTypeID()

Returns the typeID of an Float8E4M3 type.
"""
function mlirFloat8E4M3TypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirFloat8E4M3TypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAFloat8E4M3(type)

Checks whether the given type is an f8E4M3 type.
"""
function mlirTypeIsAFloat8E4M3(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAFloat8E4M3(type::MlirType)::Bool
end

"""
    mlirFloat8E4M3TypeGet(ctx)

Creates an f8E4M3 type in the given context. The type is owned by the context.
"""
function mlirFloat8E4M3TypeGet(ctx)
    @ccall Reactant_jll.libReactantExtra.mlirFloat8E4M3TypeGet(ctx::MlirContext)::MlirType
end

function mlirFloat8E4M3TypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirFloat8E4M3TypeGetName()::MlirStringRef
end

"""
    mlirFloat8E4M3FNTypeGetTypeID()

Returns the typeID of an Float8E4M3FN type.
"""
function mlirFloat8E4M3FNTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirFloat8E4M3FNTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAFloat8E4M3FN(type)

Checks whether the given type is an f8E4M3FN type.
"""
function mlirTypeIsAFloat8E4M3FN(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAFloat8E4M3FN(type::MlirType)::Bool
end

"""
    mlirFloat8E4M3FNTypeGet(ctx)

Creates an f8E4M3FN type in the given context. The type is owned by the context.
"""
function mlirFloat8E4M3FNTypeGet(ctx)
    @ccall Reactant_jll.libReactantExtra.mlirFloat8E4M3FNTypeGet(ctx::MlirContext)::MlirType
end

function mlirFloat8E4M3FNTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirFloat8E4M3FNTypeGetName()::MlirStringRef
end

"""
    mlirFloat8E5M2FNUZTypeGetTypeID()

Returns the typeID of an Float8E5M2FNUZ type.
"""
function mlirFloat8E5M2FNUZTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirFloat8E5M2FNUZTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAFloat8E5M2FNUZ(type)

Checks whether the given type is an f8E5M2FNUZ type.
"""
function mlirTypeIsAFloat8E5M2FNUZ(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAFloat8E5M2FNUZ(type::MlirType)::Bool
end

"""
    mlirFloat8E5M2FNUZTypeGet(ctx)

Creates an f8E5M2FNUZ type in the given context. The type is owned by the context.
"""
function mlirFloat8E5M2FNUZTypeGet(ctx)
    @ccall Reactant_jll.libReactantExtra.mlirFloat8E5M2FNUZTypeGet(
        ctx::MlirContext
    )::MlirType
end

function mlirFloat8E5M2FNUZTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirFloat8E5M2FNUZTypeGetName()::MlirStringRef
end

"""
    mlirFloat8E4M3FNUZTypeGetTypeID()

Returns the typeID of an Float8E4M3FNUZ type.
"""
function mlirFloat8E4M3FNUZTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirFloat8E4M3FNUZTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAFloat8E4M3FNUZ(type)

Checks whether the given type is an f8E4M3FNUZ type.
"""
function mlirTypeIsAFloat8E4M3FNUZ(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAFloat8E4M3FNUZ(type::MlirType)::Bool
end

"""
    mlirFloat8E4M3FNUZTypeGet(ctx)

Creates an f8E4M3FNUZ type in the given context. The type is owned by the context.
"""
function mlirFloat8E4M3FNUZTypeGet(ctx)
    @ccall Reactant_jll.libReactantExtra.mlirFloat8E4M3FNUZTypeGet(
        ctx::MlirContext
    )::MlirType
end

function mlirFloat8E4M3FNUZTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirFloat8E4M3FNUZTypeGetName()::MlirStringRef
end

"""
    mlirFloat8E4M3B11FNUZTypeGetTypeID()

Returns the typeID of an Float8E4M3B11FNUZ type.
"""
function mlirFloat8E4M3B11FNUZTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirFloat8E4M3B11FNUZTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAFloat8E4M3B11FNUZ(type)

Checks whether the given type is an f8E4M3B11FNUZ type.
"""
function mlirTypeIsAFloat8E4M3B11FNUZ(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAFloat8E4M3B11FNUZ(type::MlirType)::Bool
end

"""
    mlirFloat8E4M3B11FNUZTypeGet(ctx)

Creates an f8E4M3B11FNUZ type in the given context. The type is owned by the context.
"""
function mlirFloat8E4M3B11FNUZTypeGet(ctx)
    @ccall Reactant_jll.libReactantExtra.mlirFloat8E4M3B11FNUZTypeGet(
        ctx::MlirContext
    )::MlirType
end

function mlirFloat8E4M3B11FNUZTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirFloat8E4M3B11FNUZTypeGetName()::MlirStringRef
end

"""
    mlirFloat8E3M4TypeGetTypeID()

Returns the typeID of an Float8E3M4 type.
"""
function mlirFloat8E3M4TypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirFloat8E3M4TypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAFloat8E3M4(type)

Checks whether the given type is an f8E3M4 type.
"""
function mlirTypeIsAFloat8E3M4(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAFloat8E3M4(type::MlirType)::Bool
end

"""
    mlirFloat8E3M4TypeGet(ctx)

Creates an f8E3M4 type in the given context. The type is owned by the context.
"""
function mlirFloat8E3M4TypeGet(ctx)
    @ccall Reactant_jll.libReactantExtra.mlirFloat8E3M4TypeGet(ctx::MlirContext)::MlirType
end

function mlirFloat8E3M4TypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirFloat8E3M4TypeGetName()::MlirStringRef
end

"""
    mlirFloat8E8M0FNUTypeGetTypeID()

Returns the typeID of an Float8E8M0FNU type.
"""
function mlirFloat8E8M0FNUTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirFloat8E8M0FNUTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAFloat8E8M0FNU(type)

Checks whether the given type is an f8E8M0FNU type.
"""
function mlirTypeIsAFloat8E8M0FNU(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAFloat8E8M0FNU(type::MlirType)::Bool
end

"""
    mlirFloat8E8M0FNUTypeGet(ctx)

Creates an f8E8M0FNU type in the given context. The type is owned by the context.
"""
function mlirFloat8E8M0FNUTypeGet(ctx)
    @ccall Reactant_jll.libReactantExtra.mlirFloat8E8M0FNUTypeGet(
        ctx::MlirContext
    )::MlirType
end

function mlirFloat8E8M0FNUTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirFloat8E8M0FNUTypeGetName()::MlirStringRef
end

"""
    mlirFloat8E5M3FNUTypeGetTypeID()

Returns the typeID of a Float8E5M3FNU type.
"""
function mlirFloat8E5M3FNUTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirFloat8E5M3FNUTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAFloat8E5M3FNU(type)

Checks whether the given type is an f8E5M3FNU type.
"""
function mlirTypeIsAFloat8E5M3FNU(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAFloat8E5M3FNU(type::MlirType)::Bool
end

"""
    mlirFloat8E5M3FNUTypeGet(ctx)

Creates an f8E5M3FNU type in the given context. The type is owned by the context.
"""
function mlirFloat8E5M3FNUTypeGet(ctx)
    @ccall Reactant_jll.libReactantExtra.mlirFloat8E5M3FNUTypeGet(
        ctx::MlirContext
    )::MlirType
end

function mlirFloat8E5M3FNUTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirFloat8E5M3FNUTypeGetName()::MlirStringRef
end

"""
    mlirBFloat16TypeGetTypeID()

Returns the typeID of an BFloat16 type.
"""
function mlirBFloat16TypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirBFloat16TypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsABF16(type)

Checks whether the given type is a bf16 type.
"""
function mlirTypeIsABF16(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsABF16(type::MlirType)::Bool
end

"""
    mlirBF16TypeGet(ctx)

Creates a bf16 type in the given context. The type is owned by the context.
"""
function mlirBF16TypeGet(ctx)
    @ccall Reactant_jll.libReactantExtra.mlirBF16TypeGet(ctx::MlirContext)::MlirType
end

function mlirBF16TypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirBF16TypeGetName()::MlirStringRef
end

"""
    mlirFloat16TypeGetTypeID()

Returns the typeID of an Float16 type.
"""
function mlirFloat16TypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirFloat16TypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAF16(type)

Checks whether the given type is an f16 type.
"""
function mlirTypeIsAF16(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAF16(type::MlirType)::Bool
end

"""
    mlirF16TypeGet(ctx)

Creates an f16 type in the given context. The type is owned by the context.
"""
function mlirF16TypeGet(ctx)
    @ccall Reactant_jll.libReactantExtra.mlirF16TypeGet(ctx::MlirContext)::MlirType
end

function mlirF16TypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirF16TypeGetName()::MlirStringRef
end

"""
    mlirFloat32TypeGetTypeID()

Returns the typeID of an Float32 type.
"""
function mlirFloat32TypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirFloat32TypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAF32(type)

Checks whether the given type is an f32 type.
"""
function mlirTypeIsAF32(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAF32(type::MlirType)::Bool
end

"""
    mlirF32TypeGet(ctx)

Creates an f32 type in the given context. The type is owned by the context.
"""
function mlirF32TypeGet(ctx)
    @ccall Reactant_jll.libReactantExtra.mlirF32TypeGet(ctx::MlirContext)::MlirType
end

function mlirF32TypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirF32TypeGetName()::MlirStringRef
end

"""
    mlirFloat64TypeGetTypeID()

Returns the typeID of an Float64 type.
"""
function mlirFloat64TypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirFloat64TypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAF64(type)

Checks whether the given type is an f64 type.
"""
function mlirTypeIsAF64(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAF64(type::MlirType)::Bool
end

"""
    mlirF64TypeGet(ctx)

Creates a f64 type in the given context. The type is owned by the context.
"""
function mlirF64TypeGet(ctx)
    @ccall Reactant_jll.libReactantExtra.mlirF64TypeGet(ctx::MlirContext)::MlirType
end

function mlirF64TypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirF64TypeGetName()::MlirStringRef
end

"""
    mlirFloatTF32TypeGetTypeID()

Returns the typeID of a TF32 type.
"""
function mlirFloatTF32TypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirFloatTF32TypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsATF32(type)

Checks whether the given type is an TF32 type.
"""
function mlirTypeIsATF32(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsATF32(type::MlirType)::Bool
end

"""
    mlirTF32TypeGet(ctx)

Creates a TF32 type in the given context. The type is owned by the context.
"""
function mlirTF32TypeGet(ctx)
    @ccall Reactant_jll.libReactantExtra.mlirTF32TypeGet(ctx::MlirContext)::MlirType
end

function mlirTF32TypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirTF32TypeGetName()::MlirStringRef
end

"""
    mlirNoneTypeGetTypeID()

Returns the typeID of an None type.
"""
function mlirNoneTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirNoneTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsANone(type)

Checks whether the given type is a None type.
"""
function mlirTypeIsANone(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsANone(type::MlirType)::Bool
end

"""
    mlirNoneTypeGet(ctx)

Creates a None type in the given context. The type is owned by the context.
"""
function mlirNoneTypeGet(ctx)
    @ccall Reactant_jll.libReactantExtra.mlirNoneTypeGet(ctx::MlirContext)::MlirType
end

function mlirNoneTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirNoneTypeGetName()::MlirStringRef
end

"""
    mlirComplexTypeGetTypeID()

Returns the typeID of an Complex type.
"""
function mlirComplexTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirComplexTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAComplex(type)

Checks whether the given type is a Complex type.
"""
function mlirTypeIsAComplex(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAComplex(type::MlirType)::Bool
end

"""
    mlirComplexTypeGet(elementType)

Creates a complex type with the given element type in the same context as the element type. The type is owned by the context.
"""
function mlirComplexTypeGet(elementType)
    @ccall Reactant_jll.libReactantExtra.mlirComplexTypeGet(elementType::MlirType)::MlirType
end

function mlirComplexTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirComplexTypeGetName()::MlirStringRef
end

"""
    mlirComplexTypeGetElementType(type)

Returns the element type of the given complex type.
"""
function mlirComplexTypeGetElementType(type)
    @ccall Reactant_jll.libReactantExtra.mlirComplexTypeGetElementType(
        type::MlirType
    )::MlirType
end

"""
    mlirTypeIsAShaped(type)

Checks whether the given type is a Shaped type.
"""
function mlirTypeIsAShaped(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAShaped(type::MlirType)::Bool
end

"""
    mlirShapedTypeGetElementType(type)

Returns the element type of the shaped type.
"""
function mlirShapedTypeGetElementType(type)
    @ccall Reactant_jll.libReactantExtra.mlirShapedTypeGetElementType(
        type::MlirType
    )::MlirType
end

"""
    mlirShapedTypeHasRank(type)

Checks whether the given shaped type is ranked.
"""
function mlirShapedTypeHasRank(type)
    @ccall Reactant_jll.libReactantExtra.mlirShapedTypeHasRank(type::MlirType)::Bool
end

"""
    mlirShapedTypeGetRank(type)

Returns the rank of the given ranked shaped type.
"""
function mlirShapedTypeGetRank(type)
    @ccall Reactant_jll.libReactantExtra.mlirShapedTypeGetRank(type::MlirType)::Int64
end

"""
    mlirShapedTypeHasStaticShape(type)

Checks whether the given shaped type has a static shape.
"""
function mlirShapedTypeHasStaticShape(type)
    @ccall Reactant_jll.libReactantExtra.mlirShapedTypeHasStaticShape(type::MlirType)::Bool
end

"""
    mlirShapedTypeIsDynamicDim(type, dim)

Checks whether the dim-th dimension of the given shaped type is dynamic.
"""
function mlirShapedTypeIsDynamicDim(type, dim)
    @ccall Reactant_jll.libReactantExtra.mlirShapedTypeIsDynamicDim(
        type::MlirType, dim::Cptrdiff_t
    )::Bool
end

"""
    mlirShapedTypeIsStaticDim(type, dim)

Checks whether the dim-th dimension of the given shaped type is static.
"""
function mlirShapedTypeIsStaticDim(type, dim)
    @ccall Reactant_jll.libReactantExtra.mlirShapedTypeIsStaticDim(
        type::MlirType, dim::Cptrdiff_t
    )::Bool
end

"""
    mlirShapedTypeGetDimSize(type, dim)

Returns the dim-th dimension of the given ranked shaped type.
"""
function mlirShapedTypeGetDimSize(type, dim)
    @ccall Reactant_jll.libReactantExtra.mlirShapedTypeGetDimSize(
        type::MlirType, dim::Cptrdiff_t
    )::Int64
end

"""
    mlirShapedTypeIsDynamicSize(size)

Checks whether the given value is used as a placeholder for dynamic sizes in shaped types.
"""
function mlirShapedTypeIsDynamicSize(size)
    @ccall Reactant_jll.libReactantExtra.mlirShapedTypeIsDynamicSize(size::Int64)::Bool
end

"""
    mlirShapedTypeIsStaticSize(size)

Checks whether the given shaped type dimension value is statically-sized.
"""
function mlirShapedTypeIsStaticSize(size)
    @ccall Reactant_jll.libReactantExtra.mlirShapedTypeIsStaticSize(size::Int64)::Bool
end

"""
    mlirShapedTypeGetDynamicSize()

Returns the value indicating a dynamic size in a shaped type. Prefer [`mlirShapedTypeIsDynamicSize`](@ref) and [`mlirShapedTypeIsStaticSize`](@ref) to direct comparisons with this value.
"""
function mlirShapedTypeGetDynamicSize()
    @ccall Reactant_jll.libReactantExtra.mlirShapedTypeGetDynamicSize()::Int64
end

"""
    mlirShapedTypeIsDynamicStrideOrOffset(val)

Checks whether the given value is used as a placeholder for dynamic strides and offsets in shaped types.
"""
function mlirShapedTypeIsDynamicStrideOrOffset(val)
    @ccall Reactant_jll.libReactantExtra.mlirShapedTypeIsDynamicStrideOrOffset(
        val::Int64
    )::Bool
end

"""
    mlirShapedTypeIsStaticStrideOrOffset(val)

Checks whether the given dimension value of a stride or an offset is statically-sized.
"""
function mlirShapedTypeIsStaticStrideOrOffset(val)
    @ccall Reactant_jll.libReactantExtra.mlirShapedTypeIsStaticStrideOrOffset(
        val::Int64
    )::Bool
end

"""
    mlirShapedTypeGetDynamicStrideOrOffset()

Returns the value indicating a dynamic stride or offset in a shaped type. Prefer [`mlirShapedTypeIsDynamicStrideOrOffset`](@ref) and [`mlirShapedTypeIsStaticStrideOrOffset`](@ref) to direct comparisons with this value.
"""
function mlirShapedTypeGetDynamicStrideOrOffset()
    @ccall Reactant_jll.libReactantExtra.mlirShapedTypeGetDynamicStrideOrOffset()::Int64
end

"""
    mlirVectorTypeGetTypeID()

Returns the typeID of an Vector type.
"""
function mlirVectorTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirVectorTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAVector(type)

Checks whether the given type is a Vector type.
"""
function mlirTypeIsAVector(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAVector(type::MlirType)::Bool
end

"""
    mlirVectorTypeGet(rank, shape, elementType)

Creates a vector type of the shape identified by its rank and dimensions, with the given element type in the same context as the element type. The type is owned by the context.
"""
function mlirVectorTypeGet(rank, shape, elementType)
    @ccall Reactant_jll.libReactantExtra.mlirVectorTypeGet(
        rank::Cptrdiff_t, shape::Ptr{Int64}, elementType::MlirType
    )::MlirType
end

function mlirVectorTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirVectorTypeGetName()::MlirStringRef
end

"""
    mlirVectorTypeGetChecked(loc, rank, shape, elementType)

Same as "[`mlirVectorTypeGet`](@ref)" but returns a nullptr wrapping [`MlirType`](@ref) on illegal arguments, emitting appropriate diagnostics.
"""
function mlirVectorTypeGetChecked(loc, rank, shape, elementType)
    @ccall Reactant_jll.libReactantExtra.mlirVectorTypeGetChecked(
        loc::MlirLocation, rank::Cptrdiff_t, shape::Ptr{Int64}, elementType::MlirType
    )::MlirType
end

"""
    mlirVectorTypeGetScalable(rank, shape, scalable, elementType)

Creates a scalable vector type with the shape identified by its rank and dimensions. A subset of dimensions may be marked as scalable via the corresponding flag list, which is expected to have as many entries as the rank of the vector. The vector is created in the same context as the element type.
"""
function mlirVectorTypeGetScalable(rank, shape, scalable, elementType)
    @ccall Reactant_jll.libReactantExtra.mlirVectorTypeGetScalable(
        rank::Cptrdiff_t, shape::Ptr{Int64}, scalable::Ptr{Bool}, elementType::MlirType
    )::MlirType
end

"""
    mlirVectorTypeGetScalableChecked(loc, rank, shape, scalable, elementType)

Same as "[`mlirVectorTypeGetScalable`](@ref)" but returns a nullptr wrapping [`MlirType`](@ref) on illegal arguments, emitting appropriate diagnostics.
"""
function mlirVectorTypeGetScalableChecked(loc, rank, shape, scalable, elementType)
    @ccall Reactant_jll.libReactantExtra.mlirVectorTypeGetScalableChecked(
        loc::MlirLocation,
        rank::Cptrdiff_t,
        shape::Ptr{Int64},
        scalable::Ptr{Bool},
        elementType::MlirType,
    )::MlirType
end

"""
    mlirVectorTypeIsScalable(type)

Checks whether the given vector type is scalable, i.e., has at least one scalable dimension.
"""
function mlirVectorTypeIsScalable(type)
    @ccall Reactant_jll.libReactantExtra.mlirVectorTypeIsScalable(type::MlirType)::Bool
end

"""
    mlirVectorTypeIsDimScalable(type, dim)

Checks whether the "dim"-th dimension of the given vector is scalable.
"""
function mlirVectorTypeIsDimScalable(type, dim)
    @ccall Reactant_jll.libReactantExtra.mlirVectorTypeIsDimScalable(
        type::MlirType, dim::Cptrdiff_t
    )::Bool
end

"""
    mlirTypeIsATensor(type)

Checks whether the given type is a Tensor type.
"""
function mlirTypeIsATensor(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsATensor(type::MlirType)::Bool
end

"""
    mlirRankedTensorTypeGetTypeID()

Returns the typeID of an RankedTensor type.
"""
function mlirRankedTensorTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirRankedTensorTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsARankedTensor(type)

Checks whether the given type is a ranked tensor type.
"""
function mlirTypeIsARankedTensor(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsARankedTensor(type::MlirType)::Bool
end

"""
    mlirUnrankedTensorTypeGetTypeID()

Returns the typeID of an UnrankedTensor type.
"""
function mlirUnrankedTensorTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirUnrankedTensorTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAUnrankedTensor(type)

Checks whether the given type is an unranked tensor type.
"""
function mlirTypeIsAUnrankedTensor(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAUnrankedTensor(type::MlirType)::Bool
end

"""
    mlirRankedTensorTypeGet(rank, shape, elementType, encoding)

Creates a tensor type of a fixed rank with the given shape, element type, and optional encoding in the same context as the element type. The type is owned by the context. Tensor types without any specific encoding field should assign [`mlirAttributeGetNull`](@ref)() to this parameter.
"""
function mlirRankedTensorTypeGet(rank, shape, elementType, encoding)
    @ccall Reactant_jll.libReactantExtra.mlirRankedTensorTypeGet(
        rank::Cptrdiff_t, shape::Ptr{Int64}, elementType::MlirType, encoding::MlirAttribute
    )::MlirType
end

function mlirRankedTensorTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirRankedTensorTypeGetName()::MlirStringRef
end

"""
    mlirRankedTensorTypeGetChecked(loc, rank, shape, elementType, encoding)

Same as "[`mlirRankedTensorTypeGet`](@ref)" but returns a nullptr wrapping [`MlirType`](@ref) on illegal arguments, emitting appropriate diagnostics.
"""
function mlirRankedTensorTypeGetChecked(loc, rank, shape, elementType, encoding)
    @ccall Reactant_jll.libReactantExtra.mlirRankedTensorTypeGetChecked(
        loc::MlirLocation,
        rank::Cptrdiff_t,
        shape::Ptr{Int64},
        elementType::MlirType,
        encoding::MlirAttribute,
    )::MlirType
end

"""
    mlirRankedTensorTypeGetEncoding(type)

Gets the 'encoding' attribute from the ranked tensor type, returning a null attribute if none.
"""
function mlirRankedTensorTypeGetEncoding(type)
    @ccall Reactant_jll.libReactantExtra.mlirRankedTensorTypeGetEncoding(
        type::MlirType
    )::MlirAttribute
end

"""
    mlirUnrankedTensorTypeGet(elementType)

Creates an unranked tensor type with the given element type in the same context as the element type. The type is owned by the context.
"""
function mlirUnrankedTensorTypeGet(elementType)
    @ccall Reactant_jll.libReactantExtra.mlirUnrankedTensorTypeGet(
        elementType::MlirType
    )::MlirType
end

function mlirUnrankedTensorTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirUnrankedTensorTypeGetName()::MlirStringRef
end

"""
    mlirUnrankedTensorTypeGetChecked(loc, elementType)

Same as "[`mlirUnrankedTensorTypeGet`](@ref)" but returns a nullptr wrapping [`MlirType`](@ref) on illegal arguments, emitting appropriate diagnostics.
"""
function mlirUnrankedTensorTypeGetChecked(loc, elementType)
    @ccall Reactant_jll.libReactantExtra.mlirUnrankedTensorTypeGetChecked(
        loc::MlirLocation, elementType::MlirType
    )::MlirType
end

"""
    mlirMemRefTypeGetTypeID()

Returns the typeID of an MemRef type.
"""
function mlirMemRefTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirMemRefTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAMemRef(type)

Checks whether the given type is a MemRef type.
"""
function mlirTypeIsAMemRef(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAMemRef(type::MlirType)::Bool
end

"""
    mlirUnrankedMemRefTypeGetTypeID()

Returns the typeID of an UnrankedMemRef type.
"""
function mlirUnrankedMemRefTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirUnrankedMemRefTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAUnrankedMemRef(type)

Checks whether the given type is an UnrankedMemRef type.
"""
function mlirTypeIsAUnrankedMemRef(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAUnrankedMemRef(type::MlirType)::Bool
end

"""
    mlirMemRefTypeGet(elementType, rank, shape, layout, memorySpace)

Creates a MemRef type with the given rank and shape, a potentially empty list of affine layout maps, the given memory space and element type, in the same context as element type. The type is owned by the context.
"""
function mlirMemRefTypeGet(elementType, rank, shape, layout, memorySpace)
    @ccall Reactant_jll.libReactantExtra.mlirMemRefTypeGet(
        elementType::MlirType,
        rank::Cptrdiff_t,
        shape::Ptr{Int64},
        layout::MlirAttribute,
        memorySpace::MlirAttribute,
    )::MlirType
end

function mlirMemRefTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirMemRefTypeGetName()::MlirStringRef
end

"""
    mlirMemRefTypeGetChecked(loc, elementType, rank, shape, layout, memorySpace)

Same as "[`mlirMemRefTypeGet`](@ref)" but returns a nullptr-wrapping [`MlirType`](@ref) o illegal arguments, emitting appropriate diagnostics.
"""
function mlirMemRefTypeGetChecked(loc, elementType, rank, shape, layout, memorySpace)
    @ccall Reactant_jll.libReactantExtra.mlirMemRefTypeGetChecked(
        loc::MlirLocation,
        elementType::MlirType,
        rank::Cptrdiff_t,
        shape::Ptr{Int64},
        layout::MlirAttribute,
        memorySpace::MlirAttribute,
    )::MlirType
end

"""
    mlirMemRefTypeContiguousGet(elementType, rank, shape, memorySpace)

Creates a MemRef type with the given rank, shape, memory space and element type in the same context as the element type. The type has no affine maps, i.e. represents a default row-major contiguous memref. The type is owned by the context.
"""
function mlirMemRefTypeContiguousGet(elementType, rank, shape, memorySpace)
    @ccall Reactant_jll.libReactantExtra.mlirMemRefTypeContiguousGet(
        elementType::MlirType,
        rank::Cptrdiff_t,
        shape::Ptr{Int64},
        memorySpace::MlirAttribute,
    )::MlirType
end

"""
    mlirMemRefTypeContiguousGetChecked(loc, elementType, rank, shape, memorySpace)

Same as "[`mlirMemRefTypeContiguousGet`](@ref)" but returns a nullptr wrapping [`MlirType`](@ref) on illegal arguments, emitting appropriate diagnostics.
"""
function mlirMemRefTypeContiguousGetChecked(loc, elementType, rank, shape, memorySpace)
    @ccall Reactant_jll.libReactantExtra.mlirMemRefTypeContiguousGetChecked(
        loc::MlirLocation,
        elementType::MlirType,
        rank::Cptrdiff_t,
        shape::Ptr{Int64},
        memorySpace::MlirAttribute,
    )::MlirType
end

"""
    mlirUnrankedMemRefTypeGet(elementType, memorySpace)

Creates an Unranked MemRef type with the given element type and in the given memory space. The type is owned by the context of element type.
"""
function mlirUnrankedMemRefTypeGet(elementType, memorySpace)
    @ccall Reactant_jll.libReactantExtra.mlirUnrankedMemRefTypeGet(
        elementType::MlirType, memorySpace::MlirAttribute
    )::MlirType
end

function mlirUnrankedMemRefTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirUnrankedMemRefTypeGetName()::MlirStringRef
end

"""
    mlirUnrankedMemRefTypeGetChecked(loc, elementType, memorySpace)

Same as "[`mlirUnrankedMemRefTypeGet`](@ref)" but returns a nullptr wrapping [`MlirType`](@ref) on illegal arguments, emitting appropriate diagnostics.
"""
function mlirUnrankedMemRefTypeGetChecked(loc, elementType, memorySpace)
    @ccall Reactant_jll.libReactantExtra.mlirUnrankedMemRefTypeGetChecked(
        loc::MlirLocation, elementType::MlirType, memorySpace::MlirAttribute
    )::MlirType
end

"""
    mlirMemRefTypeGetLayout(type)

Returns the layout of the given MemRef type.
"""
function mlirMemRefTypeGetLayout(type)
    @ccall Reactant_jll.libReactantExtra.mlirMemRefTypeGetLayout(
        type::MlirType
    )::MlirAttribute
end

"""
    mlirMemRefTypeGetAffineMap(type)

Returns the affine map of the given MemRef type.
"""
function mlirMemRefTypeGetAffineMap(type)
    @ccall Reactant_jll.libReactantExtra.mlirMemRefTypeGetAffineMap(
        type::MlirType
    )::MlirAffineMap
end

"""
    mlirMemRefTypeGetMemorySpace(type)

Returns the memory space of the given MemRef type.
"""
function mlirMemRefTypeGetMemorySpace(type)
    @ccall Reactant_jll.libReactantExtra.mlirMemRefTypeGetMemorySpace(
        type::MlirType
    )::MlirAttribute
end

"""
    mlirMemRefTypeGetStridesAndOffset(type, strides, offset)

Returns the strides of the MemRef if the layout map is in strided form. Both strides and offset are out params. strides must point to pre-allocated memory of length equal to the rank of the memref.
"""
function mlirMemRefTypeGetStridesAndOffset(type, strides, offset)
    @ccall Reactant_jll.libReactantExtra.mlirMemRefTypeGetStridesAndOffset(
        type::MlirType, strides::Ptr{Int64}, offset::Ptr{Int64}
    )::MlirLogicalResult
end

"""
    mlirUnrankedMemrefGetMemorySpace(type)

Returns the memory spcae of the given Unranked MemRef type.
"""
function mlirUnrankedMemrefGetMemorySpace(type)
    @ccall Reactant_jll.libReactantExtra.mlirUnrankedMemrefGetMemorySpace(
        type::MlirType
    )::MlirAttribute
end

"""
    mlirTupleTypeGetTypeID()

Returns the typeID of an Tuple type.
"""
function mlirTupleTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirTupleTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsATuple(type)

Checks whether the given type is a tuple type.
"""
function mlirTypeIsATuple(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsATuple(type::MlirType)::Bool
end

"""
    mlirTupleTypeGet(ctx, numElements, elements)

Creates a tuple type that consists of the given list of elemental types. The type is owned by the context.
"""
function mlirTupleTypeGet(ctx, numElements, elements)
    @ccall Reactant_jll.libReactantExtra.mlirTupleTypeGet(
        ctx::MlirContext, numElements::Cptrdiff_t, elements::Ptr{MlirType}
    )::MlirType
end

function mlirTupleTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirTupleTypeGetName()::MlirStringRef
end

"""
    mlirTupleTypeGetNumTypes(type)

Returns the number of types contained in a tuple.
"""
function mlirTupleTypeGetNumTypes(type)
    @ccall Reactant_jll.libReactantExtra.mlirTupleTypeGetNumTypes(
        type::MlirType
    )::Cptrdiff_t
end

"""
    mlirTupleTypeGetType(type, pos)

Returns the pos-th type in the tuple type.
"""
function mlirTupleTypeGetType(type, pos)
    @ccall Reactant_jll.libReactantExtra.mlirTupleTypeGetType(
        type::MlirType, pos::Cptrdiff_t
    )::MlirType
end

"""
    mlirFunctionTypeGetTypeID()

Returns the typeID of an Function type.
"""
function mlirFunctionTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirFunctionTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAFunction(type)

Checks whether the given type is a function type.
"""
function mlirTypeIsAFunction(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAFunction(type::MlirType)::Bool
end

"""
    mlirFunctionTypeGet(ctx, numInputs, inputs, numResults, results)

Creates a function type, mapping a list of input types to result types.
"""
function mlirFunctionTypeGet(ctx, numInputs, inputs, numResults, results)
    @ccall Reactant_jll.libReactantExtra.mlirFunctionTypeGet(
        ctx::MlirContext,
        numInputs::Cptrdiff_t,
        inputs::Ptr{MlirType},
        numResults::Cptrdiff_t,
        results::Ptr{MlirType},
    )::MlirType
end

function mlirFunctionTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirFunctionTypeGetName()::MlirStringRef
end

"""
    mlirFunctionTypeGetNumInputs(type)

Returns the number of input types.
"""
function mlirFunctionTypeGetNumInputs(type)
    @ccall Reactant_jll.libReactantExtra.mlirFunctionTypeGetNumInputs(
        type::MlirType
    )::Cptrdiff_t
end

"""
    mlirFunctionTypeGetNumResults(type)

Returns the number of result types.
"""
function mlirFunctionTypeGetNumResults(type)
    @ccall Reactant_jll.libReactantExtra.mlirFunctionTypeGetNumResults(
        type::MlirType
    )::Cptrdiff_t
end

"""
    mlirFunctionTypeGetInput(type, pos)

Returns the pos-th input type.
"""
function mlirFunctionTypeGetInput(type, pos)
    @ccall Reactant_jll.libReactantExtra.mlirFunctionTypeGetInput(
        type::MlirType, pos::Cptrdiff_t
    )::MlirType
end

"""
    mlirFunctionTypeGetResult(type, pos)

Returns the pos-th result type.
"""
function mlirFunctionTypeGetResult(type, pos)
    @ccall Reactant_jll.libReactantExtra.mlirFunctionTypeGetResult(
        type::MlirType, pos::Cptrdiff_t
    )::MlirType
end

"""
    mlirOpaqueTypeGetTypeID()

Returns the typeID of an Opaque type.
"""
function mlirOpaqueTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirOpaqueTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAOpaque(type)

Checks whether the given type is an opaque type.
"""
function mlirTypeIsAOpaque(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAOpaque(type::MlirType)::Bool
end

"""
    mlirOpaqueTypeGet(ctx, dialectNamespace, typeData)

Creates an opaque type in the given context associated with the dialect identified by its namespace. The type contains opaque byte data of the specified length (data need not be null-terminated).
"""
function mlirOpaqueTypeGet(ctx, dialectNamespace, typeData)
    @ccall Reactant_jll.libReactantExtra.mlirOpaqueTypeGet(
        ctx::MlirContext, dialectNamespace::MlirStringRef, typeData::MlirStringRef
    )::MlirType
end

function mlirOpaqueTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirOpaqueTypeGetName()::MlirStringRef
end

"""
    mlirOpaqueTypeGetDialectNamespace(type)

Returns the namespace of the dialect with which the given opaque type is associated. The namespace string is owned by the context.
"""
function mlirOpaqueTypeGetDialectNamespace(type)
    @ccall Reactant_jll.libReactantExtra.mlirOpaqueTypeGetDialectNamespace(
        type::MlirType
    )::MlirStringRef
end

"""
    mlirOpaqueTypeGetData(type)

Returns the raw data as a string reference. The data remains live as long as the context in which the type lives.
"""
function mlirOpaqueTypeGetData(type)
    @ccall Reactant_jll.libReactantExtra.mlirOpaqueTypeGetData(
        type::MlirType
    )::MlirStringRef
end

"""
    mlirEnableGlobalDebug(enable)

Sets the global debugging flag.
"""
function mlirEnableGlobalDebug(enable)
    @ccall Reactant_jll.libReactantExtra.mlirEnableGlobalDebug(enable::Bool)::Cvoid
end

"""
    mlirIsGlobalDebugEnabled()

Retuns `true` if the global debugging flag is set, false otherwise.
"""
function mlirIsGlobalDebugEnabled()
    @ccall Reactant_jll.libReactantExtra.mlirIsGlobalDebugEnabled()::Bool
end

"""
    mlirSetGlobalDebugType(type)

Sets the current debug type, similarly to `-debug-only=type` in the command-line tools. Note that global debug should be enabled for any output to be produced.
"""
function mlirSetGlobalDebugType(type)
    @ccall Reactant_jll.libReactantExtra.mlirSetGlobalDebugType(type::Cstring)::Cvoid
end

"""
    mlirSetGlobalDebugTypes(types, n)

Sets multiple current debug types, similarly to `-debug-only=type1,type2" in the command-line tools. Note that global debug should be enabled for any output to be produced.
"""
function mlirSetGlobalDebugTypes(types, n)
    @ccall Reactant_jll.libReactantExtra.mlirSetGlobalDebugTypes(
        types::Ptr{Cstring}, n::Cptrdiff_t
    )::Cvoid
end

"""
    mlirIsCurrentDebugType(type)

Checks if `type` is set as the current debug type.
"""
function mlirIsCurrentDebugType(type)
    @ccall Reactant_jll.libReactantExtra.mlirIsCurrentDebugType(type::Cstring)::Bool
end

"""
    MlirDiagnostic

An opaque reference to a diagnostic, always owned by the diagnostics engine (context). Must not be stored outside of the diagnostic handler.
"""
struct MlirDiagnostic
    ptr::Ptr{Cvoid}
end

"""
    MlirDiagnosticSeverity

Severity of a diagnostic.
"""
@cenum MlirDiagnosticSeverity::UInt32 begin
    MlirDiagnosticError = 0x0000000000000000
    MlirDiagnosticWarning = 0x0000000000000001
    MlirDiagnosticNote = 0x0000000000000002
    MlirDiagnosticRemark = 0x0000000000000003
end

"""
Opaque identifier of a diagnostic handler, useful to detach a handler.
"""
const MlirDiagnosticHandlerID = UInt64

# typedef MlirLogicalResult ( * MlirDiagnosticHandler ) ( MlirDiagnostic , void * userData )
"""
Diagnostic handler type. Accepts a reference to a diagnostic, which is only guaranteed to be live during the call. The handler is passed the `userData` that was provided when the handler was attached to a context. If the handler processed the diagnostic completely, it is expected to return success. Otherwise, it is expected to return failure to indicate that other handlers should attempt to process the diagnostic.
"""
const MlirDiagnosticHandler = Ptr{Cvoid}

"""
    mlirDiagnosticPrint(diagnostic, callback, userData)

Prints a diagnostic using the provided callback.
"""
function mlirDiagnosticPrint(diagnostic, callback, userData)
    @ccall Reactant_jll.libReactantExtra.mlirDiagnosticPrint(
        diagnostic::MlirDiagnostic, callback::MlirStringCallback, userData::Ptr{Cvoid}
    )::Cvoid
end

"""
    mlirDiagnosticGetLocation(diagnostic)

Returns the location at which the diagnostic is reported.
"""
function mlirDiagnosticGetLocation(diagnostic)
    @ccall Reactant_jll.libReactantExtra.mlirDiagnosticGetLocation(
        diagnostic::MlirDiagnostic
    )::MlirLocation
end

"""
    mlirDiagnosticGetSeverity(diagnostic)

Returns the severity of the diagnostic.
"""
function mlirDiagnosticGetSeverity(diagnostic)
    @ccall Reactant_jll.libReactantExtra.mlirDiagnosticGetSeverity(
        diagnostic::MlirDiagnostic
    )::MlirDiagnosticSeverity
end

"""
    mlirDiagnosticGetNumNotes(diagnostic)

Returns the number of notes attached to the diagnostic.
"""
function mlirDiagnosticGetNumNotes(diagnostic)
    @ccall Reactant_jll.libReactantExtra.mlirDiagnosticGetNumNotes(
        diagnostic::MlirDiagnostic
    )::Cptrdiff_t
end

"""
    mlirDiagnosticGetNote(diagnostic, pos)

Returns `pos`-th note attached to the diagnostic. Expects `pos` to be a valid zero-based index into the list of notes.
"""
function mlirDiagnosticGetNote(diagnostic, pos)
    @ccall Reactant_jll.libReactantExtra.mlirDiagnosticGetNote(
        diagnostic::MlirDiagnostic, pos::Cptrdiff_t
    )::MlirDiagnostic
end

"""
    mlirContextAttachDiagnosticHandler(context, handler, userData, deleteUserData)

Attaches the diagnostic handler to the context. Handlers are invoked in the reverse order of attachment until one of them processes the diagnostic completely. When a handler is invoked it is passed the `userData` that was provided when it was attached. If non-NULL, `deleteUserData` is called once the system no longer needs to call the handler (for instance after the handler is detached or the context is destroyed). Returns an identifier that can be used to detach the handler.
"""
function mlirContextAttachDiagnosticHandler(context, handler, userData, deleteUserData)
    @ccall Reactant_jll.libReactantExtra.mlirContextAttachDiagnosticHandler(
        context::MlirContext,
        handler::MlirDiagnosticHandler,
        userData::Ptr{Cvoid},
        deleteUserData::Ptr{Cvoid},
    )::MlirDiagnosticHandlerID
end

"""
    mlirContextDetachDiagnosticHandler(context, id)

Detaches an attached diagnostic handler from the context given its identifier.
"""
function mlirContextDetachDiagnosticHandler(context, id)
    @ccall Reactant_jll.libReactantExtra.mlirContextDetachDiagnosticHandler(
        context::MlirContext, id::MlirDiagnosticHandlerID
    )::Cvoid
end

"""
    mlirEmitError(location, message)

Emits an error at the given location through the diagnostics engine. Used for testing purposes.
"""
function mlirEmitError(location, message)
    @ccall Reactant_jll.libReactantExtra.mlirEmitError(
        location::MlirLocation, message::Cstring
    )::Cvoid
end

function mlirGetDialectHandle__amdgpu__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__amdgpu__()::MlirDialectHandle
end

function mlirTypeIsAAMDGPUTDMBaseType(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAAMDGPUTDMBaseType(type::MlirType)::Bool
end

function mlirAMDGPUTDMBaseTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirAMDGPUTDMBaseTypeGetTypeID()::MlirTypeID
end

function mlirAMDGPUTDMBaseTypeGet(ctx, elementType)
    @ccall Reactant_jll.libReactantExtra.mlirAMDGPUTDMBaseTypeGet(
        ctx::MlirContext, elementType::MlirType
    )::MlirType
end

function mlirAMDGPUTDMBaseTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirAMDGPUTDMBaseTypeGetName()::MlirStringRef
end

function mlirTypeIsAAMDGPUTDMDescriptorType(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAAMDGPUTDMDescriptorType(
        type::MlirType
    )::Bool
end

function mlirAMDGPUTDMDescriptorTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirAMDGPUTDMDescriptorTypeGetTypeID()::MlirTypeID
end

function mlirAMDGPUTDMDescriptorTypeGet(ctx)
    @ccall Reactant_jll.libReactantExtra.mlirAMDGPUTDMDescriptorTypeGet(
        ctx::MlirContext
    )::MlirType
end

function mlirAMDGPUTDMDescriptorTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirAMDGPUTDMDescriptorTypeGetName()::MlirStringRef
end

function mlirTypeIsAAMDGPUTDMGatherBaseType(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAAMDGPUTDMGatherBaseType(
        type::MlirType
    )::Bool
end

function mlirAMDGPUTDMGatherBaseTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirAMDGPUTDMGatherBaseTypeGetTypeID()::MlirTypeID
end

function mlirAMDGPUTDMGatherBaseTypeGet(ctx, elementType, indexType)
    @ccall Reactant_jll.libReactantExtra.mlirAMDGPUTDMGatherBaseTypeGet(
        ctx::MlirContext, elementType::MlirType, indexType::MlirType
    )::MlirType
end

function mlirAMDGPUTDMGatherBaseTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirAMDGPUTDMGatherBaseTypeGetName()::MlirStringRef
end

function mlirGetDialectHandle__affine__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__affine__()::MlirDialectHandle
end

function mlirGetDialectHandle__arith__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__arith__()::MlirDialectHandle
end

function mlirGetDialectHandle__arm_neon__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__arm_neon__()::MlirDialectHandle
end

function mlirGetDialectHandle__arm_sme__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__arm_sme__()::MlirDialectHandle
end

function mlirGetDialectHandle__arm_sve__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__arm_sve__()::MlirDialectHandle
end

function mlirGetDialectHandle__async__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__async__()::MlirDialectHandle
end

function mlirGetDialectHandle__bufferization__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__bufferization__()::MlirDialectHandle
end

function mlirGetDialectHandle__complex__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__complex__()::MlirDialectHandle
end

"""
    mlirAttributeIsAComplex(attr)

Checks whether the given attribute is a complex attribute.
"""
function mlirAttributeIsAComplex(attr)
    @ccall Reactant_jll.libReactantExtra.mlirAttributeIsAComplex(attr::MlirAttribute)::Bool
end

"""
    mlirComplexAttrDoubleGet(ctx, type, real, imag)

Creates a complex attribute in the given context with the given double real and imaginary values and double-precision FP semantics.
"""
function mlirComplexAttrDoubleGet(ctx, type, real, imag)
    @ccall Reactant_jll.libReactantExtra.mlirComplexAttrDoubleGet(
        ctx::MlirContext, type::MlirType, real::Cdouble, imag::Cdouble
    )::MlirAttribute
end

"""
    mlirComplexAttrDoubleGetChecked(loc, type, real, imag)

Same as "[`mlirComplexAttrDoubleGet`](@ref)", but if the type is not valid for a construction of a ComplexAttr, returns a null [`MlirAttribute`](@ref).
"""
function mlirComplexAttrDoubleGetChecked(loc, type, real, imag)
    @ccall Reactant_jll.libReactantExtra.mlirComplexAttrDoubleGetChecked(
        loc::MlirLocation, type::MlirType, real::Cdouble, imag::Cdouble
    )::MlirAttribute
end

"""
    mlirComplexAttrGetRealDouble(attr)

Returns the real value stored in the given complex attribute, interpreting the value as double.
"""
function mlirComplexAttrGetRealDouble(attr)
    @ccall Reactant_jll.libReactantExtra.mlirComplexAttrGetRealDouble(
        attr::MlirAttribute
    )::Cdouble
end

"""
    mlirComplexAttrGetImagDouble(attr)

Returns the imaginaryvalue stored in the given complex attribute, interpreting the value as double.
"""
function mlirComplexAttrGetImagDouble(attr)
    @ccall Reactant_jll.libReactantExtra.mlirComplexAttrGetImagDouble(
        attr::MlirAttribute
    )::Cdouble
end

"""
    mlirComplexAttrGetTypeID()

Returns the typeID of a Complex attribute.
"""
function mlirComplexAttrGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirComplexAttrGetTypeID()::MlirTypeID
end

function mlirGetDialectHandle__cf__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__cf__()::MlirDialectHandle
end

function mlirGetDialectHandle__dlti__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__dlti__()::MlirDialectHandle
end

function mlirGetDialectHandle__emitc__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__emitc__()::MlirDialectHandle
end

@cenum MlirEmitCCmpPredicate::UInt64 begin
    MLIR_EMITC_CMP_PREDICATE_EQ = 0x0000000000000000
    MLIR_EMITC_CMP_PREDICATE_NE = 0x0000000000000001
    MLIR_EMITC_CMP_PREDICATE_LT = 0x0000000000000002
    MLIR_EMITC_CMP_PREDICATE_LE = 0x0000000000000003
    MLIR_EMITC_CMP_PREDICATE_GT = 0x0000000000000004
    MLIR_EMITC_CMP_PREDICATE_GE = 0x0000000000000005
    MLIR_EMITC_CMP_PREDICATE_THREE_WAY = 0x0000000000000006
end

function mlirTypeIsAEmitCArrayType(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAEmitCArrayType(type::MlirType)::Bool
end

function mlirEmitCArrayTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirEmitCArrayTypeGetTypeID()::MlirTypeID
end

function mlirEmitCArrayTypeGet(nDims, shape, elementType)
    @ccall Reactant_jll.libReactantExtra.mlirEmitCArrayTypeGet(
        nDims::Cptrdiff_t, shape::Ptr{Int64}, elementType::MlirType
    )::MlirType
end

function mlirEmitCArrayTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirEmitCArrayTypeGetName()::MlirStringRef
end

function mlirTypeIsAEmitCLValueType(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAEmitCLValueType(type::MlirType)::Bool
end

function mlirEmitCLValueTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirEmitCLValueTypeGetTypeID()::MlirTypeID
end

function mlirEmitCLValueTypeGet(valueType)
    @ccall Reactant_jll.libReactantExtra.mlirEmitCLValueTypeGet(
        valueType::MlirType
    )::MlirType
end

function mlirEmitCLValueTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirEmitCLValueTypeGetName()::MlirStringRef
end

function mlirTypeIsAEmitCOpaqueType(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAEmitCOpaqueType(type::MlirType)::Bool
end

function mlirEmitCOpaqueTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirEmitCOpaqueTypeGetTypeID()::MlirTypeID
end

function mlirEmitCOpaqueTypeGet(ctx, value)
    @ccall Reactant_jll.libReactantExtra.mlirEmitCOpaqueTypeGet(
        ctx::MlirContext, value::MlirStringRef
    )::MlirType
end

function mlirEmitCOpaqueTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirEmitCOpaqueTypeGetName()::MlirStringRef
end

function mlirTypeIsAEmitCPointerType(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAEmitCPointerType(type::MlirType)::Bool
end

function mlirEmitCPointerTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirEmitCPointerTypeGetTypeID()::MlirTypeID
end

function mlirEmitCPointerTypeGet(pointee)
    @ccall Reactant_jll.libReactantExtra.mlirEmitCPointerTypeGet(
        pointee::MlirType
    )::MlirType
end

function mlirEmitCPointerTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirEmitCPointerTypeGetName()::MlirStringRef
end

function mlirTypeIsAEmitCPtrDiffTType(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAEmitCPtrDiffTType(type::MlirType)::Bool
end

function mlirEmitCPtrDiffTTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirEmitCPtrDiffTTypeGetTypeID()::MlirTypeID
end

function mlirEmitCPtrDiffTTypeGet(ctx)
    @ccall Reactant_jll.libReactantExtra.mlirEmitCPtrDiffTTypeGet(
        ctx::MlirContext
    )::MlirType
end

function mlirEmitCPtrDiffTTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirEmitCPtrDiffTTypeGetName()::MlirStringRef
end

function mlirTypeIsAEmitCSignedSizeTType(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAEmitCSignedSizeTType(
        type::MlirType
    )::Bool
end

function mlirEmitCSignedSizeTTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirEmitCSignedSizeTTypeGetTypeID()::MlirTypeID
end

function mlirEmitCSignedSizeTTypeGet(ctx)
    @ccall Reactant_jll.libReactantExtra.mlirEmitCSignedSizeTTypeGet(
        ctx::MlirContext
    )::MlirType
end

function mlirEmitCSignedSizeTTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirEmitCSignedSizeTTypeGetName()::MlirStringRef
end

function mlirTypeIsAEmitCSizeTType(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAEmitCSizeTType(type::MlirType)::Bool
end

function mlirEmitCSizeTTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirEmitCSizeTTypeGetTypeID()::MlirTypeID
end

function mlirEmitCSizeTTypeGet(ctx)
    @ccall Reactant_jll.libReactantExtra.mlirEmitCSizeTTypeGet(ctx::MlirContext)::MlirType
end

function mlirEmitCSizeTTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirEmitCSizeTTypeGetName()::MlirStringRef
end

function mlirAttributeIsAEmitCCmpPredicate(attr)
    @ccall Reactant_jll.libReactantExtra.mlirAttributeIsAEmitCCmpPredicate(
        attr::MlirAttribute
    )::Bool
end

function mlirEmitCCmpPredicateAttrGet(ctx, val)
    @ccall Reactant_jll.libReactantExtra.mlirEmitCCmpPredicateAttrGet(
        ctx::MlirContext, val::MlirEmitCCmpPredicate
    )::MlirAttribute
end

function mlirEmitCCmpPredicateAttrGetName()
    @ccall Reactant_jll.libReactantExtra.mlirEmitCCmpPredicateAttrGetName()::MlirStringRef
end

function mlirEmitCCmpPredicateAttrGetValue(attr)
    @ccall Reactant_jll.libReactantExtra.mlirEmitCCmpPredicateAttrGetValue(
        attr::MlirAttribute
    )::MlirEmitCCmpPredicate
end

function mlirEmitCCmpPredicateAttrGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirEmitCCmpPredicateAttrGetTypeID()::MlirTypeID
end

function mlirAttributeIsAEmitCOpaque(attr)
    @ccall Reactant_jll.libReactantExtra.mlirAttributeIsAEmitCOpaque(
        attr::MlirAttribute
    )::Bool
end

function mlirEmitCOpaqueAttrGet(ctx, value)
    @ccall Reactant_jll.libReactantExtra.mlirEmitCOpaqueAttrGet(
        ctx::MlirContext, value::MlirStringRef
    )::MlirAttribute
end

function mlirEmitCOpaqueAttrGetName()
    @ccall Reactant_jll.libReactantExtra.mlirEmitCOpaqueAttrGetName()::MlirStringRef
end

function mlirEmitCOpaqueAttrGetValue(attr)
    @ccall Reactant_jll.libReactantExtra.mlirEmitCOpaqueAttrGetValue(
        attr::MlirAttribute
    )::MlirStringRef
end

function mlirEmitCOpaqueAttrGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirEmitCOpaqueAttrGetTypeID()::MlirTypeID
end

function mlirGetDialectHandle__func__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__func__()::MlirDialectHandle
end

"""
    mlirFuncSetArgAttr(op, pos, name, attr)

Sets the argument attribute 'name' of an argument at index 'pos'. Asserts that the operation is a FuncOp.
"""
function mlirFuncSetArgAttr(op, pos, name, attr)
    @ccall Reactant_jll.libReactantExtra.mlirFuncSetArgAttr(
        op::MlirOperation, pos::Cptrdiff_t, name::MlirStringRef, attr::MlirAttribute
    )::Cvoid
end

function mlirFuncSetResultAttr(op, pos, name, attr)
    @ccall Reactant_jll.libReactantExtra.mlirFuncSetResultAttr(
        op::MlirOperation, pos::Cptrdiff_t, name::MlirStringRef, attr::MlirAttribute
    )::Cvoid
end

function mlirGetDialectHandle__gpu__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__gpu__()::MlirDialectHandle
end

function mlirTypeIsAGPUAsyncTokenType(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAGPUAsyncTokenType(type::MlirType)::Bool
end

function mlirGPUAsyncTokenTypeGet(ctx)
    @ccall Reactant_jll.libReactantExtra.mlirGPUAsyncTokenTypeGet(
        ctx::MlirContext
    )::MlirType
end

function mlirGPUAsyncTokenTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirGPUAsyncTokenTypeGetName()::MlirStringRef
end

function mlirAttributeIsAGPUObjectAttr(attr)
    @ccall Reactant_jll.libReactantExtra.mlirAttributeIsAGPUObjectAttr(
        attr::MlirAttribute
    )::Bool
end

function mlirGPUObjectAttrGet(mlirCtx, target, format, objectStrRef, mlirObjectProps)
    @ccall Reactant_jll.libReactantExtra.mlirGPUObjectAttrGet(
        mlirCtx::MlirContext,
        target::MlirAttribute,
        format::UInt32,
        objectStrRef::MlirStringRef,
        mlirObjectProps::MlirAttribute,
    )::MlirAttribute
end

function mlirGPUObjectAttrGetName()
    @ccall Reactant_jll.libReactantExtra.mlirGPUObjectAttrGetName()::MlirStringRef
end

function mlirGPUObjectAttrGetWithKernels(
    mlirCtx, target, format, objectStrRef, mlirObjectProps, mlirKernelsAttr
)
    @ccall Reactant_jll.libReactantExtra.mlirGPUObjectAttrGetWithKernels(
        mlirCtx::MlirContext,
        target::MlirAttribute,
        format::UInt32,
        objectStrRef::MlirStringRef,
        mlirObjectProps::MlirAttribute,
        mlirKernelsAttr::MlirAttribute,
    )::MlirAttribute
end

function mlirGPUObjectAttrGetTarget(mlirObjectAttr)
    @ccall Reactant_jll.libReactantExtra.mlirGPUObjectAttrGetTarget(
        mlirObjectAttr::MlirAttribute
    )::MlirAttribute
end

function mlirGPUObjectAttrGetFormat(mlirObjectAttr)
    @ccall Reactant_jll.libReactantExtra.mlirGPUObjectAttrGetFormat(
        mlirObjectAttr::MlirAttribute
    )::UInt32
end

function mlirGPUObjectAttrGetObject(mlirObjectAttr)
    @ccall Reactant_jll.libReactantExtra.mlirGPUObjectAttrGetObject(
        mlirObjectAttr::MlirAttribute
    )::MlirStringRef
end

function mlirGPUObjectAttrHasProperties(mlirObjectAttr)
    @ccall Reactant_jll.libReactantExtra.mlirGPUObjectAttrHasProperties(
        mlirObjectAttr::MlirAttribute
    )::Bool
end

function mlirGPUObjectAttrGetProperties(mlirObjectAttr)
    @ccall Reactant_jll.libReactantExtra.mlirGPUObjectAttrGetProperties(
        mlirObjectAttr::MlirAttribute
    )::MlirAttribute
end

function mlirGPUObjectAttrHasKernels(mlirObjectAttr)
    @ccall Reactant_jll.libReactantExtra.mlirGPUObjectAttrHasKernels(
        mlirObjectAttr::MlirAttribute
    )::Bool
end

function mlirGPUObjectAttrGetKernels(mlirObjectAttr)
    @ccall Reactant_jll.libReactantExtra.mlirGPUObjectAttrGetKernels(
        mlirObjectAttr::MlirAttribute
    )::MlirAttribute
end

function mlirGetDialectHandle__irdl__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__irdl__()::MlirDialectHandle
end

"""
    mlirLoadIRDLDialects(_module)

Loads all IRDL dialects in the provided module, registering the dialects in the module's associated context.
"""
function mlirLoadIRDLDialects(_module)
    @ccall Reactant_jll.libReactantExtra.mlirLoadIRDLDialects(
        _module::MlirModule
    )::MlirLogicalResult
end

function mlirIRDLVariadicityAttrGet(ctx, value)
    @ccall Reactant_jll.libReactantExtra.mlirIRDLVariadicityAttrGet(
        ctx::MlirContext, value::MlirStringRef
    )::MlirAttribute
end

function mlirIRDLVariadicityAttrGetName()
    @ccall Reactant_jll.libReactantExtra.mlirIRDLVariadicityAttrGetName()::MlirStringRef
end

function mlirIRDLVariadicityArrayAttrGet(ctx, nValues, values)
    @ccall Reactant_jll.libReactantExtra.mlirIRDLVariadicityArrayAttrGet(
        ctx::MlirContext, nValues::Cptrdiff_t, values::Ptr{MlirAttribute}
    )::MlirAttribute
end

function mlirIRDLVariadicityArrayAttrGetName()
    @ccall Reactant_jll.libReactantExtra.mlirIRDLVariadicityArrayAttrGetName()::MlirStringRef
end

function mlirGetDialectHandle__index__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__index__()::MlirDialectHandle
end

function mlirGetDialectHandle__llvm__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__llvm__()::MlirDialectHandle
end

"""
    mlirLLVMPointerTypeGet(ctx, addressSpace)

Creates an llvm.ptr type.
"""
function mlirLLVMPointerTypeGet(ctx, addressSpace)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMPointerTypeGet(
        ctx::MlirContext, addressSpace::Cuint
    )::MlirType
end

function mlirLLVMPointerTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirLLVMPointerTypeGetName()::MlirStringRef
end

function mlirLLVMPointerTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirLLVMPointerTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsALLVMPointerType(type)

Returns `true` if the type is an LLVM dialect pointer type.
"""
function mlirTypeIsALLVMPointerType(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsALLVMPointerType(type::MlirType)::Bool
end

"""
    mlirLLVMPointerTypeGetAddressSpace(pointerType)

Returns address space of llvm.ptr
"""
function mlirLLVMPointerTypeGetAddressSpace(pointerType)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMPointerTypeGetAddressSpace(
        pointerType::MlirType
    )::Cuint
end

"""
    mlirLLVMVoidTypeGet(ctx)

Creates an llmv.void type.
"""
function mlirLLVMVoidTypeGet(ctx)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMVoidTypeGet(ctx::MlirContext)::MlirType
end

function mlirLLVMVoidTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirLLVMVoidTypeGetName()::MlirStringRef
end

"""
    mlirTypeIsALLVMArrayType(type)

Returns `true` if the type is an LLVM dialect array type.
"""
function mlirTypeIsALLVMArrayType(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsALLVMArrayType(type::MlirType)::Bool
end

function mlirLLVMArrayTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirLLVMArrayTypeGetTypeID()::MlirTypeID
end

"""
    mlirLLVMArrayTypeGet(elementType, numElements)

Creates an llvm.array type.
"""
function mlirLLVMArrayTypeGet(elementType, numElements)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMArrayTypeGet(
        elementType::MlirType, numElements::Cuint
    )::MlirType
end

function mlirLLVMArrayTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirLLVMArrayTypeGetName()::MlirStringRef
end

"""
    mlirLLVMArrayTypeGetElementType(type)

Returns the element type of the llvm.array type.
"""
function mlirLLVMArrayTypeGetElementType(type)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMArrayTypeGetElementType(
        type::MlirType
    )::MlirType
end

"""
    mlirLLVMArrayTypeGetNumElements(type)

Returns the number of elements in the llvm.array type.
"""
function mlirLLVMArrayTypeGetNumElements(type)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMArrayTypeGetNumElements(
        type::MlirType
    )::Cuint
end

"""
    mlirLLVMFunctionTypeGet(resultType, nArgumentTypes, argumentTypes, isVarArg)

Creates an llvm.func type.
"""
function mlirLLVMFunctionTypeGet(resultType, nArgumentTypes, argumentTypes, isVarArg)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMFunctionTypeGet(
        resultType::MlirType,
        nArgumentTypes::Cptrdiff_t,
        argumentTypes::Ptr{MlirType},
        isVarArg::Bool,
    )::MlirType
end

function mlirLLVMFunctionTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirLLVMFunctionTypeGetName()::MlirStringRef
end

"""
    mlirTypeIsALLVMFunctionType(type)

Returns `true` if the type is an LLVM dialect function type.
"""
function mlirTypeIsALLVMFunctionType(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsALLVMFunctionType(type::MlirType)::Bool
end

"""
    mlirLLVMFunctionTypeGetTypeID()

Returns the TypeID of an LLVM function type.
"""
function mlirLLVMFunctionTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirLLVMFunctionTypeGetTypeID()::MlirTypeID
end

"""
    mlirLLVMFunctionTypeGetNumInputs(type)

Returns the number of input types.
"""
function mlirLLVMFunctionTypeGetNumInputs(type)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMFunctionTypeGetNumInputs(
        type::MlirType
    )::Cptrdiff_t
end

"""
    mlirLLVMFunctionTypeGetInput(type, pos)

Returns the pos-th input type.
"""
function mlirLLVMFunctionTypeGetInput(type, pos)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMFunctionTypeGetInput(
        type::MlirType, pos::Cptrdiff_t
    )::MlirType
end

"""
    mlirLLVMFunctionTypeIsVarArg(type)

Returns `true` if the function type is variadic.
"""
function mlirLLVMFunctionTypeIsVarArg(type)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMFunctionTypeIsVarArg(type::MlirType)::Bool
end

"""
    mlirLLVMFunctionTypeGetReturnType(type)

Returns the return type of the function type.
"""
function mlirLLVMFunctionTypeGetReturnType(type)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMFunctionTypeGetReturnType(
        type::MlirType
    )::MlirType
end

"""
    mlirTypeIsALLVMStructType(type)

Returns `true` if the type is an LLVM dialect struct type.
"""
function mlirTypeIsALLVMStructType(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsALLVMStructType(type::MlirType)::Bool
end

function mlirLLVMStructTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirLLVMStructTypeGetTypeID()::MlirTypeID
end

function mlirLLVMStructTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirLLVMStructTypeGetName()::MlirStringRef
end

"""
    mlirLLVMStructTypeIsLiteral(type)

Returns `true` if the type is a literal (unnamed) LLVM struct type.
"""
function mlirLLVMStructTypeIsLiteral(type)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMStructTypeIsLiteral(type::MlirType)::Bool
end

"""
    mlirLLVMStructTypeGetNumElementTypes(type)

Returns the number of fields in the struct. Asserts if the struct is opaque or not yet initialized.
"""
function mlirLLVMStructTypeGetNumElementTypes(type)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMStructTypeGetNumElementTypes(
        type::MlirType
    )::Cptrdiff_t
end

"""
    mlirLLVMStructTypeGetElementType(type, position)

Returns the `positions`-th field of the struct. Asserts if the struct is opaque, not yet initialized or if the position is out of range.
"""
function mlirLLVMStructTypeGetElementType(type, position)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMStructTypeGetElementType(
        type::MlirType, position::Cptrdiff_t
    )::MlirType
end

"""
    mlirLLVMStructTypeIsPacked(type)

Returns `true` if the struct is packed.
"""
function mlirLLVMStructTypeIsPacked(type)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMStructTypeIsPacked(type::MlirType)::Bool
end

"""
    mlirLLVMStructTypeGetIdentifier(type)

Returns the identifier of the identified struct. Asserts that the struct is identified, i.e., not literal.
"""
function mlirLLVMStructTypeGetIdentifier(type)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMStructTypeGetIdentifier(
        type::MlirType
    )::MlirStringRef
end

"""
    mlirLLVMStructTypeIsOpaque(type)

Returns `true` is the struct is explicitly opaque (will not have a body) or uninitialized (will eventually have a body).
"""
function mlirLLVMStructTypeIsOpaque(type)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMStructTypeIsOpaque(type::MlirType)::Bool
end

"""
    mlirLLVMStructTypeLiteralGet(ctx, nFieldTypes, fieldTypes, isPacked)

Creates an LLVM literal (unnamed) struct type. This may assert if the fields have types not compatible with the LLVM dialect. For a graceful failure, use the checked version.
"""
function mlirLLVMStructTypeLiteralGet(ctx, nFieldTypes, fieldTypes, isPacked)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMStructTypeLiteralGet(
        ctx::MlirContext, nFieldTypes::Cptrdiff_t, fieldTypes::Ptr{MlirType}, isPacked::Bool
    )::MlirType
end

"""
    mlirLLVMStructTypeLiteralGetChecked(loc, nFieldTypes, fieldTypes, isPacked)

Creates an LLVM literal (unnamed) struct type if possible. Emits a diagnostic at the given location and returns null otherwise.
"""
function mlirLLVMStructTypeLiteralGetChecked(loc, nFieldTypes, fieldTypes, isPacked)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMStructTypeLiteralGetChecked(
        loc::MlirLocation,
        nFieldTypes::Cptrdiff_t,
        fieldTypes::Ptr{MlirType},
        isPacked::Bool,
    )::MlirType
end

"""
    mlirLLVMStructTypeIdentifiedGet(ctx, name)

Creates an LLVM identified struct type with no body. If a struct type with this name already exists in the context, returns that type. Use [`mlirLLVMStructTypeIdentifiedNewGet`](@ref) to create a fresh struct type, potentially renaming it. The body should be set separatelty by calling [`mlirLLVMStructTypeSetBody`](@ref), if it isn't set already.
"""
function mlirLLVMStructTypeIdentifiedGet(ctx, name)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMStructTypeIdentifiedGet(
        ctx::MlirContext, name::MlirStringRef
    )::MlirType
end

"""
    mlirLLVMStructTypeIdentifiedNewGet(ctx, name, nFieldTypes, fieldTypes, isPacked)

Creates an LLVM identified struct type with no body and a name starting with the given prefix. If a struct with the exact name as the given prefix already exists, appends an unspecified suffix to the name so that the name is unique in context.
"""
function mlirLLVMStructTypeIdentifiedNewGet(ctx, name, nFieldTypes, fieldTypes, isPacked)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMStructTypeIdentifiedNewGet(
        ctx::MlirContext,
        name::MlirStringRef,
        nFieldTypes::Cptrdiff_t,
        fieldTypes::Ptr{MlirType},
        isPacked::Bool,
    )::MlirType
end

function mlirLLVMStructTypeOpaqueGet(ctx, name)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMStructTypeOpaqueGet(
        ctx::MlirContext, name::MlirStringRef
    )::MlirType
end

"""
    mlirLLVMStructTypeSetBody(structType, nFieldTypes, fieldTypes, isPacked)

Sets the body of the identified struct if it hasn't been set yet. Returns whether the operation was successful.
"""
function mlirLLVMStructTypeSetBody(structType, nFieldTypes, fieldTypes, isPacked)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMStructTypeSetBody(
        structType::MlirType,
        nFieldTypes::Cptrdiff_t,
        fieldTypes::Ptr{MlirType},
        isPacked::Bool,
    )::MlirLogicalResult
end

@cenum MlirLLVMCConv::UInt32 begin
    MlirLLVMCConvC = 0x0000000000000000
    MlirLLVMCConvFast = 0x0000000000000008
    MlirLLVMCConvCold = 0x0000000000000009
    MlirLLVMCConvGHC = 0x000000000000000a
    MlirLLVMCConvHiPE = 0x000000000000000b
    MlirLLVMCConvAnyReg = 0x000000000000000d
    MlirLLVMCConvPreserveMost = 0x000000000000000e
    MlirLLVMCConvPreserveAll = 0x000000000000000f
    MlirLLVMCConvSwift = 0x0000000000000010
    MlirLLVMCConvCXX_FAST_TLS = 0x0000000000000011
    MlirLLVMCConvTail = 0x0000000000000012
    MlirLLVMCConvCFGuard_Check = 0x0000000000000013
    MlirLLVMCConvSwiftTail = 0x0000000000000014
    MlirLLVMCConvX86_StdCall = 0x0000000000000040
    MlirLLVMCConvX86_FastCall = 0x0000000000000041
    MlirLLVMCConvARM_APCS = 0x0000000000000042
    MlirLLVMCConvARM_AAPCS = 0x0000000000000043
    MlirLLVMCConvARM_AAPCS_VFP = 0x0000000000000044
    MlirLLVMCConvMSP430_INTR = 0x0000000000000045
    MlirLLVMCConvX86_ThisCall = 0x0000000000000046
    MlirLLVMCConvPTX_Kernel = 0x0000000000000047
    MlirLLVMCConvPTX_Device = 0x0000000000000048
    MlirLLVMCConvSPIR_FUNC = 0x000000000000004b
    MlirLLVMCConvSPIR_KERNEL = 0x000000000000004c
    MlirLLVMCConvIntel_OCL_BI = 0x000000000000004d
    MlirLLVMCConvX86_64_SysV = 0x000000000000004e
    MlirLLVMCConvWin64 = 0x000000000000004f
    MlirLLVMCConvX86_VectorCall = 0x0000000000000050
    MlirLLVMCConvDUMMY_HHVM = 0x0000000000000051
    MlirLLVMCConvDUMMY_HHVM_C = 0x0000000000000052
    MlirLLVMCConvX86_INTR = 0x0000000000000053
    MlirLLVMCConvAVR_INTR = 0x0000000000000054
    MlirLLVMCConvAVR_BUILTIN = 0x0000000000000056
    MlirLLVMCConvAMDGPU_VS = 0x0000000000000057
    MlirLLVMCConvAMDGPU_GS = 0x0000000000000058
    MlirLLVMCConvAMDGPU_CS = 0x000000000000005a
    MlirLLVMCConvAMDGPU_KERNEL = 0x000000000000005b
    MlirLLVMCConvX86_RegCall = 0x000000000000005c
    MlirLLVMCConvAMDGPU_HS = 0x000000000000005d
    MlirLLVMCConvMSP430_BUILTIN = 0x000000000000005e
    MlirLLVMCConvAMDGPU_LS = 0x000000000000005f
    MlirLLVMCConvAMDGPU_ES = 0x0000000000000060
    MlirLLVMCConvAArch64_VectorCall = 0x0000000000000061
    MlirLLVMCConvAArch64_SVE_VectorCall = 0x0000000000000062
    MlirLLVMCConvWASM_EmscriptenInvoke = 0x0000000000000063
    MlirLLVMCConvAMDGPU_Gfx = 0x0000000000000064
    MlirLLVMCConvM68k_INTR = 0x0000000000000065
end

"""
    mlirLLVMCConvAttrGet(ctx, cconv)

Creates a LLVM CConv attribute.
"""
function mlirLLVMCConvAttrGet(ctx, cconv)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMCConvAttrGet(
        ctx::MlirContext, cconv::MlirLLVMCConv
    )::MlirAttribute
end

function mlirLLVMCConvAttrGetName()
    @ccall Reactant_jll.libReactantExtra.mlirLLVMCConvAttrGetName()::MlirStringRef
end

@cenum MlirLLVMComdat::UInt32 begin
    MlirLLVMComdatAny = 0x0000000000000000
    MlirLLVMComdatExactMatch = 0x0000000000000001
    MlirLLVMComdatLargest = 0x0000000000000002
    MlirLLVMComdatNoDeduplicate = 0x0000000000000003
    MlirLLVMComdatSameSize = 0x0000000000000004
end

"""
    mlirLLVMComdatAttrGet(ctx, comdat)

Creates a LLVM Comdat attribute.
"""
function mlirLLVMComdatAttrGet(ctx, comdat)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMComdatAttrGet(
        ctx::MlirContext, comdat::MlirLLVMComdat
    )::MlirAttribute
end

function mlirLLVMComdatAttrGetName()
    @ccall Reactant_jll.libReactantExtra.mlirLLVMComdatAttrGetName()::MlirStringRef
end

@cenum MlirLLVMLinkage::UInt32 begin
    MlirLLVMLinkageExternal = 0x0000000000000000
    MlirLLVMLinkageAvailableExternally = 0x0000000000000001
    MlirLLVMLinkageLinkonce = 0x0000000000000002
    MlirLLVMLinkageLinkonceODR = 0x0000000000000003
    MlirLLVMLinkageWeak = 0x0000000000000004
    MlirLLVMLinkageWeakODR = 0x0000000000000005
    MlirLLVMLinkageAppending = 0x0000000000000006
    MlirLLVMLinkageInternal = 0x0000000000000007
    MlirLLVMLinkagePrivate = 0x0000000000000008
    MlirLLVMLinkageExternWeak = 0x0000000000000009
    MlirLLVMLinkageCommon = 0x000000000000000a
end

"""
    mlirLLVMLinkageAttrGet(ctx, linkage)

Creates a LLVM Linkage attribute.
"""
function mlirLLVMLinkageAttrGet(ctx, linkage)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMLinkageAttrGet(
        ctx::MlirContext, linkage::MlirLLVMLinkage
    )::MlirAttribute
end

function mlirLLVMLinkageAttrGetName()
    @ccall Reactant_jll.libReactantExtra.mlirLLVMLinkageAttrGetName()::MlirStringRef
end

"""
    mlirLLVMDINullTypeAttrGet(ctx)

Creates a LLVM DINullType attribute.
"""
function mlirLLVMDINullTypeAttrGet(ctx)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDINullTypeAttrGet(
        ctx::MlirContext
    )::MlirAttribute
end

function mlirLLVMDINullTypeAttrGetName()
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDINullTypeAttrGetName()::MlirStringRef
end

"""
    mlirLLVMDIExpressionElemAttrGet(ctx, opcode, nArguments, arguments)

Creates a LLVM DIExpressionElem attribute.
"""
function mlirLLVMDIExpressionElemAttrGet(ctx, opcode, nArguments, arguments)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDIExpressionElemAttrGet(
        ctx::MlirContext, opcode::Cuint, nArguments::Cptrdiff_t, arguments::Ptr{UInt64}
    )::MlirAttribute
end

function mlirLLVMDIExpressionElemAttrGetName()
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDIExpressionElemAttrGetName()::MlirStringRef
end

"""
    mlirLLVMDIExpressionAttrGet(ctx, nOperations, operations)

Creates a LLVM DIExpression attribute.
"""
function mlirLLVMDIExpressionAttrGet(ctx, nOperations, operations)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDIExpressionAttrGet(
        ctx::MlirContext, nOperations::Cptrdiff_t, operations::Ptr{MlirAttribute}
    )::MlirAttribute
end

function mlirLLVMDIExpressionAttrGetName()
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDIExpressionAttrGetName()::MlirStringRef
end

@cenum MlirLLVMTypeEncoding::UInt32 begin
    MlirLLVMTypeEncodingAddress = 0x0000000000000001
    MlirLLVMTypeEncodingBoolean = 0x0000000000000002
    MlirLLVMTypeEncodingComplexFloat = 0x0000000000000031
    MlirLLVMTypeEncodingFloatT = 0x0000000000000004
    MlirLLVMTypeEncodingSigned = 0x0000000000000005
    MlirLLVMTypeEncodingSignedChar = 0x0000000000000006
    MlirLLVMTypeEncodingUnsigned = 0x0000000000000007
    MlirLLVMTypeEncodingUnsignedChar = 0x0000000000000008
    MlirLLVMTypeEncodingImaginaryFloat = 0x0000000000000009
    MlirLLVMTypeEncodingPackedDecimal = 0x000000000000000a
    MlirLLVMTypeEncodingNumericString = 0x000000000000000b
    MlirLLVMTypeEncodingEdited = 0x000000000000000c
    MlirLLVMTypeEncodingSignedFixed = 0x000000000000000d
    MlirLLVMTypeEncodingUnsignedFixed = 0x000000000000000e
    MlirLLVMTypeEncodingDecimalFloat = 0x000000000000000f
    MlirLLVMTypeEncodingUTF = 0x0000000000000010
    MlirLLVMTypeEncodingUCS = 0x0000000000000011
    MlirLLVMTypeEncodingASCII = 0x0000000000000012
    MlirLLVMTypeEncodingLoUser = 0x0000000000000080
    MlirLLVMTypeEncodingHiUser = 0x00000000000000ff
end

"""
    mlirLLVMDIBasicTypeAttrGet(ctx, tag, name, sizeInBits, encoding)

Creates a LLVM DIBasicType attribute.
"""
function mlirLLVMDIBasicTypeAttrGet(ctx, tag, name, sizeInBits, encoding)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDIBasicTypeAttrGet(
        ctx::MlirContext,
        tag::Cuint,
        name::MlirAttribute,
        sizeInBits::UInt64,
        encoding::MlirLLVMTypeEncoding,
    )::MlirAttribute
end

function mlirLLVMDIBasicTypeAttrGetName()
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDIBasicTypeAttrGetName()::MlirStringRef
end

"""
    mlirLLVMDICompositeTypeAttrGetRecSelf(recId)

Creates a self-referencing LLVM DICompositeType attribute.
"""
function mlirLLVMDICompositeTypeAttrGetRecSelf(recId)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDICompositeTypeAttrGetRecSelf(
        recId::MlirAttribute
    )::MlirAttribute
end

"""
    mlirLLVMDICompositeTypeAttrGet(ctx, recId, isRecSelf, tag, name, file, line, scope, baseType, flags, sizeInBits, alignInBits, nElements, elements, dataLocation, rank, allocated, associated, identifier, discriminator)

Creates a LLVM DICompositeType attribute.
"""
function mlirLLVMDICompositeTypeAttrGet(
    ctx,
    recId,
    isRecSelf,
    tag,
    name,
    file,
    line,
    scope,
    baseType,
    flags,
    sizeInBits,
    alignInBits,
    nElements,
    elements,
    dataLocation,
    rank,
    allocated,
    associated,
    identifier,
    discriminator,
)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDICompositeTypeAttrGet(
        ctx::MlirContext,
        recId::MlirAttribute,
        isRecSelf::Bool,
        tag::Cuint,
        name::MlirAttribute,
        file::MlirAttribute,
        line::UInt32,
        scope::MlirAttribute,
        baseType::MlirAttribute,
        flags::Int64,
        sizeInBits::UInt64,
        alignInBits::UInt64,
        nElements::Cptrdiff_t,
        elements::Ptr{MlirAttribute},
        dataLocation::MlirAttribute,
        rank::MlirAttribute,
        allocated::MlirAttribute,
        associated::MlirAttribute,
        identifier::MlirAttribute,
        discriminator::MlirAttribute,
    )::MlirAttribute
end

function mlirLLVMDICompositeTypeAttrGetName()
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDICompositeTypeAttrGetName()::MlirStringRef
end

"""
    mlirLLVMDIDerivedTypeAttrGet(ctx, tag, name, file, line, scope, baseType, sizeInBits, alignInBits, offsetInBits, dwarfAddressSpace, flags, extraData)

Creates a LLVM DIDerivedType attribute. Note that `dwarfAddressSpace` is an optional field, where [`MLIR_CAPI_DWARF_ADDRESS_SPACE_NULL`](@ref) indicates null and non-negative values indicate a value present.
"""
function mlirLLVMDIDerivedTypeAttrGet(
    ctx,
    tag,
    name,
    file,
    line,
    scope,
    baseType,
    sizeInBits,
    alignInBits,
    offsetInBits,
    dwarfAddressSpace,
    flags,
    extraData,
)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDIDerivedTypeAttrGet(
        ctx::MlirContext,
        tag::Cuint,
        name::MlirAttribute,
        file::MlirAttribute,
        line::UInt32,
        scope::MlirAttribute,
        baseType::MlirAttribute,
        sizeInBits::UInt64,
        alignInBits::UInt32,
        offsetInBits::UInt64,
        dwarfAddressSpace::Int64,
        flags::Int64,
        extraData::MlirAttribute,
    )::MlirAttribute
end

function mlirLLVMDIDerivedTypeAttrGetName()
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDIDerivedTypeAttrGetName()::MlirStringRef
end

function mlirLLVMDIStringTypeAttrGet(
    ctx,
    tag,
    name,
    sizeInBits,
    alignInBits,
    stringLength,
    stringLengthExp,
    stringLocationExp,
    encoding,
)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDIStringTypeAttrGet(
        ctx::MlirContext,
        tag::Cuint,
        name::MlirAttribute,
        sizeInBits::UInt64,
        alignInBits::UInt32,
        stringLength::MlirAttribute,
        stringLengthExp::MlirAttribute,
        stringLocationExp::MlirAttribute,
        encoding::MlirLLVMTypeEncoding,
    )::MlirAttribute
end

function mlirLLVMDIStringTypeAttrGetName()
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDIStringTypeAttrGetName()::MlirStringRef
end

"""
    mlirLLVMDIDerivedTypeAttrGetBaseType(diDerivedType)

Gets the base type from a LLVM DIDerivedType attribute.
"""
function mlirLLVMDIDerivedTypeAttrGetBaseType(diDerivedType)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDIDerivedTypeAttrGetBaseType(
        diDerivedType::MlirAttribute
    )::MlirAttribute
end

"""
    mlirLLVMDIFileAttrGet(ctx, name, directory)

Creates a LLVM DIFileAttr attribute.
"""
function mlirLLVMDIFileAttrGet(ctx, name, directory)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDIFileAttrGet(
        ctx::MlirContext, name::MlirAttribute, directory::MlirAttribute
    )::MlirAttribute
end

function mlirLLVMDIFileAttrGetName()
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDIFileAttrGetName()::MlirStringRef
end

@cenum MlirLLVMDIEmissionKind::UInt32 begin
    MlirLLVMDIEmissionKindNone = 0x0000000000000000
    MlirLLVMDIEmissionKindFull = 0x0000000000000001
    MlirLLVMDIEmissionKindLineTablesOnly = 0x0000000000000002
    MlirLLVMDIEmissionKindDebugDirectivesOnly = 0x0000000000000003
end

@cenum MlirLLVMDINameTableKind::UInt32 begin
    MlirLLVMDINameTableKindDefault = 0x0000000000000000
    MlirLLVMDINameTableKindGNU = 0x0000000000000001
    MlirLLVMDINameTableKindNone = 0x0000000000000002
    MlirLLVMDINameTableKindApple = 0x0000000000000003
end

"""
    mlirLLVMDICompileUnitAttrGetRecSelf(recId)

Creates a self-referencing LLVM DICompileUnitAttr attribute.
"""
function mlirLLVMDICompileUnitAttrGetRecSelf(recId)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDICompileUnitAttrGetRecSelf(
        recId::MlirAttribute
    )::MlirAttribute
end

"""
    mlirLLVMDICompileUnitAttrGet(ctx, recId, isRecSelf, id, sourceLanguage, file, producer, isOptimized, emissionKind, isDebugInfoForProfiling, nameTableKind, splitDebugFilename, nImportedEntities, importedEntities)

Creates a LLVM DICompileUnit attribute.
"""
function mlirLLVMDICompileUnitAttrGet(
    ctx,
    recId,
    isRecSelf,
    id,
    sourceLanguage,
    file,
    producer,
    isOptimized,
    emissionKind,
    isDebugInfoForProfiling,
    nameTableKind,
    splitDebugFilename,
    nImportedEntities,
    importedEntities,
)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDICompileUnitAttrGet(
        ctx::MlirContext,
        recId::MlirAttribute,
        isRecSelf::Bool,
        id::MlirAttribute,
        sourceLanguage::Cuint,
        file::MlirAttribute,
        producer::MlirAttribute,
        isOptimized::Bool,
        emissionKind::MlirLLVMDIEmissionKind,
        isDebugInfoForProfiling::Bool,
        nameTableKind::MlirLLVMDINameTableKind,
        splitDebugFilename::MlirAttribute,
        nImportedEntities::Cptrdiff_t,
        importedEntities::Ptr{MlirAttribute},
    )::MlirAttribute
end

function mlirLLVMDICompileUnitAttrGetName()
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDICompileUnitAttrGetName()::MlirStringRef
end

"""
    mlirLLVMDIFlagsAttrGet(ctx, value)

Creates a LLVM DIFlags attribute.
"""
function mlirLLVMDIFlagsAttrGet(ctx, value)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDIFlagsAttrGet(
        ctx::MlirContext, value::UInt64
    )::MlirAttribute
end

function mlirLLVMDIFlagsAttrGetName()
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDIFlagsAttrGetName()::MlirStringRef
end

"""
    mlirLLVMDILexicalBlockAttrGet(ctx, scope, file, line, column)

Creates a LLVM DILexicalBlock attribute.
"""
function mlirLLVMDILexicalBlockAttrGet(ctx, scope, file, line, column)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDILexicalBlockAttrGet(
        ctx::MlirContext,
        scope::MlirAttribute,
        file::MlirAttribute,
        line::Cuint,
        column::Cuint,
    )::MlirAttribute
end

function mlirLLVMDILexicalBlockAttrGetName()
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDILexicalBlockAttrGetName()::MlirStringRef
end

"""
    mlirLLVMDILexicalBlockFileAttrGet(ctx, scope, file, discriminator)

Creates a LLVM DILexicalBlockFile attribute.
"""
function mlirLLVMDILexicalBlockFileAttrGet(ctx, scope, file, discriminator)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDILexicalBlockFileAttrGet(
        ctx::MlirContext, scope::MlirAttribute, file::MlirAttribute, discriminator::Cuint
    )::MlirAttribute
end

function mlirLLVMDILexicalBlockFileAttrGetName()
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDILexicalBlockFileAttrGetName()::MlirStringRef
end

"""
    mlirLLVMDILocalVariableAttrGet(ctx, scope, name, diFile, line, arg, alignInBits, diType, flags)

Creates a LLVM DILocalVariableAttr attribute.
"""
function mlirLLVMDILocalVariableAttrGet(
    ctx, scope, name, diFile, line, arg, alignInBits, diType, flags
)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDILocalVariableAttrGet(
        ctx::MlirContext,
        scope::MlirAttribute,
        name::MlirAttribute,
        diFile::MlirAttribute,
        line::Cuint,
        arg::Cuint,
        alignInBits::Cuint,
        diType::MlirAttribute,
        flags::Int64,
    )::MlirAttribute
end

function mlirLLVMDILocalVariableAttrGetName()
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDILocalVariableAttrGetName()::MlirStringRef
end

"""
    mlirLLVMDISubprogramAttrGetRecSelf(recId)

Creates a self-referencing LLVM DISubprogramAttr attribute.
"""
function mlirLLVMDISubprogramAttrGetRecSelf(recId)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDISubprogramAttrGetRecSelf(
        recId::MlirAttribute
    )::MlirAttribute
end

"""
    mlirLLVMDISubprogramAttrGet(ctx, recId, isRecSelf, id, compileUnit, scope, name, linkageName, file, line, scopeLine, subprogramFlags, type, nRetainedNodes, retainedNodes, nAnnotations, annotations)

Creates a LLVM DISubprogramAttr attribute.
"""
function mlirLLVMDISubprogramAttrGet(
    ctx,
    recId,
    isRecSelf,
    id,
    compileUnit,
    scope,
    name,
    linkageName,
    file,
    line,
    scopeLine,
    subprogramFlags,
    type,
    nRetainedNodes,
    retainedNodes,
    nAnnotations,
    annotations,
)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDISubprogramAttrGet(
        ctx::MlirContext,
        recId::MlirAttribute,
        isRecSelf::Bool,
        id::MlirAttribute,
        compileUnit::MlirAttribute,
        scope::MlirAttribute,
        name::MlirAttribute,
        linkageName::MlirAttribute,
        file::MlirAttribute,
        line::Cuint,
        scopeLine::Cuint,
        subprogramFlags::UInt64,
        type::MlirAttribute,
        nRetainedNodes::Cptrdiff_t,
        retainedNodes::Ptr{MlirAttribute},
        nAnnotations::Cptrdiff_t,
        annotations::Ptr{MlirAttribute},
    )::MlirAttribute
end

function mlirLLVMDISubprogramAttrGetName()
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDISubprogramAttrGetName()::MlirStringRef
end

"""
    mlirLLVMDIAnnotationAttrGet(ctx, name, value)

Creates a LLVM DIAnnotation attribute.
"""
function mlirLLVMDIAnnotationAttrGet(ctx, name, value)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDIAnnotationAttrGet(
        ctx::MlirContext, name::MlirAttribute, value::MlirAttribute
    )::MlirAttribute
end

function mlirLLVMDIAnnotationAttrGetName()
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDIAnnotationAttrGetName()::MlirStringRef
end

"""
    mlirLLVMDISubprogramAttrGetScope(diSubprogram)

Gets the scope from this DISubprogramAttr.
"""
function mlirLLVMDISubprogramAttrGetScope(diSubprogram)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDISubprogramAttrGetScope(
        diSubprogram::MlirAttribute
    )::MlirAttribute
end

"""
    mlirLLVMDISubprogramAttrGetLine(diSubprogram)

Gets the line from this DISubprogramAttr.
"""
function mlirLLVMDISubprogramAttrGetLine(diSubprogram)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDISubprogramAttrGetLine(
        diSubprogram::MlirAttribute
    )::Cuint
end

"""
    mlirLLVMDISubprogramAttrGetScopeLine(diSubprogram)

Gets the scope line from this DISubprogram.
"""
function mlirLLVMDISubprogramAttrGetScopeLine(diSubprogram)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDISubprogramAttrGetScopeLine(
        diSubprogram::MlirAttribute
    )::Cuint
end

"""
    mlirLLVMDISubprogramAttrGetCompileUnit(diSubprogram)

Gets the compile unit from this DISubprogram.
"""
function mlirLLVMDISubprogramAttrGetCompileUnit(diSubprogram)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDISubprogramAttrGetCompileUnit(
        diSubprogram::MlirAttribute
    )::MlirAttribute
end

"""
    mlirLLVMDISubprogramAttrGetFile(diSubprogram)

Gets the file from this DISubprogramAttr.
"""
function mlirLLVMDISubprogramAttrGetFile(diSubprogram)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDISubprogramAttrGetFile(
        diSubprogram::MlirAttribute
    )::MlirAttribute
end

"""
    mlirLLVMDISubprogramAttrGetType(diSubprogram)

Gets the type from this DISubprogramAttr.
"""
function mlirLLVMDISubprogramAttrGetType(diSubprogram)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDISubprogramAttrGetType(
        diSubprogram::MlirAttribute
    )::MlirAttribute
end

"""
    mlirLLVMDISubroutineTypeAttrGet(ctx, callingConvention, nTypes, types)

Creates a LLVM DISubroutineTypeAttr attribute.
"""
function mlirLLVMDISubroutineTypeAttrGet(ctx, callingConvention, nTypes, types)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDISubroutineTypeAttrGet(
        ctx::MlirContext,
        callingConvention::Cuint,
        nTypes::Cptrdiff_t,
        types::Ptr{MlirAttribute},
    )::MlirAttribute
end

function mlirLLVMDISubroutineTypeAttrGetName()
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDISubroutineTypeAttrGetName()::MlirStringRef
end

"""
    mlirLLVMDIModuleAttrGet(ctx, file, scope, name, configMacros, includePath, apinotes, line, isDecl)

Creates a LLVM DIModuleAttr attribute.
"""
function mlirLLVMDIModuleAttrGet(
    ctx, file, scope, name, configMacros, includePath, apinotes, line, isDecl
)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDIModuleAttrGet(
        ctx::MlirContext,
        file::MlirAttribute,
        scope::MlirAttribute,
        name::MlirAttribute,
        configMacros::MlirAttribute,
        includePath::MlirAttribute,
        apinotes::MlirAttribute,
        line::Cuint,
        isDecl::Bool,
    )::MlirAttribute
end

function mlirLLVMDIModuleAttrGetName()
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDIModuleAttrGetName()::MlirStringRef
end

"""
    mlirLLVMDIImportedEntityAttrGet(ctx, tag, scope, entity, file, line, name, nElements, elements)

Creates a LLVM DIImportedEntityAttr attribute.
"""
function mlirLLVMDIImportedEntityAttrGet(
    ctx, tag, scope, entity, file, line, name, nElements, elements
)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDIImportedEntityAttrGet(
        ctx::MlirContext,
        tag::Cuint,
        scope::MlirAttribute,
        entity::MlirAttribute,
        file::MlirAttribute,
        line::Cuint,
        name::MlirAttribute,
        nElements::Cptrdiff_t,
        elements::Ptr{MlirAttribute},
    )::MlirAttribute
end

function mlirLLVMDIImportedEntityAttrGetName()
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDIImportedEntityAttrGetName()::MlirStringRef
end

"""
    mlirLLVMDIModuleAttrGetScope(diModule)

Gets the scope of this DIModuleAttr.
"""
function mlirLLVMDIModuleAttrGetScope(diModule)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMDIModuleAttrGetScope(
        diModule::MlirAttribute
    )::MlirAttribute
end

"""
    mlirLLVMMDStringAttrGet(ctx, value)

Creates an LLVM MDStringAttr.
"""
function mlirLLVMMDStringAttrGet(ctx, value)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMMDStringAttrGet(
        ctx::MlirContext, value::MlirStringRef
    )::MlirAttribute
end

"""
    mlirLLVMAttrIsAMDStringAttr(attr)

Returns `true` if the attribute is an LLVM MDStringAttr.
"""
function mlirLLVMAttrIsAMDStringAttr(attr)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMAttrIsAMDStringAttr(
        attr::MlirAttribute
    )::Bool
end

"""
    mlirLLVMMDStringAttrGetTypeID()

Returns the TypeID of MDStringAttr.
"""
function mlirLLVMMDStringAttrGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirLLVMMDStringAttrGetTypeID()::MlirTypeID
end

"""
    mlirLLVMMDStringAttrGetValue(attr)

Returns the string value of an LLVM MDStringAttr.
"""
function mlirLLVMMDStringAttrGetValue(attr)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMMDStringAttrGetValue(
        attr::MlirAttribute
    )::MlirStringRef
end

"""
    mlirLLVMMDConstantAttrGet(ctx, valueAttr)

Creates an LLVM MDConstantAttr wrapping an attribute.
"""
function mlirLLVMMDConstantAttrGet(ctx, valueAttr)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMMDConstantAttrGet(
        ctx::MlirContext, valueAttr::MlirAttribute
    )::MlirAttribute
end

"""
    mlirLLVMAttrIsAMDConstantAttr(attr)

Returns `true` if the attribute is an LLVM MDConstantAttr.
"""
function mlirLLVMAttrIsAMDConstantAttr(attr)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMAttrIsAMDConstantAttr(
        attr::MlirAttribute
    )::Bool
end

"""
    mlirLLVMMDConstantAttrGetTypeID()

Returns the TypeID of MDConstantAttr.
"""
function mlirLLVMMDConstantAttrGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirLLVMMDConstantAttrGetTypeID()::MlirTypeID
end

"""
    mlirLLVMMDConstantAttrGetValue(attr)

Returns the attribute value of an LLVM MDConstantAttr.
"""
function mlirLLVMMDConstantAttrGetValue(attr)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMMDConstantAttrGetValue(
        attr::MlirAttribute
    )::MlirAttribute
end

"""
    mlirLLVMMDGlobalValueAttrGet(ctx, name)

Creates an LLVM MDGlobalValueAttr referencing a symbol-backed global value.
"""
function mlirLLVMMDGlobalValueAttrGet(ctx, name)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMMDGlobalValueAttrGet(
        ctx::MlirContext, name::MlirAttribute
    )::MlirAttribute
end

"""
    mlirLLVMAttrIsAMDGlobalValueAttr(attr)

Returns `true` if the attribute is an LLVM MDGlobalValueAttr.
"""
function mlirLLVMAttrIsAMDGlobalValueAttr(attr)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMAttrIsAMDGlobalValueAttr(
        attr::MlirAttribute
    )::Bool
end

"""
    mlirLLVMMDGlobalValueAttrGetTypeID()

Returns the TypeID of MDGlobalValueAttr.
"""
function mlirLLVMMDGlobalValueAttrGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirLLVMMDGlobalValueAttrGetTypeID()::MlirTypeID
end

"""
    mlirLLVMMDGlobalValueAttrGetName(attr)

Returns the symbol name of an LLVM MDGlobalValueAttr.
"""
function mlirLLVMMDGlobalValueAttrGetName(attr)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMMDGlobalValueAttrGetName(
        attr::MlirAttribute
    )::MlirAttribute
end

"""
    mlirLLVMMDNodeAttrGet(ctx, nOperands, operands)

Creates an LLVM MDNodeAttr.
"""
function mlirLLVMMDNodeAttrGet(ctx, nOperands, operands)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMMDNodeAttrGet(
        ctx::MlirContext, nOperands::Cptrdiff_t, operands::Ptr{MlirAttribute}
    )::MlirAttribute
end

"""
    mlirLLVMAttrIsAMDNodeAttr(attr)

Returns `true` if the attribute is an LLVM MDNodeAttr.
"""
function mlirLLVMAttrIsAMDNodeAttr(attr)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMAttrIsAMDNodeAttr(
        attr::MlirAttribute
    )::Bool
end

"""
    mlirLLVMMDNodeAttrGetTypeID()

Returns the TypeID of MDNodeAttr.
"""
function mlirLLVMMDNodeAttrGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirLLVMMDNodeAttrGetTypeID()::MlirTypeID
end

"""
    mlirLLVMMDNodeAttrGetNumOperands(attr)

Returns the number of operands in an LLVM MDNodeAttr.
"""
function mlirLLVMMDNodeAttrGetNumOperands(attr)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMMDNodeAttrGetNumOperands(
        attr::MlirAttribute
    )::Cptrdiff_t
end

"""
    mlirLLVMMDNodeAttrGetOperand(attr, index)

Returns the operand at the given index of an LLVM MDNodeAttr.
"""
function mlirLLVMMDNodeAttrGetOperand(attr, index)
    @ccall Reactant_jll.libReactantExtra.mlirLLVMMDNodeAttrGetOperand(
        attr::MlirAttribute, index::Cptrdiff_t
    )::MlirAttribute
end

"""
    mlirLinalgFillBuiltinNamedOpRegion(mlirOp)

Apply the special region builder for the builtin named Linalg op. Assert that `mlirOp` is a builtin named Linalg op.
"""
function mlirLinalgFillBuiltinNamedOpRegion(mlirOp)
    @ccall Reactant_jll.libReactantExtra.mlirLinalgFillBuiltinNamedOpRegion(
        mlirOp::MlirOperation
    )::Cvoid
end

function mlirLinalgIsAContractionOp(op)
    @ccall Reactant_jll.libReactantExtra.mlirLinalgIsAContractionOp(op::MlirOperation)::Bool
end

struct MlirLinalgContractionDimensions
    batch::MlirAttribute
    m::MlirAttribute
    n::MlirAttribute
    k::MlirAttribute
end

function mlirLinalgInferContractionDimensions(op)
    @ccall Reactant_jll.libReactantExtra.mlirLinalgInferContractionDimensions(
        op::MlirOperation
    )::MlirLinalgContractionDimensions
end

function mlirLinalgInferContractionDimensionsFromMaps(indexingMaps, numMaps)
    @ccall Reactant_jll.libReactantExtra.mlirLinalgInferContractionDimensionsFromMaps(
        indexingMaps::Ptr{MlirAffineMap}, numMaps::Csize_t
    )::MlirLinalgContractionDimensions
end

function mlirLinalgIsAConvolutionOp(op)
    @ccall Reactant_jll.libReactantExtra.mlirLinalgIsAConvolutionOp(op::MlirOperation)::Bool
end

struct MlirLinalgConvolutionDimensions
    batch::MlirAttribute
    outputImage::MlirAttribute
    outputChannel::MlirAttribute
    filterLoop::MlirAttribute
    inputChannel::MlirAttribute
    depth::MlirAttribute
    strides::MlirAttribute
    dilations::MlirAttribute
end

function mlirLinalgInferConvolutionDimensions(op)
    @ccall Reactant_jll.libReactantExtra.mlirLinalgInferConvolutionDimensions(
        op::MlirOperation
    )::MlirLinalgConvolutionDimensions
end

function mlirLinalgInferConvolutionDimensionsFromMaps(indexingMaps, numMaps)
    @ccall Reactant_jll.libReactantExtra.mlirLinalgInferConvolutionDimensionsFromMaps(
        indexingMaps::Ptr{MlirAffineMap}, numMaps::Csize_t
    )::MlirLinalgConvolutionDimensions
end

function mlirLinalgGetIndexingMapsAttribute(op)
    @ccall Reactant_jll.libReactantExtra.mlirLinalgGetIndexingMapsAttribute(
        op::MlirOperation
    )::MlirAttribute
end

function mlirGetDialectHandle__linalg__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__linalg__()::MlirDialectHandle
end

function mlirGetDialectHandle__ml_program__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__ml_program__()::MlirDialectHandle
end

function mlirGetDialectHandle__mpi__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__mpi__()::MlirDialectHandle
end

function mlirGetDialectHandle__math__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__math__()::MlirDialectHandle
end

function mlirGetDialectHandle__memref__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__memref__()::MlirDialectHandle
end

function mlirGetDialectHandle__nvgpu__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__nvgpu__()::MlirDialectHandle
end

function mlirTypeIsANVGPUTensorMapDescriptorType(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsANVGPUTensorMapDescriptorType(
        type::MlirType
    )::Bool
end

function mlirNVGPUTensorMapDescriptorTypeGet(
    ctx, tensorMemrefType, swizzle, l2promo, oobFill, interleave
)
    @ccall Reactant_jll.libReactantExtra.mlirNVGPUTensorMapDescriptorTypeGet(
        ctx::MlirContext,
        tensorMemrefType::MlirType,
        swizzle::Cint,
        l2promo::Cint,
        oobFill::Cint,
        interleave::Cint,
    )::MlirType
end

function mlirNVGPUTensorMapDescriptorTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirNVGPUTensorMapDescriptorTypeGetName()::MlirStringRef
end

function mlirGetDialectHandle__nvvm__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__nvvm__()::MlirDialectHandle
end

function mlirGetDialectHandle__acc__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__acc__()::MlirDialectHandle
end

function mlirGetDialectHandle__omp__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__omp__()::MlirDialectHandle
end

function mlirGetDialectHandle__pdl__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__pdl__()::MlirDialectHandle
end

function mlirTypeIsAPDLType(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAPDLType(type::MlirType)::Bool
end

function mlirTypeIsAPDLAttributeType(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAPDLAttributeType(type::MlirType)::Bool
end

function mlirPDLAttributeTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirPDLAttributeTypeGetTypeID()::MlirTypeID
end

function mlirPDLAttributeTypeGet(ctx)
    @ccall Reactant_jll.libReactantExtra.mlirPDLAttributeTypeGet(ctx::MlirContext)::MlirType
end

function mlirPDLAttributeTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirPDLAttributeTypeGetName()::MlirStringRef
end

function mlirTypeIsAPDLOperationType(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAPDLOperationType(type::MlirType)::Bool
end

function mlirPDLOperationTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirPDLOperationTypeGetTypeID()::MlirTypeID
end

function mlirPDLOperationTypeGet(ctx)
    @ccall Reactant_jll.libReactantExtra.mlirPDLOperationTypeGet(ctx::MlirContext)::MlirType
end

function mlirPDLOperationTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirPDLOperationTypeGetName()::MlirStringRef
end

function mlirTypeIsAPDLRangeType(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAPDLRangeType(type::MlirType)::Bool
end

function mlirPDLRangeTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirPDLRangeTypeGetTypeID()::MlirTypeID
end

function mlirPDLRangeTypeGet(elementType)
    @ccall Reactant_jll.libReactantExtra.mlirPDLRangeTypeGet(
        elementType::MlirType
    )::MlirType
end

function mlirPDLRangeTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirPDLRangeTypeGetName()::MlirStringRef
end

function mlirPDLRangeTypeGetElementType(type)
    @ccall Reactant_jll.libReactantExtra.mlirPDLRangeTypeGetElementType(
        type::MlirType
    )::MlirType
end

function mlirTypeIsAPDLTypeType(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAPDLTypeType(type::MlirType)::Bool
end

function mlirPDLTypeTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirPDLTypeTypeGetTypeID()::MlirTypeID
end

function mlirPDLTypeTypeGet(ctx)
    @ccall Reactant_jll.libReactantExtra.mlirPDLTypeTypeGet(ctx::MlirContext)::MlirType
end

function mlirPDLTypeTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirPDLTypeTypeGetName()::MlirStringRef
end

function mlirTypeIsAPDLValueType(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAPDLValueType(type::MlirType)::Bool
end

function mlirPDLValueTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirPDLValueTypeGetTypeID()::MlirTypeID
end

function mlirPDLValueTypeGet(ctx)
    @ccall Reactant_jll.libReactantExtra.mlirPDLValueTypeGet(ctx::MlirContext)::MlirType
end

function mlirPDLValueTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirPDLValueTypeGetName()::MlirStringRef
end

function mlirGetDialectHandle__pdl_interp__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__pdl_interp__()::MlirDialectHandle
end

function mlirGetDialectHandle__ptr__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__ptr__()::MlirDialectHandle
end

function mlirGetDialectHandle__quant__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__quant__()::MlirDialectHandle
end

"""
    mlirTypeIsAQuantizedType(type)

Returns `true` if the given type is a quantization dialect type.
"""
function mlirTypeIsAQuantizedType(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAQuantizedType(type::MlirType)::Bool
end

"""
    mlirQuantizedTypeGetSignedFlag()

Returns the bit flag used to indicate signedness of a quantized type.
"""
function mlirQuantizedTypeGetSignedFlag()
    @ccall Reactant_jll.libReactantExtra.mlirQuantizedTypeGetSignedFlag()::Cuint
end

"""
    mlirQuantizedTypeGetDefaultMinimumForInteger(isSigned, integralWidth)

Returns the minimum possible value stored by a quantized type.
"""
function mlirQuantizedTypeGetDefaultMinimumForInteger(isSigned, integralWidth)
    @ccall Reactant_jll.libReactantExtra.mlirQuantizedTypeGetDefaultMinimumForInteger(
        isSigned::Bool, integralWidth::Cuint
    )::Int64
end

"""
    mlirQuantizedTypeGetDefaultMaximumForInteger(isSigned, integralWidth)

Returns the maximum possible value stored by a quantized type.
"""
function mlirQuantizedTypeGetDefaultMaximumForInteger(isSigned, integralWidth)
    @ccall Reactant_jll.libReactantExtra.mlirQuantizedTypeGetDefaultMaximumForInteger(
        isSigned::Bool, integralWidth::Cuint
    )::Int64
end

"""
    mlirQuantizedTypeGetExpressedType(type)

Gets the original type approximated by the given quantized type.
"""
function mlirQuantizedTypeGetExpressedType(type)
    @ccall Reactant_jll.libReactantExtra.mlirQuantizedTypeGetExpressedType(
        type::MlirType
    )::MlirType
end

"""
    mlirQuantizedTypeGetFlags(type)

Gets the flags associated with the given quantized type.
"""
function mlirQuantizedTypeGetFlags(type)
    @ccall Reactant_jll.libReactantExtra.mlirQuantizedTypeGetFlags(type::MlirType)::Cuint
end

"""
    mlirQuantizedTypeIsSigned(type)

Returns `true` if the given type is signed, `false` otherwise.
"""
function mlirQuantizedTypeIsSigned(type)
    @ccall Reactant_jll.libReactantExtra.mlirQuantizedTypeIsSigned(type::MlirType)::Bool
end

"""
    mlirQuantizedTypeGetStorageType(type)

Returns the underlying type used to store the values.
"""
function mlirQuantizedTypeGetStorageType(type)
    @ccall Reactant_jll.libReactantExtra.mlirQuantizedTypeGetStorageType(
        type::MlirType
    )::MlirType
end

"""
    mlirQuantizedTypeGetStorageTypeMin(type)

Returns the minimum value that the storage type of the given quantized type can take.
"""
function mlirQuantizedTypeGetStorageTypeMin(type)
    @ccall Reactant_jll.libReactantExtra.mlirQuantizedTypeGetStorageTypeMin(
        type::MlirType
    )::Int64
end

"""
    mlirQuantizedTypeGetStorageTypeMax(type)

Returns the maximum value that the storage type of the given quantized type can take.
"""
function mlirQuantizedTypeGetStorageTypeMax(type)
    @ccall Reactant_jll.libReactantExtra.mlirQuantizedTypeGetStorageTypeMax(
        type::MlirType
    )::Int64
end

"""
    mlirQuantizedTypeGetStorageTypeIntegralWidth(type)

Returns the integral bitwidth that the storage type of the given quantized type can represent exactly.
"""
function mlirQuantizedTypeGetStorageTypeIntegralWidth(type)
    @ccall Reactant_jll.libReactantExtra.mlirQuantizedTypeGetStorageTypeIntegralWidth(
        type::MlirType
    )::Cuint
end

"""
    mlirQuantizedTypeIsCompatibleExpressedType(type, candidate)

Returns `true` if the `candidate` type is compatible with the given quantized `type`.
"""
function mlirQuantizedTypeIsCompatibleExpressedType(type, candidate)
    @ccall Reactant_jll.libReactantExtra.mlirQuantizedTypeIsCompatibleExpressedType(
        type::MlirType, candidate::MlirType
    )::Bool
end

"""
    mlirQuantizedTypeGetQuantizedElementType(type)

Returns the element type of the given quantized type as another quantized type.
"""
function mlirQuantizedTypeGetQuantizedElementType(type)
    @ccall Reactant_jll.libReactantExtra.mlirQuantizedTypeGetQuantizedElementType(
        type::MlirType
    )::MlirType
end

"""
    mlirQuantizedTypeCastFromStorageType(type, candidate)

Casts from a type based on the storage type of the given type to a corresponding type based on the given type. Returns a null type if the cast is not valid.
"""
function mlirQuantizedTypeCastFromStorageType(type, candidate)
    @ccall Reactant_jll.libReactantExtra.mlirQuantizedTypeCastFromStorageType(
        type::MlirType, candidate::MlirType
    )::MlirType
end

"""
    mlirQuantizedTypeCastToStorageType(type)

Casts from a type based on a quantized type to a corresponding typed based on the storage type. Returns a null type if the cast is not valid.
"""
function mlirQuantizedTypeCastToStorageType(type)
    @ccall Reactant_jll.libReactantExtra.mlirQuantizedTypeCastToStorageType(
        type::MlirType
    )::MlirType
end

"""
    mlirQuantizedTypeCastFromExpressedType(type, candidate)

Casts from a type based on the expressed type of the given type to a corresponding type based on the given type. Returns a null type if the cast is not valid.
"""
function mlirQuantizedTypeCastFromExpressedType(type, candidate)
    @ccall Reactant_jll.libReactantExtra.mlirQuantizedTypeCastFromExpressedType(
        type::MlirType, candidate::MlirType
    )::MlirType
end

"""
    mlirQuantizedTypeCastToExpressedType(type)

Casts from a type based on a quantized type to a corresponding typed based on the expressed type. Returns a null type if the cast is not valid.
"""
function mlirQuantizedTypeCastToExpressedType(type)
    @ccall Reactant_jll.libReactantExtra.mlirQuantizedTypeCastToExpressedType(
        type::MlirType
    )::MlirType
end

"""
    mlirQuantizedTypeCastExpressedToStorageType(type, candidate)

Casts from a type based on the expressed type of the given quantized type to equivalent type based on storage type of the same quantized type.
"""
function mlirQuantizedTypeCastExpressedToStorageType(type, candidate)
    @ccall Reactant_jll.libReactantExtra.mlirQuantizedTypeCastExpressedToStorageType(
        type::MlirType, candidate::MlirType
    )::MlirType
end

"""
    mlirTypeIsAAnyQuantizedType(type)

Returns `true` if the given type is an AnyQuantizedType.
"""
function mlirTypeIsAAnyQuantizedType(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAAnyQuantizedType(type::MlirType)::Bool
end

function mlirAnyQuantizedTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirAnyQuantizedTypeGetTypeID()::MlirTypeID
end

"""
    mlirAnyQuantizedTypeGet(flags, storageType, expressedType, storageTypeMin, storageTypeMax)

Creates an instance of AnyQuantizedType with the given parameters in the same context as `storageType` and returns it. The instance is owned by the context.
"""
function mlirAnyQuantizedTypeGet(
    flags, storageType, expressedType, storageTypeMin, storageTypeMax
)
    @ccall Reactant_jll.libReactantExtra.mlirAnyQuantizedTypeGet(
        flags::Cuint,
        storageType::MlirType,
        expressedType::MlirType,
        storageTypeMin::Int64,
        storageTypeMax::Int64,
    )::MlirType
end

function mlirAnyQuantizedTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirAnyQuantizedTypeGetName()::MlirStringRef
end

"""
    mlirTypeIsAUniformQuantizedType(type)

Returns `true` if the given type is a UniformQuantizedType.
"""
function mlirTypeIsAUniformQuantizedType(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAUniformQuantizedType(
        type::MlirType
    )::Bool
end

function mlirUniformQuantizedTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirUniformQuantizedTypeGetTypeID()::MlirTypeID
end

"""
    mlirUniformQuantizedTypeGet(flags, storageType, expressedType, scale, zeroPoint, storageTypeMin, storageTypeMax)

Creates an instance of UniformQuantizedType with the given parameters in the same context as `storageType` and returns it. The instance is owned by the context.
"""
function mlirUniformQuantizedTypeGet(
    flags, storageType, expressedType, scale, zeroPoint, storageTypeMin, storageTypeMax
)
    @ccall Reactant_jll.libReactantExtra.mlirUniformQuantizedTypeGet(
        flags::Cuint,
        storageType::MlirType,
        expressedType::MlirType,
        scale::Cdouble,
        zeroPoint::Int64,
        storageTypeMin::Int64,
        storageTypeMax::Int64,
    )::MlirType
end

function mlirUniformQuantizedTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirUniformQuantizedTypeGetName()::MlirStringRef
end

"""
    mlirUniformQuantizedTypeGetScale(type)

Returns the scale of the given uniform quantized type.
"""
function mlirUniformQuantizedTypeGetScale(type)
    @ccall Reactant_jll.libReactantExtra.mlirUniformQuantizedTypeGetScale(
        type::MlirType
    )::Cdouble
end

"""
    mlirUniformQuantizedTypeGetZeroPoint(type)

Returns the zero point of the given uniform quantized type.
"""
function mlirUniformQuantizedTypeGetZeroPoint(type)
    @ccall Reactant_jll.libReactantExtra.mlirUniformQuantizedTypeGetZeroPoint(
        type::MlirType
    )::Int64
end

"""
    mlirUniformQuantizedTypeIsFixedPoint(type)

Returns `true` if the given uniform quantized type is fixed-point.
"""
function mlirUniformQuantizedTypeIsFixedPoint(type)
    @ccall Reactant_jll.libReactantExtra.mlirUniformQuantizedTypeIsFixedPoint(
        type::MlirType
    )::Bool
end

"""
    mlirTypeIsAUniformQuantizedPerAxisType(type)

Returns `true` if the given type is a UniformQuantizedPerAxisType.
"""
function mlirTypeIsAUniformQuantizedPerAxisType(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAUniformQuantizedPerAxisType(
        type::MlirType
    )::Bool
end

function mlirUniformQuantizedPerAxisTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirUniformQuantizedPerAxisTypeGetTypeID()::MlirTypeID
end

"""
    mlirUniformQuantizedPerAxisTypeGet(flags, storageType, expressedType, nDims, scales, zeroPoints, quantizedDimension, storageTypeMin, storageTypeMax)

Creates an instance of UniformQuantizedPerAxisType with the given parameters in the same context as `storageType` and returns it. `scales` and `zeroPoints` point to `nDims` number of elements. The instance is owned by the context.
"""
function mlirUniformQuantizedPerAxisTypeGet(
    flags,
    storageType,
    expressedType,
    nDims,
    scales,
    zeroPoints,
    quantizedDimension,
    storageTypeMin,
    storageTypeMax,
)
    @ccall Reactant_jll.libReactantExtra.mlirUniformQuantizedPerAxisTypeGet(
        flags::Cuint,
        storageType::MlirType,
        expressedType::MlirType,
        nDims::Cptrdiff_t,
        scales::Ptr{Cdouble},
        zeroPoints::Ptr{Int64},
        quantizedDimension::Int32,
        storageTypeMin::Int64,
        storageTypeMax::Int64,
    )::MlirType
end

function mlirUniformQuantizedPerAxisTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirUniformQuantizedPerAxisTypeGetName()::MlirStringRef
end

"""
    mlirUniformQuantizedPerAxisTypeGetNumDims(type)

Returns the number of axes in the given quantized per-axis type.
"""
function mlirUniformQuantizedPerAxisTypeGetNumDims(type)
    @ccall Reactant_jll.libReactantExtra.mlirUniformQuantizedPerAxisTypeGetNumDims(
        type::MlirType
    )::Cptrdiff_t
end

"""
    mlirUniformQuantizedPerAxisTypeGetScale(type, pos)

Returns `pos`-th scale of the given quantized per-axis type.
"""
function mlirUniformQuantizedPerAxisTypeGetScale(type, pos)
    @ccall Reactant_jll.libReactantExtra.mlirUniformQuantizedPerAxisTypeGetScale(
        type::MlirType, pos::Cptrdiff_t
    )::Cdouble
end

"""
    mlirUniformQuantizedPerAxisTypeGetZeroPoint(type, pos)

Returns `pos`-th zero point of the given quantized per-axis type.
"""
function mlirUniformQuantizedPerAxisTypeGetZeroPoint(type, pos)
    @ccall Reactant_jll.libReactantExtra.mlirUniformQuantizedPerAxisTypeGetZeroPoint(
        type::MlirType, pos::Cptrdiff_t
    )::Int64
end

"""
    mlirUniformQuantizedPerAxisTypeGetQuantizedDimension(type)

Returns the index of the quantized dimension in the given quantized per-axis type.
"""
function mlirUniformQuantizedPerAxisTypeGetQuantizedDimension(type)
    @ccall Reactant_jll.libReactantExtra.mlirUniformQuantizedPerAxisTypeGetQuantizedDimension(
        type::MlirType
    )::Int32
end

"""
    mlirUniformQuantizedPerAxisTypeIsFixedPoint(type)

Returns `true` if the given uniform quantized per-axis type is fixed-point.
"""
function mlirUniformQuantizedPerAxisTypeIsFixedPoint(type)
    @ccall Reactant_jll.libReactantExtra.mlirUniformQuantizedPerAxisTypeIsFixedPoint(
        type::MlirType
    )::Bool
end

"""
    mlirTypeIsAUniformQuantizedSubChannelType(type)

Returns `true` if the given type is a UniformQuantizedSubChannel.
"""
function mlirTypeIsAUniformQuantizedSubChannelType(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsAUniformQuantizedSubChannelType(
        type::MlirType
    )::Bool
end

function mlirUniformQuantizedSubChannelTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirUniformQuantizedSubChannelTypeGetTypeID()::MlirTypeID
end

"""
    mlirUniformQuantizedSubChannelTypeGet(flags, storageType, expressedType, scalesAttr, zeroPointsAttr, blockSizeInfoLength, quantizedDimensions, blockSizes, storageTypeMin, storageTypeMax)

Creates a UniformQuantizedSubChannelType with the given parameters.

The type is owned by the context. `scalesAttr` and `zeroPointsAttr` must be DenseElementsAttrs. `quantizedDimensions` and `blockSizes` point to `blockSizeInfoLength` number of elements, describing respectively the quantization axis and corresponding block size.
"""
function mlirUniformQuantizedSubChannelTypeGet(
    flags,
    storageType,
    expressedType,
    scalesAttr,
    zeroPointsAttr,
    blockSizeInfoLength,
    quantizedDimensions,
    blockSizes,
    storageTypeMin,
    storageTypeMax,
)
    @ccall Reactant_jll.libReactantExtra.mlirUniformQuantizedSubChannelTypeGet(
        flags::Cuint,
        storageType::MlirType,
        expressedType::MlirType,
        scalesAttr::MlirAttribute,
        zeroPointsAttr::MlirAttribute,
        blockSizeInfoLength::Cptrdiff_t,
        quantizedDimensions::Ptr{Int32},
        blockSizes::Ptr{Int64},
        storageTypeMin::Int64,
        storageTypeMax::Int64,
    )::MlirType
end

function mlirUniformQuantizedSubChannelTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirUniformQuantizedSubChannelTypeGetName()::MlirStringRef
end

"""
    mlirUniformQuantizedSubChannelTypeGetNumBlockSizes(type)

Returns the number of block sizes provided in type.
"""
function mlirUniformQuantizedSubChannelTypeGetNumBlockSizes(type)
    @ccall Reactant_jll.libReactantExtra.mlirUniformQuantizedSubChannelTypeGetNumBlockSizes(
        type::MlirType
    )::Cptrdiff_t
end

"""
    mlirUniformQuantizedSubChannelTypeGetQuantizedDimension(type, pos)

Returns the quantized dimension at the given position.
"""
function mlirUniformQuantizedSubChannelTypeGetQuantizedDimension(type, pos)
    @ccall Reactant_jll.libReactantExtra.mlirUniformQuantizedSubChannelTypeGetQuantizedDimension(
        type::MlirType, pos::Cptrdiff_t
    )::Int32
end

"""
    mlirUniformQuantizedSubChannelTypeGetBlockSize(type, pos)

Returns the block size at the given position.
"""
function mlirUniformQuantizedSubChannelTypeGetBlockSize(type, pos)
    @ccall Reactant_jll.libReactantExtra.mlirUniformQuantizedSubChannelTypeGetBlockSize(
        type::MlirType, pos::Cptrdiff_t
    )::Int64
end

"""
    mlirUniformQuantizedSubChannelTypeGetScales(type)

Returns the scales of the quantized type.
"""
function mlirUniformQuantizedSubChannelTypeGetScales(type)
    @ccall Reactant_jll.libReactantExtra.mlirUniformQuantizedSubChannelTypeGetScales(
        type::MlirType
    )::MlirAttribute
end

"""
    mlirUniformQuantizedSubChannelTypeGetZeroPoints(type)

Returns the zero-points of the quantized type.
"""
function mlirUniformQuantizedSubChannelTypeGetZeroPoints(type)
    @ccall Reactant_jll.libReactantExtra.mlirUniformQuantizedSubChannelTypeGetZeroPoints(
        type::MlirType
    )::MlirAttribute
end

"""
    mlirTypeIsACalibratedQuantizedType(type)

Returns `true` if the given type is a CalibratedQuantizedType.
"""
function mlirTypeIsACalibratedQuantizedType(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsACalibratedQuantizedType(
        type::MlirType
    )::Bool
end

function mlirCalibratedQuantizedTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirCalibratedQuantizedTypeGetTypeID()::MlirTypeID
end

"""
    mlirCalibratedQuantizedTypeGet(expressedType, min, max)

Creates an instance of CalibratedQuantizedType with the given parameters in the same context as `expressedType` and returns it. The instance is owned by the context.
"""
function mlirCalibratedQuantizedTypeGet(expressedType, min, max)
    @ccall Reactant_jll.libReactantExtra.mlirCalibratedQuantizedTypeGet(
        expressedType::MlirType, min::Cdouble, max::Cdouble
    )::MlirType
end

function mlirCalibratedQuantizedTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirCalibratedQuantizedTypeGetName()::MlirStringRef
end

"""
    mlirCalibratedQuantizedTypeGetMin(type)

Returns the min value of the given calibrated quantized type.
"""
function mlirCalibratedQuantizedTypeGetMin(type)
    @ccall Reactant_jll.libReactantExtra.mlirCalibratedQuantizedTypeGetMin(
        type::MlirType
    )::Cdouble
end

"""
    mlirCalibratedQuantizedTypeGetMax(type)

Returns the max value of the given calibrated quantized type.
"""
function mlirCalibratedQuantizedTypeGetMax(type)
    @ccall Reactant_jll.libReactantExtra.mlirCalibratedQuantizedTypeGetMax(
        type::MlirType
    )::Cdouble
end

function mlirGetDialectHandle__rocdl__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__rocdl__()::MlirDialectHandle
end

function mlirGetDialectHandle__scf__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__scf__()::MlirDialectHandle
end

function mlirGetDialectHandle__smt__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__smt__()::MlirDialectHandle
end

"""
    mlirSMTTypeIsAnyNonFuncSMTValueType(type)

Checks if the given type is any non-func SMT value type.
"""
function mlirSMTTypeIsAnyNonFuncSMTValueType(type)
    @ccall Reactant_jll.libReactantExtra.mlirSMTTypeIsAnyNonFuncSMTValueType(
        type::MlirType
    )::Bool
end

"""
    mlirSMTTypeIsAnySMTValueType(type)

Checks if the given type is any SMT value type.
"""
function mlirSMTTypeIsAnySMTValueType(type)
    @ccall Reactant_jll.libReactantExtra.mlirSMTTypeIsAnySMTValueType(type::MlirType)::Bool
end

"""
    mlirSMTTypeIsAArray(type)

Checks if the given type is a smt::ArrayType.
"""
function mlirSMTTypeIsAArray(type)
    @ccall Reactant_jll.libReactantExtra.mlirSMTTypeIsAArray(type::MlirType)::Bool
end

"""
    mlirSMTTypeGetArray(ctx, domainType, rangeType)

Creates an array type with the given domain and range types.
"""
function mlirSMTTypeGetArray(ctx, domainType, rangeType)
    @ccall Reactant_jll.libReactantExtra.mlirSMTTypeGetArray(
        ctx::MlirContext, domainType::MlirType, rangeType::MlirType
    )::MlirType
end

"""
    mlirSMTTypeIsABitVector(type)

Checks if the given type is a smt::BitVectorType.
"""
function mlirSMTTypeIsABitVector(type)
    @ccall Reactant_jll.libReactantExtra.mlirSMTTypeIsABitVector(type::MlirType)::Bool
end

"""
    mlirSMTTypeGetBitVector(ctx, width)

Creates a smt::BitVectorType with the given width.
"""
function mlirSMTTypeGetBitVector(ctx, width)
    @ccall Reactant_jll.libReactantExtra.mlirSMTTypeGetBitVector(
        ctx::MlirContext, width::Int32
    )::MlirType
end

function mlirSMTBitVectorTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirSMTBitVectorTypeGetName()::MlirStringRef
end

function mlirSMTBitVectorTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirSMTBitVectorTypeGetTypeID()::MlirTypeID
end

"""
    mlirSMTTypeIsABool(type)

Checks if the given type is a smt::BoolType.
"""
function mlirSMTTypeIsABool(type)
    @ccall Reactant_jll.libReactantExtra.mlirSMTTypeIsABool(type::MlirType)::Bool
end

"""
    mlirSMTTypeGetBool(ctx)

Creates a smt::BoolType.
"""
function mlirSMTTypeGetBool(ctx)
    @ccall Reactant_jll.libReactantExtra.mlirSMTTypeGetBool(ctx::MlirContext)::MlirType
end

function mlirSMTBoolTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirSMTBoolTypeGetName()::MlirStringRef
end

function mlirSMTBoolTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirSMTBoolTypeGetTypeID()::MlirTypeID
end

"""
    mlirSMTTypeIsAInt(type)

Checks if the given type is a smt::IntType.
"""
function mlirSMTTypeIsAInt(type)
    @ccall Reactant_jll.libReactantExtra.mlirSMTTypeIsAInt(type::MlirType)::Bool
end

"""
    mlirSMTTypeGetInt(ctx)

Creates a smt::IntType.
"""
function mlirSMTTypeGetInt(ctx)
    @ccall Reactant_jll.libReactantExtra.mlirSMTTypeGetInt(ctx::MlirContext)::MlirType
end

function mlirSMTIntTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirSMTIntTypeGetName()::MlirStringRef
end

function mlirSMTIntTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirSMTIntTypeGetTypeID()::MlirTypeID
end

"""
    mlirSMTTypeIsASMTFunc(type)

Checks if the given type is a smt::FuncType.
"""
function mlirSMTTypeIsASMTFunc(type)
    @ccall Reactant_jll.libReactantExtra.mlirSMTTypeIsASMTFunc(type::MlirType)::Bool
end

"""
    mlirSMTTypeGetSMTFunc(ctx, numberOfDomainTypes, domainTypes, rangeType)

Creates a smt::FuncType with the given domain and range types.
"""
function mlirSMTTypeGetSMTFunc(ctx, numberOfDomainTypes, domainTypes, rangeType)
    @ccall Reactant_jll.libReactantExtra.mlirSMTTypeGetSMTFunc(
        ctx::MlirContext,
        numberOfDomainTypes::Csize_t,
        domainTypes::Ptr{MlirType},
        rangeType::MlirType,
    )::MlirType
end

"""
    mlirSMTTypeIsASort(type)

Checks if the given type is a smt::SortType.
"""
function mlirSMTTypeIsASort(type)
    @ccall Reactant_jll.libReactantExtra.mlirSMTTypeIsASort(type::MlirType)::Bool
end

"""
    mlirSMTTypeGetSort(ctx, identifier, numberOfSortParams, sortParams)

Creates a smt::SortType with the given identifier and sort parameters.
"""
function mlirSMTTypeGetSort(ctx, identifier, numberOfSortParams, sortParams)
    @ccall Reactant_jll.libReactantExtra.mlirSMTTypeGetSort(
        ctx::MlirContext,
        identifier::MlirIdentifier,
        numberOfSortParams::Csize_t,
        sortParams::Ptr{MlirType},
    )::MlirType
end

"""
    mlirSMTAttrCheckBVCmpPredicate(ctx, str)

Checks if the given string is a valid smt::BVCmpPredicate.
"""
function mlirSMTAttrCheckBVCmpPredicate(ctx, str)
    @ccall Reactant_jll.libReactantExtra.mlirSMTAttrCheckBVCmpPredicate(
        ctx::MlirContext, str::MlirStringRef
    )::Bool
end

"""
    mlirSMTAttrCheckIntPredicate(ctx, str)

Checks if the given string is a valid smt::IntPredicate.
"""
function mlirSMTAttrCheckIntPredicate(ctx, str)
    @ccall Reactant_jll.libReactantExtra.mlirSMTAttrCheckIntPredicate(
        ctx::MlirContext, str::MlirStringRef
    )::Bool
end

"""
    mlirSMTAttrIsASMTAttribute(attr)

Checks if the given attribute is a smt::SMTAttribute.
"""
function mlirSMTAttrIsASMTAttribute(attr)
    @ccall Reactant_jll.libReactantExtra.mlirSMTAttrIsASMTAttribute(
        attr::MlirAttribute
    )::Bool
end

"""
    mlirSMTAttrGetBitVector(ctx, value, width)

Creates a smt::BitVectorAttr with the given value and width.
"""
function mlirSMTAttrGetBitVector(ctx, value, width)
    @ccall Reactant_jll.libReactantExtra.mlirSMTAttrGetBitVector(
        ctx::MlirContext, value::UInt64, width::Cuint
    )::MlirAttribute
end

"""
    mlirSMTAttrGetBVCmpPredicate(ctx, str)

Creates a smt::BVCmpPredicateAttr with the given string.
"""
function mlirSMTAttrGetBVCmpPredicate(ctx, str)
    @ccall Reactant_jll.libReactantExtra.mlirSMTAttrGetBVCmpPredicate(
        ctx::MlirContext, str::MlirStringRef
    )::MlirAttribute
end

"""
    mlirSMTAttrGetIntPredicate(ctx, str)

Creates a smt::IntPredicateAttr with the given string.
"""
function mlirSMTAttrGetIntPredicate(ctx, str)
    @ccall Reactant_jll.libReactantExtra.mlirSMTAttrGetIntPredicate(
        ctx::MlirContext, str::MlirStringRef
    )::MlirAttribute
end

function mlirGetDialectHandle__spirv__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__spirv__()::MlirDialectHandle
end

function mlirGetDialectHandle__shape__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__shape__()::MlirDialectHandle
end

function mlirGetDialectHandle__shard__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__shard__()::MlirDialectHandle
end

function mlirGetDialectHandle__sparse_tensor__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__sparse_tensor__()::MlirDialectHandle
end

"""
Dimension level types (and properties) that define sparse tensors. See the documentation in SparseTensorAttrDefs.td for their meaning.

These correspond to SparseTensorEncodingAttr::LevelType in the C++ API. If updating, keep them in sync and update the static\\_assert in the impl file.
"""
const MlirSparseTensorLevelType = UInt64

@cenum MlirSparseTensorLevelFormat::UInt32 begin
    MLIR_SPARSE_TENSOR_LEVEL_DENSE = 0x0000000000010000
    MLIR_SPARSE_TENSOR_LEVEL_BATCH = 0x0000000000020000
    MLIR_SPARSE_TENSOR_LEVEL_COMPRESSED = 0x0000000000040000
    MLIR_SPARSE_TENSOR_LEVEL_SINGLETON = 0x0000000000080000
    MLIR_SPARSE_TENSOR_LEVEL_LOOSE_COMPRESSED = 0x0000000000100000
    MLIR_SPARSE_TENSOR_LEVEL_N_OUT_OF_M = 0x0000000000200000
end

@cenum MlirSparseTensorLevelPropertyNondefault::UInt32 begin
    MLIR_SPARSE_PROPERTY_NON_UNIQUE = 0x0000000000000001
    MLIR_SPARSE_PROPERTY_NON_ORDERED = 0x0000000000000002
    MLIR_SPARSE_PROPERTY_SOA = 0x0000000000000004
end

"""
    mlirAttributeIsASparseTensorEncodingAttr(attr)

Checks whether the given attribute is a `sparse\\_tensor.encoding` attribute.
"""
function mlirAttributeIsASparseTensorEncodingAttr(attr)
    @ccall Reactant_jll.libReactantExtra.mlirAttributeIsASparseTensorEncodingAttr(
        attr::MlirAttribute
    )::Bool
end

"""
    mlirSparseTensorEncodingAttrGet(ctx, lvlRank, lvlTypes, dimToLvl, lvlTodim, posWidth, crdWidth, explicitVal, implicitVal)

Creates a `sparse\\_tensor.encoding` attribute with the given parameters.
"""
function mlirSparseTensorEncodingAttrGet(
    ctx, lvlRank, lvlTypes, dimToLvl, lvlTodim, posWidth, crdWidth, explicitVal, implicitVal
)
    @ccall Reactant_jll.libReactantExtra.mlirSparseTensorEncodingAttrGet(
        ctx::MlirContext,
        lvlRank::Cptrdiff_t,
        lvlTypes::Ptr{MlirSparseTensorLevelType},
        dimToLvl::MlirAffineMap,
        lvlTodim::MlirAffineMap,
        posWidth::Cint,
        crdWidth::Cint,
        explicitVal::MlirAttribute,
        implicitVal::MlirAttribute,
    )::MlirAttribute
end

function mlirSparseTensorEncodingAttrGetName()
    @ccall Reactant_jll.libReactantExtra.mlirSparseTensorEncodingAttrGetName()::MlirStringRef
end

"""
    mlirSparseTensorEncodingGetLvlRank(attr)

Returns the level-rank of the `sparse\\_tensor.encoding` attribute.
"""
function mlirSparseTensorEncodingGetLvlRank(attr)
    @ccall Reactant_jll.libReactantExtra.mlirSparseTensorEncodingGetLvlRank(
        attr::MlirAttribute
    )::Cptrdiff_t
end

"""
    mlirSparseTensorEncodingAttrGetLvlType(attr, lvl)

Returns a specified level-type of the `sparse\\_tensor.encoding` attribute.
"""
function mlirSparseTensorEncodingAttrGetLvlType(attr, lvl)
    @ccall Reactant_jll.libReactantExtra.mlirSparseTensorEncodingAttrGetLvlType(
        attr::MlirAttribute, lvl::Cptrdiff_t
    )::MlirSparseTensorLevelType
end

"""
    mlirSparseTensorEncodingAttrGetLvlFmt(attr, lvl)

Returns a specified level-format of the `sparse\\_tensor.encoding` attribute.
"""
function mlirSparseTensorEncodingAttrGetLvlFmt(attr, lvl)
    @ccall Reactant_jll.libReactantExtra.mlirSparseTensorEncodingAttrGetLvlFmt(
        attr::MlirAttribute, lvl::Cptrdiff_t
    )::MlirSparseTensorLevelFormat
end

"""
    mlirSparseTensorEncodingAttrGetDimToLvl(attr)

Returns the dimension-to-level mapping of the `sparse\\_tensor.encoding` attribute.
"""
function mlirSparseTensorEncodingAttrGetDimToLvl(attr)
    @ccall Reactant_jll.libReactantExtra.mlirSparseTensorEncodingAttrGetDimToLvl(
        attr::MlirAttribute
    )::MlirAffineMap
end

"""
    mlirSparseTensorEncodingAttrGetLvlToDim(attr)

Returns the level-to-dimension mapping of the `sparse\\_tensor.encoding` attribute.
"""
function mlirSparseTensorEncodingAttrGetLvlToDim(attr)
    @ccall Reactant_jll.libReactantExtra.mlirSparseTensorEncodingAttrGetLvlToDim(
        attr::MlirAttribute
    )::MlirAffineMap
end

"""
    mlirSparseTensorEncodingAttrGetPosWidth(attr)

Returns the position bitwidth of the `sparse\\_tensor.encoding` attribute.
"""
function mlirSparseTensorEncodingAttrGetPosWidth(attr)
    @ccall Reactant_jll.libReactantExtra.mlirSparseTensorEncodingAttrGetPosWidth(
        attr::MlirAttribute
    )::Cint
end

"""
    mlirSparseTensorEncodingAttrGetCrdWidth(attr)

Returns the coordinate bitwidth of the `sparse\\_tensor.encoding` attribute.
"""
function mlirSparseTensorEncodingAttrGetCrdWidth(attr)
    @ccall Reactant_jll.libReactantExtra.mlirSparseTensorEncodingAttrGetCrdWidth(
        attr::MlirAttribute
    )::Cint
end

"""
    mlirSparseTensorEncodingAttrGetExplicitVal(attr)

Returns the explicit value of the `sparse\\_tensor.encoding` attribute.
"""
function mlirSparseTensorEncodingAttrGetExplicitVal(attr)
    @ccall Reactant_jll.libReactantExtra.mlirSparseTensorEncodingAttrGetExplicitVal(
        attr::MlirAttribute
    )::MlirAttribute
end

"""
    mlirSparseTensorEncodingAttrGetImplicitVal(attr)

Returns the implicit value of the `sparse\\_tensor.encoding` attribute.
"""
function mlirSparseTensorEncodingAttrGetImplicitVal(attr)
    @ccall Reactant_jll.libReactantExtra.mlirSparseTensorEncodingAttrGetImplicitVal(
        attr::MlirAttribute
    )::MlirAttribute
end

function mlirSparseTensorEncodingAttrGetStructuredN(lvlType)
    @ccall Reactant_jll.libReactantExtra.mlirSparseTensorEncodingAttrGetStructuredN(
        lvlType::MlirSparseTensorLevelType
    )::Cuint
end

function mlirSparseTensorEncodingAttrGetStructuredM(lvlType)
    @ccall Reactant_jll.libReactantExtra.mlirSparseTensorEncodingAttrGetStructuredM(
        lvlType::MlirSparseTensorLevelType
    )::Cuint
end

function mlirSparseTensorEncodingAttrBuildLvlType(lvlFmt, properties, propSize, n, m)
    @ccall Reactant_jll.libReactantExtra.mlirSparseTensorEncodingAttrBuildLvlType(
        lvlFmt::MlirSparseTensorLevelFormat,
        properties::Ptr{MlirSparseTensorLevelPropertyNondefault},
        propSize::Cuint,
        n::Cuint,
        m::Cuint,
    )::MlirSparseTensorLevelType
end

function mlirGetDialectHandle__tensor__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__tensor__()::MlirDialectHandle
end

function mlirGetDialectHandle__tosa__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__tosa__()::MlirDialectHandle
end

struct MlirMemoryEffect
    ptr::Ptr{Cvoid}
end

struct MlirMemoryEffectInstance
    ptr::Ptr{Cvoid}
end

struct MlirMemoryEffectInstancesList
    ptr::Ptr{Cvoid}
end

struct MlirSideEffectResource
    ptr::Ptr{Cvoid}
end

"""
    mlirOperationImplementsInterface(operation, interfaceTypeID)

Returns `true` if the given operation implements an interface identified by its TypeID.
"""
function mlirOperationImplementsInterface(operation, interfaceTypeID)
    @ccall Reactant_jll.libReactantExtra.mlirOperationImplementsInterface(
        operation::MlirOperation, interfaceTypeID::MlirTypeID
    )::Bool
end

"""
    mlirOperationImplementsInterfaceStatic(operationName, context, interfaceTypeID)

Returns `true` if the operation identified by its canonical string name implements the interface identified by its TypeID in the given context. Note that interfaces may be attached to operations in some contexts and not others.
"""
function mlirOperationImplementsInterfaceStatic(operationName, context, interfaceTypeID)
    @ccall Reactant_jll.libReactantExtra.mlirOperationImplementsInterfaceStatic(
        operationName::MlirStringRef, context::MlirContext, interfaceTypeID::MlirTypeID
    )::Bool
end

"""
    mlirInferTypeOpInterfaceTypeID()

Returns the interface TypeID of the InferTypeOpInterface.
"""
function mlirInferTypeOpInterfaceTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirInferTypeOpInterfaceTypeID()::MlirTypeID
end

# typedef void ( * MlirTypesCallback ) ( intptr_t , MlirType * , void * )
"""
These callbacks are used to return multiple types from functions while transferring ownership to the caller. The first argument is the number of consecutive elements pointed to by the second argument. The third argument is an opaque pointer forwarded to the callback by the caller.
"""
const MlirTypesCallback = Ptr{Cvoid}

"""
    mlirInferTypeOpInterfaceInferReturnTypes(opName, context, location, nOperands, operands, attributes, properties, nRegions, regions, callback, userData)

Infers the return types of the operation identified by its canonical given the arguments that will be supplied to its generic builder. Calls `callback` with the types of inferred arguments, potentially several times, on success. Returns failure otherwise.
"""
function mlirInferTypeOpInterfaceInferReturnTypes(
    opName,
    context,
    location,
    nOperands,
    operands,
    attributes,
    properties,
    nRegions,
    regions,
    callback,
    userData,
)
    @ccall Reactant_jll.libReactantExtra.mlirInferTypeOpInterfaceInferReturnTypes(
        opName::MlirStringRef,
        context::MlirContext,
        location::MlirLocation,
        nOperands::Cptrdiff_t,
        operands::Ptr{MlirValue},
        attributes::MlirAttribute,
        properties::Ptr{Cvoid},
        nRegions::Cptrdiff_t,
        regions::Ptr{MlirRegion},
        callback::MlirTypesCallback,
        userData::Ptr{Cvoid},
    )::MlirLogicalResult
end

"""
    mlirInferShapedTypeOpInterfaceTypeID()

Returns the interface TypeID of the InferShapedTypeOpInterface.
"""
function mlirInferShapedTypeOpInterfaceTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirInferShapedTypeOpInterfaceTypeID()::MlirTypeID
end

# typedef void ( * MlirShapedTypeComponentsCallback ) ( bool , intptr_t , const int64_t * , MlirType , MlirAttribute , void * )
"""
These callbacks are used to return multiple shaped type components from functions while transferring ownership to the caller. The first argument is the has rank boolean followed by the the rank and a pointer to the shape (if applicable). The next argument is the element type, then the attribute. The last argument is an opaque pointer forwarded to the callback by the caller. This callback will be called potentially multiple times for each shaped type components.
"""
const MlirShapedTypeComponentsCallback = Ptr{Cvoid}

"""
    mlirInferShapedTypeOpInterfaceInferReturnTypes(opName, context, location, nOperands, operands, attributes, properties, nRegions, regions, callback, userData)

Infers the return shaped type components of the operation. Calls `callback` with the types of inferred arguments on success. Returns failure otherwise.
"""
function mlirInferShapedTypeOpInterfaceInferReturnTypes(
    opName,
    context,
    location,
    nOperands,
    operands,
    attributes,
    properties,
    nRegions,
    regions,
    callback,
    userData,
)
    @ccall Reactant_jll.libReactantExtra.mlirInferShapedTypeOpInterfaceInferReturnTypes(
        opName::MlirStringRef,
        context::MlirContext,
        location::MlirLocation,
        nOperands::Cptrdiff_t,
        operands::Ptr{MlirValue},
        attributes::MlirAttribute,
        properties::Ptr{Cvoid},
        nRegions::Cptrdiff_t,
        regions::Ptr{MlirRegion},
        callback::MlirShapedTypeComponentsCallback,
        userData::Ptr{Cvoid},
    )::MlirLogicalResult
end

"""
    MlirSpeculatability

Enum representing the speculatability of an operation.
"""
@cenum MlirSpeculatability::UInt32 begin
    MlirSpeculatabilityNotSpeculatable = 0x0000000000000000
    MlirSpeculatabilitySpeculatable = 0x0000000000000001
    MlirSpeculatabilityRecursivelySpeculatable = 0x0000000000000002
end

"""
    mlirConditionallySpeculatableOpInterfaceTypeID()

Returns the interface TypeID of the ConditionallySpeculatable interface.
"""
function mlirConditionallySpeculatableOpInterfaceTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirConditionallySpeculatableOpInterfaceTypeID()::MlirTypeID
end

"""
    MlirConditionallySpeculatableOpInterfaceCallbacks

Callbacks for implementing ConditionallySpeculatable from external code.

| Field              | Note                                                               |
| :----------------- | :----------------------------------------------------------------- |
| construct          | Optional constructor for user data. Set to nullptr to disable it.  |
| destruct           | Optional destructor for user data. Set to nullptr to disable it.   |
| getSpeculatability | Returns the speculatability of the given operation.                |
"""
struct MlirConditionallySpeculatableOpInterfaceCallbacks
    construct::Ptr{Cvoid}
    destruct::Ptr{Cvoid}
    getSpeculatability::Ptr{Cvoid}
    userData::Ptr{Cvoid}
end

"""
    mlirConditionallySpeculatableOpInterfaceAttachFallbackModel(ctx, opName, callbacks)

Attach a new FallbackModel for the ConditionallySpeculatable interface to the named operation. The FallbackModel will call the provided callbacks.
"""
function mlirConditionallySpeculatableOpInterfaceAttachFallbackModel(ctx, opName, callbacks)
    @ccall Reactant_jll.libReactantExtra.mlirConditionallySpeculatableOpInterfaceAttachFallbackModel(
        ctx::MlirContext,
        opName::MlirStringRef,
        callbacks::MlirConditionallySpeculatableOpInterfaceCallbacks,
    )::Cvoid
end

"""
    mlirConditionallySpeculatableOpInterfaceGetSpeculatability(operation)

Returns the speculatability of the given operation.

The operation must implement the ConditionallySpeculatable interface.
"""
function mlirConditionallySpeculatableOpInterfaceGetSpeculatability(operation)
    @ccall Reactant_jll.libReactantExtra.mlirConditionallySpeculatableOpInterfaceGetSpeculatability(
        operation::MlirOperation
    )::MlirSpeculatability
end

"""
    mlirMemoryEffectsAllocateGet()

Returns the borrowed singleton instance of the allocate memory effect.
"""
function mlirMemoryEffectsAllocateGet()
    @ccall Reactant_jll.libReactantExtra.mlirMemoryEffectsAllocateGet()::MlirMemoryEffect
end

"""
    mlirMemoryEffectsFreeGet()

Returns the borrowed singleton instance of the free memory effect.
"""
function mlirMemoryEffectsFreeGet()
    @ccall Reactant_jll.libReactantExtra.mlirMemoryEffectsFreeGet()::MlirMemoryEffect
end

"""
    mlirMemoryEffectsReadGet()

Returns the borrowed singleton instance of the read memory effect.
"""
function mlirMemoryEffectsReadGet()
    @ccall Reactant_jll.libReactantExtra.mlirMemoryEffectsReadGet()::MlirMemoryEffect
end

"""
    mlirMemoryEffectsWriteGet()

Returns the borrowed singleton instance of the write memory effect.
"""
function mlirMemoryEffectsWriteGet()
    @ccall Reactant_jll.libReactantExtra.mlirMemoryEffectsWriteGet()::MlirMemoryEffect
end

"""
    mlirSideEffectsDefaultResourceGet()

Returns the borrowed singleton instance of the default side effect resource.
"""
function mlirSideEffectsDefaultResourceGet()
    @ccall Reactant_jll.libReactantExtra.mlirSideEffectsDefaultResourceGet()::MlirSideEffectResource
end

"""
    mlirMemoryEffectInstanceCreate(effect, parameters, stage, effectOnFullRegion, resource)

Creates a memory effect instance without an associated IR entity. `parameters` may be a null attribute. The caller owns the returned instance and must destroy it with [`mlirMemoryEffectInstanceDestroy`](@ref).
"""
function mlirMemoryEffectInstanceCreate(
    effect, parameters, stage, effectOnFullRegion, resource
)
    @ccall Reactant_jll.libReactantExtra.mlirMemoryEffectInstanceCreate(
        effect::MlirMemoryEffect,
        parameters::MlirAttribute,
        stage::Cint,
        effectOnFullRegion::Bool,
        resource::MlirSideEffectResource,
    )::MlirMemoryEffectInstance
end

"""
    mlirMemoryEffectInstanceCreateForOpOperand(effect, opOperand, parameters, stage, effectOnFullRegion, resource)

Creates a memory effect instance associated with an operation operand. `parameters` may be a null attribute. The caller owns the returned instance and must destroy it with [`mlirMemoryEffectInstanceDestroy`](@ref).
"""
function mlirMemoryEffectInstanceCreateForOpOperand(
    effect, opOperand, parameters, stage, effectOnFullRegion, resource
)
    @ccall Reactant_jll.libReactantExtra.mlirMemoryEffectInstanceCreateForOpOperand(
        effect::MlirMemoryEffect,
        opOperand::MlirOpOperand,
        parameters::MlirAttribute,
        stage::Cint,
        effectOnFullRegion::Bool,
        resource::MlirSideEffectResource,
    )::MlirMemoryEffectInstance
end

"""
    mlirMemoryEffectInstanceCreateForOpResult(effect, result, parameters, stage, effectOnFullRegion, resource)

Creates a memory effect instance associated with an operation result. `result` must wrap an OpResult. `parameters` may be a null attribute. The caller owns the returned instance and must destroy it with [`mlirMemoryEffectInstanceDestroy`](@ref).
"""
function mlirMemoryEffectInstanceCreateForOpResult(
    effect, result, parameters, stage, effectOnFullRegion, resource
)
    @ccall Reactant_jll.libReactantExtra.mlirMemoryEffectInstanceCreateForOpResult(
        effect::MlirMemoryEffect,
        result::MlirValue,
        parameters::MlirAttribute,
        stage::Cint,
        effectOnFullRegion::Bool,
        resource::MlirSideEffectResource,
    )::MlirMemoryEffectInstance
end

"""
    mlirMemoryEffectInstanceCreateForBlockArgument(effect, blockArgument, parameters, stage, effectOnFullRegion, resource)

Creates a memory effect instance associated with a block argument. `blockArgument` must wrap a BlockArgument. `parameters` may be a null attribute. The caller owns the returned instance and must destroy it with [`mlirMemoryEffectInstanceDestroy`](@ref).
"""
function mlirMemoryEffectInstanceCreateForBlockArgument(
    effect, blockArgument, parameters, stage, effectOnFullRegion, resource
)
    @ccall Reactant_jll.libReactantExtra.mlirMemoryEffectInstanceCreateForBlockArgument(
        effect::MlirMemoryEffect,
        blockArgument::MlirValue,
        parameters::MlirAttribute,
        stage::Cint,
        effectOnFullRegion::Bool,
        resource::MlirSideEffectResource,
    )::MlirMemoryEffectInstance
end

"""
    mlirMemoryEffectInstanceCreateForSymbol(effect, symbol, parameters, stage, effectOnFullRegion, resource)

Creates a memory effect instance associated with a symbol. `symbol` must be a SymbolRefAttr. `parameters` may be a null attribute. The caller owns the returned instance and must destroy it with [`mlirMemoryEffectInstanceDestroy`](@ref).
"""
function mlirMemoryEffectInstanceCreateForSymbol(
    effect, symbol, parameters, stage, effectOnFullRegion, resource
)
    @ccall Reactant_jll.libReactantExtra.mlirMemoryEffectInstanceCreateForSymbol(
        effect::MlirMemoryEffect,
        symbol::MlirAttribute,
        parameters::MlirAttribute,
        stage::Cint,
        effectOnFullRegion::Bool,
        resource::MlirSideEffectResource,
    )::MlirMemoryEffectInstance
end

"""
    mlirMemoryEffectInstanceDestroy(instance)

Destroys a memory effect instance created by one of the functions above.
"""
function mlirMemoryEffectInstanceDestroy(instance)
    @ccall Reactant_jll.libReactantExtra.mlirMemoryEffectInstanceDestroy(
        instance::MlirMemoryEffectInstance
    )::Cvoid
end

"""
    mlirMemoryEffectInstancesListAppend(list, instance)

Appends a copy of `instance` to the given list. This does not take ownership of `instance`; the caller remains responsible for destroying it.
"""
function mlirMemoryEffectInstancesListAppend(list, instance)
    @ccall Reactant_jll.libReactantExtra.mlirMemoryEffectInstancesListAppend(
        list::MlirMemoryEffectInstancesList, instance::MlirMemoryEffectInstance
    )::Cvoid
end

"""
    mlirMemoryEffectsOpInterfaceTypeID()

Returns the interface TypeID of the MemoryEffectsOpInterface.
"""
function mlirMemoryEffectsOpInterfaceTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirMemoryEffectsOpInterfaceTypeID()::MlirTypeID
end

"""
    MlirMemoryEffectsOpInterfaceCallbacks

Callbacks for implementing MemoryEffectsOpInterface from external code.

| Field      | Note                                                               |
| :--------- | :----------------------------------------------------------------- |
| construct  | Optional constructor for user data. Set to nullptr to disable it.  |
| destruct   | Optional destructor for user data. Set to nullptr to disable it.   |
| getEffects | Get memory effects callback.                                       |
"""
struct MlirMemoryEffectsOpInterfaceCallbacks
    construct::Ptr{Cvoid}
    destruct::Ptr{Cvoid}
    getEffects::Ptr{Cvoid}
    userData::Ptr{Cvoid}
end

"""
    mlirMemoryEffectsOpInterfaceAttachFallbackModel(ctx, opName, callbacks)

Attach a new FallbackModel for the MemoryEffectsOpInterface to the named operation. The FallbackModel will call the provided callbacks.
"""
function mlirMemoryEffectsOpInterfaceAttachFallbackModel(ctx, opName, callbacks)
    @ccall Reactant_jll.libReactantExtra.mlirMemoryEffectsOpInterfaceAttachFallbackModel(
        ctx::MlirContext,
        opName::MlirStringRef,
        callbacks::MlirMemoryEffectsOpInterfaceCallbacks,
    )::Cvoid
end

struct MlirRewriterBase
    ptr::Ptr{Cvoid}
end

struct MlirFrozenRewritePatternSet
    ptr::Ptr{Cvoid}
end

struct MlirGreedyRewriteDriverConfig
    ptr::Ptr{Cvoid}
end

"""
    MlirGreedyRewriteStrictness

Greedy rewrite strictness levels.
"""
@cenum MlirGreedyRewriteStrictness::UInt32 begin
    MLIR_GREEDY_REWRITE_STRICTNESS_ANY_OP = 0x0000000000000000
    MLIR_GREEDY_REWRITE_STRICTNESS_EXISTING_AND_NEW_OPS = 0x0000000000000001
    MLIR_GREEDY_REWRITE_STRICTNESS_EXISTING_OPS = 0x0000000000000002
end

"""
    MlirGreedySimplifyRegionLevel

Greedy simplify region levels.
"""
@cenum MlirGreedySimplifyRegionLevel::UInt32 begin
    MLIR_GREEDY_SIMPLIFY_REGION_LEVEL_DISABLED = 0x0000000000000000
    MLIR_GREEDY_SIMPLIFY_REGION_LEVEL_NORMAL = 0x0000000000000001
    MLIR_GREEDY_SIMPLIFY_REGION_LEVEL_AGGRESSIVE = 0x0000000000000002
end

struct MlirRewritePatternSet
    ptr::Ptr{Cvoid}
end

struct MlirPatternRewriter
    ptr::Ptr{Cvoid}
end

struct MlirRewritePattern
    ptr::Ptr{Cvoid}
end

struct MlirConversionTarget
    ptr::Ptr{Cvoid}
end

struct MlirConversionPattern
    ptr::Ptr{Cvoid}
end

struct MlirTypeConverter
    ptr::Ptr{Cvoid}
end

struct MlirConversionPatternRewriter
    ptr::Ptr{Cvoid}
end

struct MlirConversionConfig
    ptr::Ptr{Cvoid}
end

"""
    mlirRewriterBaseGetContext(rewriter)

Get the MLIR context referenced by the rewriter.
"""
function mlirRewriterBaseGetContext(rewriter)
    @ccall Reactant_jll.libReactantExtra.mlirRewriterBaseGetContext(
        rewriter::MlirRewriterBase
    )::MlirContext
end

"""
    mlirRewriterBaseClearInsertionPoint(rewriter)

Reset the insertion point to no location. Creating an operation without a set insertion point is an error, but this can still be useful when the current insertion point a builder refers to is being removed.
"""
function mlirRewriterBaseClearInsertionPoint(rewriter)
    @ccall Reactant_jll.libReactantExtra.mlirRewriterBaseClearInsertionPoint(
        rewriter::MlirRewriterBase
    )::Cvoid
end

"""
    mlirRewriterBaseSetInsertionPointBefore(rewriter, op)

Sets the insertion point to the specified operation, which will cause subsequent insertions to go right before it.
"""
function mlirRewriterBaseSetInsertionPointBefore(rewriter, op)
    @ccall Reactant_jll.libReactantExtra.mlirRewriterBaseSetInsertionPointBefore(
        rewriter::MlirRewriterBase, op::MlirOperation
    )::Cvoid
end

"""
    mlirRewriterBaseSetInsertionPointAfter(rewriter, op)

Sets the insertion point to the node after the specified operation, which will cause subsequent insertions to go right after it.
"""
function mlirRewriterBaseSetInsertionPointAfter(rewriter, op)
    @ccall Reactant_jll.libReactantExtra.mlirRewriterBaseSetInsertionPointAfter(
        rewriter::MlirRewriterBase, op::MlirOperation
    )::Cvoid
end

"""
    mlirRewriterBaseSetInsertionPointAfterValue(rewriter, value)

Sets the insertion point to the node after the specified value. If value has a defining operation, sets the insertion point to the node after such defining operation. This will cause subsequent insertions to go right after it. Otherwise, value is a BlockArgument. Sets the insertion point to the start of its block.
"""
function mlirRewriterBaseSetInsertionPointAfterValue(rewriter, value)
    @ccall Reactant_jll.libReactantExtra.mlirRewriterBaseSetInsertionPointAfterValue(
        rewriter::MlirRewriterBase, value::MlirValue
    )::Cvoid
end

"""
    mlirRewriterBaseSetInsertionPointToStart(rewriter, block)

Sets the insertion point to the start of the specified block.
"""
function mlirRewriterBaseSetInsertionPointToStart(rewriter, block)
    @ccall Reactant_jll.libReactantExtra.mlirRewriterBaseSetInsertionPointToStart(
        rewriter::MlirRewriterBase, block::MlirBlock
    )::Cvoid
end

"""
    mlirRewriterBaseSetInsertionPointToEnd(rewriter, block)

Sets the insertion point to the end of the specified block.
"""
function mlirRewriterBaseSetInsertionPointToEnd(rewriter, block)
    @ccall Reactant_jll.libReactantExtra.mlirRewriterBaseSetInsertionPointToEnd(
        rewriter::MlirRewriterBase, block::MlirBlock
    )::Cvoid
end

"""
    mlirRewriterBaseGetInsertionBlock(rewriter)

Return the block the current insertion point belongs to. Note that the insertion point is not necessarily the end of the block.
"""
function mlirRewriterBaseGetInsertionBlock(rewriter)
    @ccall Reactant_jll.libReactantExtra.mlirRewriterBaseGetInsertionBlock(
        rewriter::MlirRewriterBase
    )::MlirBlock
end

"""
    mlirRewriterBaseGetBlock(rewriter)

Returns the current block of the rewriter.
"""
function mlirRewriterBaseGetBlock(rewriter)
    @ccall Reactant_jll.libReactantExtra.mlirRewriterBaseGetBlock(
        rewriter::MlirRewriterBase
    )::MlirBlock
end

"""
    mlirRewriterBaseGetOperationAfterInsertion(rewriter)

Returns the operation right after the current insertion point of the rewriter. A null [`MlirOperation`](@ref) will be returned
"""
function mlirRewriterBaseGetOperationAfterInsertion(rewriter)
    @ccall Reactant_jll.libReactantExtra.mlirRewriterBaseGetOperationAfterInsertion(
        rewriter::MlirRewriterBase
    )::MlirOperation
end

"""
    MlirRewriterBaseInsertPoint

A saved insertion point: a (block, operationAfter) pair. `operationAfter` is the operation that subsequent insertions go before. If `operationAfter` is null, the insertion point is at the end of `block`. If `block` is null, the insertion point is not set (cleared).
"""
struct MlirRewriterBaseInsertPoint
    block::MlirBlock
    operationAfter::MlirOperation
end

"""
    mlirRewriterBaseSaveInsertionPoint(rewriter)

Returns the current insertion point of the rewriter so that it can be restored later with [`mlirRewriterBaseRestoreInsertionPoint`](@ref).
"""
function mlirRewriterBaseSaveInsertionPoint(rewriter)
    @ccall Reactant_jll.libReactantExtra.mlirRewriterBaseSaveInsertionPoint(
        rewriter::MlirRewriterBase
    )::MlirRewriterBaseInsertPoint
end

"""
    mlirRewriterBaseRestoreInsertionPoint(rewriter, insertPoint)

Restores a previously saved insertion point.
"""
function mlirRewriterBaseRestoreInsertionPoint(rewriter, insertPoint)
    @ccall Reactant_jll.libReactantExtra.mlirRewriterBaseRestoreInsertionPoint(
        rewriter::MlirRewriterBase, insertPoint::MlirRewriterBaseInsertPoint
    )::Cvoid
end

"""
    mlirRewriterBaseCreateBlockBefore(rewriter, insertBefore, nArgTypes, argTypes, locations)

Add new block with 'argTypes' arguments and set the insertion point to the end of it. The block is placed before 'insertBefore'. `locs` contains the locations of the inserted arguments, and should match the size of `argTypes`.
"""
function mlirRewriterBaseCreateBlockBefore(
    rewriter, insertBefore, nArgTypes, argTypes, locations
)
    @ccall Reactant_jll.libReactantExtra.mlirRewriterBaseCreateBlockBefore(
        rewriter::MlirRewriterBase,
        insertBefore::MlirBlock,
        nArgTypes::Cptrdiff_t,
        argTypes::Ptr{MlirType},
        locations::Ptr{MlirLocation},
    )::MlirBlock
end

"""
    mlirRewriterBaseInsert(rewriter, op)

Insert the given operation at the current insertion point and return it.
"""
function mlirRewriterBaseInsert(rewriter, op)
    @ccall Reactant_jll.libReactantExtra.mlirRewriterBaseInsert(
        rewriter::MlirRewriterBase, op::MlirOperation
    )::MlirOperation
end

"""
    mlirRewriterBaseClone(rewriter, op)

Creates a deep copy of the specified operation.
"""
function mlirRewriterBaseClone(rewriter, op)
    @ccall Reactant_jll.libReactantExtra.mlirRewriterBaseClone(
        rewriter::MlirRewriterBase, op::MlirOperation
    )::MlirOperation
end

"""
    mlirRewriterBaseCloneWithoutRegions(rewriter, op)

Creates a deep copy of this operation but keep the operation regions empty.
"""
function mlirRewriterBaseCloneWithoutRegions(rewriter, op)
    @ccall Reactant_jll.libReactantExtra.mlirRewriterBaseCloneWithoutRegions(
        rewriter::MlirRewriterBase, op::MlirOperation
    )::MlirOperation
end

"""
    mlirRewriterBaseCloneWithMapping(rewriter, op, mapping)

Clones the given operation using the rewriter and the provided IRMapping. The mapping is updated with the results of the cloned operation.
"""
function mlirRewriterBaseCloneWithMapping(rewriter, op, mapping)
    @ccall Reactant_jll.libReactantExtra.mlirRewriterBaseCloneWithMapping(
        rewriter::MlirRewriterBase, op::MlirOperation, mapping::MlirIRMapping
    )::MlirOperation
end

"""
    mlirRewriterBaseCloneRegionBefore(rewriter, region, before)

Clone the blocks that belong to "region" before the given position in another region "parent".
"""
function mlirRewriterBaseCloneRegionBefore(rewriter, region, before)
    @ccall Reactant_jll.libReactantExtra.mlirRewriterBaseCloneRegionBefore(
        rewriter::MlirRewriterBase, region::MlirRegion, before::MlirBlock
    )::Cvoid
end

"""
    mlirRewriterBaseInlineRegionBefore(rewriter, region, before)

Move the blocks that belong to "region" before the given position in another region "parent". The two regions must be different. The caller is responsible for creating or updating the operation transferring flow of control to the region and passing it the correct block arguments.
"""
function mlirRewriterBaseInlineRegionBefore(rewriter, region, before)
    @ccall Reactant_jll.libReactantExtra.mlirRewriterBaseInlineRegionBefore(
        rewriter::MlirRewriterBase, region::MlirRegion, before::MlirBlock
    )::Cvoid
end

"""
    mlirRewriterBaseReplaceOpWithValues(rewriter, op, nValues, values)

Replace the results of the given (original) operation with the specified list of values (replacements). The result types of the given op and the replacements must match. The original op is erased.
"""
function mlirRewriterBaseReplaceOpWithValues(rewriter, op, nValues, values)
    @ccall Reactant_jll.libReactantExtra.mlirRewriterBaseReplaceOpWithValues(
        rewriter::MlirRewriterBase,
        op::MlirOperation,
        nValues::Cptrdiff_t,
        values::Ptr{MlirValue},
    )::Cvoid
end

"""
    mlirRewriterBaseReplaceOpWithOperation(rewriter, op, newOp)

Replace the results of the given (original) operation with the specified new op (replacement). The result types of the two ops must match. The original op is erased.
"""
function mlirRewriterBaseReplaceOpWithOperation(rewriter, op, newOp)
    @ccall Reactant_jll.libReactantExtra.mlirRewriterBaseReplaceOpWithOperation(
        rewriter::MlirRewriterBase, op::MlirOperation, newOp::MlirOperation
    )::Cvoid
end

"""
    mlirRewriterBaseEraseOp(rewriter, op)

Erases an operation that is known to have no uses.
"""
function mlirRewriterBaseEraseOp(rewriter, op)
    @ccall Reactant_jll.libReactantExtra.mlirRewriterBaseEraseOp(
        rewriter::MlirRewriterBase, op::MlirOperation
    )::Cvoid
end

"""
    mlirRewriterBaseEraseBlock(rewriter, block)

Erases a block along with all operations inside it.
"""
function mlirRewriterBaseEraseBlock(rewriter, block)
    @ccall Reactant_jll.libReactantExtra.mlirRewriterBaseEraseBlock(
        rewriter::MlirRewriterBase, block::MlirBlock
    )::Cvoid
end

"""
    mlirRewriterBaseInlineBlockBefore(rewriter, source, op, nArgValues, argValues)

Inline the operations of block 'source' before the operation 'op'. The source block will be deleted and must have no uses. 'argValues' is used to replace the block arguments of 'source'

The source block must have no successors. Otherwise, the resulting IR would have unreachable operations.
"""
function mlirRewriterBaseInlineBlockBefore(rewriter, source, op, nArgValues, argValues)
    @ccall Reactant_jll.libReactantExtra.mlirRewriterBaseInlineBlockBefore(
        rewriter::MlirRewriterBase,
        source::MlirBlock,
        op::MlirOperation,
        nArgValues::Cptrdiff_t,
        argValues::Ptr{MlirValue},
    )::Cvoid
end

"""
    mlirRewriterBaseMergeBlocks(rewriter, source, dest, nArgValues, argValues)

Inline the operations of block 'source' into the end of block 'dest'. The source block will be deleted and must have no uses. 'argValues' is used to replace the block arguments of 'source'

The dest block must have no successors. Otherwise, the resulting IR would have unreachable operation.
"""
function mlirRewriterBaseMergeBlocks(rewriter, source, dest, nArgValues, argValues)
    @ccall Reactant_jll.libReactantExtra.mlirRewriterBaseMergeBlocks(
        rewriter::MlirRewriterBase,
        source::MlirBlock,
        dest::MlirBlock,
        nArgValues::Cptrdiff_t,
        argValues::Ptr{MlirValue},
    )::Cvoid
end

"""
    mlirRewriterBaseMoveOpBefore(rewriter, op, existingOp)

Unlink this operation from its current block and insert it right before `existingOp` which may be in the same or another block in the same function.
"""
function mlirRewriterBaseMoveOpBefore(rewriter, op, existingOp)
    @ccall Reactant_jll.libReactantExtra.mlirRewriterBaseMoveOpBefore(
        rewriter::MlirRewriterBase, op::MlirOperation, existingOp::MlirOperation
    )::Cvoid
end

"""
    mlirRewriterBaseMoveOpAfter(rewriter, op, existingOp)

Unlink this operation from its current block and insert it right after `existingOp` which may be in the same or another block in the same function.
"""
function mlirRewriterBaseMoveOpAfter(rewriter, op, existingOp)
    @ccall Reactant_jll.libReactantExtra.mlirRewriterBaseMoveOpAfter(
        rewriter::MlirRewriterBase, op::MlirOperation, existingOp::MlirOperation
    )::Cvoid
end

"""
    mlirRewriterBaseMoveBlockBefore(rewriter, block, existingBlock)

Unlink this block and insert it right before `existingBlock`.
"""
function mlirRewriterBaseMoveBlockBefore(rewriter, block, existingBlock)
    @ccall Reactant_jll.libReactantExtra.mlirRewriterBaseMoveBlockBefore(
        rewriter::MlirRewriterBase, block::MlirBlock, existingBlock::MlirBlock
    )::Cvoid
end

"""
    mlirRewriterBaseStartOpModification(rewriter, op)

This method is used to notify the rewriter that an in-place operation modification is about to happen. A call to this function *must* be followed by a call to either `finalizeOpModification` or `cancelOpModification`. This is a minor efficiency win (it avoids creating a new operation and removing the old one) but also often allows simpler code in the client.
"""
function mlirRewriterBaseStartOpModification(rewriter, op)
    @ccall Reactant_jll.libReactantExtra.mlirRewriterBaseStartOpModification(
        rewriter::MlirRewriterBase, op::MlirOperation
    )::Cvoid
end

"""
    mlirRewriterBaseFinalizeOpModification(rewriter, op)

This method is used to signal the end of an in-place modification of the given operation. This can only be called on operations that were provided to a call to `startOpModification`.
"""
function mlirRewriterBaseFinalizeOpModification(rewriter, op)
    @ccall Reactant_jll.libReactantExtra.mlirRewriterBaseFinalizeOpModification(
        rewriter::MlirRewriterBase, op::MlirOperation
    )::Cvoid
end

"""
    mlirRewriterBaseCancelOpModification(rewriter, op)

This method cancels a pending in-place modification. This can only be called on operations that were provided to a call to `startOpModification`.
"""
function mlirRewriterBaseCancelOpModification(rewriter, op)
    @ccall Reactant_jll.libReactantExtra.mlirRewriterBaseCancelOpModification(
        rewriter::MlirRewriterBase, op::MlirOperation
    )::Cvoid
end

"""
    mlirRewriterBaseReplaceAllUsesWith(rewriter, from, to)

Find uses of `from` and replace them with `to`. Also notify the listener about every in-place op modification (for every use that was replaced).
"""
function mlirRewriterBaseReplaceAllUsesWith(rewriter, from, to)
    @ccall Reactant_jll.libReactantExtra.mlirRewriterBaseReplaceAllUsesWith(
        rewriter::MlirRewriterBase, from::MlirValue, to::MlirValue
    )::Cvoid
end

"""
    mlirRewriterBaseReplaceAllValueRangeUsesWith(rewriter, nValues, from, to)

Find uses of `from` and replace them with `to`. Also notify the listener about every in-place op modification (for every use that was replaced).
"""
function mlirRewriterBaseReplaceAllValueRangeUsesWith(rewriter, nValues, from, to)
    @ccall Reactant_jll.libReactantExtra.mlirRewriterBaseReplaceAllValueRangeUsesWith(
        rewriter::MlirRewriterBase,
        nValues::Cptrdiff_t,
        from::Ptr{MlirValue},
        to::Ptr{MlirValue},
    )::Cvoid
end

"""
    mlirRewriterBaseReplaceAllOpUsesWithValueRange(rewriter, from, nTo, to)

Find uses of `from` and replace them with `to`. Also notify the listener about every in-place op modification (for every use that was replaced) and that the `from` operation is about to be replaced.
"""
function mlirRewriterBaseReplaceAllOpUsesWithValueRange(rewriter, from, nTo, to)
    @ccall Reactant_jll.libReactantExtra.mlirRewriterBaseReplaceAllOpUsesWithValueRange(
        rewriter::MlirRewriterBase, from::MlirOperation, nTo::Cptrdiff_t, to::Ptr{MlirValue}
    )::Cvoid
end

"""
    mlirRewriterBaseReplaceAllOpUsesWithOperation(rewriter, from, to)

Find uses of `from` and replace them with `to`. Also notify the listener about every in-place op modification (for every use that was replaced) and that the `from` operation is about to be replaced.
"""
function mlirRewriterBaseReplaceAllOpUsesWithOperation(rewriter, from, to)
    @ccall Reactant_jll.libReactantExtra.mlirRewriterBaseReplaceAllOpUsesWithOperation(
        rewriter::MlirRewriterBase, from::MlirOperation, to::MlirOperation
    )::Cvoid
end

"""
    mlirRewriterBaseReplaceOpUsesWithinBlock(rewriter, op, nNewValues, newValues, block)

Find uses of `from` within `block` and replace them with `to`. Also notify the listener about every in-place op modification (for every use that was replaced). The optional `allUsesReplaced` flag is set to "true" if all uses were replaced.
"""
function mlirRewriterBaseReplaceOpUsesWithinBlock(
    rewriter, op, nNewValues, newValues, block
)
    @ccall Reactant_jll.libReactantExtra.mlirRewriterBaseReplaceOpUsesWithinBlock(
        rewriter::MlirRewriterBase,
        op::MlirOperation,
        nNewValues::Cptrdiff_t,
        newValues::Ptr{MlirValue},
        block::MlirBlock,
    )::Cvoid
end

"""
    mlirRewriterBaseReplaceAllUsesExcept(rewriter, from, to, exceptedUser)

Find uses of `from` and replace them with `to` except if the user is `exceptedUser`. Also notify the listener about every in-place op modification (for every use that was replaced).
"""
function mlirRewriterBaseReplaceAllUsesExcept(rewriter, from, to, exceptedUser)
    @ccall Reactant_jll.libReactantExtra.mlirRewriterBaseReplaceAllUsesExcept(
        rewriter::MlirRewriterBase,
        from::MlirValue,
        to::MlirValue,
        exceptedUser::MlirOperation,
    )::Cvoid
end

"""
    mlirIRRewriterCreate(context)

Create an IRRewriter and transfer ownership to the caller.
"""
function mlirIRRewriterCreate(context)
    @ccall Reactant_jll.libReactantExtra.mlirIRRewriterCreate(
        context::MlirContext
    )::MlirRewriterBase
end

"""
    mlirIRRewriterCreateFromOp(op)

Create an IRRewriter and transfer ownership to the caller. Additionally set the insertion point before the operation.
"""
function mlirIRRewriterCreateFromOp(op)
    @ccall Reactant_jll.libReactantExtra.mlirIRRewriterCreateFromOp(
        op::MlirOperation
    )::MlirRewriterBase
end

"""
    mlirIRRewriterDestroy(rewriter)

Takes an IRRewriter owned by the caller and destroys it. It is the responsibility of the user to only pass an IRRewriter class.
"""
function mlirIRRewriterDestroy(rewriter)
    @ccall Reactant_jll.libReactantExtra.mlirIRRewriterDestroy(
        rewriter::MlirRewriterBase
    )::Cvoid
end

"""
    mlirFreezeRewritePattern(set)

Freeze the given [`MlirRewritePatternSet`](@ref) to a [`MlirFrozenRewritePatternSet`](@ref). Note that the ownership of the input set is transferred into the frozen set after this call.
"""
function mlirFreezeRewritePattern(set)
    @ccall Reactant_jll.libReactantExtra.mlirFreezeRewritePattern(
        set::MlirRewritePatternSet
    )::MlirFrozenRewritePatternSet
end

"""
    mlirFrozenRewritePatternSetDestroy(set)

Destroy the given [`MlirFrozenRewritePatternSet`](@ref).
"""
function mlirFrozenRewritePatternSetDestroy(set)
    @ccall Reactant_jll.libReactantExtra.mlirFrozenRewritePatternSetDestroy(
        set::MlirFrozenRewritePatternSet
    )::Cvoid
end

function mlirApplyPatternsAndFoldGreedilyWithOp(op, patterns, arg3)
    @ccall Reactant_jll.libReactantExtra.mlirApplyPatternsAndFoldGreedilyWithOp(
        op::MlirOperation,
        patterns::MlirFrozenRewritePatternSet,
        arg3::MlirGreedyRewriteDriverConfig,
    )::MlirLogicalResult
end

function mlirApplyPatternsAndFoldGreedily(op, patterns, config)
    @ccall Reactant_jll.libReactantExtra.mlirApplyPatternsAndFoldGreedily(
        op::MlirModule,
        patterns::MlirFrozenRewritePatternSet,
        config::MlirGreedyRewriteDriverConfig,
    )::MlirLogicalResult
end

"""
    mlirGreedyRewriteDriverConfigCreate()

Creates a greedy rewrite driver configuration with default settings.
"""
function mlirGreedyRewriteDriverConfigCreate()
    @ccall Reactant_jll.libReactantExtra.mlirGreedyRewriteDriverConfigCreate()::MlirGreedyRewriteDriverConfig
end

"""
    mlirGreedyRewriteDriverConfigDestroy(config)

Destroys a greedy rewrite driver configuration.
"""
function mlirGreedyRewriteDriverConfigDestroy(config)
    @ccall Reactant_jll.libReactantExtra.mlirGreedyRewriteDriverConfigDestroy(
        config::MlirGreedyRewriteDriverConfig
    )::Cvoid
end

"""
    mlirGreedyRewriteDriverConfigSetMaxIterations(config, maxIterations)

Sets the maximum number of iterations for the greedy rewrite driver. Use -1 for no limit.
"""
function mlirGreedyRewriteDriverConfigSetMaxIterations(config, maxIterations)
    @ccall Reactant_jll.libReactantExtra.mlirGreedyRewriteDriverConfigSetMaxIterations(
        config::MlirGreedyRewriteDriverConfig, maxIterations::Int64
    )::Cvoid
end

"""
    mlirGreedyRewriteDriverConfigSetMaxNumRewrites(config, maxNumRewrites)

Sets the maximum number of rewrites within an iteration. Use -1 for no limit.
"""
function mlirGreedyRewriteDriverConfigSetMaxNumRewrites(config, maxNumRewrites)
    @ccall Reactant_jll.libReactantExtra.mlirGreedyRewriteDriverConfigSetMaxNumRewrites(
        config::MlirGreedyRewriteDriverConfig, maxNumRewrites::Int64
    )::Cvoid
end

"""
    mlirGreedyRewriteDriverConfigSetUseTopDownTraversal(config, useTopDownTraversal)

Sets whether to use top-down traversal for the initial population of the worklist.
"""
function mlirGreedyRewriteDriverConfigSetUseTopDownTraversal(config, useTopDownTraversal)
    @ccall Reactant_jll.libReactantExtra.mlirGreedyRewriteDriverConfigSetUseTopDownTraversal(
        config::MlirGreedyRewriteDriverConfig, useTopDownTraversal::Bool
    )::Cvoid
end

"""
    mlirGreedyRewriteDriverConfigEnableFolding(config, enable)

Enables or disables folding during greedy rewriting.
"""
function mlirGreedyRewriteDriverConfigEnableFolding(config, enable)
    @ccall Reactant_jll.libReactantExtra.mlirGreedyRewriteDriverConfigEnableFolding(
        config::MlirGreedyRewriteDriverConfig, enable::Bool
    )::Cvoid
end

"""
    mlirGreedyRewriteDriverConfigSetStrictness(config, strictness)

Sets the strictness level for the greedy rewrite driver.
"""
function mlirGreedyRewriteDriverConfigSetStrictness(config, strictness)
    @ccall Reactant_jll.libReactantExtra.mlirGreedyRewriteDriverConfigSetStrictness(
        config::MlirGreedyRewriteDriverConfig, strictness::MlirGreedyRewriteStrictness
    )::Cvoid
end

"""
    mlirGreedyRewriteDriverConfigSetRegionSimplificationLevel(config, level)

Sets the region simplification level.
"""
function mlirGreedyRewriteDriverConfigSetRegionSimplificationLevel(config, level)
    @ccall Reactant_jll.libReactantExtra.mlirGreedyRewriteDriverConfigSetRegionSimplificationLevel(
        config::MlirGreedyRewriteDriverConfig, level::MlirGreedySimplifyRegionLevel
    )::Cvoid
end

"""
    mlirGreedyRewriteDriverConfigEnableConstantCSE(config, enable)

Enables or disables constant CSE.
"""
function mlirGreedyRewriteDriverConfigEnableConstantCSE(config, enable)
    @ccall Reactant_jll.libReactantExtra.mlirGreedyRewriteDriverConfigEnableConstantCSE(
        config::MlirGreedyRewriteDriverConfig, enable::Bool
    )::Cvoid
end

"""
    mlirGreedyRewriteDriverConfigGetMaxIterations(config)

Gets the maximum number of iterations for the greedy rewrite driver.
"""
function mlirGreedyRewriteDriverConfigGetMaxIterations(config)
    @ccall Reactant_jll.libReactantExtra.mlirGreedyRewriteDriverConfigGetMaxIterations(
        config::MlirGreedyRewriteDriverConfig
    )::Int64
end

"""
    mlirGreedyRewriteDriverConfigGetMaxNumRewrites(config)

Gets the maximum number of rewrites within an iteration.
"""
function mlirGreedyRewriteDriverConfigGetMaxNumRewrites(config)
    @ccall Reactant_jll.libReactantExtra.mlirGreedyRewriteDriverConfigGetMaxNumRewrites(
        config::MlirGreedyRewriteDriverConfig
    )::Int64
end

"""
    mlirGreedyRewriteDriverConfigGetUseTopDownTraversal(config)

Gets whether top-down traversal is used for initial worklist population.
"""
function mlirGreedyRewriteDriverConfigGetUseTopDownTraversal(config)
    @ccall Reactant_jll.libReactantExtra.mlirGreedyRewriteDriverConfigGetUseTopDownTraversal(
        config::MlirGreedyRewriteDriverConfig
    )::Bool
end

"""
    mlirGreedyRewriteDriverConfigIsFoldingEnabled(config)

Gets whether folding is enabled during greedy rewriting.
"""
function mlirGreedyRewriteDriverConfigIsFoldingEnabled(config)
    @ccall Reactant_jll.libReactantExtra.mlirGreedyRewriteDriverConfigIsFoldingEnabled(
        config::MlirGreedyRewriteDriverConfig
    )::Bool
end

"""
    mlirGreedyRewriteDriverConfigGetStrictness(config)

Gets the strictness level for the greedy rewrite driver.
"""
function mlirGreedyRewriteDriverConfigGetStrictness(config)
    @ccall Reactant_jll.libReactantExtra.mlirGreedyRewriteDriverConfigGetStrictness(
        config::MlirGreedyRewriteDriverConfig
    )::MlirGreedyRewriteStrictness
end

"""
    mlirGreedyRewriteDriverConfigGetRegionSimplificationLevel(config)

Gets the region simplification level.
"""
function mlirGreedyRewriteDriverConfigGetRegionSimplificationLevel(config)
    @ccall Reactant_jll.libReactantExtra.mlirGreedyRewriteDriverConfigGetRegionSimplificationLevel(
        config::MlirGreedyRewriteDriverConfig
    )::MlirGreedySimplifyRegionLevel
end

"""
    mlirGreedyRewriteDriverConfigIsConstantCSEEnabled(config)

Gets whether constant CSE is enabled.
"""
function mlirGreedyRewriteDriverConfigIsConstantCSEEnabled(config)
    @ccall Reactant_jll.libReactantExtra.mlirGreedyRewriteDriverConfigIsConstantCSEEnabled(
        config::MlirGreedyRewriteDriverConfig
    )::Bool
end

"""
    mlirWalkAndApplyPatterns(op, patterns)

Applies the given patterns to the given op by a fast walk-based pattern rewrite driver.
"""
function mlirWalkAndApplyPatterns(op, patterns)
    @ccall Reactant_jll.libReactantExtra.mlirWalkAndApplyPatterns(
        op::MlirOperation, patterns::MlirFrozenRewritePatternSet
    )::Cvoid
end

"""
    mlirApplyPartialConversion(op, target, patterns, config)

Apply a partial conversion on the given operation.
"""
function mlirApplyPartialConversion(op, target, patterns, config)
    @ccall Reactant_jll.libReactantExtra.mlirApplyPartialConversion(
        op::MlirOperation,
        target::MlirConversionTarget,
        patterns::MlirFrozenRewritePatternSet,
        config::MlirConversionConfig,
    )::MlirLogicalResult
end

"""
    mlirApplyFullConversion(op, target, patterns, config)

Apply a full conversion on the given operation.
"""
function mlirApplyFullConversion(op, target, patterns, config)
    @ccall Reactant_jll.libReactantExtra.mlirApplyFullConversion(
        op::MlirOperation,
        target::MlirConversionTarget,
        patterns::MlirFrozenRewritePatternSet,
        config::MlirConversionConfig,
    )::MlirLogicalResult
end

"""
    mlirConversionConfigCreate()

Create a default ConversionConfig.
"""
function mlirConversionConfigCreate()
    @ccall Reactant_jll.libReactantExtra.mlirConversionConfigCreate()::MlirConversionConfig
end

"""
    mlirConversionConfigDestroy(config)

Destroy the given ConversionConfig.
"""
function mlirConversionConfigDestroy(config)
    @ccall Reactant_jll.libReactantExtra.mlirConversionConfigDestroy(
        config::MlirConversionConfig
    )::Cvoid
end

@cenum MlirDialectConversionFoldingMode::UInt32 begin
    MLIR_DIALECT_CONVERSION_FOLDING_MODE_NEVER = 0x0000000000000000
    MLIR_DIALECT_CONVERSION_FOLDING_MODE_BEFORE_PATTERNS = 0x0000000000000001
    MLIR_DIALECT_CONVERSION_FOLDING_MODE_AFTER_PATTERNS = 0x0000000000000002
end

"""
    mlirConversionConfigSetFoldingMode(config, mode)

Set the folding mode for the given ConversionConfig.
"""
function mlirConversionConfigSetFoldingMode(config, mode)
    @ccall Reactant_jll.libReactantExtra.mlirConversionConfigSetFoldingMode(
        config::MlirConversionConfig, mode::MlirDialectConversionFoldingMode
    )::Cvoid
end

"""
    mlirConversionConfigGetFoldingMode(config)

Get the folding mode for the given ConversionConfig.
"""
function mlirConversionConfigGetFoldingMode(config)
    @ccall Reactant_jll.libReactantExtra.mlirConversionConfigGetFoldingMode(
        config::MlirConversionConfig
    )::MlirDialectConversionFoldingMode
end

"""
    mlirConversionConfigEnableBuildMaterializations(config, enable)

Enable or disable building materializations during conversion.
"""
function mlirConversionConfigEnableBuildMaterializations(config, enable)
    @ccall Reactant_jll.libReactantExtra.mlirConversionConfigEnableBuildMaterializations(
        config::MlirConversionConfig, enable::Bool
    )::Cvoid
end

"""
    mlirConversionConfigIsBuildMaterializationsEnabled(config)

Check if building materializations during conversion is enabled.
"""
function mlirConversionConfigIsBuildMaterializationsEnabled(config)
    @ccall Reactant_jll.libReactantExtra.mlirConversionConfigIsBuildMaterializationsEnabled(
        config::MlirConversionConfig
    )::Bool
end

"""
    mlirPatternRewriterAsBase(rewriter)

Cast the PatternRewriter to a RewriterBase
"""
function mlirPatternRewriterAsBase(rewriter)
    @ccall Reactant_jll.libReactantExtra.mlirPatternRewriterAsBase(
        rewriter::MlirPatternRewriter
    )::MlirRewriterBase
end

"""
    mlirConversionPatternRewriterAsPatternRewriter(rewriter)

Cast the ConversionPatternRewriter to a PatternRewriter
"""
function mlirConversionPatternRewriterAsPatternRewriter(rewriter)
    @ccall Reactant_jll.libReactantExtra.mlirConversionPatternRewriterAsPatternRewriter(
        rewriter::MlirConversionPatternRewriter
    )::MlirPatternRewriter
end

"""
    mlirConversionPatternRewriterConvertRegionTypes(rewriter, region, typeConverter)

Apply a signature conversion to each block in the given region.
"""
function mlirConversionPatternRewriterConvertRegionTypes(rewriter, region, typeConverter)
    @ccall Reactant_jll.libReactantExtra.mlirConversionPatternRewriterConvertRegionTypes(
        rewriter::MlirConversionPatternRewriter,
        region::MlirRegion,
        typeConverter::MlirTypeConverter,
    )::MlirLogicalResult
end

"""
    mlirConversionPatternRewriterReplaceOpWithMultiple(rewriter, op, nRanges, rangeSizes, values)

Replace the given operation with multiple value ranges -- one range per result of `op` -- and erase it. `nRanges` must equal the number of results of `op`. `rangeSizes[i]` is the number of values in the i-th range, and `values` is the flat concatenation of all ranges (its length is the sum of `rangeSizes[0..nRanges)`).
"""
function mlirConversionPatternRewriterReplaceOpWithMultiple(
    rewriter, op, nRanges, rangeSizes, values
)
    @ccall Reactant_jll.libReactantExtra.mlirConversionPatternRewriterReplaceOpWithMultiple(
        rewriter::MlirConversionPatternRewriter,
        op::MlirOperation,
        nRanges::Cptrdiff_t,
        rangeSizes::Ptr{Cptrdiff_t},
        values::Ptr{MlirValue},
    )::Cvoid
end

"""
    mlirConversionTargetCreate(context)

Create an empty ConversionTarget.
"""
function mlirConversionTargetCreate(context)
    @ccall Reactant_jll.libReactantExtra.mlirConversionTargetCreate(
        context::MlirContext
    )::MlirConversionTarget
end

"""
    mlirConversionTargetDestroy(target)

Destroy the given ConversionTarget.
"""
function mlirConversionTargetDestroy(target)
    @ccall Reactant_jll.libReactantExtra.mlirConversionTargetDestroy(
        target::MlirConversionTarget
    )::Cvoid
end

"""
    mlirConversionTargetAddLegalOp(target, opName)

Register the given operations as legal.
"""
function mlirConversionTargetAddLegalOp(target, opName)
    @ccall Reactant_jll.libReactantExtra.mlirConversionTargetAddLegalOp(
        target::MlirConversionTarget, opName::MlirStringRef
    )::Cvoid
end

"""
    mlirConversionTargetAddIllegalOp(target, opName)

Register the given operations as illegal.
"""
function mlirConversionTargetAddIllegalOp(target, opName)
    @ccall Reactant_jll.libReactantExtra.mlirConversionTargetAddIllegalOp(
        target::MlirConversionTarget, opName::MlirStringRef
    )::Cvoid
end

"""
    mlirConversionTargetAddLegalDialect(target, dialectName)

Register the operations of the given dialect as legal.
"""
function mlirConversionTargetAddLegalDialect(target, dialectName)
    @ccall Reactant_jll.libReactantExtra.mlirConversionTargetAddLegalDialect(
        target::MlirConversionTarget, dialectName::MlirStringRef
    )::Cvoid
end

"""
    mlirConversionTargetAddIllegalDialect(target, dialectName)

Register the operations of the given dialect as illegal.
"""
function mlirConversionTargetAddIllegalDialect(target, dialectName)
    @ccall Reactant_jll.libReactantExtra.mlirConversionTargetAddIllegalDialect(
        target::MlirConversionTarget, dialectName::MlirStringRef
    )::Cvoid
end

"""
    MlirConversionTargetLegality

Result of a dynamic legality callback.
"""
@cenum MlirConversionTargetLegality::UInt32 begin
    MLIR_CONVERSION_TARGET_LEGALITY_LEGAL = 0x0000000000000000
    MLIR_CONVERSION_TARGET_LEGALITY_ILLEGAL = 0x0000000000000001
    MLIR_CONVERSION_TARGET_LEGALITY_NO_OPINION = 0x0000000000000002
end

# typedef MlirConversionTargetLegality ( * MlirConversionTargetDynamicLegalityCallback ) ( MlirOperation op , void * userData )
"""
Callback for dynamic legality checks. Returns the legality of the given operation instance (see [`MlirConversionTargetLegality`](@ref)).
"""
const MlirConversionTargetDynamicLegalityCallback = Ptr{Cvoid}

"""
    mlirConversionTargetAddDynamicallyLegalOp(target, opName, callback, userData)

Register the given operation as dynamically legal, with a callback to determine per-instance legality. The callback must not be NULL.
"""
function mlirConversionTargetAddDynamicallyLegalOp(target, opName, callback, userData)
    @ccall Reactant_jll.libReactantExtra.mlirConversionTargetAddDynamicallyLegalOp(
        target::MlirConversionTarget,
        opName::MlirStringRef,
        callback::MlirConversionTargetDynamicLegalityCallback,
        userData::Ptr{Cvoid},
    )::Cvoid
end

"""
    mlirConversionTargetAddDynamicallyLegalDialect(target, dialectName, callback, userData)

Register the given dialect as dynamically legal, with a callback to determine per-instance legality for all operations in the dialect. The callback must not be NULL.
"""
function mlirConversionTargetAddDynamicallyLegalDialect(
    target, dialectName, callback, userData
)
    @ccall Reactant_jll.libReactantExtra.mlirConversionTargetAddDynamicallyLegalDialect(
        target::MlirConversionTarget,
        dialectName::MlirStringRef,
        callback::MlirConversionTargetDynamicLegalityCallback,
        userData::Ptr{Cvoid},
    )::Cvoid
end

"""
    mlirConversionTargetMarkOpRecursivelyLegal(target, opName, callback, userData)

Mark the given operation as recursively legal. The optional callback (may be NULL) determines whether a specific instance is recursively legal; a NULL callback marks the operation as unconditionally recursively legal.
"""
function mlirConversionTargetMarkOpRecursivelyLegal(target, opName, callback, userData)
    @ccall Reactant_jll.libReactantExtra.mlirConversionTargetMarkOpRecursivelyLegal(
        target::MlirConversionTarget,
        opName::MlirStringRef,
        callback::MlirConversionTargetDynamicLegalityCallback,
        userData::Ptr{Cvoid},
    )::Cvoid
end

"""
    mlirConversionTargetMarkUnknownOpDynamicallyLegal(target, callback, userData)

Mark unknown operations as dynamically legal, with a callback. The callback must not be NULL.
"""
function mlirConversionTargetMarkUnknownOpDynamicallyLegal(target, callback, userData)
    @ccall Reactant_jll.libReactantExtra.mlirConversionTargetMarkUnknownOpDynamicallyLegal(
        target::MlirConversionTarget,
        callback::MlirConversionTargetDynamicLegalityCallback,
        userData::Ptr{Cvoid},
    )::Cvoid
end

"""
    mlirTypeConverterCreate()

Create a TypeConverter.
"""
function mlirTypeConverterCreate()
    @ccall Reactant_jll.libReactantExtra.mlirTypeConverterCreate()::MlirTypeConverter
end

"""
    mlirTypeConverterDestroy(typeConverter)

Destroy the given TypeConverter.
"""
function mlirTypeConverterDestroy(typeConverter)
    @ccall Reactant_jll.libReactantExtra.mlirTypeConverterDestroy(
        typeConverter::MlirTypeConverter
    )::Cvoid
end

"""
    MlirTypeConverterConversionStatus

Outcome of a type conversion callback. Mirrors the three states of the underlying C++ `std::optional<LogicalResult>` conversion result.
"""
@cenum MlirTypeConverterConversionStatus::UInt32 begin
    MlirTypeConverterConversionStatusSuccess = 0x0000000000000000
    MlirTypeConverterConversionStatusFailure = 0x0000000000000001
    MlirTypeConverterConversionStatusDeclined = 0x0000000000000002
end

# typedef MlirTypeConverterConversionStatus ( * MlirTypeConverterConversionCallback ) ( MlirType type , MlirType * convertedType , void * userData )
"""
Callback type for type conversion functions. On success the callback sets `*convertedType` to the converted type and returns MlirTypeConverterConversionStatusSuccess. Returning MlirTypeConverterConversionStatusDeclined leaves the type unconverted and allows another registered conversion function to be tried; returning MlirTypeConverterConversionStatusFailure fails the conversion without trying any further function.
"""
const MlirTypeConverterConversionCallback = Ptr{Cvoid}

"""
    mlirTypeConverterAddConversion(typeConverter, convertType, userData)

Add a type conversion function to the given TypeConverter.
"""
function mlirTypeConverterAddConversion(typeConverter, convertType, userData)
    @ccall Reactant_jll.libReactantExtra.mlirTypeConverterAddConversion(
        typeConverter::MlirTypeConverter,
        convertType::MlirTypeConverterConversionCallback,
        userData::Ptr{Cvoid},
    )::Cvoid
end

"""
    MlirTypeConverterConversionResults

Opaque accumulator for the result types of a 1:N type conversion. It is passed to a [`MlirTypeConverter1ToNConversionCallback`](@ref), which appends converted types to it via [`mlirTypeConverterConversionResultsAppend`](@ref).
"""
struct MlirTypeConverterConversionResults
    ptr::Ptr{Cvoid}
end

"""
    mlirTypeConverterConversionResultsAppend(results, type)

Append a converted result type to the given 1:N conversion result accumulator.
"""
function mlirTypeConverterConversionResultsAppend(results, type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeConverterConversionResultsAppend(
        results::MlirTypeConverterConversionResults, type::MlirType
    )::Cvoid
end

# typedef MlirTypeConverterConversionStatus ( * MlirTypeConverter1ToNConversionCallback ) ( MlirType type , MlirTypeConverterConversionResults results , void * userData )
"""
Callback type for 1:N type conversion functions. For the given `type`, the callback appends zero or more converted result types to `results` (via [`mlirTypeConverterConversionResultsAppend`](@ref)) and returns a status. On MlirTypeConverterConversionStatusSuccess the appended types make up the conversion: appending a single type is a 1:1 conversion, appending several is a 1:N conversion, and appending none erases the type. Returning MlirTypeConverterConversionStatusDeclined lets another conversion function be tried; MlirTypeConverterConversionStatusFailure fails the conversion without trying another. Any types appended before a non-success status are discarded.
"""
const MlirTypeConverter1ToNConversionCallback = Ptr{Cvoid}

"""
    mlirTypeConverterAdd1ToNConversion(typeConverter, convertType, userData)

Add a 1:N type conversion function to the given TypeConverter.
"""
function mlirTypeConverterAdd1ToNConversion(typeConverter, convertType, userData)
    @ccall Reactant_jll.libReactantExtra.mlirTypeConverterAdd1ToNConversion(
        typeConverter::MlirTypeConverter,
        convertType::MlirTypeConverter1ToNConversionCallback,
        userData::Ptr{Cvoid},
    )::Cvoid
end

"""
    mlirTypeConverterConvertType(typeConverter, type)

Convert the given type using the given TypeConverter. This is the 1:1 convenience form: it returns the single converted type, or a null [`MlirType`](@ref) on failure or if the type converts to anything other than exactly one type (e.g. a 1:N conversion registered via [`mlirTypeConverterAdd1ToNConversion`](@ref), or an erasure to zero types).
"""
function mlirTypeConverterConvertType(typeConverter, type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeConverterConvertType(
        typeConverter::MlirTypeConverter, type::MlirType
    )::MlirType
end

# typedef MlirValue ( * MlirTypeConverterSourceMaterializationCallback ) ( MlirRewriterBase rewriter , MlirType outputType , intptr_t nInputs , MlirValue * inputs , MlirLocation loc , void * userData )
"""
Callback type for source materializations. Given a builder (passed as a rewriter), the desired output type, the input values, and a location, the callback must build a cast-like operation that produces a single value of `outputType` and return it. Returning a null [`MlirValue`](@ref) indicates failure, in which case another registered materialization may be attempted.
"""
const MlirTypeConverterSourceMaterializationCallback = Ptr{Cvoid}

# typedef MlirValue ( * MlirTypeConverterTargetMaterializationCallback ) ( MlirRewriterBase rewriter , MlirType outputType , intptr_t nInputs , MlirValue * inputs , MlirLocation loc , MlirType originalType , void * userData )
"""
Callback type for 1:1 target materializations. Behaves like [`MlirTypeConverterSourceMaterializationCallback`](@ref), but additionally receives `originalType`: the original type of the SSA value being materialized.

Note: This callback is single-output. For the 1:N (multiple-output) form, use [`MlirTypeConverter1ToNTargetMaterializationCallback`](@ref).
"""
const MlirTypeConverterTargetMaterializationCallback = Ptr{Cvoid}

"""
    mlirTypeConverterAddSourceMaterialization(typeConverter, callback, userData)

Register a source materialization with the given TypeConverter. This is invoked when a replacement value must be converted back to its original source type because some uses persist beyond the main conversion.
"""
function mlirTypeConverterAddSourceMaterialization(typeConverter, callback, userData)
    @ccall Reactant_jll.libReactantExtra.mlirTypeConverterAddSourceMaterialization(
        typeConverter::MlirTypeConverter,
        callback::MlirTypeConverterSourceMaterializationCallback,
        userData::Ptr{Cvoid},
    )::Cvoid
end

"""
    mlirTypeConverterAddTargetMaterialization(typeConverter, callback, userData)

Register a target materialization with the given TypeConverter. This is invoked when a value must be converted to a target type according to a pattern's type converter.
"""
function mlirTypeConverterAddTargetMaterialization(typeConverter, callback, userData)
    @ccall Reactant_jll.libReactantExtra.mlirTypeConverterAddTargetMaterialization(
        typeConverter::MlirTypeConverter,
        callback::MlirTypeConverterTargetMaterializationCallback,
        userData::Ptr{Cvoid},
    )::Cvoid
end

# typedef MlirLogicalResult ( * MlirTypeConverter1ToNTargetMaterializationCallback ) ( MlirRewriterBase rewriter , intptr_t nOutputTypes , MlirType * outputTypes , intptr_t nInputs , MlirValue * inputs , MlirLocation loc , MlirType originalType , MlirValue * outputs , void * userData )
"""
Callback type for 1:N target materializations. Like [`MlirTypeConverterTargetMaterializationCallback`](@ref), but produces a value for each of the `nOutputTypes` requested output types instead of a single value. On success the callback must fill `outputs` -- a caller-allocated array of length `nOutputTypes` -- with that many non-null values; succeeding while leaving any entry null asserts. Returning failure signals that this materialization declined (so another may be attempted); in that case `outputs` is ignored. `originalType` carries the original type of the value being materialized and may be a null [`MlirType`](@ref).
"""
const MlirTypeConverter1ToNTargetMaterializationCallback = Ptr{Cvoid}

"""
    mlirTypeConverterAdd1ToNTargetMaterialization(typeConverter, callback, userData)

Register a 1:N target materialization with the given TypeConverter.
"""
function mlirTypeConverterAdd1ToNTargetMaterialization(typeConverter, callback, userData)
    @ccall Reactant_jll.libReactantExtra.mlirTypeConverterAdd1ToNTargetMaterialization(
        typeConverter::MlirTypeConverter,
        callback::MlirTypeConverter1ToNTargetMaterializationCallback,
        userData::Ptr{Cvoid},
    )::Cvoid
end

"""
    MlirConversionPatternCallbacks

ConversionPattern API

| Field               | Note                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| :------------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| construct           | Optional constructor for the user data. Set to nullptr to disable it.                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| destruct            | Optional destructor for the user data. Set to nullptr to disable it.                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| matchAndRewrite     | The callback function to match against code rooted at the specified operation, and perform the conversion rewrite if the match is successful, corresponding to ConversionPattern::matchAndRewrite.                                                                                                                                                                                                                                                                                                                                            |
| matchAndRewrite1ToN | Optional callback corresponding to the 1:N ConversionPattern::matchAndRewrite([`Operation`](@ref) *, ArrayRef<ValueRange>, ...) overload, used when one or more operands are remapped to several values (e.g. under a 1:N type conversion). `operands` is the flat concatenation of all operand ranges; there are `nRanges` ranges (one per original operand) and `rangeSizes[i]` is the number of values in the i-th range. When this is non-null it takes precedence; when null, the driver falls back to the 1:1 `matchAndRewrite` above.  |
"""
struct MlirConversionPatternCallbacks
    construct::Ptr{Cvoid}
    destruct::Ptr{Cvoid}
    matchAndRewrite::Ptr{Cvoid}
    matchAndRewrite1ToN::Ptr{Cvoid}
end

"""
    mlirOpConversionPatternCreate(rootName, benefit, context, typeConverter, callbacks, userData, nGeneratedNames, generatedNames)

Create a conversion pattern that matches the operation with the given rootName, corresponding to mlir::OpConversionPattern.
"""
function mlirOpConversionPatternCreate(
    rootName,
    benefit,
    context,
    typeConverter,
    callbacks,
    userData,
    nGeneratedNames,
    generatedNames,
)
    @ccall Reactant_jll.libReactantExtra.mlirOpConversionPatternCreate(
        rootName::MlirStringRef,
        benefit::Cuint,
        context::MlirContext,
        typeConverter::MlirTypeConverter,
        callbacks::MlirConversionPatternCallbacks,
        userData::Ptr{Cvoid},
        nGeneratedNames::Csize_t,
        generatedNames::Ptr{MlirStringRef},
    )::MlirConversionPattern
end

"""
    mlirConversionPatternGetTypeConverter(pattern)

Get the type converter used by this conversion pattern.
"""
function mlirConversionPatternGetTypeConverter(pattern)
    @ccall Reactant_jll.libReactantExtra.mlirConversionPatternGetTypeConverter(
        pattern::MlirConversionPattern
    )::MlirTypeConverter
end

"""
    mlirConversionPatternAsRewritePattern(pattern)

Cast the ConversionPattern to a RewritePattern.
"""
function mlirConversionPatternAsRewritePattern(pattern)
    @ccall Reactant_jll.libReactantExtra.mlirConversionPatternAsRewritePattern(
        pattern::MlirConversionPattern
    )::MlirRewritePattern
end

"""
    MlirRewritePatternCallbacks

Callbacks to construct a rewrite pattern.

| Field           | Note                                                                                                                                                                                  |
| :-------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| construct       | Optional constructor for the user data. Set to nullptr to disable it.                                                                                                                 |
| destruct        | Optional destructor for the user data. Set to nullptr to disable it.                                                                                                                  |
| matchAndRewrite | The callback function to match against code rooted at the specified operation, and perform the rewrite if the match is successful, corresponding to RewritePattern::matchAndRewrite.  |
"""
struct MlirRewritePatternCallbacks
    construct::Ptr{Cvoid}
    destruct::Ptr{Cvoid}
    matchAndRewrite::Ptr{Cvoid}
end

"""
    mlirOpRewritePatternCreate(rootName, benefit, context, callbacks, userData, nGeneratedNames, generatedNames)

Create a rewrite pattern that matches the operation with the given rootName, corresponding to mlir::OpRewritePattern.
"""
function mlirOpRewritePatternCreate(
    rootName, benefit, context, callbacks, userData, nGeneratedNames, generatedNames
)
    @ccall Reactant_jll.libReactantExtra.mlirOpRewritePatternCreate(
        rootName::MlirStringRef,
        benefit::Cuint,
        context::MlirContext,
        callbacks::MlirRewritePatternCallbacks,
        userData::Ptr{Cvoid},
        nGeneratedNames::Csize_t,
        generatedNames::Ptr{MlirStringRef},
    )::MlirRewritePattern
end

"""
    mlirRewritePatternSetCreate(context)

Create an empty [`MlirRewritePatternSet`](@ref).
"""
function mlirRewritePatternSetCreate(context)
    @ccall Reactant_jll.libReactantExtra.mlirRewritePatternSetCreate(
        context::MlirContext
    )::MlirRewritePatternSet
end

"""
    mlirRewritePatternSetGetContext(set)

Get the context associated with a [`MlirRewritePatternSet`](@ref).
"""
function mlirRewritePatternSetGetContext(set)
    @ccall Reactant_jll.libReactantExtra.mlirRewritePatternSetGetContext(
        set::MlirRewritePatternSet
    )::MlirContext
end

"""
    mlirRewritePatternSetDestroy(set)

Destruct the given [`MlirRewritePatternSet`](@ref).
"""
function mlirRewritePatternSetDestroy(set)
    @ccall Reactant_jll.libReactantExtra.mlirRewritePatternSetDestroy(
        set::MlirRewritePatternSet
    )::Cvoid
end

"""
    mlirRewritePatternSetAdd(set, pattern)

Add the given [`MlirRewritePattern`](@ref) into a [`MlirRewritePatternSet`](@ref). Note that the ownership of the pattern is transferred to the set after this call.
"""
function mlirRewritePatternSetAdd(set, pattern)
    @ccall Reactant_jll.libReactantExtra.mlirRewritePatternSetAdd(
        set::MlirRewritePatternSet, pattern::MlirRewritePattern
    )::Cvoid
end

function mlirGetDialectHandle__transform__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__transform__()::MlirDialectHandle
end

struct MlirTransformResults
    ptr::Ptr{Cvoid}
end

struct MlirTransformRewriter
    ptr::Ptr{Cvoid}
end

struct MlirTransformState
    ptr::Ptr{Cvoid}
end

"""
    MlirDiagnosedSilenceableFailure

Enum representing the result of a transform operation.
"""
@cenum MlirDiagnosedSilenceableFailure::UInt32 begin
    MlirDiagnosedSilenceableFailureSuccess = 0x0000000000000000
    MlirDiagnosedSilenceableFailureSilenceableFailure = 0x0000000000000001
    MlirDiagnosedSilenceableFailureDefiniteFailure = 0x0000000000000002
end

function mlirTypeIsATransformAnyOpType(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsATransformAnyOpType(type::MlirType)::Bool
end

function mlirTransformAnyOpTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirTransformAnyOpTypeGetTypeID()::MlirTypeID
end

function mlirTransformAnyOpTypeGet(ctx)
    @ccall Reactant_jll.libReactantExtra.mlirTransformAnyOpTypeGet(
        ctx::MlirContext
    )::MlirType
end

function mlirTransformAnyOpTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirTransformAnyOpTypeGetName()::MlirStringRef
end

function mlirTypeIsATransformAnyParamType(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsATransformAnyParamType(
        type::MlirType
    )::Bool
end

function mlirTransformAnyParamTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirTransformAnyParamTypeGetTypeID()::MlirTypeID
end

function mlirTransformAnyParamTypeGet(ctx)
    @ccall Reactant_jll.libReactantExtra.mlirTransformAnyParamTypeGet(
        ctx::MlirContext
    )::MlirType
end

function mlirTransformAnyParamTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirTransformAnyParamTypeGetName()::MlirStringRef
end

function mlirTypeIsATransformAnyValueType(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsATransformAnyValueType(
        type::MlirType
    )::Bool
end

function mlirTransformAnyValueTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirTransformAnyValueTypeGetTypeID()::MlirTypeID
end

function mlirTransformAnyValueTypeGet(ctx)
    @ccall Reactant_jll.libReactantExtra.mlirTransformAnyValueTypeGet(
        ctx::MlirContext
    )::MlirType
end

function mlirTransformAnyValueTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirTransformAnyValueTypeGetName()::MlirStringRef
end

function mlirTypeIsATransformOperationType(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsATransformOperationType(
        type::MlirType
    )::Bool
end

function mlirTransformOperationTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirTransformOperationTypeGetTypeID()::MlirTypeID
end

function mlirTransformOperationTypeGet(ctx, operationName)
    @ccall Reactant_jll.libReactantExtra.mlirTransformOperationTypeGet(
        ctx::MlirContext, operationName::MlirStringRef
    )::MlirType
end

function mlirTransformOperationTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirTransformOperationTypeGetName()::MlirStringRef
end

function mlirTransformOperationTypeGetOperationName(type)
    @ccall Reactant_jll.libReactantExtra.mlirTransformOperationTypeGetOperationName(
        type::MlirType
    )::MlirStringRef
end

function mlirTypeIsATransformParamType(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsATransformParamType(type::MlirType)::Bool
end

function mlirTransformParamTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirTransformParamTypeGetTypeID()::MlirTypeID
end

function mlirTransformParamTypeGet(ctx, type)
    @ccall Reactant_jll.libReactantExtra.mlirTransformParamTypeGet(
        ctx::MlirContext, type::MlirType
    )::MlirType
end

function mlirTransformParamTypeGetName()
    @ccall Reactant_jll.libReactantExtra.mlirTransformParamTypeGetName()::MlirStringRef
end

function mlirTransformParamTypeGetType(type)
    @ccall Reactant_jll.libReactantExtra.mlirTransformParamTypeGetType(
        type::MlirType
    )::MlirType
end

"""
    mlirTransformRewriterAsBase(rewriter)

Cast the TransformRewriter to a RewriterBase
"""
function mlirTransformRewriterAsBase(rewriter)
    @ccall Reactant_jll.libReactantExtra.mlirTransformRewriterAsBase(
        rewriter::MlirTransformRewriter
    )::MlirRewriterBase
end

"""
    mlirTransformResultsSetOps(results, result, numOps, ops)

Set the payload operations for a transform result by iterating over a list.
"""
function mlirTransformResultsSetOps(results, result, numOps, ops)
    @ccall Reactant_jll.libReactantExtra.mlirTransformResultsSetOps(
        results::MlirTransformResults,
        result::MlirValue,
        numOps::Cptrdiff_t,
        ops::Ptr{MlirOperation},
    )::Cvoid
end

"""
    mlirTransformResultsSetValues(results, result, numValues, values)

Set the payload values for a transform result by iterating over a list.
"""
function mlirTransformResultsSetValues(results, result, numValues, values)
    @ccall Reactant_jll.libReactantExtra.mlirTransformResultsSetValues(
        results::MlirTransformResults,
        result::MlirValue,
        numValues::Cptrdiff_t,
        values::Ptr{MlirValue},
    )::Cvoid
end

"""
    mlirTransformResultsSetParams(results, result, numParams, params)

Set the parameters for a transform result by iterating over a list.
"""
function mlirTransformResultsSetParams(results, result, numParams, params)
    @ccall Reactant_jll.libReactantExtra.mlirTransformResultsSetParams(
        results::MlirTransformResults,
        result::MlirValue,
        numParams::Cptrdiff_t,
        params::Ptr{MlirAttribute},
    )::Cvoid
end

# typedef void ( * MlirOperationCallback ) ( MlirOperation , void * userData )
"""
Callback for iterating over payload operations.
"""
const MlirOperationCallback = Ptr{Cvoid}

"""
    mlirTransformStateForEachPayloadOp(state, value, callback, userData)

Iterate over payload operations associated with the transform IR value. Calls the callback for each payload operation.
"""
function mlirTransformStateForEachPayloadOp(state, value, callback, userData)
    @ccall Reactant_jll.libReactantExtra.mlirTransformStateForEachPayloadOp(
        state::MlirTransformState,
        value::MlirValue,
        callback::MlirOperationCallback,
        userData::Ptr{Cvoid},
    )::Cvoid
end

# typedef void ( * MlirValueCallback ) ( MlirValue , void * userData )
"""
Callback for iterating over payload values.
"""
const MlirValueCallback = Ptr{Cvoid}

"""
    mlirTransformStateForEachPayloadValue(state, value, callback, userData)

Iterate over payload values associated with the transform IR value. Calls the callback for each payload value.
"""
function mlirTransformStateForEachPayloadValue(state, value, callback, userData)
    @ccall Reactant_jll.libReactantExtra.mlirTransformStateForEachPayloadValue(
        state::MlirTransformState,
        value::MlirValue,
        callback::MlirValueCallback,
        userData::Ptr{Cvoid},
    )::Cvoid
end

# typedef void ( * MlirAttributeCallback ) ( MlirAttribute , void * userData )
"""
Callback for iterating over parameters.
"""
const MlirAttributeCallback = Ptr{Cvoid}

"""
    mlirTransformStateForEachParam(state, value, callback, userData)

Iterate over parameters associated with the transform IR value. Calls the callback for each parameter.
"""
function mlirTransformStateForEachParam(state, value, callback, userData)
    @ccall Reactant_jll.libReactantExtra.mlirTransformStateForEachParam(
        state::MlirTransformState,
        value::MlirValue,
        callback::MlirAttributeCallback,
        userData::Ptr{Cvoid},
    )::Cvoid
end

"""
    mlirTransformOpInterfaceTypeID()

Returns the interface TypeID of the TransformOpInterface.
"""
function mlirTransformOpInterfaceTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirTransformOpInterfaceTypeID()::MlirTypeID
end

"""
    MlirTransformOpInterfaceCallbacks

Callbacks for implementing TransformOpInterface from external code.

| Field                        | Note                                                                   |
| :--------------------------- | :--------------------------------------------------------------------- |
| construct                    | Optional constructor for the user data. Set to nullptr to disable it.  |
| destruct                     | Optional destructor for the user data. Set to nullptr to disable it.   |
| apply                        | Apply callback that implements the transformation.                     |
| allowsRepeatedHandleOperands | Callback to check if repeated handle operands are allowed.             |
"""
struct MlirTransformOpInterfaceCallbacks
    construct::Ptr{Cvoid}
    destruct::Ptr{Cvoid}
    apply::Ptr{Cvoid}
    allowsRepeatedHandleOperands::Ptr{Cvoid}
    userData::Ptr{Cvoid}
end

"""
    mlirTransformOpInterfaceAttachFallbackModel(ctx, opName, callbacks)

Attach TransformOpInterface to the operation with the given name using the provided callbacks.
"""
function mlirTransformOpInterfaceAttachFallbackModel(ctx, opName, callbacks)
    @ccall Reactant_jll.libReactantExtra.mlirTransformOpInterfaceAttachFallbackModel(
        ctx::MlirContext,
        opName::MlirStringRef,
        callbacks::MlirTransformOpInterfaceCallbacks,
    )::Cvoid
end

"""
    mlirPatternDescriptorOpInterfaceTypeID()

Returns the interface TypeID of the PatternDescriptorOpInterface.
"""
function mlirPatternDescriptorOpInterfaceTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirPatternDescriptorOpInterfaceTypeID()::MlirTypeID
end

"""
    MlirPatternDescriptorOpInterfaceCallbacks

Callbacks for implementing PatternDescriptorOpInterface from external code.

| Field                     | Note                                                                                                                                             |
| :------------------------ | :----------------------------------------------------------------------------------------------------------------------------------------------- |
| construct                 | Optional constructor for the user data. Set to nullptr to disable it.                                                                            |
| destruct                  | Optional destructor for the user data. Set to nullptr to disable it.                                                                             |
| populatePatterns          | Callback to populate rewrite patterns into the given pattern set.                                                                                |
| populatePatternsWithState | Optional callback to populate rewrite patterns with transform state. Set to nullptr to use the default implementation (calls populatePatterns).  |
"""
struct MlirPatternDescriptorOpInterfaceCallbacks
    construct::Ptr{Cvoid}
    destruct::Ptr{Cvoid}
    populatePatterns::Ptr{Cvoid}
    populatePatternsWithState::Ptr{Cvoid}
    userData::Ptr{Cvoid}
end

"""
    mlirPatternDescriptorOpInterfaceAttachFallbackModel(ctx, opName, callbacks)

Attach PatternDescriptorOpInterface to the operation with the given name using the provided callbacks.
"""
function mlirPatternDescriptorOpInterfaceAttachFallbackModel(ctx, opName, callbacks)
    @ccall Reactant_jll.libReactantExtra.mlirPatternDescriptorOpInterfaceAttachFallbackModel(
        ctx::MlirContext,
        opName::MlirStringRef,
        callbacks::MlirPatternDescriptorOpInterfaceCallbacks,
    )::Cvoid
end

"""
    mlirTransformOnlyReadsHandle(operands, numOperands, effects)

Helper to mark operands as only reading handles.
"""
function mlirTransformOnlyReadsHandle(operands, numOperands, effects)
    @ccall Reactant_jll.libReactantExtra.mlirTransformOnlyReadsHandle(
        operands::Ptr{MlirOpOperand},
        numOperands::Cptrdiff_t,
        effects::MlirMemoryEffectInstancesList,
    )::Cvoid
end

"""
    mlirTransformConsumesHandle(operands, numOperands, effects)

Helper to mark operands as consuming handles.
"""
function mlirTransformConsumesHandle(operands, numOperands, effects)
    @ccall Reactant_jll.libReactantExtra.mlirTransformConsumesHandle(
        operands::Ptr{MlirOpOperand},
        numOperands::Cptrdiff_t,
        effects::MlirMemoryEffectInstancesList,
    )::Cvoid
end

"""
    mlirTransformProducesHandle(results, numResults, effects)

Helper to mark results as producing handles.
"""
function mlirTransformProducesHandle(results, numResults, effects)
    @ccall Reactant_jll.libReactantExtra.mlirTransformProducesHandle(
        results::Ptr{MlirValue},
        numResults::Cptrdiff_t,
        effects::MlirMemoryEffectInstancesList,
    )::Cvoid
end

"""
    mlirTransformModifiesPayload(effects)

Helper to mark potential modifications to the payload IR.
"""
function mlirTransformModifiesPayload(effects)
    @ccall Reactant_jll.libReactantExtra.mlirTransformModifiesPayload(
        effects::MlirMemoryEffectInstancesList
    )::Cvoid
end

"""
    mlirTransformOnlyReadsPayload(effects)

Helper to mark potential reads from the payload IR.
"""
function mlirTransformOnlyReadsPayload(effects)
    @ccall Reactant_jll.libReactantExtra.mlirTransformOnlyReadsPayload(
        effects::MlirMemoryEffectInstancesList
    )::Cvoid
end

struct MlirTransformOptions
    ptr::Ptr{Cvoid}
end

"""
    mlirTransformOptionsCreate()

Creates a default-initialized transform options object.
"""
function mlirTransformOptionsCreate()
    @ccall Reactant_jll.libReactantExtra.mlirTransformOptionsCreate()::MlirTransformOptions
end

"""
    mlirTransformOptionsEnableExpensiveChecks(transformOptions, enable)

Enables or disables expensive checks in transform options.
"""
function mlirTransformOptionsEnableExpensiveChecks(transformOptions, enable)
    @ccall Reactant_jll.libReactantExtra.mlirTransformOptionsEnableExpensiveChecks(
        transformOptions::MlirTransformOptions, enable::Bool
    )::Cvoid
end

"""
    mlirTransformOptionsGetExpensiveChecksEnabled(transformOptions)

Returns true if expensive checks are enabled in transform options.
"""
function mlirTransformOptionsGetExpensiveChecksEnabled(transformOptions)
    @ccall Reactant_jll.libReactantExtra.mlirTransformOptionsGetExpensiveChecksEnabled(
        transformOptions::MlirTransformOptions
    )::Bool
end

"""
    mlirTransformOptionsEnforceSingleTopLevelTransformOp(transformOptions, enable)

Enables or disables the enforcement of the top-level transform op being single in transform options.
"""
function mlirTransformOptionsEnforceSingleTopLevelTransformOp(transformOptions, enable)
    @ccall Reactant_jll.libReactantExtra.mlirTransformOptionsEnforceSingleTopLevelTransformOp(
        transformOptions::MlirTransformOptions, enable::Bool
    )::Cvoid
end

"""
    mlirTransformOptionsGetEnforceSingleTopLevelTransformOp(transformOptions)

Returns true if the enforcement of the top-level transform op being single is enabled in transform options.
"""
function mlirTransformOptionsGetEnforceSingleTopLevelTransformOp(transformOptions)
    @ccall Reactant_jll.libReactantExtra.mlirTransformOptionsGetEnforceSingleTopLevelTransformOp(
        transformOptions::MlirTransformOptions
    )::Bool
end

"""
    mlirTransformOptionsDestroy(transformOptions)

Destroys a transform options object previously created by [`mlirTransformOptionsCreate`](@ref).
"""
function mlirTransformOptionsDestroy(transformOptions)
    @ccall Reactant_jll.libReactantExtra.mlirTransformOptionsDestroy(
        transformOptions::MlirTransformOptions
    )::Cvoid
end

"""
    mlirTransformApplyNamedSequence(payload, transformRoot, transformModule, transformOptions)

Applies the transformation script starting at the given transform root operation to the given payload operation. The module containing the transform root as well as the transform options should be provided. The transform operation must implement TransformOpInterface and the module must be a ModuleOp. Returns the status of the application.
"""
function mlirTransformApplyNamedSequence(
    payload, transformRoot, transformModule, transformOptions
)
    @ccall Reactant_jll.libReactantExtra.mlirTransformApplyNamedSequence(
        payload::MlirOperation,
        transformRoot::MlirOperation,
        transformModule::MlirOperation,
        transformOptions::MlirTransformOptions,
    )::MlirLogicalResult
end

"""
    mlirMergeSymbolsIntoFromClone(target, other)

Merge the symbols from `other` into `target`, potentially renaming them to avoid conflicts. Private symbols may be renamed during the merge, public symbols must have at most one declaration. A name conflict in public symbols is reported as an error before returning a failure.

Note that this clones the `other` operation unlike the C++ counterpart that takes ownership.
"""
function mlirMergeSymbolsIntoFromClone(target, other)
    @ccall Reactant_jll.libReactantExtra.mlirMergeSymbolsIntoFromClone(
        target::MlirOperation, other::MlirOperation
    )::MlirLogicalResult
end

function mlirGetDialectHandle__ub__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__ub__()::MlirDialectHandle
end

function mlirGetDialectHandle__vcix__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__vcix__()::MlirDialectHandle
end

function mlirGetDialectHandle__vector__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__vector__()::MlirDialectHandle
end

function mlirGetDialectHandle__wasmssa__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__wasmssa__()::MlirDialectHandle
end

function mlirGetDialectHandle__x86__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__x86__()::MlirDialectHandle
end

function mlirGetDialectHandle__xegpu__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__xegpu__()::MlirDialectHandle
end

function mlirGetDialectHandle__xevm__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__xevm__()::MlirDialectHandle
end

struct MlirDominanceInfo
    ptr::Ptr{Cvoid}
end

struct MlirPostDominanceInfo
    ptr::Ptr{Cvoid}
end

"""
    mlirDominanceInfoCreate(op)

Creates a DominanceInfo for the given operation (typically a FuncOp or ModuleOp). The caller owns the returned object and must destroy it.
"""
function mlirDominanceInfoCreate(op)
    @ccall Reactant_jll.libReactantExtra.mlirDominanceInfoCreate(
        op::MlirOperation
    )::MlirDominanceInfo
end

"""
    mlirDominanceInfoDestroy(info)

Destroys the given DominanceInfo.
"""
function mlirDominanceInfoDestroy(info)
    @ccall Reactant_jll.libReactantExtra.mlirDominanceInfoDestroy(
        info::MlirDominanceInfo
    )::Cvoid
end

"""
    mlirDominanceInfoProperlyDominatesOperation(info, a, b)

Returns true if operation A properly dominates operation B.
"""
function mlirDominanceInfoProperlyDominatesOperation(info, a, b)
    @ccall Reactant_jll.libReactantExtra.mlirDominanceInfoProperlyDominatesOperation(
        info::MlirDominanceInfo, a::MlirOperation, b::MlirOperation
    )::Bool
end

"""
    mlirDominanceInfoDominatesOperation(info, a, b)

Returns true if operation A dominates operation B (A == B or A properly dominates B).
"""
function mlirDominanceInfoDominatesOperation(info, a, b)
    @ccall Reactant_jll.libReactantExtra.mlirDominanceInfoDominatesOperation(
        info::MlirDominanceInfo, a::MlirOperation, b::MlirOperation
    )::Bool
end

"""
    mlirDominanceInfoValueProperlyDominates(info, a, b)

Returns true if value A properly dominates operation B.
"""
function mlirDominanceInfoValueProperlyDominates(info, a, b)
    @ccall Reactant_jll.libReactantExtra.mlirDominanceInfoValueProperlyDominates(
        info::MlirDominanceInfo, a::MlirValue, b::MlirOperation
    )::Bool
end

"""
    mlirDominanceInfoValueDominates(info, a, b)

Returns true if value A dominates operation B (the operation defining A is B or A properly dominates B).
"""
function mlirDominanceInfoValueDominates(info, a, b)
    @ccall Reactant_jll.libReactantExtra.mlirDominanceInfoValueDominates(
        info::MlirDominanceInfo, a::MlirValue, b::MlirOperation
    )::Bool
end

"""
    mlirDominanceInfoProperlyDominatesBlock(info, a, b)

Returns true if block A properly dominates block B.
"""
function mlirDominanceInfoProperlyDominatesBlock(info, a, b)
    @ccall Reactant_jll.libReactantExtra.mlirDominanceInfoProperlyDominatesBlock(
        info::MlirDominanceInfo, a::MlirBlock, b::MlirBlock
    )::Bool
end

"""
    mlirDominanceInfoDominatesBlock(info, a, b)

Returns true if block A dominates block B.
"""
function mlirDominanceInfoDominatesBlock(info, a, b)
    @ccall Reactant_jll.libReactantExtra.mlirDominanceInfoDominatesBlock(
        info::MlirDominanceInfo, a::MlirBlock, b::MlirBlock
    )::Bool
end

"""
    mlirDominanceInfoFindNearestCommonDominator(info, a, b)

Finds the nearest common dominator of blocks A and B. Returns a null block if none exists.
"""
function mlirDominanceInfoFindNearestCommonDominator(info, a, b)
    @ccall Reactant_jll.libReactantExtra.mlirDominanceInfoFindNearestCommonDominator(
        info::MlirDominanceInfo, a::MlirBlock, b::MlirBlock
    )::MlirBlock
end

"""
    mlirDominanceInfoIsReachableFromEntry(info, block)

Returns true if the given block is reachable from the entry block of its region.
"""
function mlirDominanceInfoIsReachableFromEntry(info, block)
    @ccall Reactant_jll.libReactantExtra.mlirDominanceInfoIsReachableFromEntry(
        info::MlirDominanceInfo, block::MlirBlock
    )::Bool
end

"""
    mlirDominanceInfoInvalidate(info)

Invalidates all cached dominance information.
"""
function mlirDominanceInfoInvalidate(info)
    @ccall Reactant_jll.libReactantExtra.mlirDominanceInfoInvalidate(
        info::MlirDominanceInfo
    )::Cvoid
end

"""
    mlirPostDominanceInfoCreate(op)

Creates a PostDominanceInfo for the given operation.
"""
function mlirPostDominanceInfoCreate(op)
    @ccall Reactant_jll.libReactantExtra.mlirPostDominanceInfoCreate(
        op::MlirOperation
    )::MlirPostDominanceInfo
end

"""
    mlirPostDominanceInfoDestroy(info)

Destroys the given PostDominanceInfo.
"""
function mlirPostDominanceInfoDestroy(info)
    @ccall Reactant_jll.libReactantExtra.mlirPostDominanceInfoDestroy(
        info::MlirPostDominanceInfo
    )::Cvoid
end

"""
    mlirPostDominanceInfoProperlyPostDominatesOperation(info, a, b)

Returns true if operation A properly post-dominates operation B.
"""
function mlirPostDominanceInfoProperlyPostDominatesOperation(info, a, b)
    @ccall Reactant_jll.libReactantExtra.mlirPostDominanceInfoProperlyPostDominatesOperation(
        info::MlirPostDominanceInfo, a::MlirOperation, b::MlirOperation
    )::Bool
end

"""
    mlirPostDominanceInfoPostDominatesOperation(info, a, b)

Returns true if operation A post-dominates operation B.
"""
function mlirPostDominanceInfoPostDominatesOperation(info, a, b)
    @ccall Reactant_jll.libReactantExtra.mlirPostDominanceInfoPostDominatesOperation(
        info::MlirPostDominanceInfo, a::MlirOperation, b::MlirOperation
    )::Bool
end

"""
    mlirPostDominanceInfoProperlyPostDominatesBlock(info, a, b)

Returns true if block A properly post-dominates block B.
"""
function mlirPostDominanceInfoProperlyPostDominatesBlock(info, a, b)
    @ccall Reactant_jll.libReactantExtra.mlirPostDominanceInfoProperlyPostDominatesBlock(
        info::MlirPostDominanceInfo, a::MlirBlock, b::MlirBlock
    )::Bool
end

"""
    mlirPostDominanceInfoPostDominatesBlock(info, a, b)

Returns true if block A post-dominates block B.
"""
function mlirPostDominanceInfoPostDominatesBlock(info, a, b)
    @ccall Reactant_jll.libReactantExtra.mlirPostDominanceInfoPostDominatesBlock(
        info::MlirPostDominanceInfo, a::MlirBlock, b::MlirBlock
    )::Bool
end

"""
    mlirPostDominanceInfoInvalidate(info)

Invalidates all cached post-dominance information.
"""
function mlirPostDominanceInfoInvalidate(info)
    @ccall Reactant_jll.libReactantExtra.mlirPostDominanceInfoInvalidate(
        info::MlirPostDominanceInfo
    )::Cvoid
end

struct MlirExecutionEngine
    ptr::Ptr{Cvoid}
end

"""
    mlirExecutionEngineCreate(op, optLevel, numPaths, sharedLibPaths, enableObjectDump, enablePIC)

Creates an ExecutionEngine for the provided ModuleOp. The ModuleOp is expected to be "translatable" to LLVM IR (only contains operations in dialects that implement the `LLVMTranslationDialectInterface`). The module ownership stays with the client and can be destroyed as soon as the call returns. `optLevel` is the optimization level to be used for transformation and code generation. LLVM passes at `optLevel` are run before code generation. The number and array of paths corresponding to shared libraries that will be loaded are specified via `numPaths` and `sharedLibPaths` respectively. The `enablePIC` arguments controls the relocation model, when true the generated code is emitted as "position independent", making it possible to save it and reload it as a shared object in another process. TODO: figure out other options.
"""
function mlirExecutionEngineCreate(
    op, optLevel, numPaths, sharedLibPaths, enableObjectDump, enablePIC
)
    @ccall Reactant_jll.libReactantExtra.mlirExecutionEngineCreate(
        op::MlirModule,
        optLevel::Cint,
        numPaths::Cint,
        sharedLibPaths::Ptr{MlirStringRef},
        enableObjectDump::Bool,
        enablePIC::Bool,
    )::MlirExecutionEngine
end

"""
    mlirExecutionEngineInitialize(jit)

Initialize the ExecutionEngine. Global constructors specified by `llvm.mlir.global\\_ctors` will be run. One common scenario is that kernel binary compiled from `gpu.module` gets loaded during initialization. Make sure all symbols are resolvable before initialization by calling [`mlirExecutionEngineRegisterSymbol`](@ref) or including shared libraries.
"""
function mlirExecutionEngineInitialize(jit)
    @ccall Reactant_jll.libReactantExtra.mlirExecutionEngineInitialize(
        jit::MlirExecutionEngine
    )::Cvoid
end

"""
    mlirExecutionEngineDestroy(jit)

Destroy an ExecutionEngine instance.
"""
function mlirExecutionEngineDestroy(jit)
    @ccall Reactant_jll.libReactantExtra.mlirExecutionEngineDestroy(
        jit::MlirExecutionEngine
    )::Cvoid
end

"""
    mlirExecutionEngineIsNull(jit)

Checks whether an execution engine is null.
"""
function mlirExecutionEngineIsNull(jit)
    @ccall Reactant_jll.libReactantExtra.mlirExecutionEngineIsNull(
        jit::MlirExecutionEngine
    )::Bool
end

"""
    mlirExecutionEngineInvokePacked(jit, name, arguments)

Invoke a native function in the execution engine by name with the arguments and result of the invoked function passed as an array of pointers. The function must have been tagged with the `llvm.emit\\_c\\_interface` attribute. Returns a failure if the execution fails for any reason (the function name can't be resolved for instance).
"""
function mlirExecutionEngineInvokePacked(jit, name, arguments)
    @ccall Reactant_jll.libReactantExtra.mlirExecutionEngineInvokePacked(
        jit::MlirExecutionEngine, name::MlirStringRef, arguments::Ptr{Ptr{Cvoid}}
    )::MlirLogicalResult
end

"""
    mlirExecutionEngineLookupPacked(jit, name)

Lookup the wrapper of the native function in the execution engine with the given name, returns nullptr if the function can't be looked-up.
"""
function mlirExecutionEngineLookupPacked(jit, name)
    @ccall Reactant_jll.libReactantExtra.mlirExecutionEngineLookupPacked(
        jit::MlirExecutionEngine, name::MlirStringRef
    )::Ptr{Cvoid}
end

"""
    mlirExecutionEngineLookup(jit, name)

Lookup a native function in the execution engine by name, returns nullptr if the name can't be looked-up.
"""
function mlirExecutionEngineLookup(jit, name)
    @ccall Reactant_jll.libReactantExtra.mlirExecutionEngineLookup(
        jit::MlirExecutionEngine, name::MlirStringRef
    )::Ptr{Cvoid}
end

"""
    mlirExecutionEngineRegisterSymbol(jit, name, sym)

Register a symbol with the jit: this symbol will be accessible to the jitted code.
"""
function mlirExecutionEngineRegisterSymbol(jit, name, sym)
    @ccall Reactant_jll.libReactantExtra.mlirExecutionEngineRegisterSymbol(
        jit::MlirExecutionEngine, name::MlirStringRef, sym::Ptr{Cvoid}
    )::Cvoid
end

"""
    mlirExecutionEngineDumpToObjectFile(jit, fileName)

Dump as an object in `fileName`.
"""
function mlirExecutionEngineDumpToObjectFile(jit, fileName)
    @ccall Reactant_jll.libReactantExtra.mlirExecutionEngineDumpToObjectFile(
        jit::MlirExecutionEngine, fileName::MlirStringRef
    )::Cvoid
end

struct MlirDynamicOpTrait
    ptr::Ptr{Cvoid}
end

struct MlirDynamicTypeDefinition
    ptr::Ptr{Cvoid}
end

struct MlirDynamicAttrDefinition
    ptr::Ptr{Cvoid}
end

"""
    mlirDynamicOpTraitAttach(dynamicOpTrait, opName, context)

Attach a dynamic op trait to the given operation name. Note that the operation name must be modeled by dynamic dialect and must be registered. The ownership of the trait will be transferred to the operation name after this call.
"""
function mlirDynamicOpTraitAttach(dynamicOpTrait, opName, context)
    @ccall Reactant_jll.libReactantExtra.mlirDynamicOpTraitAttach(
        dynamicOpTrait::MlirDynamicOpTrait, opName::MlirStringRef, context::MlirContext
    )::Bool
end

"""
    mlirDynamicOpTraitIsTerminatorCreate()

Get the dynamic op trait that indicates the operation is a terminator.
"""
function mlirDynamicOpTraitIsTerminatorCreate()
    @ccall Reactant_jll.libReactantExtra.mlirDynamicOpTraitIsTerminatorCreate()::MlirDynamicOpTrait
end

"""
    mlirDynamicOpTraitIsTerminatorGetTypeID()

Get the type ID of the dynamic op trait that indicates the operation is a terminator.
"""
function mlirDynamicOpTraitIsTerminatorGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirDynamicOpTraitIsTerminatorGetTypeID()::MlirTypeID
end

"""
    mlirDynamicOpTraitIsIsolatedFromAboveCreate()

Get the dynamic op trait that indicates regions are isolated from above.
"""
function mlirDynamicOpTraitIsIsolatedFromAboveCreate()
    @ccall Reactant_jll.libReactantExtra.mlirDynamicOpTraitIsIsolatedFromAboveCreate()::MlirDynamicOpTrait
end

"""
    mlirDynamicOpTraitIsIsolatedFromAboveGetTypeID()

Get the type ID of the dynamic op trait that indicates regions are isolated from above.
"""
function mlirDynamicOpTraitIsIsolatedFromAboveGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirDynamicOpTraitIsIsolatedFromAboveGetTypeID()::MlirTypeID
end

"""
    mlirDynamicOpTraitNoTerminatorCreate()

Get the dynamic op trait that indicates regions have no terminator.
"""
function mlirDynamicOpTraitNoTerminatorCreate()
    @ccall Reactant_jll.libReactantExtra.mlirDynamicOpTraitNoTerminatorCreate()::MlirDynamicOpTrait
end

"""
    mlirDynamicOpTraitNoTerminatorGetTypeID()

Get the type ID of the dynamic op trait that indicates regions have no terminator.
"""
function mlirDynamicOpTraitNoTerminatorGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirDynamicOpTraitNoTerminatorGetTypeID()::MlirTypeID
end

"""
    mlirDynamicOpTraitDestroy(dynamicOpTrait)

Destroy the dynamic op trait.
"""
function mlirDynamicOpTraitDestroy(dynamicOpTrait)
    @ccall Reactant_jll.libReactantExtra.mlirDynamicOpTraitDestroy(
        dynamicOpTrait::MlirDynamicOpTrait
    )::Cvoid
end

"""
    MlirDynamicOpTraitCallbacks

| Field             | Note                                                                   |
| :---------------- | :--------------------------------------------------------------------- |
| construct         | Optional constructor for the user data. Set to nullptr to disable it.  |
| destruct          | Optional destructor for the user data. Set to nullptr to disable it.   |
| verifyTrait       | The callback function to verify the operation.                         |
| verifyRegionTrait | The callback function to verify the operation with access to regions.  |
"""
struct MlirDynamicOpTraitCallbacks
    construct::Ptr{Cvoid}
    destruct::Ptr{Cvoid}
    verifyTrait::Ptr{Cvoid}
    verifyRegionTrait::Ptr{Cvoid}
end

"""
    mlirDynamicOpTraitCreate(typeID, callbacks, userData)

Create a custom dynamic op trait with the given type ID and callbacks.
"""
function mlirDynamicOpTraitCreate(typeID, callbacks, userData)
    @ccall Reactant_jll.libReactantExtra.mlirDynamicOpTraitCreate(
        typeID::MlirTypeID, callbacks::MlirDynamicOpTraitCallbacks, userData::Ptr{Cvoid}
    )::MlirDynamicOpTrait
end

"""
    mlirDialectIsAExtensibleDialect(dialect)

Check if the given dialect is an extensible dialect.
"""
function mlirDialectIsAExtensibleDialect(dialect)
    @ccall Reactant_jll.libReactantExtra.mlirDialectIsAExtensibleDialect(
        dialect::MlirDialect
    )::Bool
end

"""
    mlirExtensibleDialectLookupTypeDefinition(dialect, typeName)

Look up a registered type definition by type name in the given dialect. Note that the dialect must be an extensible dialect.
"""
function mlirExtensibleDialectLookupTypeDefinition(dialect, typeName)
    @ccall Reactant_jll.libReactantExtra.mlirExtensibleDialectLookupTypeDefinition(
        dialect::MlirDialect, typeName::MlirStringRef
    )::MlirDynamicTypeDefinition
end

"""
    mlirTypeIsADynamicType(type)

Check if the given type is a dynamic type.
"""
function mlirTypeIsADynamicType(type)
    @ccall Reactant_jll.libReactantExtra.mlirTypeIsADynamicType(type::MlirType)::Bool
end

"""
    mlirDynamicTypeGet(typeDef, attrs, numAttrs)

Get a dynamic type by instantiating the given type definition with the provided attributes.
"""
function mlirDynamicTypeGet(typeDef, attrs, numAttrs)
    @ccall Reactant_jll.libReactantExtra.mlirDynamicTypeGet(
        typeDef::MlirDynamicTypeDefinition, attrs::Ptr{MlirAttribute}, numAttrs::Cptrdiff_t
    )::MlirType
end

"""
    mlirDynamicTypeGetNumParams(type)

Get the number of parameters in the given dynamic type.
"""
function mlirDynamicTypeGetNumParams(type)
    @ccall Reactant_jll.libReactantExtra.mlirDynamicTypeGetNumParams(
        type::MlirType
    )::Cptrdiff_t
end

"""
    mlirDynamicTypeGetParam(type, index)

Get the parameter at the given index in the provided dynamic type.
"""
function mlirDynamicTypeGetParam(type, index)
    @ccall Reactant_jll.libReactantExtra.mlirDynamicTypeGetParam(
        type::MlirType, index::Cptrdiff_t
    )::MlirAttribute
end

"""
    mlirDynamicTypeGetTypeDef(type)

Get the type definition of the given dynamic type.
"""
function mlirDynamicTypeGetTypeDef(type)
    @ccall Reactant_jll.libReactantExtra.mlirDynamicTypeGetTypeDef(
        type::MlirType
    )::MlirDynamicTypeDefinition
end

"""
    mlirDynamicTypeDefinitionGetTypeID(typeDef)

Get the type ID of a dynamic type definition.
"""
function mlirDynamicTypeDefinitionGetTypeID(typeDef)
    @ccall Reactant_jll.libReactantExtra.mlirDynamicTypeDefinitionGetTypeID(
        typeDef::MlirDynamicTypeDefinition
    )::MlirTypeID
end

"""
    mlirDynamicTypeDefinitionGetName(typeDef)

Get the name of the given dynamic type definition.
"""
function mlirDynamicTypeDefinitionGetName(typeDef)
    @ccall Reactant_jll.libReactantExtra.mlirDynamicTypeDefinitionGetName(
        typeDef::MlirDynamicTypeDefinition
    )::MlirStringRef
end

"""
    mlirDynamicTypeDefinitionGetDialect(typeDef)

Get the dialect that the given dynamic type definition belongs to.
"""
function mlirDynamicTypeDefinitionGetDialect(typeDef)
    @ccall Reactant_jll.libReactantExtra.mlirDynamicTypeDefinitionGetDialect(
        typeDef::MlirDynamicTypeDefinition
    )::MlirDialect
end

"""
    mlirExtensibleDialectLookupAttrDefinition(dialect, attrName)

Look up a registered attribute definition by attribute name in the given dialect. Note that the dialect must be an extensible dialect.
"""
function mlirExtensibleDialectLookupAttrDefinition(dialect, attrName)
    @ccall Reactant_jll.libReactantExtra.mlirExtensibleDialectLookupAttrDefinition(
        dialect::MlirDialect, attrName::MlirStringRef
    )::MlirDynamicAttrDefinition
end

"""
    mlirAttributeIsADynamicAttr(attr)

Check if the given attribute is a dynamic attribute.
"""
function mlirAttributeIsADynamicAttr(attr)
    @ccall Reactant_jll.libReactantExtra.mlirAttributeIsADynamicAttr(
        attr::MlirAttribute
    )::Bool
end

"""
    mlirDynamicAttrGet(attrDef, attrs, numAttrs)

Get a dynamic attribute by instantiating the given attribute definition with the provided attributes.
"""
function mlirDynamicAttrGet(attrDef, attrs, numAttrs)
    @ccall Reactant_jll.libReactantExtra.mlirDynamicAttrGet(
        attrDef::MlirDynamicAttrDefinition, attrs::Ptr{MlirAttribute}, numAttrs::Cptrdiff_t
    )::MlirAttribute
end

"""
    mlirDynamicAttrGetNumParams(attr)

Get the number of parameters in the given dynamic attribute.
"""
function mlirDynamicAttrGetNumParams(attr)
    @ccall Reactant_jll.libReactantExtra.mlirDynamicAttrGetNumParams(
        attr::MlirAttribute
    )::Cptrdiff_t
end

"""
    mlirDynamicAttrGetParam(attr, index)

Get the parameter at the given index in the provided dynamic attribute.
"""
function mlirDynamicAttrGetParam(attr, index)
    @ccall Reactant_jll.libReactantExtra.mlirDynamicAttrGetParam(
        attr::MlirAttribute, index::Cptrdiff_t
    )::MlirAttribute
end

"""
    mlirDynamicAttrGetAttrDef(attr)

Get the attribute definition of the given dynamic attribute.
"""
function mlirDynamicAttrGetAttrDef(attr)
    @ccall Reactant_jll.libReactantExtra.mlirDynamicAttrGetAttrDef(
        attr::MlirAttribute
    )::MlirDynamicAttrDefinition
end

"""
    mlirDynamicAttrDefinitionGetTypeID(attrDef)

Get the type ID of a dynamic attribute definition.
"""
function mlirDynamicAttrDefinitionGetTypeID(attrDef)
    @ccall Reactant_jll.libReactantExtra.mlirDynamicAttrDefinitionGetTypeID(
        attrDef::MlirDynamicAttrDefinition
    )::MlirTypeID
end

"""
    mlirDynamicAttrDefinitionGetName(attrDef)

Get the name of the given dynamic attribute definition.
"""
function mlirDynamicAttrDefinitionGetName(attrDef)
    @ccall Reactant_jll.libReactantExtra.mlirDynamicAttrDefinitionGetName(
        attrDef::MlirDynamicAttrDefinition
    )::MlirStringRef
end

"""
    mlirDynamicAttrDefinitionGetDialect(attrDef)

Get the dialect that the given dynamic attribute definition belongs to.
"""
function mlirDynamicAttrDefinitionGetDialect(attrDef)
    @ccall Reactant_jll.libReactantExtra.mlirDynamicAttrDefinitionGetDialect(
        attrDef::MlirDynamicAttrDefinition
    )::MlirDialect
end

struct MlirPass
    ptr::Ptr{Cvoid}
end

struct MlirExternalPass
    ptr::Ptr{Cvoid}
end

struct MlirPassManager
    ptr::Ptr{Cvoid}
end

struct MlirOpPassManager
    ptr::Ptr{Cvoid}
end

"""
    mlirPassManagerCreate(ctx)

Create a new top-level PassManager with the default anchor.
"""
function mlirPassManagerCreate(ctx)
    @ccall Reactant_jll.libReactantExtra.mlirPassManagerCreate(
        ctx::MlirContext
    )::MlirPassManager
end

"""
    mlirPassManagerCreateOnOperation(ctx, anchorOp)

Create a new top-level PassManager anchored on `anchorOp`.
"""
function mlirPassManagerCreateOnOperation(ctx, anchorOp)
    @ccall Reactant_jll.libReactantExtra.mlirPassManagerCreateOnOperation(
        ctx::MlirContext, anchorOp::MlirStringRef
    )::MlirPassManager
end

"""
    mlirPassManagerDestroy(passManager)

Destroy the provided PassManager.
"""
function mlirPassManagerDestroy(passManager)
    @ccall Reactant_jll.libReactantExtra.mlirPassManagerDestroy(
        passManager::MlirPassManager
    )::Cvoid
end

"""
    mlirPassManagerIsNull(passManager)

Checks if a PassManager is null.
"""
function mlirPassManagerIsNull(passManager)
    @ccall Reactant_jll.libReactantExtra.mlirPassManagerIsNull(
        passManager::MlirPassManager
    )::Bool
end

"""
    mlirPassManagerGetAsOpPassManager(passManager)

Cast a top-level PassManager to a generic OpPassManager.
"""
function mlirPassManagerGetAsOpPassManager(passManager)
    @ccall Reactant_jll.libReactantExtra.mlirPassManagerGetAsOpPassManager(
        passManager::MlirPassManager
    )::MlirOpPassManager
end

"""
    mlirPassManagerRunOnOp(passManager, op)

Run the provided `passManager` on the given `op`.
"""
function mlirPassManagerRunOnOp(passManager, op)
    @ccall Reactant_jll.libReactantExtra.mlirPassManagerRunOnOp(
        passManager::MlirPassManager, op::MlirOperation
    )::MlirLogicalResult
end

"""
    mlirPassManagerEnableIRPrinting(passManager, printBeforeAll, printAfterAll, printModuleScope, printAfterOnlyOnChange, printAfterOnlyOnFailure, flags, treePrintingPath)

Enable IR printing. The treePrintingPath argument is an optional path to a directory where the dumps will be produced. If it isn't provided then dumps are produced to stderr.
"""
function mlirPassManagerEnableIRPrinting(
    passManager,
    printBeforeAll,
    printAfterAll,
    printModuleScope,
    printAfterOnlyOnChange,
    printAfterOnlyOnFailure,
    flags,
    treePrintingPath,
)
    @ccall Reactant_jll.libReactantExtra.mlirPassManagerEnableIRPrinting(
        passManager::MlirPassManager,
        printBeforeAll::Bool,
        printAfterAll::Bool,
        printModuleScope::Bool,
        printAfterOnlyOnChange::Bool,
        printAfterOnlyOnFailure::Bool,
        flags::MlirOpPrintingFlags,
        treePrintingPath::MlirStringRef,
    )::Cvoid
end

"""
    mlirPassManagerEnableVerifier(passManager, enable)

Enable / disable verify-each.
"""
function mlirPassManagerEnableVerifier(passManager, enable)
    @ccall Reactant_jll.libReactantExtra.mlirPassManagerEnableVerifier(
        passManager::MlirPassManager, enable::Bool
    )::Cvoid
end

"""
    mlirPassManagerEnableTiming(passManager)

Enable pass timing.
"""
function mlirPassManagerEnableTiming(passManager)
    @ccall Reactant_jll.libReactantExtra.mlirPassManagerEnableTiming(
        passManager::MlirPassManager
    )::Cvoid
end

"""
    MlirPassDisplayMode

Enumerated type of pass display modes. Mainly used in [`mlirPassManagerEnableStatistics`](@ref).
"""
@cenum MlirPassDisplayMode::UInt32 begin
    MLIR_PASS_DISPLAY_MODE_LIST = 0x0000000000000000
    MLIR_PASS_DISPLAY_MODE_PIPELINE = 0x0000000000000001
end

"""
    mlirPassManagerEnableStatistics(passManager, displayMode)

Enable pass statistics.
"""
function mlirPassManagerEnableStatistics(passManager, displayMode)
    @ccall Reactant_jll.libReactantExtra.mlirPassManagerEnableStatistics(
        passManager::MlirPassManager, displayMode::MlirPassDisplayMode
    )::Cvoid
end

"""
    mlirPassManagerGetNestedUnder(passManager, operationName)

Nest an OpPassManager under the top-level PassManager, the nested passmanager will only run on operations matching the provided name. The returned OpPassManager will be destroyed when the parent is destroyed. To further nest more OpPassManager under the newly returned one, see `mlirOpPassManagerNest` below.
"""
function mlirPassManagerGetNestedUnder(passManager, operationName)
    @ccall Reactant_jll.libReactantExtra.mlirPassManagerGetNestedUnder(
        passManager::MlirPassManager, operationName::MlirStringRef
    )::MlirOpPassManager
end

"""
    mlirOpPassManagerGetNestedUnder(passManager, operationName)

Nest an OpPassManager under the provided OpPassManager, the nested passmanager will only run on operations matching the provided name. The returned OpPassManager will be destroyed when the parent is destroyed.
"""
function mlirOpPassManagerGetNestedUnder(passManager, operationName)
    @ccall Reactant_jll.libReactantExtra.mlirOpPassManagerGetNestedUnder(
        passManager::MlirOpPassManager, operationName::MlirStringRef
    )::MlirOpPassManager
end

"""
    mlirPassManagerAddOwnedPass(passManager, pass)

Add a pass and transfer ownership to the provided top-level mlirPassManager. If the pass is not a generic operation pass or a ModulePass, a new OpPassManager is implicitly nested under the provided PassManager.
"""
function mlirPassManagerAddOwnedPass(passManager, pass)
    @ccall Reactant_jll.libReactantExtra.mlirPassManagerAddOwnedPass(
        passManager::MlirPassManager, pass::MlirPass
    )::Cvoid
end

"""
    mlirOpPassManagerAddOwnedPass(passManager, pass)

Add a pass and transfer ownership to the provided mlirOpPassManager. If the pass is not a generic operation pass or matching the type of the provided PassManager, a new OpPassManager is implicitly nested under the provided PassManager.
"""
function mlirOpPassManagerAddOwnedPass(passManager, pass)
    @ccall Reactant_jll.libReactantExtra.mlirOpPassManagerAddOwnedPass(
        passManager::MlirOpPassManager, pass::MlirPass
    )::Cvoid
end

"""
    mlirOpPassManagerAddPipeline(passManager, pipelineElements, callback, userData)

Parse a sequence of textual MLIR pass pipeline elements and add them to the provided OpPassManager. If parsing fails an error message is reported using the provided callback.
"""
function mlirOpPassManagerAddPipeline(passManager, pipelineElements, callback, userData)
    @ccall Reactant_jll.libReactantExtra.mlirOpPassManagerAddPipeline(
        passManager::MlirOpPassManager,
        pipelineElements::MlirStringRef,
        callback::MlirStringCallback,
        userData::Ptr{Cvoid},
    )::MlirLogicalResult
end

"""
    mlirPrintPassPipeline(passManager, callback, userData)

Print a textual MLIR pass pipeline by sending chunks of the string representation and forwarding `userData to `callback`. Note that the callback may be called several times with consecutive chunks of the string.
"""
function mlirPrintPassPipeline(passManager, callback, userData)
    @ccall Reactant_jll.libReactantExtra.mlirPrintPassPipeline(
        passManager::MlirOpPassManager, callback::MlirStringCallback, userData::Ptr{Cvoid}
    )::Cvoid
end

"""
    mlirParsePassPipeline(passManager, pipeline, callback, userData)

Parse a textual MLIR pass pipeline and assign it to the provided OpPassManager. If parsing fails an error message is reported using the provided callback.
"""
function mlirParsePassPipeline(passManager, pipeline, callback, userData)
    @ccall Reactant_jll.libReactantExtra.mlirParsePassPipeline(
        passManager::MlirOpPassManager,
        pipeline::MlirStringRef,
        callback::MlirStringCallback,
        userData::Ptr{Cvoid},
    )::MlirLogicalResult
end

"""
    MlirExternalPassCallbacks

Structure of external [`MlirPass`](@ref) callbacks. All callbacks are required to be set unless otherwise specified.

| Field      | Note                                                                                                                                                                                              |
| :--------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| construct  | This callback is called from the pass is created. This is analogous to a C++ pass constructor.                                                                                                    |
| destruct   | This callback is called when the pass is destroyed This is analogous to a C++ pass destructor.                                                                                                    |
| initialize | This callback is optional. The callback is called before the pass is run, allowing a chance to initialize any complex state necessary for running the pass. See Pass::initialize(MLIRContext *).  |
| clone      | This callback is called when the pass is cloned. See Pass::clonePass().                                                                                                                           |
| run        | This callback is called when the pass is run. See Pass::runOnOperation().                                                                                                                         |
"""
struct MlirExternalPassCallbacks
    construct::Ptr{Cvoid}
    destruct::Ptr{Cvoid}
    initialize::Ptr{Cvoid}
    clone::Ptr{Cvoid}
    run::Ptr{Cvoid}
end

"""
    mlirCreateExternalPass(passID, name, argument, description, opName, nDependentDialects, dependentDialects, callbacks, userData)

Creates an external [`MlirPass`](@ref) that calls the supplied `callbacks` using the supplied `userData`. If `opName` is empty, the pass is a generic operation pass. Otherwise it is an operation pass specific to the specified pass name.
"""
function mlirCreateExternalPass(
    passID,
    name,
    argument,
    description,
    opName,
    nDependentDialects,
    dependentDialects,
    callbacks,
    userData,
)
    @ccall Reactant_jll.libReactantExtra.mlirCreateExternalPass(
        passID::MlirTypeID,
        name::MlirStringRef,
        argument::MlirStringRef,
        description::MlirStringRef,
        opName::MlirStringRef,
        nDependentDialects::Cptrdiff_t,
        dependentDialects::Ptr{MlirDialectHandle},
        callbacks::MlirExternalPassCallbacks,
        userData::Ptr{Cvoid},
    )::MlirPass
end

"""
    mlirExternalPassSignalFailure(pass)

This signals that the pass has failed. This is only valid to call during the `run` callback of [`MlirExternalPassCallbacks`](@ref). See Pass::signalPassFailure().
"""
function mlirExternalPassSignalFailure(pass)
    @ccall Reactant_jll.libReactantExtra.mlirExternalPassSignalFailure(
        pass::MlirExternalPass
    )::Cvoid
end

"""
    mlirRegisterAllDialects(registry)

Appends all upstream dialects and extensions to the dialect registry.
"""
function mlirRegisterAllDialects(registry)
    @ccall Reactant_jll.libReactantExtra.mlirRegisterAllDialects(
        registry::MlirDialectRegistry
    )::Cvoid
end

"""
    mlirRegisterAllLLVMTranslations(context)

Register all translations to LLVM IR for dialects that can support it.
"""
function mlirRegisterAllLLVMTranslations(context)
    @ccall Reactant_jll.libReactantExtra.mlirRegisterAllLLVMTranslations(
        context::MlirContext
    )::Cvoid
end

"""
    mlirRegisterAllPasses()

Register all compiler passes of MLIR.
"""
function mlirRegisterAllPasses()
    @ccall Reactant_jll.libReactantExtra.mlirRegisterAllPasses()::Cvoid
end

"""
    mlirTranslateModuleToSMTLIB(arg1, arg2, userData, inlineSingleUseValues, indentLetBody, emitReset)

Emits SMTLIB for the specified module using the provided callback and user data
"""
function mlirTranslateModuleToSMTLIB(
    arg1, arg2, userData, inlineSingleUseValues, indentLetBody, emitReset
)
    @ccall Reactant_jll.libReactantExtra.mlirTranslateModuleToSMTLIB(
        arg1::MlirModule,
        arg2::MlirStringCallback,
        userData::Ptr{Cvoid},
        inlineSingleUseValues::Bool,
        indentLetBody::Bool,
        emitReset::Bool,
    )::MlirLogicalResult
end

function mlirTranslateOperationToSMTLIB(
    arg1, arg2, userData, inlineSingleUseValues, indentLetBody, emitReset
)
    @ccall Reactant_jll.libReactantExtra.mlirTranslateOperationToSMTLIB(
        arg1::MlirOperation,
        arg2::MlirStringCallback,
        userData::Ptr{Cvoid},
        inlineSingleUseValues::Bool,
        indentLetBody::Bool,
        emitReset::Bool,
    )::MlirLogicalResult
end

"""
` LLVMCSupportTypes Types and Enumerations`

@{
"""
const LLVMBool = Cint

mutable struct LLVMOpaqueMemoryBuffer end

"""
Used to pass regions of memory through LLVM interfaces.

# See also
llvm::MemoryBuffer
"""
const LLVMMemoryBufferRef = Ptr{LLVMOpaqueMemoryBuffer}

mutable struct LLVMOpaqueContext end

"""
The top-level container for all LLVM global data. See the [`LLVMContext`](@ref) class.
"""
const LLVMContextRef = Ptr{LLVMOpaqueContext}

mutable struct LLVMOpaqueModule end

"""
The top-level container for all other LLVM Intermediate Representation (IR) objects.

# See also
llvm::[`Module`](@ref)
"""
const LLVMModuleRef = Ptr{LLVMOpaqueModule}

mutable struct LLVMOpaqueType end

"""
Each value in the LLVM IR has a type, an [`LLVMTypeRef`](@ref).

# See also
llvm::Type
"""
const LLVMTypeRef = Ptr{LLVMOpaqueType}

mutable struct LLVMOpaqueValue end

"""
Represents an individual value in LLVM IR.

This models llvm::Value.
"""
const LLVMValueRef = Ptr{LLVMOpaqueValue}

mutable struct LLVMOpaqueBasicBlock end

"""
Represents a basic block of instructions in LLVM IR.

This models llvm::BasicBlock.
"""
const LLVMBasicBlockRef = Ptr{LLVMOpaqueBasicBlock}

mutable struct LLVMOpaqueMetadata end

"""
Represents an LLVM Metadata.

This models llvm::Metadata.
"""
const LLVMMetadataRef = Ptr{LLVMOpaqueMetadata}

mutable struct LLVMOpaqueNamedMDNode end

"""
Represents an LLVM Named Metadata Node.

This models llvm::NamedMDNode.
"""
const LLVMNamedMDNodeRef = Ptr{LLVMOpaqueNamedMDNode}

mutable struct LLVMOpaqueValueMetadataEntry end

"""
Represents an entry in a Global Object's metadata attachments.

This models std::pair<unsigned, MDNode *>
"""
const LLVMValueMetadataEntry = LLVMOpaqueValueMetadataEntry

mutable struct LLVMOpaqueBuilder end

"""
Represents an LLVM basic block builder.

This models llvm::IRBuilder.
"""
const LLVMBuilderRef = Ptr{LLVMOpaqueBuilder}

mutable struct LLVMOpaqueDIBuilder end

"""
Represents an LLVM debug info builder.

This models llvm::DIBuilder.
"""
const LLVMDIBuilderRef = Ptr{LLVMOpaqueDIBuilder}

mutable struct LLVMOpaqueModuleProvider end

"""
Interface used to provide a module to JIT or interpreter. This is now just a synonym for llvm::[`Module`](@ref), but we have to keep using the different type to keep binary compatibility.
"""
const LLVMModuleProviderRef = Ptr{LLVMOpaqueModuleProvider}

mutable struct LLVMOpaquePassManager end

"""
# See also
llvm::PassManagerBase
"""
const LLVMPassManagerRef = Ptr{LLVMOpaquePassManager}

mutable struct LLVMOpaqueUse end

"""
Used to get the users and usees of a Value.

# See also
llvm::Use
"""
const LLVMUseRef = Ptr{LLVMOpaqueUse}

mutable struct LLVMOpaqueOperandBundle end

"""
# See also
llvm::OperandBundleDef
"""
const LLVMOperandBundleRef = Ptr{LLVMOpaqueOperandBundle}

mutable struct LLVMOpaqueAttributeRef end

"""
Used to represent an attributes.

# See also
llvm::Attribute
"""
const LLVMAttributeRef = Ptr{LLVMOpaqueAttributeRef}

mutable struct LLVMOpaqueDiagnosticInfo end

"""
# See also
llvm::DiagnosticInfo
"""
const LLVMDiagnosticInfoRef = Ptr{LLVMOpaqueDiagnosticInfo}

mutable struct LLVMComdat end

"""
# See also
llvm::Comdat
"""
const LLVMComdatRef = Ptr{LLVMComdat}

mutable struct LLVMOpaqueModuleFlagEntry end

"""
# See also
llvm::[`Module`](@ref)::ModuleFlagEntry
"""
const LLVMModuleFlagEntry = LLVMOpaqueModuleFlagEntry

mutable struct LLVMOpaqueJITEventListener end

"""
# See also
llvm::JITEventListener
"""
const LLVMJITEventListenerRef = Ptr{LLVMOpaqueJITEventListener}

mutable struct LLVMOpaqueBinary end

"""
# See also
llvm::object::Binary
"""
const LLVMBinaryRef = Ptr{LLVMOpaqueBinary}

mutable struct LLVMOpaqueDbgRecord end

"""
# See also
llvm::DbgRecord
"""
const LLVMDbgRecordRef = Ptr{LLVMOpaqueDbgRecord}

function LLVMParseCommandLineOptions(argc, argv, Overview)
    @ccall Reactant_jll.libReactantExtra.LLVMParseCommandLineOptions(
        argc::Cint, argv::Ptr{Cstring}, Overview::Cstring
    )::Cint
end

function LLVMSearchForAddressOfSymbol(symbolName)
    @ccall Reactant_jll.libReactantExtra.LLVMSearchForAddressOfSymbol(
        symbolName::Cstring
    )::Ptr{Cint}
end

function LLVMAddSymbol(symbolName, symbolValue)
    @ccall Reactant_jll.libReactantExtra.LLVMAddSymbol(
        symbolName::Cstring, symbolValue::Ptr{Cvoid}
    )::Cint
end

"""
    mlirTranslateModuleToLLVMIR(_module, context)

Translate operation that satisfies LLVM dialect module requirements into an LLVM IR module living in the given context. This translates operations from any dilalect that has a registered implementation of LLVMTranslationDialectInterface.

# Returns
the generated LLVM IR [`Module`](@ref) from the translated MLIR module, it is owned by the caller.
"""
function mlirTranslateModuleToLLVMIR(_module, context)
    @ccall Reactant_jll.libReactantExtra.mlirTranslateModuleToLLVMIR(
        _module::MlirOperation, context::LLVMContextRef
    )::LLVMModuleRef
end

function mlirTranslateModuleToLLVMIRToString(_module)
    @ccall Reactant_jll.libReactantExtra.mlirTranslateModuleToLLVMIRToString(
        _module::MlirOperation
    )::Cstring
end

struct MlirTypeFromLLVMIRTranslator
    ptr::Ptr{Cvoid}
end

"""
    mlirTypeFromLLVMIRTranslatorCreate(ctx)

Create an LLVM::TypeFromLLVMIRTranslator and transfer ownership to the caller.
"""
function mlirTypeFromLLVMIRTranslatorCreate(ctx)
    @ccall Reactant_jll.libReactantExtra.mlirTypeFromLLVMIRTranslatorCreate(
        ctx::MlirContext
    )::MlirTypeFromLLVMIRTranslator
end

"""
    mlirTypeFromLLVMIRTranslatorDestroy(translator)

Takes an LLVM::TypeFromLLVMIRTranslator owned by the caller and destroys it. It is the responsibility of the user to only pass an LLVM::TypeFromLLVMIRTranslator class.
"""
function mlirTypeFromLLVMIRTranslatorDestroy(translator)
    @ccall Reactant_jll.libReactantExtra.mlirTypeFromLLVMIRTranslatorDestroy(
        translator::MlirTypeFromLLVMIRTranslator
    )::Cvoid
end

"""
    mlirTypeFromLLVMIRTranslatorTranslateType(translator, llvmType)

Translates the given LLVM IR type to the MLIR LLVM dialect.
"""
function mlirTypeFromLLVMIRTranslatorTranslateType(translator, llvmType)
    @ccall Reactant_jll.libReactantExtra.mlirTypeFromLLVMIRTranslatorTranslateType(
        translator::MlirTypeFromLLVMIRTranslator, llvmType::LLVMTypeRef
    )::MlirType
end

struct MlirTypeToLLVMIRTranslator
    ptr::Ptr{Cvoid}
end

"""
    mlirTypeToLLVMIRTranslatorCreate(ctx)

Create an LLVM::TypeToLLVMIRTranslator and transfer ownership to the caller.
"""
function mlirTypeToLLVMIRTranslatorCreate(ctx)
    @ccall Reactant_jll.libReactantExtra.mlirTypeToLLVMIRTranslatorCreate(
        ctx::LLVMContextRef
    )::MlirTypeToLLVMIRTranslator
end

"""
    mlirTypeToLLVMIRTranslatorDestroy(translator)

Takes an LLVM::TypeToLLVMIRTranslator owned by the caller and destroys it. It is the responsibility of the user to only pass an LLVM::TypeToLLVMIRTranslator class.
"""
function mlirTypeToLLVMIRTranslatorDestroy(translator)
    @ccall Reactant_jll.libReactantExtra.mlirTypeToLLVMIRTranslatorDestroy(
        translator::MlirTypeToLLVMIRTranslator
    )::Cvoid
end

"""
    mlirTypeToLLVMIRTranslatorTranslateType(translator, mlirType)

Translates the given MLIR LLVM dialect to the LLVM IR type.
"""
function mlirTypeToLLVMIRTranslatorTranslateType(translator, mlirType)
    @ccall Reactant_jll.libReactantExtra.mlirTypeToLLVMIRTranslatorTranslateType(
        translator::MlirTypeToLLVMIRTranslator, mlirType::MlirType
    )::LLVMTypeRef
end

function stablehloScatterDimensionNumbersGet(
    ctx,
    nUpdateWindowDims,
    updateWindowDims,
    nInsertedWindowDims,
    insertedWindowDims,
    nInputBatchingDims,
    inputBatchingDims,
    nScatterIndicesBatchingDims,
    scatterIndicesBatchingDims,
    nScatteredDimsToOperandDims,
    scatteredDimsToOperandDims,
    indexVectorDim,
)
    @ccall Reactant_jll.libReactantExtra.stablehloScatterDimensionNumbersGet(
        ctx::MlirContext,
        nUpdateWindowDims::Cptrdiff_t,
        updateWindowDims::Ptr{Int64},
        nInsertedWindowDims::Cptrdiff_t,
        insertedWindowDims::Ptr{Int64},
        nInputBatchingDims::Cptrdiff_t,
        inputBatchingDims::Ptr{Int64},
        nScatterIndicesBatchingDims::Cptrdiff_t,
        scatterIndicesBatchingDims::Ptr{Int64},
        nScatteredDimsToOperandDims::Cptrdiff_t,
        scatteredDimsToOperandDims::Ptr{Int64},
        indexVectorDim::Int64,
    )::MlirAttribute
end

function stablehloAttributeIsAScatterDimensionNumbers(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloAttributeIsAScatterDimensionNumbers(
        attr::MlirAttribute
    )::Bool
end

function stablehloScatterDimensionNumbersGetUpdateWindowDimsSize(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloScatterDimensionNumbersGetUpdateWindowDimsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function stablehloScatterDimensionNumbersGetUpdateWindowDimsElem(attr, pos)
    @ccall Reactant_jll.libReactantExtra.stablehloScatterDimensionNumbersGetUpdateWindowDimsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function stablehloScatterDimensionNumbersGetInsertedWindowDimsSize(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloScatterDimensionNumbersGetInsertedWindowDimsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function stablehloScatterDimensionNumbersGetInsertedWindowDimsElem(attr, pos)
    @ccall Reactant_jll.libReactantExtra.stablehloScatterDimensionNumbersGetInsertedWindowDimsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function stablehloScatterDimensionNumbersGetInputBatchingDimsSize(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloScatterDimensionNumbersGetInputBatchingDimsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function stablehloScatterDimensionNumbersGetInputBatchingDimsElem(attr, pos)
    @ccall Reactant_jll.libReactantExtra.stablehloScatterDimensionNumbersGetInputBatchingDimsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function stablehloScatterDimensionNumbersGetScatterIndicesBatchingDimsSize(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloScatterDimensionNumbersGetScatterIndicesBatchingDimsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function stablehloScatterDimensionNumbersGetScatterIndicesBatchingDimsElem(attr, pos)
    @ccall Reactant_jll.libReactantExtra.stablehloScatterDimensionNumbersGetScatterIndicesBatchingDimsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function stablehloScatterDimensionNumbersGetScatteredDimsToOperandDimsSize(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloScatterDimensionNumbersGetScatteredDimsToOperandDimsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function stablehloScatterDimensionNumbersGetScatteredDimsToOperandDimsElem(attr, pos)
    @ccall Reactant_jll.libReactantExtra.stablehloScatterDimensionNumbersGetScatteredDimsToOperandDimsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function stablehloDimensionNumbersGetIndexVectorDim(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloDimensionNumbersGetIndexVectorDim(
        attr::MlirAttribute
    )::Int64
end

function stablehloGatherDimensionNumbersGet(
    ctx,
    nOffsetDims,
    offsetDims,
    nCollapsedSliceDims,
    collapsedSliceDims,
    nOperandBatchingDims,
    operandBatchingDims,
    nStartIndicesBatchingDims,
    startIndicesBatchingDims,
    nStartIndexMap,
    startIndexMap,
    indexVectorDim,
)
    @ccall Reactant_jll.libReactantExtra.stablehloGatherDimensionNumbersGet(
        ctx::MlirContext,
        nOffsetDims::Cptrdiff_t,
        offsetDims::Ptr{Int64},
        nCollapsedSliceDims::Cptrdiff_t,
        collapsedSliceDims::Ptr{Int64},
        nOperandBatchingDims::Cptrdiff_t,
        operandBatchingDims::Ptr{Int64},
        nStartIndicesBatchingDims::Cptrdiff_t,
        startIndicesBatchingDims::Ptr{Int64},
        nStartIndexMap::Cptrdiff_t,
        startIndexMap::Ptr{Int64},
        indexVectorDim::Int64,
    )::MlirAttribute
end

function stablehloAttributeIsAGatherDimensionNumbers(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloAttributeIsAGatherDimensionNumbers(
        attr::MlirAttribute
    )::Bool
end

function stablehloGatherDimensionNumbersGetOffsetDimsSize(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloGatherDimensionNumbersGetOffsetDimsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function stablehloGatherDimensionNumbersGetOffsetDimsElem(attr, pos)
    @ccall Reactant_jll.libReactantExtra.stablehloGatherDimensionNumbersGetOffsetDimsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function stablehloGatherDimensionNumbersGetCollapsedSliceDimsSize(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloGatherDimensionNumbersGetCollapsedSliceDimsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function stablehloGatherDimensionNumbersGetCollapsedSliceDimsElem(attr, pos)
    @ccall Reactant_jll.libReactantExtra.stablehloGatherDimensionNumbersGetCollapsedSliceDimsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function stablehloGatherDimensionNumbersGetOperandBatchingDimsSize(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloGatherDimensionNumbersGetOperandBatchingDimsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function stablehloGatherDimensionNumbersGetOperandBatchingDimsElem(attr, pos)
    @ccall Reactant_jll.libReactantExtra.stablehloGatherDimensionNumbersGetOperandBatchingDimsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function stablehloGatherDimensionNumbersGetStartIndicesBatchingDimsSize(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloGatherDimensionNumbersGetStartIndicesBatchingDimsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function stablehloGatherDimensionNumbersGetStartIndicesBatchingDimsElem(attr, pos)
    @ccall Reactant_jll.libReactantExtra.stablehloGatherDimensionNumbersGetStartIndicesBatchingDimsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function stablehloGatherDimensionNumbersGetStartIndexMapSize(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloGatherDimensionNumbersGetStartIndexMapSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function stablehloGatherDimensionNumbersGetStartIndexMapElem(attr, pos)
    @ccall Reactant_jll.libReactantExtra.stablehloGatherDimensionNumbersGetStartIndexMapElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function stablehloGatherDimensionNumbersGetIndexVectorDim(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloGatherDimensionNumbersGetIndexVectorDim(
        attr::MlirAttribute
    )::Int64
end

function stablehloDotAlgorithmGet(
    ctx,
    lhsPrecisionType,
    rhsPrecisionType,
    accumulationType,
    lhsComponentCount,
    rhsComponentCount,
    numPrimitiveOperations,
    allowImpreciseAccumulation,
)
    @ccall Reactant_jll.libReactantExtra.stablehloDotAlgorithmGet(
        ctx::MlirContext,
        lhsPrecisionType::MlirType,
        rhsPrecisionType::MlirType,
        accumulationType::MlirType,
        lhsComponentCount::Int64,
        rhsComponentCount::Int64,
        numPrimitiveOperations::Int64,
        allowImpreciseAccumulation::Bool,
    )::MlirAttribute
end

function stablehloAttributeIsADotAlgorithm(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloAttributeIsADotAlgorithm(
        attr::MlirAttribute
    )::Bool
end

function stablehloDotAlgorithmGetLhsPrecisionType(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloDotAlgorithmGetLhsPrecisionType(
        attr::MlirAttribute
    )::MlirType
end

function stablehloDotAlgorithmGetRhsPrecisionType(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloDotAlgorithmGetRhsPrecisionType(
        attr::MlirAttribute
    )::MlirType
end

function stablehloDotAlgorithmGetAccumulationType(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloDotAlgorithmGetAccumulationType(
        attr::MlirAttribute
    )::MlirType
end

function stablehloDotAlgorithmGetLhsComponentCount(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloDotAlgorithmGetLhsComponentCount(
        attr::MlirAttribute
    )::Int64
end

function stablehloDotAlgorithmGetRhsComponentCount(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloDotAlgorithmGetRhsComponentCount(
        attr::MlirAttribute
    )::Int64
end

function stablehloDotAlgorithmGetNumPrimitiveOperations(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloDotAlgorithmGetNumPrimitiveOperations(
        attr::MlirAttribute
    )::Int64
end

function stablehloDotAlgorithmGetAllowImpreciseAccumulation(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloDotAlgorithmGetAllowImpreciseAccumulation(
        attr::MlirAttribute
    )::Bool
end

function stablehloDotDimensionNumbersGet(
    ctx,
    nLhsBatchingDimensions,
    lhsBatchingDimensions,
    nRhsBatchingDimensions,
    rhsBatchingDimensions,
    nLhsContractingDimensions,
    lhsContractingDimensions,
    nRhsContractingDimensions,
    rhsContractingDimensions,
)
    @ccall Reactant_jll.libReactantExtra.stablehloDotDimensionNumbersGet(
        ctx::MlirContext,
        nLhsBatchingDimensions::Cptrdiff_t,
        lhsBatchingDimensions::Ptr{Int64},
        nRhsBatchingDimensions::Cptrdiff_t,
        rhsBatchingDimensions::Ptr{Int64},
        nLhsContractingDimensions::Cptrdiff_t,
        lhsContractingDimensions::Ptr{Int64},
        nRhsContractingDimensions::Cptrdiff_t,
        rhsContractingDimensions::Ptr{Int64},
    )::MlirAttribute
end

function stablehloAttributeIsADotDimensionNumbers(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloAttributeIsADotDimensionNumbers(
        attr::MlirAttribute
    )::Bool
end

function stablehloDotDimensionNumbersGetLhsBatchingDimensionsSize(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloDotDimensionNumbersGetLhsBatchingDimensionsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function stablehloDotDimensionNumbersGetLhsBatchingDimensionsElem(attr, pos)
    @ccall Reactant_jll.libReactantExtra.stablehloDotDimensionNumbersGetLhsBatchingDimensionsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function stablehloDotDimensionNumbersGetRhsBatchingDimensionsSize(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloDotDimensionNumbersGetRhsBatchingDimensionsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function stablehloDotDimensionNumbersGetRhsBatchingDimensionsElem(attr, pos)
    @ccall Reactant_jll.libReactantExtra.stablehloDotDimensionNumbersGetRhsBatchingDimensionsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function stablehloDotDimensionNumbersGetLhsContractingDimensionsSize(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloDotDimensionNumbersGetLhsContractingDimensionsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function stablehloDotDimensionNumbersGetLhsContractingDimensionsElem(attr, pos)
    @ccall Reactant_jll.libReactantExtra.stablehloDotDimensionNumbersGetLhsContractingDimensionsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function stablehloDotDimensionNumbersGetRhsContractingDimensionsSize(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloDotDimensionNumbersGetRhsContractingDimensionsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function stablehloDotDimensionNumbersGetRhsContractingDimensionsElem(attr, pos)
    @ccall Reactant_jll.libReactantExtra.stablehloDotDimensionNumbersGetRhsContractingDimensionsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function stablehloConvDimensionNumbersGet(
    ctx,
    inputBatchDimension,
    inputFeatureDimension,
    nInputSpatialDimensions,
    inputSpatialDimensions,
    kernelInputFeatureDimension,
    kernelOutputFeatureDimension,
    nKernelSpatialDimensions,
    kernelSpatialDimensions,
    outputBatchDimension,
    outputFeatureDimension,
    nOutputSpatialDimensions,
    outputSpatialDimensions,
)
    @ccall Reactant_jll.libReactantExtra.stablehloConvDimensionNumbersGet(
        ctx::MlirContext,
        inputBatchDimension::Int64,
        inputFeatureDimension::Int64,
        nInputSpatialDimensions::Cptrdiff_t,
        inputSpatialDimensions::Ptr{Int64},
        kernelInputFeatureDimension::Int64,
        kernelOutputFeatureDimension::Int64,
        nKernelSpatialDimensions::Cptrdiff_t,
        kernelSpatialDimensions::Ptr{Int64},
        outputBatchDimension::Int64,
        outputFeatureDimension::Int64,
        nOutputSpatialDimensions::Cptrdiff_t,
        outputSpatialDimensions::Ptr{Int64},
    )::MlirAttribute
end

function stablehloAttributeIsAConvDimensionNumbers(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloAttributeIsAConvDimensionNumbers(
        attr::MlirAttribute
    )::Bool
end

function stablehloConvDimensionNumbersGetInputBatchDimension(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloConvDimensionNumbersGetInputBatchDimension(
        attr::MlirAttribute
    )::Int64
end

function stablehloConvDimensionNumbersGetInputFeatureDimension(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloConvDimensionNumbersGetInputFeatureDimension(
        attr::MlirAttribute
    )::Int64
end

function stablehloConvDimensionNumbersGetInputSpatialDimensionsSize(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloConvDimensionNumbersGetInputSpatialDimensionsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function stablehloConvDimensionNumbersGetInputSpatialDimensionsElem(attr, pos)
    @ccall Reactant_jll.libReactantExtra.stablehloConvDimensionNumbersGetInputSpatialDimensionsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function stablehloConvDimensionNumbersGetKernelInputFeatureDimension(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloConvDimensionNumbersGetKernelInputFeatureDimension(
        attr::MlirAttribute
    )::Int64
end

function stablehloConvDimensionNumbersGetKernelOutputFeatureDimension(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloConvDimensionNumbersGetKernelOutputFeatureDimension(
        attr::MlirAttribute
    )::Int64
end

function stablehloConvDimensionNumbersGetKernelSpatialDimensionsSize(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloConvDimensionNumbersGetKernelSpatialDimensionsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function stablehloConvDimensionNumbersGetKernelSpatialDimensionsElem(attr, pos)
    @ccall Reactant_jll.libReactantExtra.stablehloConvDimensionNumbersGetKernelSpatialDimensionsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function stablehloConvDimensionNumbersGetOutputBatchDimension(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloConvDimensionNumbersGetOutputBatchDimension(
        attr::MlirAttribute
    )::Int64
end

function stablehloConvDimensionNumbersGetOutputFeatureDimension(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloConvDimensionNumbersGetOutputFeatureDimension(
        attr::MlirAttribute
    )::Int64
end

function stablehloConvDimensionNumbersGetOutputSpatialDimensionsSize(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloConvDimensionNumbersGetOutputSpatialDimensionsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function stablehloConvDimensionNumbersGetOutputSpatialDimensionsElem(attr, pos)
    @ccall Reactant_jll.libReactantExtra.stablehloConvDimensionNumbersGetOutputSpatialDimensionsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function stablehloOutputOperandAliasGet(
    ctx,
    nOutputTupleIndices,
    outputTupleIndices,
    operandIndex,
    nOperandTupleIndices,
    operandTupleIndices,
)
    @ccall Reactant_jll.libReactantExtra.stablehloOutputOperandAliasGet(
        ctx::MlirContext,
        nOutputTupleIndices::Cptrdiff_t,
        outputTupleIndices::Ptr{Int64},
        operandIndex::Int64,
        nOperandTupleIndices::Cptrdiff_t,
        operandTupleIndices::Ptr{Int64},
    )::MlirAttribute
end

function stablehloAttributeIsAOutputOperandAlias(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloAttributeIsAOutputOperandAlias(
        attr::MlirAttribute
    )::Bool
end

function stablehloOutputOperandAliasGetOutputTupleIndicesSize(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloOutputOperandAliasGetOutputTupleIndicesSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function stablehloOutputOperandAliasGetOutputTupleIndicesElem(attr, pos)
    @ccall Reactant_jll.libReactantExtra.stablehloOutputOperandAliasGetOutputTupleIndicesElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function stablehloOutputOperandAliasGetOperandIndex(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloOutputOperandAliasGetOperandIndex(
        attr::MlirAttribute
    )::Int64
end

function stablehloOutputOperandAliasGetOperandTupleIndicesSize(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloOutputOperandAliasGetOperandTupleIndicesSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function stablehloOutputOperandAliasGetOperandTupleIndicesElem(attr, pos)
    @ccall Reactant_jll.libReactantExtra.stablehloOutputOperandAliasGetOperandTupleIndicesElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function stablehloComparisonDirectionAttrGet(ctx, value)
    @ccall Reactant_jll.libReactantExtra.stablehloComparisonDirectionAttrGet(
        ctx::MlirContext, value::MlirStringRef
    )::MlirAttribute
end

function stablehloAttributeIsAComparisonDirectionAttr(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloAttributeIsAComparisonDirectionAttr(
        attr::MlirAttribute
    )::Bool
end

function stablehloComparisonDirectionAttrGetValue(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloComparisonDirectionAttrGetValue(
        attr::MlirAttribute
    )::MlirStringRef
end

function stablehloComparisonTypeAttrGet(ctx, value)
    @ccall Reactant_jll.libReactantExtra.stablehloComparisonTypeAttrGet(
        ctx::MlirContext, value::MlirStringRef
    )::MlirAttribute
end

function stablehloAttributeIsAComparisonTypeAttr(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloAttributeIsAComparisonTypeAttr(
        attr::MlirAttribute
    )::Bool
end

function stablehloComparisonTypeAttrGetValue(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloComparisonTypeAttrGetValue(
        attr::MlirAttribute
    )::MlirStringRef
end

function stablehloPrecisionAttrGet(ctx, value)
    @ccall Reactant_jll.libReactantExtra.stablehloPrecisionAttrGet(
        ctx::MlirContext, value::MlirStringRef
    )::MlirAttribute
end

function stablehloAttributeIsAPrecisionAttr(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloAttributeIsAPrecisionAttr(
        attr::MlirAttribute
    )::Bool
end

function stablehloPrecisionAttrGetValue(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloPrecisionAttrGetValue(
        attr::MlirAttribute
    )::MlirStringRef
end

function stablehloFftTypeAttrGet(ctx, value)
    @ccall Reactant_jll.libReactantExtra.stablehloFftTypeAttrGet(
        ctx::MlirContext, value::MlirStringRef
    )::MlirAttribute
end

function stablehloAttributeIsAFftTypeAttr(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloAttributeIsAFftTypeAttr(
        attr::MlirAttribute
    )::Bool
end

function stablehloFftTypeAttrGetValue(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloFftTypeAttrGetValue(
        attr::MlirAttribute
    )::MlirStringRef
end

function stablehloTransposeAttrGet(ctx, value)
    @ccall Reactant_jll.libReactantExtra.stablehloTransposeAttrGet(
        ctx::MlirContext, value::MlirStringRef
    )::MlirAttribute
end

function stablehloAttributeIsATransposeAttr(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloAttributeIsATransposeAttr(
        attr::MlirAttribute
    )::Bool
end

function stablehloTransposeAttrGetValue(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloTransposeAttrGetValue(
        attr::MlirAttribute
    )::MlirStringRef
end

function stablehloRngDistributionAttrGet(ctx, value)
    @ccall Reactant_jll.libReactantExtra.stablehloRngDistributionAttrGet(
        ctx::MlirContext, value::MlirStringRef
    )::MlirAttribute
end

function stablehloAttributeIsARngDistributionAttr(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloAttributeIsARngDistributionAttr(
        attr::MlirAttribute
    )::Bool
end

function stablehloRngDistributionAttrGetValue(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloRngDistributionAttrGetValue(
        attr::MlirAttribute
    )::MlirStringRef
end

function stablehloRngAlgorithmAttrGet(ctx, value)
    @ccall Reactant_jll.libReactantExtra.stablehloRngAlgorithmAttrGet(
        ctx::MlirContext, value::MlirStringRef
    )::MlirAttribute
end

function stablehloAttributeIsARngAlgorithmAttr(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloAttributeIsARngAlgorithmAttr(
        attr::MlirAttribute
    )::Bool
end

function stablehloRngAlgorithmAttrGetValue(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloRngAlgorithmAttrGetValue(
        attr::MlirAttribute
    )::MlirStringRef
end

function stablehloChannelHandleGet(ctx, handle, type)
    @ccall Reactant_jll.libReactantExtra.stablehloChannelHandleGet(
        ctx::MlirContext, handle::Int64, type::Int64
    )::MlirAttribute
end

function stablehloAttributeIsChannelHandle(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloAttributeIsChannelHandle(
        attr::MlirAttribute
    )::Bool
end

function stablehloChannelHandleGetHandle(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloChannelHandleGetHandle(
        attr::MlirAttribute
    )::Int64
end

function stablehloChannelHandleGetType(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloChannelHandleGetType(
        attr::MlirAttribute
    )::Int64
end

function stablehloTypeExtensionsGet(ctx, nBounds, bounds)
    @ccall Reactant_jll.libReactantExtra.stablehloTypeExtensionsGet(
        ctx::MlirContext, nBounds::Cptrdiff_t, bounds::Ptr{Int64}
    )::MlirAttribute
end

function stablehloAttributeIsTypeExtensions(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloAttributeIsTypeExtensions(
        attr::MlirAttribute
    )::Bool
end

function stablehloTypeExtensionsGetBoundsSize(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloTypeExtensionsGetBoundsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function stablehloTypeExtensionsGetBoundsElem(attr, pos)
    @ccall Reactant_jll.libReactantExtra.stablehloTypeExtensionsGetBoundsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function stablehloResultAccuracyModeAttrGet(ctx, value)
    @ccall Reactant_jll.libReactantExtra.stablehloResultAccuracyModeAttrGet(
        ctx::MlirContext, value::MlirStringRef
    )::MlirAttribute
end

function stablehloAttributeIsAResultAccuracyModeAttr(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloAttributeIsAResultAccuracyModeAttr(
        attr::MlirAttribute
    )::Bool
end

function stablehloResultAccuracyModeAttrGetValue(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloResultAccuracyModeAttrGetValue(
        attr::MlirAttribute
    )::MlirStringRef
end

function stablehloResultAccuracyAttrGet(ctx, atol, rtol, ulps, value)
    @ccall Reactant_jll.libReactantExtra.stablehloResultAccuracyAttrGet(
        ctx::MlirContext, atol::Cdouble, rtol::Cdouble, ulps::Int64, value::MlirStringRef
    )::MlirAttribute
end

function stablehloAttributeIsAResultAccuracyAttr(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloAttributeIsAResultAccuracyAttr(
        attr::MlirAttribute
    )::Bool
end

function stablehloResultAccuracyAttrGetAtol(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloResultAccuracyAttrGetAtol(
        attr::MlirAttribute
    )::Cdouble
end

function stablehloResultAccuracyAttrGetRtol(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloResultAccuracyAttrGetRtol(
        attr::MlirAttribute
    )::Cdouble
end

function stablehloResultAccuracyAttrGetUlps(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloResultAccuracyAttrGetUlps(
        attr::MlirAttribute
    )::Int64
end

function stablehloResultAccuracyAttrGetMode(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloResultAccuracyAttrGetMode(
        attr::MlirAttribute
    )::MlirAttribute
end

function stablehloSubAxisInfoAttrGet(ctx, preSize, size)
    @ccall Reactant_jll.libReactantExtra.stablehloSubAxisInfoAttrGet(
        ctx::MlirContext, preSize::Int64, size::Int64
    )::MlirAttribute
end

function stablehloAttributeIsASubAxisInfoAttr(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloAttributeIsASubAxisInfoAttr(
        attr::MlirAttribute
    )::Bool
end

function stablehloSubAxisInfoAttrGetPreSize(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloSubAxisInfoAttrGetPreSize(
        attr::MlirAttribute
    )::Int64
end

function stablehloSubAxisInfoAttrGetSize(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloSubAxisInfoAttrGetSize(
        attr::MlirAttribute
    )::Int64
end

function stablehloAxisRefAttrGet(ctx, name, subAxisInfo)
    @ccall Reactant_jll.libReactantExtra.stablehloAxisRefAttrGet(
        ctx::MlirContext, name::MlirStringRef, subAxisInfo::MlirAttribute
    )::MlirAttribute
end

function stablehloAttributeIsAnAxisRefAttr(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloAttributeIsAnAxisRefAttr(
        attr::MlirAttribute
    )::Bool
end

function stablehloAxisRefAttrGetName(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloAxisRefAttrGetName(
        attr::MlirAttribute
    )::MlirStringRef
end

function stablehloAxisRefAttrGetSubAxisInfo(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloAxisRefAttrGetSubAxisInfo(
        attr::MlirAttribute
    )::MlirAttribute
end

function stablehloReplicaGroupMeshAxesAttrGet(ctx, mesh, axes)
    @ccall Reactant_jll.libReactantExtra.stablehloReplicaGroupMeshAxesAttrGet(
        ctx::MlirContext, mesh::MlirAttribute, axes::MlirAttribute
    )::MlirAttribute
end

function stablehloAttributeIsAReplicaGroupMeshAxesAttr(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloAttributeIsAReplicaGroupMeshAxesAttr(
        attr::MlirAttribute
    )::Bool
end

function stablehloReplicaGroupMeshAxesAttrGetMesh(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloReplicaGroupMeshAxesAttrGetMesh(
        attr::MlirAttribute
    )::MlirAttribute
end

function stablehloReplicaGroupMeshAxesAttrGetAxes(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloReplicaGroupMeshAxesAttrGetAxes(
        attr::MlirAttribute
    )::MlirAttribute
end

function stablehloMeshAxisAttrGet(ctx, name, size)
    @ccall Reactant_jll.libReactantExtra.stablehloMeshAxisAttrGet(
        ctx::MlirContext, name::MlirStringRef, size::Int64
    )::MlirAttribute
end

function stablehloAttributeIsAMeshAxisAttr(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloAttributeIsAMeshAxisAttr(
        attr::MlirAttribute
    )::Bool
end

function stablehloMeshAxisAttrGetName(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloMeshAxisAttrGetName(
        attr::MlirAttribute
    )::MlirStringRef
end

function stablehloMeshAxisAttrGetSize(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloMeshAxisAttrGetSize(
        attr::MlirAttribute
    )::Int64
end

function stablehloMeshAttrGet(ctx, axes, deviceIds)
    @ccall Reactant_jll.libReactantExtra.stablehloMeshAttrGet(
        ctx::MlirContext, axes::MlirAttribute, deviceIds::MlirAttribute
    )::MlirAttribute
end

function stablehloAttributeIsAMeshAttr(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloAttributeIsAMeshAttr(
        attr::MlirAttribute
    )::Bool
end

function stablehloMeshAttrGetAxes(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloMeshAttrGetAxes(
        attr::MlirAttribute
    )::MlirAttribute
end

function stablehloMeshAttrGetDeviceIds(attr)
    @ccall Reactant_jll.libReactantExtra.stablehloMeshAttrGetDeviceIds(
        attr::MlirAttribute
    )::MlirAttribute
end

function mlirGetDialectHandle__stablehlo__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__stablehlo__()::MlirDialectHandle
end

function stablehloGetApiVersion()
    @ccall Reactant_jll.libReactantExtra.stablehloGetApiVersion()::Cint
end

@cenum MlirStablehloCompatibilityRequirement::UInt32 begin
    NONE = 0x0000000000000000
    WEEK_4 = 0x0000000000000001
    WEEK_12 = 0x0000000000000002
    MAX = 0x0000000000000003
end

function stablehloVersionFromCompatibilityRequirement(requirement, callback, userData)
    @ccall Reactant_jll.libReactantExtra.stablehloVersionFromCompatibilityRequirement(
        requirement::MlirStablehloCompatibilityRequirement,
        callback::MlirStringCallback,
        userData::Ptr{Cvoid},
    )::Cvoid
end

function stablehloGetCurrentVersion(callback, userData)
    @ccall Reactant_jll.libReactantExtra.stablehloGetCurrentVersion(
        callback::MlirStringCallback, userData::Ptr{Cvoid}
    )::Cvoid
end

function stablehloGetMinimumVersion(callback, userData)
    @ccall Reactant_jll.libReactantExtra.stablehloGetMinimumVersion(
        callback::MlirStringCallback, userData::Ptr{Cvoid}
    )::Cvoid
end

function stablehloGetSmallerVersion(version1, version2, callback, userData)
    @ccall Reactant_jll.libReactantExtra.stablehloGetSmallerVersion(
        version1::MlirStringRef,
        version2::MlirStringRef,
        callback::MlirStringCallback,
        userData::Ptr{Cvoid},
    )::MlirLogicalResult
end

function stablehloSerializePortableArtifactFromStringRef(
    moduleStr, targetVersion, callback, userData
)
    @ccall Reactant_jll.libReactantExtra.stablehloSerializePortableArtifactFromStringRef(
        moduleStr::MlirStringRef,
        targetVersion::MlirStringRef,
        callback::MlirStringCallback,
        userData::Ptr{Cvoid},
    )::MlirLogicalResult
end

function stablehloSerializePortableArtifactFromModule(
    moduleStr, targetVersion, callback, userData, allowOtherDialects
)
    @ccall Reactant_jll.libReactantExtra.stablehloSerializePortableArtifactFromModule(
        moduleStr::MlirModule,
        targetVersion::MlirStringRef,
        callback::MlirStringCallback,
        userData::Ptr{Cvoid},
        allowOtherDialects::Bool,
    )::MlirLogicalResult
end

function stablehloDeserializePortableArtifact(artifactStr, callback, userData)
    @ccall Reactant_jll.libReactantExtra.stablehloDeserializePortableArtifact(
        artifactStr::MlirStringRef, callback::MlirStringCallback, userData::Ptr{Cvoid}
    )::MlirLogicalResult
end

function stablehloDeserializePortableArtifactNoError(artifactStr, ctx)
    @ccall Reactant_jll.libReactantExtra.stablehloDeserializePortableArtifactNoError(
        artifactStr::MlirStringRef, ctx::MlirContext
    )::MlirModule
end

function stablehloTokenTypeGet(ctx)
    @ccall Reactant_jll.libReactantExtra.stablehloTokenTypeGet(ctx::MlirContext)::MlirType
end

function stablehloTypeIsAToken(type)
    @ccall Reactant_jll.libReactantExtra.stablehloTypeIsAToken(type::MlirType)::Bool
end

function stablehloFutureTypeGet(ctx, nTypes, types)
    @ccall Reactant_jll.libReactantExtra.stablehloFutureTypeGet(
        ctx::MlirContext, nTypes::Cptrdiff_t, types::Ptr{MlirType}
    )::MlirType
end

function stablehloTypeIsAFuture(type)
    @ccall Reactant_jll.libReactantExtra.stablehloTypeIsAFuture(type::MlirType)::Bool
end

function stablehloFutureTypeGetNumTypes(type)
    @ccall Reactant_jll.libReactantExtra.stablehloFutureTypeGetNumTypes(
        type::MlirType
    )::Cptrdiff_t
end

function stablehloFutureTypeGetType(type, pos)
    @ccall Reactant_jll.libReactantExtra.stablehloFutureTypeGetType(
        type::MlirType, pos::Cptrdiff_t
    )::MlirType
end

function sdyAttributeIsAMeshAxisAttr(attr)
    @ccall Reactant_jll.libReactantExtra.sdyAttributeIsAMeshAxisAttr(
        attr::MlirAttribute
    )::Bool
end

function sdyMeshAxisAttrGet(ctx, name, size)
    @ccall Reactant_jll.libReactantExtra.sdyMeshAxisAttrGet(
        ctx::MlirContext, name::MlirStringRef, size::Int64
    )::MlirAttribute
end

function sdyMeshAxisAttrGetName(attr)
    @ccall Reactant_jll.libReactantExtra.sdyMeshAxisAttrGetName(
        attr::MlirAttribute
    )::MlirStringRef
end

function sdyMeshAxisAttrGetSize(attr)
    @ccall Reactant_jll.libReactantExtra.sdyMeshAxisAttrGetSize(attr::MlirAttribute)::Int64
end

function sdyAttributeIsAMeshAttr(attr)
    @ccall Reactant_jll.libReactantExtra.sdyAttributeIsAMeshAttr(attr::MlirAttribute)::Bool
end

function sdyMeshAttrGet(ctx, nAxes, axes, nDeviceIds, deviceIds)
    @ccall Reactant_jll.libReactantExtra.sdyMeshAttrGet(
        ctx::MlirContext,
        nAxes::Cptrdiff_t,
        axes::Ptr{MlirAttribute},
        nDeviceIds::Cptrdiff_t,
        deviceIds::Ptr{Int64},
    )::MlirAttribute
end

function sdyMeshAttrGetDeviceIdsSize(attr)
    @ccall Reactant_jll.libReactantExtra.sdyMeshAttrGetDeviceIdsSize(
        attr::MlirAttribute
    )::Int64
end

function sdyMeshAttrGetDeviceIdsElem(attr, pos)
    @ccall Reactant_jll.libReactantExtra.sdyMeshAttrGetDeviceIdsElem(
        attr::MlirAttribute, pos::Int64
    )::Int64
end

function sdyMeshAttrGetAxesSize(attr)
    @ccall Reactant_jll.libReactantExtra.sdyMeshAttrGetAxesSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function sdyMeshAttrGetAxesElem(attr, pos)
    @ccall Reactant_jll.libReactantExtra.sdyMeshAttrGetAxesElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::MlirAttribute
end

function sdyAttributeIsASubAxisInfoAttr(attr)
    @ccall Reactant_jll.libReactantExtra.sdyAttributeIsASubAxisInfoAttr(
        attr::MlirAttribute
    )::Bool
end

function sdySubAxisInfoAttrGet(ctx, preSize, size)
    @ccall Reactant_jll.libReactantExtra.sdySubAxisInfoAttrGet(
        ctx::MlirContext, preSize::Int64, size::Int64
    )::MlirAttribute
end

function sdySubAxisInfoAttrGetPreSize(attr)
    @ccall Reactant_jll.libReactantExtra.sdySubAxisInfoAttrGetPreSize(
        attr::MlirAttribute
    )::Int64
end

function sdySubAxisInfoAttrGetSize(attr)
    @ccall Reactant_jll.libReactantExtra.sdySubAxisInfoAttrGetSize(
        attr::MlirAttribute
    )::Int64
end

function sdyAttributeIsAnAxisRefAttr(attr)
    @ccall Reactant_jll.libReactantExtra.sdyAttributeIsAnAxisRefAttr(
        attr::MlirAttribute
    )::Bool
end

function sdyAxisRefAttrGet(ctx, name, subAxisInfo)
    @ccall Reactant_jll.libReactantExtra.sdyAxisRefAttrGet(
        ctx::MlirContext, name::MlirStringRef, subAxisInfo::MlirAttribute
    )::MlirAttribute
end

function sdyAxisRefAttrGetName(attr)
    @ccall Reactant_jll.libReactantExtra.sdyAxisRefAttrGetName(
        attr::MlirAttribute
    )::MlirStringRef
end

function sdyAxisRefAttrGetSubAxisInfo(attr)
    @ccall Reactant_jll.libReactantExtra.sdyAxisRefAttrGetSubAxisInfo(
        attr::MlirAttribute
    )::MlirAttribute
end

function sdyAttributeIsADimensionShardingAttr(attr)
    @ccall Reactant_jll.libReactantExtra.sdyAttributeIsADimensionShardingAttr(
        attr::MlirAttribute
    )::Bool
end

function sdyDimensionShardingAttrGet(ctx, nAxes, axes, isClosed, priority)
    @ccall Reactant_jll.libReactantExtra.sdyDimensionShardingAttrGet(
        ctx::MlirContext,
        nAxes::Cptrdiff_t,
        axes::Ptr{MlirAttribute},
        isClosed::Bool,
        priority::Int64,
    )::MlirAttribute
end

function sdyDimensionShardingAttrGetAxesSize(attr)
    @ccall Reactant_jll.libReactantExtra.sdyDimensionShardingAttrGetAxesSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function sdyDimensionShardingAttrGetAxesElem(attr, pos)
    @ccall Reactant_jll.libReactantExtra.sdyDimensionShardingAttrGetAxesElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::MlirAttribute
end

function sdyDimensionShardingAttrGetIsClosed(attr)
    @ccall Reactant_jll.libReactantExtra.sdyDimensionShardingAttrGetIsClosed(
        attr::MlirAttribute
    )::Bool
end

function sdyDimensionShardingAttrGetPriority(attr)
    @ccall Reactant_jll.libReactantExtra.sdyDimensionShardingAttrGetPriority(
        attr::MlirAttribute
    )::Int64
end

function sdyAttributeIsATensorShardingAttr(attr)
    @ccall Reactant_jll.libReactantExtra.sdyAttributeIsATensorShardingAttr(
        attr::MlirAttribute
    )::Bool
end

function sdyTensorShardingAttrGet(
    ctx,
    meshOrRef,
    nDimShardings,
    dimShardings,
    nReplicatedAxes,
    replicatedAxes,
    nUnreducedAxes,
    unreducedAxes,
    reductionOp,
)
    @ccall Reactant_jll.libReactantExtra.sdyTensorShardingAttrGet(
        ctx::MlirContext,
        meshOrRef::MlirAttribute,
        nDimShardings::Cptrdiff_t,
        dimShardings::Ptr{MlirAttribute},
        nReplicatedAxes::Cptrdiff_t,
        replicatedAxes::Ptr{MlirAttribute},
        nUnreducedAxes::Cptrdiff_t,
        unreducedAxes::Ptr{MlirAttribute},
        reductionOp::UInt32,
    )::MlirAttribute
end

function sdyTensorShardingAttrGetReductionOp(attr)
    @ccall Reactant_jll.libReactantExtra.sdyTensorShardingAttrGetReductionOp(
        attr::MlirAttribute
    )::UInt32
end

function sdyTensorShardingAttrGetMeshOrRef(attr)
    @ccall Reactant_jll.libReactantExtra.sdyTensorShardingAttrGetMeshOrRef(
        attr::MlirAttribute
    )::MlirAttribute
end

function sdyTensorShardingAttrGetDimShardingsSize(attr)
    @ccall Reactant_jll.libReactantExtra.sdyTensorShardingAttrGetDimShardingsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function sdyTensorShardingAttrGetDimShardingsElem(attr, pos)
    @ccall Reactant_jll.libReactantExtra.sdyTensorShardingAttrGetDimShardingsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::MlirAttribute
end

function sdyTensorShardingAttrGetReplicatedAxesSize(attr)
    @ccall Reactant_jll.libReactantExtra.sdyTensorShardingAttrGetReplicatedAxesSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function sdyTensorShardingAttrGetReplicatedAxesElem(attr, pos)
    @ccall Reactant_jll.libReactantExtra.sdyTensorShardingAttrGetReplicatedAxesElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::MlirAttribute
end

function sdyTensorShardingAttrGetUnreducedAxesSize(attr)
    @ccall Reactant_jll.libReactantExtra.sdyTensorShardingAttrGetUnreducedAxesSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function sdyTensorShardingAttrGetUnreducedAxesElem(attr, pos)
    @ccall Reactant_jll.libReactantExtra.sdyTensorShardingAttrGetUnreducedAxesElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::MlirAttribute
end

function sdyAttributeIsATensorShardingPerValueAttr(attr)
    @ccall Reactant_jll.libReactantExtra.sdyAttributeIsATensorShardingPerValueAttr(
        attr::MlirAttribute
    )::Bool
end

function sdyTensorShardingPerValueAttrGet(ctx, nShardings, shardings)
    @ccall Reactant_jll.libReactantExtra.sdyTensorShardingPerValueAttrGet(
        ctx::MlirContext, nShardings::Cptrdiff_t, shardings::Ptr{MlirAttribute}
    )::MlirAttribute
end

function sdyTensorShardingPerValueAttrGetShardingsSize(attr)
    @ccall Reactant_jll.libReactantExtra.sdyTensorShardingPerValueAttrGetShardingsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function sdyTensorShardingPerValueAttrGetShardingsElem(attr, pos)
    @ccall Reactant_jll.libReactantExtra.sdyTensorShardingPerValueAttrGetShardingsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::MlirAttribute
end

function sdyAttributeIsADimMappingAttr(attr)
    @ccall Reactant_jll.libReactantExtra.sdyAttributeIsADimMappingAttr(
        attr::MlirAttribute
    )::Bool
end

function sdyDimMappingAttrGet(ctx, nFactorIndices, factorIndices)
    @ccall Reactant_jll.libReactantExtra.sdyDimMappingAttrGet(
        ctx::MlirContext, nFactorIndices::Cptrdiff_t, factorIndices::Ptr{Int64}
    )::MlirAttribute
end

function sdyDimMappingAttrGetFactorIndicesSize(attr)
    @ccall Reactant_jll.libReactantExtra.sdyDimMappingAttrGetFactorIndicesSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function sdyDimMappingAttrGetFactorIndicesElem(attr, pos)
    @ccall Reactant_jll.libReactantExtra.sdyDimMappingAttrGetFactorIndicesElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function sdyAttributeIsATensorMappingAttr(attr)
    @ccall Reactant_jll.libReactantExtra.sdyAttributeIsATensorMappingAttr(
        attr::MlirAttribute
    )::Bool
end

function sdyTensorMappingAttrGet(ctx, nMappings, mappings)
    @ccall Reactant_jll.libReactantExtra.sdyTensorMappingAttrGet(
        ctx::MlirContext, nMappings::Cptrdiff_t, mappings::Ptr{MlirAttribute}
    )::MlirAttribute
end

function sdyTensorMappingAttrGetRank(attr)
    @ccall Reactant_jll.libReactantExtra.sdyTensorMappingAttrGetRank(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function sdyTensorMappingAttrGetDimMappingsSize(attr)
    @ccall Reactant_jll.libReactantExtra.sdyTensorMappingAttrGetDimMappingsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function sdyTensorMappingAttrGetDimMappingsElem(attr, pos)
    @ccall Reactant_jll.libReactantExtra.sdyTensorMappingAttrGetDimMappingsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::MlirAttribute
end

function sdyAttributeIsAOpShardingRuleAttr(attr)
    @ccall Reactant_jll.libReactantExtra.sdyAttributeIsAOpShardingRuleAttr(
        attr::MlirAttribute
    )::Bool
end

function sdyOpShardingRuleAttrGet(
    ctx,
    nFactorSizes,
    factorSizes,
    nOperandMappings,
    operandMappings,
    nResultMappings,
    resultMappings,
    nReductionFactors,
    reductionFactors,
    nNeedReplicationFactors,
    needReplicationFactors,
    nPermutationFactors,
    permutationFactors,
    nBlockedPropagationFactors,
    blockedPropagationFactors,
    isCustomRule,
)
    @ccall Reactant_jll.libReactantExtra.sdyOpShardingRuleAttrGet(
        ctx::MlirContext,
        nFactorSizes::Cptrdiff_t,
        factorSizes::Ptr{Int64},
        nOperandMappings::Cptrdiff_t,
        operandMappings::Ptr{MlirAttribute},
        nResultMappings::Cptrdiff_t,
        resultMappings::Ptr{MlirAttribute},
        nReductionFactors::Cptrdiff_t,
        reductionFactors::Ptr{Int64},
        nNeedReplicationFactors::Cptrdiff_t,
        needReplicationFactors::Ptr{Int64},
        nPermutationFactors::Cptrdiff_t,
        permutationFactors::Ptr{Int64},
        nBlockedPropagationFactors::Cptrdiff_t,
        blockedPropagationFactors::Ptr{Int64},
        isCustomRule::Bool,
    )::MlirAttribute
end

function sdyOpShardingRuleAttrGetIsCustom(attr)
    @ccall Reactant_jll.libReactantExtra.sdyOpShardingRuleAttrGetIsCustom(
        attr::MlirAttribute
    )::Bool
end

function sdyOpShardingRuleAttrGetFactorSizesSize(attr)
    @ccall Reactant_jll.libReactantExtra.sdyOpShardingRuleAttrGetFactorSizesSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function sdyOpShardingRuleAttrGetFactorSizesElem(attr, pos)
    @ccall Reactant_jll.libReactantExtra.sdyOpShardingRuleAttrGetFactorSizesElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function sdyOpShardingRuleAttrGetOperandMappingsSize(attr)
    @ccall Reactant_jll.libReactantExtra.sdyOpShardingRuleAttrGetOperandMappingsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function sdyOpShardingRuleAttrGetOperandMappingsElem(attr, pos)
    @ccall Reactant_jll.libReactantExtra.sdyOpShardingRuleAttrGetOperandMappingsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::MlirAttribute
end

function sdyOpShardingRuleAttrGetResultMappingsSize(attr)
    @ccall Reactant_jll.libReactantExtra.sdyOpShardingRuleAttrGetResultMappingsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function sdyOpShardingRuleAttrGetResultMappingsElem(attr, pos)
    @ccall Reactant_jll.libReactantExtra.sdyOpShardingRuleAttrGetResultMappingsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::MlirAttribute
end

function sdyOpShardingRuleAttrGetReductionFactorsSize(attr)
    @ccall Reactant_jll.libReactantExtra.sdyOpShardingRuleAttrGetReductionFactorsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function sdyOpShardingRuleAttrGetReductionFactorsElem(attr, pos)
    @ccall Reactant_jll.libReactantExtra.sdyOpShardingRuleAttrGetReductionFactorsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function sdyOpShardingRuleAttrGetNeedReplicationFactorsSize(attr)
    @ccall Reactant_jll.libReactantExtra.sdyOpShardingRuleAttrGetNeedReplicationFactorsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function sdyOpShardingRuleAttrGetNeedReplicationFactorsElem(attr, pos)
    @ccall Reactant_jll.libReactantExtra.sdyOpShardingRuleAttrGetNeedReplicationFactorsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function sdyOpShardingRuleAttrGetPermutationFactorsSize(attr)
    @ccall Reactant_jll.libReactantExtra.sdyOpShardingRuleAttrGetPermutationFactorsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function sdyOpShardingRuleAttrGetPermutationFactorsElem(attr, pos)
    @ccall Reactant_jll.libReactantExtra.sdyOpShardingRuleAttrGetPermutationFactorsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function sdyOpShardingRuleAttrGetBlockedPropagationFactorsSize(attr)
    @ccall Reactant_jll.libReactantExtra.sdyOpShardingRuleAttrGetBlockedPropagationFactorsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function sdyOpShardingRuleAttrGetBlockedPropagationFactorsElem(attr, pos)
    @ccall Reactant_jll.libReactantExtra.sdyOpShardingRuleAttrGetBlockedPropagationFactorsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function sdyAttributeIsAManualAxesAttr(attr)
    @ccall Reactant_jll.libReactantExtra.sdyAttributeIsAManualAxesAttr(
        attr::MlirAttribute
    )::Bool
end

function sdyManualAxesAttrGet(ctx, nAxes, axes)
    @ccall Reactant_jll.libReactantExtra.sdyManualAxesAttrGet(
        ctx::MlirContext, nAxes::Cptrdiff_t, axes::Ptr{MlirAttribute}
    )::MlirAttribute
end

function sdyManualAxesAttrGetAxesSize(attr)
    @ccall Reactant_jll.libReactantExtra.sdyManualAxesAttrGetAxesSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function sdyManualAxesAttrGetAxesElem(attr, pos)
    @ccall Reactant_jll.libReactantExtra.sdyManualAxesAttrGetAxesElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::MlirStringRef
end

@cenum EnzymeXlaLapackLayout::UInt32 begin
    ENZYMEXLA_LAPACK_LAYOUT_COLUMN_MAJOR = 0x0000000000000000
    ENZYMEXLA_LAPACK_LAYOUT_ROW_MAJOR = 0x0000000000000001
end

function enzymexlaLapackLayoutAttrGet(ctx, layout)
    @ccall Reactant_jll.libReactantExtra.enzymexlaLapackLayoutAttrGet(
        ctx::MlirContext, layout::EnzymeXlaLapackLayout
    )::MlirAttribute
end

@cenum EnzymeXlaLapackTranspose::UInt32 begin
    ENZYMEXLA_LAPACK_TRANSPOSE_NONE = 0x0000000000000000
    ENZYMEXLA_LAPACK_TRANSPOSE_TRANSPOSE = 0x0000000000000001
    ENZYMEXLA_LAPACK_TRANSPOSE_CONJUGATE_TRANSPOSE = 0x0000000000000002
end

function enzymexlaLapackTransposeAttrGet(ctx, transpose)
    @ccall Reactant_jll.libReactantExtra.enzymexlaLapackTransposeAttrGet(
        ctx::MlirContext, transpose::EnzymeXlaLapackTranspose
    )::MlirAttribute
end

@cenum EnzymeXlaLapackSide::UInt32 begin
    ENZYMEXLA_LAPACK_SIDE_LEFT = 0x0000000000000000
    ENZYMEXLA_LAPACK_SIDE_RIGHT = 0x0000000000000001
end

function enzymexlaLapackSideAttrGet(ctx, side)
    @ccall Reactant_jll.libReactantExtra.enzymexlaLapackSideAttrGet(
        ctx::MlirContext, side::EnzymeXlaLapackSide
    )::MlirAttribute
end

@cenum EnzymeXlaLapackUplo::UInt32 begin
    ENZYMEXLA_LAPACK_UPLO_LOWER = 0x0000000000000000
    ENZYMEXLA_LAPACK_UPLO_UPPER = 0x0000000000000001
    ENZYMEXLA_LAPACK_UPLO_FULL = 0x0000000000000002
end

function enzymexlaLapackUploAttrGet(ctx, uplo)
    @ccall Reactant_jll.libReactantExtra.enzymexlaLapackUploAttrGet(
        ctx::MlirContext, uplo::EnzymeXlaLapackUplo
    )::MlirAttribute
end

@cenum EnzymeXlaQRAlgorithm::UInt32 begin
    ENZYMEXLA_QR_ALGORITHM_NONE = 0x0000000000000000
    ENZYMEXLA_QR_ALGORITHM_HOUSEHOLDER = 0x0000000000000001
end

function enzymexlaQRAlgorithmAttrGet(ctx, algorithm)
    @ccall Reactant_jll.libReactantExtra.enzymexlaQRAlgorithmAttrGet(
        ctx::MlirContext, algorithm::EnzymeXlaQRAlgorithm
    )::MlirAttribute
end

@cenum EnzymeXlaSVDAlgorithm::UInt32 begin
    ENZYMEXLA_SVD_ALGORITHM_NONE = 0x0000000000000000
    ENZYMEXLA_SVD_ALGORITHM_QRITERATION = 0x0000000000000001
    ENZYMEXLA_SVD_ALGORITHM_DIVIDEANDCONQUER = 0x0000000000000002
    ENZYMEXLA_SVD_ALGORITHM_JACOBI = 0x0000000000000003
end

function enzymexlaSVDAlgorithmAttrGet(ctx, algorithm)
    @ccall Reactant_jll.libReactantExtra.enzymexlaSVDAlgorithmAttrGet(
        ctx::MlirContext, algorithm::EnzymeXlaSVDAlgorithm
    )::MlirAttribute
end

@cenum EnzymeXlaGeluApproximation::UInt32 begin
    ENZYMEXLA_GELU_APPROXIMATION_NONE = 0x0000000000000000
    ENZYMEXLA_GELU_APPROXIMATION_TANH = 0x0000000000000001
    ENZYMEXLA_GELU_APPROXIMATION_SIGMOID = 0x0000000000000002
end

function enzymexlaGeluApproximationAttrGet(ctx, approximation)
    @ccall Reactant_jll.libReactantExtra.enzymexlaGeluApproximationAttrGet(
        ctx::MlirContext, approximation::EnzymeXlaGeluApproximation
    )::MlirAttribute
end

@cenum EnzymeXlaMPIDatatype::UInt32 begin
    ENZYMEXLA_MPI_DATATYPE_NULL = 0x0000000000000000
    ENZYMEXLA_MPI_INT8_T = 0x0000000000000001
    ENZYMEXLA_MPI_UINT8_T = 0x0000000000000002
    ENZYMEXLA_MPI_INT16_T = 0x0000000000000003
    ENZYMEXLA_MPI_UINT16_T = 0x0000000000000004
    ENZYMEXLA_MPI_INT32_T = 0x0000000000000005
    ENZYMEXLA_MPI_UINT32_T = 0x0000000000000006
    ENZYMEXLA_MPI_INT64_T = 0x0000000000000007
    ENZYMEXLA_MPI_UINT64_T = 0x0000000000000008
    ENZYMEXLA_MPI_BYTE = 0x0000000000000009
    ENZYMEXLA_MPI_SHORT = 0x000000000000000a
    ENZYMEXLA_MPI_UNSIGNED_SHORT = 0x000000000000000b
    ENZYMEXLA_MPI_INT = 0x000000000000000c
    ENZYMEXLA_MPI_UNSIGNED = 0x000000000000000d
    ENZYMEXLA_MPI_LONG = 0x000000000000000e
    ENZYMEXLA_MPI_UNSIGNED_LONG = 0x000000000000000f
    ENZYMEXLA_MPI_LONG_LONG_INT = 0x0000000000000010
    ENZYMEXLA_MPI_UNSIGNED_LONG_LONG = 0x0000000000000011
    ENZYMEXLA_MPI_CHAR = 0x0000000000000012
    ENZYMEXLA_MPI_SIGNED_CHAR = 0x0000000000000013
    ENZYMEXLA_MPI_UNSIGNED_CHAR = 0x0000000000000014
    ENZYMEXLA_MPI_WCHAR = 0x0000000000000015
    ENZYMEXLA_MPI_FLOAT = 0x0000000000000016
    ENZYMEXLA_MPI_DOUBLE = 0x0000000000000017
    ENZYMEXLA_MPI_C_FLOAT_COMPLEX = 0x0000000000000018
    ENZYMEXLA_MPI_C_DOUBLE_COMPLEX = 0x0000000000000019
    ENZYMEXLA_MPI_C_BOOL = 0x000000000000001a
end

function enzymexlaMPIDatatypeAttrGet(ctx, datatype)
    @ccall Reactant_jll.libReactantExtra.enzymexlaMPIDatatypeAttrGet(
        ctx::MlirContext, datatype::EnzymeXlaMPIDatatype
    )::MlirAttribute
end

@cenum EnzymeXlaMPIOp::UInt32 begin
    ENZYMEXLA_MPI_OP_NULL = 0x0000000000000000
    ENZYMEXLA_MPI_BAND = 0x0000000000000001
    ENZYMEXLA_MPI_BOR = 0x0000000000000002
    ENZYMEXLA_MPI_BXOR = 0x0000000000000003
    ENZYMEXLA_MPI_LAND = 0x0000000000000004
    ENZYMEXLA_MPI_LOR = 0x0000000000000005
    ENZYMEXLA_MPI_LXOR = 0x0000000000000006
    ENZYMEXLA_MPI_MAX = 0x0000000000000007
    ENZYMEXLA_MPI_MIN = 0x0000000000000008
    ENZYMEXLA_MPI_PROD = 0x0000000000000009
    ENZYMEXLA_MPI_REPLACE = 0x000000000000000a
    ENZYMEXLA_MPI_SUM = 0x000000000000000b
    ENZYMEXLA_MPI_NO_OP = 0x000000000000000c
end

function enzymexlaMPIOpAttrGet(ctx, op)
    @ccall Reactant_jll.libReactantExtra.enzymexlaMPIOpAttrGet(
        ctx::MlirContext, op::EnzymeXlaMPIOp
    )::MlirAttribute
end

@cenum EnzymeXlaGuaranteedAnalysisResult::UInt32 begin
    ENZYMEXLA_GUARANTEED_ANALYSIS_RESULT_GUARANTEED = 0x0000000000000000
    ENZYMEXLA_GUARANTEED_ANALYSIS_RESULT_NOTGUARANTEED = 0x0000000000000001
    ENZYMEXLA_GUARANTEED_ANALYSIS_RESULT_UNKNOWN = 0x0000000000000002
end

function enzymexlaGuaranteedAnalysisResultAttrGet(ctx, result)
    @ccall Reactant_jll.libReactantExtra.enzymexlaGuaranteedAnalysisResultAttrGet(
        ctx::MlirContext, result::EnzymeXlaGuaranteedAnalysisResult
    )::MlirAttribute
end

"""
    EnzymeXLAPropagateDirection

Enum for propagation direction (reshape/transpose).
"""
@cenum EnzymeXLAPropagateDirection::UInt32 begin
    ENZYMEXLA_PROPAGATE_NONE = 0x0000000000000000
    ENZYMEXLA_PROPAGATE_UP = 0x0000000000000001
    ENZYMEXLA_PROPAGATE_DOWN = 0x0000000000000002
end

"""
    EnzymeXLATransformPassesOptions

Options that control which transform passes are generated.
"""
struct EnzymeXLATransformPassesOptions
    max_constant_threshold::Int64
    while_unroll_threshold::Int64
    reshape_propagate::EnzymeXLAPropagateDirection
    transpose_propagate::EnzymeXLAPropagateDirection
    no_nan::Bool
    all_finite::Bool
    dus_to_concat::Bool
    dus_slice_simplify::Bool
    sum_to_reducewindow::Bool
    sum_to_conv::Bool
    aggressive_sum_to_conv::Bool
    while_concat::Bool
    aggressive_propagation::Bool
    is_sharded::Bool
    raise_shlo_to_blas_lapack::Bool
    recognize_comms::Bool
    lower_comms::Bool
    enable_self_to_convolution_like_passes::Bool
    enable_structured_tensors_detection_passes::Bool
    enable_structured_tensors_passes::Bool
    enable_scatter_gather_optimization_passes::Bool
    enable_slice_to_batch_passes::Bool
    enable_reduce_slice_fusion_passes::Bool
    enable_concat_to_batch_passes::Bool
    enable_loop_raising_passes::Bool
    enable_licm_optimization_passes::Bool
    loop_unswitch_threshold::Int64
    enable_pad_optimization_passes::Bool
    excluded_passes::Ptr{Cstring}
    num_excluded_passes::Csize_t
end

"""
    enzymexlaGetTransformPassesList(options, mainPasses, lowerPasses)

Returns the transform passes list as a semicolon-separated string. The caller must free the returned string using [`enzymexlaFreeTransformPassesList`](@ref).

Two separate lists are produced: - `mainPasses`: the primary transform pass list - `lowerPasses`: the lowering transform pass list (for lower\\_comms)

Each is returned as a semicolon-separated string of pass patterns.
"""
function enzymexlaGetTransformPassesList(options, mainPasses, lowerPasses)
    @ccall Reactant_jll.libReactantExtra.enzymexlaGetTransformPassesList(
        options::Ptr{EnzymeXLATransformPassesOptions},
        mainPasses::Ptr{Cstring},
        lowerPasses::Ptr{Cstring},
    )::Cvoid
end

"""
    enzymexlaFreeTransformPassesList(passes)

Free a string returned by [`enzymexlaGetTransformPassesList`](@ref).
"""
function enzymexlaFreeTransformPassesList(passes)
    @ccall Reactant_jll.libReactantExtra.enzymexlaFreeTransformPassesList(
        passes::Cstring
    )::Cvoid
end

function mlirGetDialectHandle__triton__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__triton__()::MlirDialectHandle
end

function mlirTritonPointerTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirTritonPointerTypeGetTypeID()::MlirTypeID
end

function mlirTritonPointerTypeGet(pointeeType, addressSpace)
    @ccall Reactant_jll.libReactantExtra.mlirTritonPointerTypeGet(
        pointeeType::MlirType, addressSpace::Cint
    )::MlirType
end

function mlirTritonIsAPointer(type)
    @ccall Reactant_jll.libReactantExtra.mlirTritonIsAPointer(type::MlirType)::Bool
end

function mlirTritonPointerTypeGetPointeeType(pointerType)
    @ccall Reactant_jll.libReactantExtra.mlirTritonPointerTypeGetPointeeType(
        pointerType::MlirType
    )::MlirType
end

function mlirTritonPointerTypeGetAddressSpace(pointerType)
    @ccall Reactant_jll.libReactantExtra.mlirTritonPointerTypeGetAddressSpace(
        pointerType::MlirType
    )::Cint
end

function mlirTritonInferReduceOpEncoding(operandEncoding, axis)
    @ccall Reactant_jll.libReactantExtra.mlirTritonInferReduceOpEncoding(
        operandEncoding::MlirAttribute, axis::Cint
    )::MlirAttribute
end

function mlirGetDialectHandle__tpu__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__tpu__()::MlirDialectHandle
end

function mlirTPUAnalyzePotentialCommunication(op, has_communication, has_custom_barrier)
    @ccall Reactant_jll.libReactantExtra.mlirTPUAnalyzePotentialCommunication(
        op::MlirOperation, has_communication::Ptr{Bool}, has_custom_barrier::Ptr{Bool}
    )::Cvoid
end

function mlirTpuRegisterMosaicSerdePass()
    @ccall Reactant_jll.libReactantExtra.mlirTpuRegisterMosaicSerdePass()::Cvoid
end

function mlirTpuFloat8EXMYTypeGetUnderlyingType(exmy_type)
    @ccall Reactant_jll.libReactantExtra.mlirTpuFloat8EXMYTypeGetUnderlyingType(
        exmy_type::MlirType
    )::MlirType
end

function mlirTpuIsAFloat8EXMYType(type)
    @ccall Reactant_jll.libReactantExtra.mlirTpuIsAFloat8EXMYType(type::MlirType)::Bool
end

function mlirTpuFloat8EXMYTypeGet(ctx, exmy_type)
    @ccall Reactant_jll.libReactantExtra.mlirTpuFloat8EXMYTypeGet(
        ctx::MlirContext, exmy_type::MlirType
    )::MlirType
end

function mlirMosaicGpuIsATileTransformAttr(attr)
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuIsATileTransformAttr(
        attr::MlirAttribute
    )::Bool
end

function mlirMosaicGpuTileTransformAttrGet(ctx, tiling)
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuTileTransformAttrGet(
        ctx::MlirContext, tiling::MlirAttribute
    )::MlirAttribute
end

function mlirMosaicGpuTileTransformAttrGetTiling(attr)
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuTileTransformAttrGetTiling(
        attr::MlirAttribute
    )::MlirAttribute
end

function mlirMosaicGpuTileTransformAttrGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuTileTransformAttrGetTypeID()::MlirTypeID
end

function mlirMosaicGpuIsASwizzleTransformAttr(attr)
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuIsASwizzleTransformAttr(
        attr::MlirAttribute
    )::Bool
end

function mlirMosaicGpuSwizzleTransformAttrGet(ctx, swizzle)
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuSwizzleTransformAttrGet(
        ctx::MlirContext, swizzle::Int32
    )::MlirAttribute
end

function mlirMosaicGpuSwizzleTransformAttrGetSwizzle(attr)
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuSwizzleTransformAttrGetSwizzle(
        attr::MlirAttribute
    )::Int32
end

function mlirMosaicGpuSwizzleTransformAttrGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuSwizzleTransformAttrGetTypeID()::MlirTypeID
end

function mlirMosaicGpuIsAWGSplatFragLayoutAttr(attr)
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuIsAWGSplatFragLayoutAttr(
        attr::MlirAttribute
    )::Bool
end

function mlirMosaicGpuWGSplatFragLayoutAttrGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuWGSplatFragLayoutAttrGetTypeID()::MlirTypeID
end

function mlirMosaicGpuWGSplatFragLayoutAttrGet(ctx, shape)
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuWGSplatFragLayoutAttrGet(
        ctx::MlirContext, shape::MlirAttribute
    )::MlirAttribute
end

function mlirMosaicGpuWGSplatFragLayoutAttrGetShape(attr)
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuWGSplatFragLayoutAttrGetShape(
        attr::MlirAttribute
    )::MlirAttribute
end

function mlirMosaicGpuIsAWGStridedFragLayoutAttr(attr)
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuIsAWGStridedFragLayoutAttr(
        attr::MlirAttribute
    )::Bool
end

function mlirMosaicGpuWGStridedFragLayoutAttrGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuWGStridedFragLayoutAttrGetTypeID()::MlirTypeID
end

function mlirMosaicGpuWGStridedFragLayoutAttrGet(ctx, shape, vector_size)
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuWGStridedFragLayoutAttrGet(
        ctx::MlirContext, shape::MlirAttribute, vector_size::Int32
    )::MlirAttribute
end

function mlirMosaicGpuWGStridedFragLayoutAttrGetShape(attr)
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuWGStridedFragLayoutAttrGetShape(
        attr::MlirAttribute
    )::MlirAttribute
end

function mlirMosaicGpuWGStridedFragLayoutAttrGetVectorSize(attr)
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuWGStridedFragLayoutAttrGetVectorSize(
        attr::MlirAttribute
    )::Int32
end

function mlirMosaicGpuIsAReplicatedAttr(attr)
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuIsAReplicatedAttr(
        attr::MlirAttribute
    )::Bool
end

function mlirMosaicGpuReplicatedAttrGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuReplicatedAttrGetTypeID()::MlirTypeID
end

function mlirMosaicGpuReplicatedAttrGet(ctx, times)
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuReplicatedAttrGet(
        ctx::MlirContext, times::Int32
    )::MlirAttribute
end

function mlirMosaicGpuReplicatedAttrGetTimes(attr)
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuReplicatedAttrGetTimes(
        attr::MlirAttribute
    )::Int32
end

function mlirMosaicGpuIsATiledLayoutAttr(attr)
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuIsATiledLayoutAttr(
        attr::MlirAttribute
    )::Bool
end

function mlirMosaicGpuTiledLayoutAttrGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuTiledLayoutAttrGetTypeID()::MlirTypeID
end

function mlirMosaicGpuTiledLayoutAttrGet(ctx, tiling, warp_dims, lane_dims, vector_dim)
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuTiledLayoutAttrGet(
        ctx::MlirContext,
        tiling::MlirAttribute,
        warp_dims::MlirAttribute,
        lane_dims::MlirAttribute,
        vector_dim::Int32,
    )::MlirAttribute
end

function mlirMosaicGpuTiledLayoutAttrGetTiling(attr)
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuTiledLayoutAttrGetTiling(
        attr::MlirAttribute
    )::MlirAttribute
end

function mlirMosaicGpuTiledLayoutAttrGetWarpDims(attr)
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuTiledLayoutAttrGetWarpDims(
        attr::MlirAttribute
    )::MlirAttribute
end

function mlirMosaicGpuTiledLayoutAttrGetLaneDims(attr)
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuTiledLayoutAttrGetLaneDims(
        attr::MlirAttribute
    )::MlirAttribute
end

function mlirMosaicGpuTiledLayoutAttrGetVectorDim(attr)
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuTiledLayoutAttrGetVectorDim(
        attr::MlirAttribute
    )::Int32
end

function mlirMosaicGpuIsACopyPartitionAttr(attr)
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuIsACopyPartitionAttr(
        attr::MlirAttribute
    )::Bool
end

function mlirMosaicGpuIsACopyReplicatedAttr(attr)
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuIsACopyReplicatedAttr(
        attr::MlirAttribute
    )::Bool
end

function mlirMosaicGpuCopyReplicatedAttrGet(ctx)
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuCopyReplicatedAttrGet(
        ctx::MlirContext
    )::MlirAttribute
end

function mlirMosaicGpuCopyReplicatedAttrGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuCopyReplicatedAttrGetTypeID()::MlirTypeID
end

function mlirMosaicGpuIsACopyPartitionedAttr(attr)
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuIsACopyPartitionedAttr(
        attr::MlirAttribute
    )::Bool
end

function mlirMosaicGpuCopyPartitionedAttrGet(ctx, axis)
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuCopyPartitionedAttrGet(
        ctx::MlirContext, axis::Int32
    )::MlirAttribute
end

function mlirMosaicGpuCopyPartitionedAttrGetAxis(attr)
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuCopyPartitionedAttrGetAxis(
        attr::MlirAttribute
    )::Int32
end

function mlirMosaicGpuCopyPartitionedAttrGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuCopyPartitionedAttrGetTypeID()::MlirTypeID
end

function mlirGetDialectHandle__mosaic_gpu__()
    @ccall Reactant_jll.libReactantExtra.mlirGetDialectHandle__mosaic_gpu__()::MlirDialectHandle
end

function mlirDialectRegistryInsertMosaicGpuInlinerExtensions(registry)
    @ccall Reactant_jll.libReactantExtra.mlirDialectRegistryInsertMosaicGpuInlinerExtensions(
        registry::MlirDialectRegistry
    )::Cvoid
end

function mlirMosaicGpuIsABarrierType(type)
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuIsABarrierType(type::MlirType)::Bool
end

function mlirMosaicGpuBarrierTypeGet(ctx, orders_tensor_core)
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuBarrierTypeGet(
        ctx::MlirContext, orders_tensor_core::Bool
    )::MlirType
end

function mlirMosaicGpuBarrierTypeGetOrdersTensorCore(type)
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuBarrierTypeGetOrdersTensorCore(
        type::MlirType
    )::Bool
end

function mlirMosaicGpuBarrierTypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuBarrierTypeGetTypeID()::MlirTypeID
end

function mlirMosaicGpuIsAB6x16P32Type(type)
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuIsAB6x16P32Type(type::MlirType)::Bool
end

function mlirMosaicGpuB6x16P32TypeGet(ctx, element_type)
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuB6x16P32TypeGet(
        ctx::MlirContext, element_type::MlirType
    )::MlirType
end

function mlirMosaicGpuB6x16P32TypeGetElementType(type)
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuB6x16P32TypeGetElementType(
        type::MlirType
    )::MlirType
end

function mlirMosaicGpuB6x16P32TypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuB6x16P32TypeGetTypeID()::MlirTypeID
end

function mlirMosaicGpuIsAP2B6Type(type)
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuIsAP2B6Type(type::MlirType)::Bool
end

function mlirMosaicGpuP2B6TypeGet(ctx, element_type)
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuP2B6TypeGet(
        ctx::MlirContext, element_type::MlirType
    )::MlirType
end

function mlirMosaicGpuP2B6TypeGetElementType(type)
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuP2B6TypeGetElementType(
        type::MlirType
    )::MlirType
end

function mlirMosaicGpuP2B6TypeGetTypeID()
    @ccall Reactant_jll.libReactantExtra.mlirMosaicGpuP2B6TypeGetTypeID()::MlirTypeID
end

@cenum EnzymeRngDistribution::UInt32 begin
    EnzymeRngDistribution_Uniform = 0x0000000000000000
    EnzymeRngDistribution_Normal = 0x0000000000000001
    EnzymeRngDistribution_MultiNormal = 0x0000000000000002
end

function enzymeRngDistributionAttrGet(ctx, dist)
    @ccall Reactant_jll.libReactantExtra.enzymeRngDistributionAttrGet(
        ctx::MlirContext, dist::EnzymeRngDistribution
    )::MlirAttribute
end

@cenum EnzymeSupportKind::UInt32 begin
    EnzymeSupportKind_Real = 0x0000000000000000
    EnzymeSupportKind_Positive = 0x0000000000000001
    EnzymeSupportKind_UnitInterval = 0x0000000000000002
    EnzymeSupportKind_Interval = 0x0000000000000003
    EnzymeSupportKind_GreaterThan = 0x0000000000000004
    EnzymeSupportKind_LessThan = 0x0000000000000005
end

function enzymeSupportAttrGet(
    ctx, kind, hasLowerBound, lowerBound, hasUpperBound, upperBound
)
    @ccall Reactant_jll.libReactantExtra.enzymeSupportAttrGet(
        ctx::MlirContext,
        kind::EnzymeSupportKind,
        hasLowerBound::Bool,
        lowerBound::Cdouble,
        hasUpperBound::Bool,
        upperBound::Cdouble,
    )::MlirAttribute
end

function enzymeHMCConfigAttrGet(ctx, trajectoryLength, adaptStepSize, adaptMassMatrix)
    @ccall Reactant_jll.libReactantExtra.enzymeHMCConfigAttrGet(
        ctx::MlirContext,
        trajectoryLength::Cdouble,
        adaptStepSize::Bool,
        adaptMassMatrix::Bool,
    )::MlirAttribute
end

function enzymeNUTSConfigAttrGet(
    ctx, maxTreeDepth, hasMaxDeltaEnergy, maxDeltaEnergy, adaptStepSize, adaptMassMatrix
)
    @ccall Reactant_jll.libReactantExtra.enzymeNUTSConfigAttrGet(
        ctx::MlirContext,
        maxTreeDepth::Int64,
        hasMaxDeltaEnergy::Bool,
        maxDeltaEnergy::Cdouble,
        adaptStepSize::Bool,
        adaptMassMatrix::Bool,
    )::MlirAttribute
end

function enzymeSymbolAttrGet(ctx, ptr)
    @ccall Reactant_jll.libReactantExtra.enzymeSymbolAttrGet(
        ctx::MlirContext, ptr::UInt64
    )::MlirAttribute
end

struct JLHloCostAnalysisProperties
    flops::Cfloat
    transcendentals::Cfloat
    bytes_accessed::Cfloat
    optimal_seconds::Cfloat
    utilization::Cfloat
    operand0_utilization::Cfloat
    operand1_utilization::Cfloat
    operand0_bytes_accessed::Cfloat
    operand1_bytes_accessed::Cfloat
    output_root_bytes_accessed::Cfloat
    reserved0::Cfloat
end

struct AllocationInfo
    buffer::Ptr{Cint}
    size::Csize_t
end

struct JLEstimateRunTimeData
    flops::Int64
    bytes_read::Int64
    bytes_written::Int64
    read_time_ns::Int64
    write_time_ns::Int64
    compute_time_ns::Int64
    execution_time_ns::Int64
end

struct JLAllocatorStats
    num_allocs::Int64
    bytes_in_use::Int64
    peak_bytes_in_use::Int64
    largest_alloc_size::Int64
    bytes_limit::Int64
    bytes_reserved::Int64
    peak_bytes_reserved::Int64
    bytes_reservable_limit::Int64
    largest_free_block_bytes::Int64
    pool_bytes::Int64
    peak_pool_bytes::Int64
end

struct DeviceProperties
    totalGlobalMem::Csize_t
    sharedMemPerBlock::Csize_t
    regsPerBlock::Cint
    warpSize::Cint
    maxThreadsPerBlock::Cint
    maxThreadsDim::NTuple{3,Cint}
    maxGridSize::NTuple{3,Cint}
    totalConstMem::Csize_t
    major::Cint
    minor::Cint
    multiProcessorCount::Cint
    canMapHostMemory::Cint
    l2CacheSize::Cint
    maxThreadsPerMultiProcessor::Cint
end

struct DistributedRuntimeClientOptions
    node_id::Int32
    rpc_timeout_in_seconds::Int32
    init_timeout_in_seconds::Int32
    shutdown_timeout_in_minutes::Int32
    heartbeat_timeout_in_seconds::Int32
    use_compression::Bool
    shutdown_on_destruction::Bool
    poll_for_error_from_service_at_startup::Bool
    recoverable::Bool
end

struct DistributedRuntimeServiceOptions
    num_nodes::Int32
    recoverable::Bool
    heartbeat_timeout_in_seconds::Int32
    cluster_register_timeout_in_minutes::Int32
    shutdown_timeout_in_minutes::Int32
end

const HeldPjRtClient = Cvoid

const HeldIfrtConstSharding = Cvoid

const LinkableRuntime = Cvoid

const Operation = Cvoid

const DeviceDescription = Cvoid

const Client = Cvoid

const Memory = Cvoid

const PjRtBuffer = Cvoid

const IfRtFutureType = Cvoid

const PjRtClient = Cvoid

const HloComputation = Cvoid

const HloModule = Cvoid

const FutureType = Cvoid

const Device = Cvoid

const HeldIfrtLoadedExecutable = Cvoid

const HeldHloModule = Cvoid

const PjRtLoadedExecutable = Cvoid

const HloSharding = Cvoid

const HeldDistributedRuntimeClient = Cvoid

const DistributedRuntimeService = Cvoid

const Module = Cvoid

const LLVMContext = Cvoid

const GrpcServer = Cvoid

const HloInstruction = Cvoid

const GPUPerformanceModel = Cvoid

const ProfilerServer = Cvoid

const HeldPjRtBuffer = Cvoid

const HeldIfrtSharding = Cvoid

const MemoryKind = Cvoid

const PjRtDevice = Cvoid

const ProfilerSession = Cvoid

const LocalExecutable = Cvoid

const OpSharding = Cvoid

const PJRT_Api = Cvoid

const HeldIfrtArray = Cvoid

function ReactantHandleCuResult(curesult)
    @ccall Reactant_jll.libReactantExtra.ReactantHandleCuResult(curesult::UInt32)::Cvoid
end

function mlirOperationInject(ctx, block, code, location, verify_after_parse)
    @ccall Reactant_jll.libReactantExtra.mlirOperationInject(
        ctx::MlirContext,
        block::MlirBlock,
        code::MlirStringRef,
        location::MlirLocation,
        verify_after_parse::Bool,
    )::Bool
end

function mlirOperationParse(ctx, block, code, location, verify_after_parse)
    @ccall Reactant_jll.libReactantExtra.mlirOperationParse(
        ctx::MlirContext,
        block::MlirBlock,
        code::MlirStringRef,
        location::MlirLocation,
        verify_after_parse::Bool,
    )::MlirOperation
end

function mlirGetFunctionTypeFromOperation(op)
    @ccall Reactant_jll.libReactantExtra.mlirGetFunctionTypeFromOperation(
        op::MlirOperation
    )::MlirType
end

function mlirIsFunctionOpInterface(op)
    @ccall Reactant_jll.libReactantExtra.mlirIsFunctionOpInterface(op::MlirOperation)::Bool
end

function ReactantFuncSetResultAttr(op, pos, name, attr)
    @ccall Reactant_jll.libReactantExtra.ReactantFuncSetResultAttr(
        op::MlirOperation, pos::Cptrdiff_t, name::MlirStringRef, attr::MlirAttribute
    )::Cvoid
end

function ReactantFuncSetArgAttr(op, pos, name, attr)
    @ccall Reactant_jll.libReactantExtra.ReactantFuncSetArgAttr(
        op::MlirOperation, pos::Cptrdiff_t, name::MlirStringRef, attr::MlirAttribute
    )::Cvoid
end

function InitializeLogs()
    @ccall Reactant_jll.libReactantExtra.InitializeLogs()::Cvoid
end

function SetLogLevel(level)
    @ccall Reactant_jll.libReactantExtra.SetLogLevel(level::Cint)::Cvoid
end

function SetModuleLogLevel(module_pattern, level)
    @ccall Reactant_jll.libReactantExtra.SetModuleLogLevel(
        module_pattern::Cstring, level::Cint
    )::Cvoid
end

function GetDefaultTargetTriple()
    @ccall Reactant_jll.libReactantExtra.GetDefaultTargetTriple()::Cstring
end

function enzymeActivityAttrGet(ctx, val)
    @ccall Reactant_jll.libReactantExtra.enzymeActivityAttrGet(
        ctx::MlirContext, val::Int32
    )::MlirAttribute
end

function CreateProfilerSession(
    device_tracer_level,
    host_tracer_level,
    advanced_config_keys,
    advanced_config_values,
    n_advanced,
)
    @ccall Reactant_jll.libReactantExtra.CreateProfilerSession(
        device_tracer_level::UInt32,
        host_tracer_level::UInt32,
        advanced_config_keys::Ptr{Cstring},
        advanced_config_values::Ptr{Cstring},
        n_advanced::Cint,
    )::Ptr{ProfilerSession}
end

function ProfilerSessionCollectData(session, path)
    @ccall Reactant_jll.libReactantExtra.ProfilerSessionCollectData(
        session::Ptr{ProfilerSession}, path::Cstring
    )::Cvoid
end

function ProfilerSessionDelete(session)
    @ccall Reactant_jll.libReactantExtra.ProfilerSessionDelete(
        session::Ptr{ProfilerSession}
    )::Cvoid
end

function ProfilerActivityStart(name, level)
    @ccall Reactant_jll.libReactantExtra.ProfilerActivityStart(
        name::Cstring, level::Cint
    )::Int64
end

function ProfilerActivityEnd(id)
    @ccall Reactant_jll.libReactantExtra.ProfilerActivityEnd(id::Int64)::Cvoid
end

function ProfilerServerStart(port)
    @ccall Reactant_jll.libReactantExtra.ProfilerServerStart(
        port::Int32
    )::Ptr{ProfilerServer}
end

function ProfilerServerStop(server)
    @ccall Reactant_jll.libReactantExtra.ProfilerServerStop(
        server::Ptr{ProfilerServer}
    )::Cvoid
end

function MakeCPUClient(asynchronous, node_id)
    @ccall Reactant_jll.libReactantExtra.MakeCPUClient(
        asynchronous::UInt8, node_id::Cint
    )::Ptr{PjRtClient}
end

function MakeGPUClient(
    node_id,
    num_nodes,
    allowed_devices,
    num_allowed_devices,
    memory_fraction,
    preallocate,
    platform_name,
    error,
    distributed_runtime_client,
)
    @ccall Reactant_jll.libReactantExtra.MakeGPUClient(
        node_id::Cint,
        num_nodes::Cint,
        allowed_devices::Ptr{Int64},
        num_allowed_devices::Int64,
        memory_fraction::Cdouble,
        preallocate::Bool,
        platform_name::Cstring,
        error::Ptr{Cstring},
        distributed_runtime_client::Ptr{Cvoid},
    )::Ptr{PjRtClient}
end

function LoadPjrtPlugin(device_type, library_path, error)
    @ccall Reactant_jll.libReactantExtra.LoadPjrtPlugin(
        device_type::Cstring, library_path::Cstring, error::Ptr{Cstring}
    )::Ptr{PJRT_Api}
end

function InitializePjrtPlugin(device_type, error)
    @ccall Reactant_jll.libReactantExtra.InitializePjrtPlugin(
        device_type::Cstring, error::Ptr{Cstring}
    )::Cint
end

function pjrt_client_register_profiler(api)
    @ccall Reactant_jll.libReactantExtra.pjrt_client_register_profiler(
        api::Ptr{PJRT_Api}
    )::Cvoid
end

function MakeClientUsingPluginAPI(device_type, library_path, client_name, error)
    @ccall Reactant_jll.libReactantExtra.MakeClientUsingPluginAPI(
        device_type::Cstring,
        library_path::Cstring,
        client_name::Cstring,
        error::Ptr{Cstring},
    )::Ptr{PjRtClient}
end

function MakeClientFromApi(api, device_type, client_name, error)
    @ccall Reactant_jll.libReactantExtra.MakeClientFromApi(
        api::Ptr{PJRT_Api}, device_type::Cstring, client_name::Cstring, error::Ptr{Cstring}
    )::Ptr{PjRtClient}
end

function MakeTPUClient(tpu_path, error)
    @ccall Reactant_jll.libReactantExtra.MakeTPUClient(
        tpu_path::Cstring, error::Ptr{Cstring}
    )::Ptr{PjRtClient}
end

function ClientNumDevices(client)
    @ccall Reactant_jll.libReactantExtra.ClientNumDevices(client::Ptr{PjRtClient})::Cint
end

function ClientNumAddressableDevices(client)
    @ccall Reactant_jll.libReactantExtra.ClientNumAddressableDevices(
        client::Ptr{PjRtClient}
    )::Cint
end

function ClientProcessIndex(client)
    @ccall Reactant_jll.libReactantExtra.ClientProcessIndex(client::Ptr{PjRtClient})::Cint
end

function ClientGetDevice(client, device_id)
    @ccall Reactant_jll.libReactantExtra.ClientGetDevice(
        client::Ptr{PjRtClient}, device_id::Cint
    )::Ptr{PjRtDevice}
end

function ClientGetAddressableDevice(client, device_id)
    @ccall Reactant_jll.libReactantExtra.ClientGetAddressableDevice(
        client::Ptr{PjRtClient}, device_id::Cint
    )::Ptr{PjRtDevice}
end

function ClientGetPlatformName(client)
    @ccall Reactant_jll.libReactantExtra.ClientGetPlatformName(
        client::Ptr{PjRtClient}
    )::Cstring
end

function DeviceGetKind(device)
    @ccall Reactant_jll.libReactantExtra.DeviceGetKind(device::Ptr{PjRtDevice})::Cstring
end

function ClientGetDevices(client, out_devices)
    @ccall Reactant_jll.libReactantExtra.ClientGetDevices(
        client::Ptr{PjRtClient}, out_devices::Ptr{Ptr{PjRtDevice}}
    )::Cvoid
end

function ClientGetAddressableDevices(client, out_devices)
    @ccall Reactant_jll.libReactantExtra.ClientGetAddressableDevices(
        client::Ptr{PjRtClient}, out_devices::Ptr{Ptr{PjRtDevice}}
    )::Cvoid
end

function PjRtDeviceGetAllocatorStats(device, jlstats)
    @ccall Reactant_jll.libReactantExtra.PjRtDeviceGetAllocatorStats(
        device::Ptr{PjRtDevice}, jlstats::Ptr{JLAllocatorStats}
    )::Cvoid
end

function ifrt_device_get_allocator_stats(device, jlstats)
    @ccall Reactant_jll.libReactantExtra.ifrt_device_get_allocator_stats(
        device::Ptr{Device}, jlstats::Ptr{JLAllocatorStats}
    )::Cvoid
end

function ExecutableFree(exec)
    @ccall Reactant_jll.libReactantExtra.ExecutableFree(
        exec::Ptr{PjRtLoadedExecutable}
    )::Cvoid
end

function BufferToDevice(Buffer)
    @ccall Reactant_jll.libReactantExtra.BufferToDevice(
        Buffer::Ptr{PjRtBuffer}
    )::Ptr{PjRtDevice}
end

function BufferToClient(Buffer)
    @ccall Reactant_jll.libReactantExtra.BufferToClient(
        Buffer::Ptr{PjRtBuffer}
    )::Ptr{PjRtClient}
end

function BufferShape(Buffer)
    @ccall Reactant_jll.libReactantExtra.BufferShape(Buffer::Ptr{PjRtBuffer})::Ptr{Int64}
end

function BufferNDimensions(Buffer)
    @ccall Reactant_jll.libReactantExtra.BufferNDimensions(Buffer::Ptr{PjRtBuffer})::Int64
end

function BufferPrimitiveType(Buffer)
    @ccall Reactant_jll.libReactantExtra.BufferPrimitiveType(Buffer::Ptr{PjRtBuffer})::Cint
end

function PjRtBufferFree(Buffer)
    @ccall Reactant_jll.libReactantExtra.PjRtBufferFree(Buffer::Ptr{PjRtBuffer})::Cvoid
end

function DeviceToClient(Device_)
    @ccall Reactant_jll.libReactantExtra.DeviceToClient(
        Device_::Ptr{PjRtDevice}
    )::Ptr{PjRtClient}
end

function PjRtLoadedExecutableGetClient(exec)
    @ccall Reactant_jll.libReactantExtra.PjRtLoadedExecutableGetClient(
        exec::Ptr{PjRtLoadedExecutable}
    )::Ptr{PjRtClient}
end

function ReactantLLVMParseCommandLineOptions(argc, argv, Overview)
    @ccall Reactant_jll.libReactantExtra.ReactantLLVMParseCommandLineOptions(
        argc::Cint, argv::Ptr{Cstring}, Overview::Cstring
    )::Cvoid
end

function ReactantCudaDriverGetVersion()
    @ccall Reactant_jll.libReactantExtra.ReactantCudaDriverGetVersion()::Int32
end

function ReactantHermeticCudaGetVersion()
    @ccall Reactant_jll.libReactantExtra.ReactantHermeticCudaGetVersion()::Int32
end

function ReactantCudaDeviceGetComputeCapalilityMajor()
    @ccall Reactant_jll.libReactantExtra.ReactantCudaDeviceGetComputeCapalilityMajor()::Int32
end

function ReactantCudaDeviceGetComputeCapalilityMinor()
    @ccall Reactant_jll.libReactantExtra.ReactantCudaDeviceGetComputeCapalilityMinor()::Int32
end

function ReactantCudaDeviceGetWarpSizeInThreads()
    @ccall Reactant_jll.libReactantExtra.ReactantCudaDeviceGetWarpSizeInThreads()::Int32
end

function ReactantCudaDeviceGetProperties(jlprops, device_id)
    @ccall Reactant_jll.libReactantExtra.ReactantCudaDeviceGetProperties(
        jlprops::Ptr{DeviceProperties}, device_id::Int32
    )::Cvoid
end

function ReactantCudaGetRegsSpillsMaxThreadsFromBinary(
    binary, fnname, regs, spills, maxThreads
)
    @ccall Reactant_jll.libReactantExtra.ReactantCudaGetRegsSpillsMaxThreadsFromBinary(
        binary::Cstring,
        fnname::Cstring,
        regs::Ptr{Int32},
        spills::Ptr{Int32},
        maxThreads::Ptr{Int32},
    )::Cvoid
end

function CudaGetStreamExecutorDeviceDescription(device_id)
    @ccall Reactant_jll.libReactantExtra.CudaGetStreamExecutorDeviceDescription(
        device_id::Int32
    )::Ptr{DeviceDescription}
end

function deviceDescriptionToString(device)
    @ccall Reactant_jll.libReactantExtra.deviceDescriptionToString(
        device::Ptr{DeviceDescription}
    )::Cstring
end

function UnsafeBufferPointer(buffer)
    @ccall Reactant_jll.libReactantExtra.UnsafeBufferPointer(
        buffer::Ptr{PjRtBuffer}
    )::Ptr{Cvoid}
end

function ArrayFromHostBuffer(client, data, ptype, dim, cshape, device)
    @ccall Reactant_jll.libReactantExtra.ArrayFromHostBuffer(
        client::Ptr{PjRtClient},
        data::Ptr{Cvoid},
        ptype::UInt64,
        dim::Csize_t,
        cshape::Ptr{Int64},
        device::Ptr{PjRtDevice},
    )::Ptr{PjRtBuffer}
end

function CopyToBuffer(client, buffer, data, offset, size, bufferP)
    @ccall Reactant_jll.libReactantExtra.CopyToBuffer(
        client::Ptr{PjRtClient},
        buffer::Ptr{PjRtBuffer},
        data::Ptr{Cvoid},
        offset::Csize_t,
        size::Csize_t,
        bufferP::Ptr{Ptr{PjRtBuffer}},
    )::Cvoid
end

function BufferToHost(buffer, data)
    @ccall Reactant_jll.libReactantExtra.BufferToHost(
        buffer::Ptr{PjRtBuffer}, data::Ptr{Cvoid}
    )::Cvoid
end

function CopyFromBuffer(client, buffer, data, offset, size, bufferP)
    @ccall Reactant_jll.libReactantExtra.CopyFromBuffer(
        client::Ptr{PjRtClient},
        buffer::Ptr{PjRtBuffer},
        data::Ptr{Cvoid},
        offset::Csize_t,
        size::Csize_t,
        bufferP::Ptr{Ptr{PjRtBuffer}},
    )::Cvoid
end

function UninitPJRTBuffer(client, device, ptype, shapeLen, shape)
    @ccall Reactant_jll.libReactantExtra.UninitPJRTBuffer(
        client::Ptr{PjRtClient},
        device::Ptr{PjRtDevice},
        ptype::UInt64,
        shapeLen::UInt64,
        shape::Ptr{UInt64},
    )::Ptr{PjRtBuffer}
end

function BufferOnCPU(buffer)
    @ccall Reactant_jll.libReactantExtra.BufferOnCPU(buffer::Ptr{PjRtBuffer})::UInt8
end

function CopyBufferToDevice(buffer, dst_device)
    @ccall Reactant_jll.libReactantExtra.CopyBufferToDevice(
        buffer::Ptr{PjRtBuffer}, dst_device::Ptr{PjRtDevice}
    )::Ptr{PjRtBuffer}
end

function BufferFromDevicePointer(client, device_ptr, ptype, dim, cshape, device, stream)
    @ccall mlir_c.BufferFromDevicePointer(
        client::Ptr{PjRtClient},
        device_ptr::Ptr{Cvoid},
        ptype::UInt64,
        dim::Csize_t,
        cshape::Ptr{Int64},
        device::Ptr{PjRtDevice},
        stream::Int64,
    )::Ptr{PjRtBuffer}
end

function AwaitBufferReady(buffer)
    @ccall mlir_c.AwaitBufferReady(buffer::Ptr{PjRtBuffer})::Ptr{Cvoid}
end

function FreeClient(client)
    @ccall Reactant_jll.libReactantExtra.FreeClient(client::Ptr{PjRtClient})::Cvoid
end

function PjRtDeviceGetLocalDeviceId(device)
    @ccall Reactant_jll.libReactantExtra.PjRtDeviceGetLocalDeviceId(
        device::Ptr{PjRtDevice}
    )::Int64
end

function PjRtDeviceGetGlobalDeviceId(device)
    @ccall Reactant_jll.libReactantExtra.PjRtDeviceGetGlobalDeviceId(
        device::Ptr{PjRtDevice}
    )::Int64
end

function PjRtDeviceGetLocalHardwareId(device)
    @ccall Reactant_jll.libReactantExtra.PjRtDeviceGetLocalHardwareId(
        device::Ptr{PjRtDevice}
    )::Int64
end

function RegisterCustomCallTarget(name, address, platform)
    @ccall Reactant_jll.libReactantExtra.RegisterCustomCallTarget(
        name::Cstring, address::Ptr{Cvoid}, platform::Cstring
    )::Cvoid
end

function ConvertLLVMToMLIR(lmod, cctx)
    @ccall Reactant_jll.libReactantExtra.ConvertLLVMToMLIR(
        lmod::Cint, cctx::MlirContext
    )::MlirModule
end

function ConvertLLVMStrToMLIR(lmod, cctx)
    @ccall Reactant_jll.libReactantExtra.ConvertLLVMStrToMLIR(
        lmod::Cstring, cctx::MlirContext
    )::MlirModule
end

function ConvertLLVMBCToMLIR(bc, len, cctx)
    @ccall Reactant_jll.libReactantExtra.ConvertLLVMBCToMLIR(
        bc::Ptr{UInt8}, len::Csize_t, cctx::MlirContext
    )::MlirModule
end

function FreeFuture(Future)
    @ccall Reactant_jll.libReactantExtra.FreeFuture(Future::Ptr{FutureType})::Cvoid
end

function FutureIsReady(Future)
    @ccall Reactant_jll.libReactantExtra.FutureIsReady(Future::Ptr{FutureType})::UInt8
end

function FutureAwait(Future)
    @ccall Reactant_jll.libReactantExtra.FutureAwait(Future::Ptr{FutureType})::Cvoid
end

function ClientCompile(
    client,
    cmod,
    device_id,
    mesh_ids,
    num_mesh_ids,
    xla_gpu_cuda_data_dir,
    use_shardy_partitioner,
    num_replicas,
    num_partitions,
    use_spmd_partitioning,
    kernel_cache_enabled,
    kernel_cache_path,
    autotune_cache_enabled,
    autotune_cache_path,
    process_id,
    enable_enzyme_comms,
)
    @ccall Reactant_jll.libReactantExtra.ClientCompile(
        client::Ptr{PjRtClient},
        cmod::MlirModule,
        device_id::Int64,
        mesh_ids::Ptr{Int64},
        num_mesh_ids::Int64,
        xla_gpu_cuda_data_dir::Cstring,
        use_shardy_partitioner::Bool,
        num_replicas::Int64,
        num_partitions::Int64,
        use_spmd_partitioning::Bool,
        kernel_cache_enabled::Bool,
        kernel_cache_path::Cstring,
        autotune_cache_enabled::Bool,
        autotune_cache_path::Cstring,
        process_id::Cint,
        enable_enzyme_comms::Bool,
    )::Ptr{PjRtLoadedExecutable}
end

function ClientCompileWithProto(
    client, cmod, compile_options_proto, compile_options_proto_size
)
    @ccall Reactant_jll.libReactantExtra.ClientCompileWithProto(
        client::Ptr{PjRtClient},
        cmod::MlirModule,
        compile_options_proto::Ptr{UInt8},
        compile_options_proto_size::Csize_t,
    )::Ptr{PjRtLoadedExecutable}
end

function PjRtLoadedExecutableGetOuputShardings(exec, op_shardings, num_op_shardings)
    @ccall Reactant_jll.libReactantExtra.PjRtLoadedExecutableGetOuputShardings(
        exec::Ptr{PjRtLoadedExecutable},
        op_shardings::Ptr{Ptr{OpSharding}},
        num_op_shardings::Int32,
    )::Cvoid
end

function PjRtLoadedExecutableGetParameterShardings(exec, op_shardings, num_op_shardings)
    @ccall Reactant_jll.libReactantExtra.PjRtLoadedExecutableGetParameterShardings(
        exec::Ptr{PjRtLoadedExecutable},
        op_shardings::Ptr{Ptr{OpSharding}},
        num_op_shardings::Int32,
    )::Cvoid
end

function XLAExecuteSharded(
    exec,
    num_args,
    op_args,
    device,
    is_arg_donatable,
    num_results,
    op_results,
    futures,
    future_results,
)
    @ccall Reactant_jll.libReactantExtra.XLAExecuteSharded(
        exec::Ptr{PjRtLoadedExecutable},
        num_args::Cint,
        op_args::Ptr{Ptr{PjRtBuffer}},
        device::Ptr{PjRtDevice},
        is_arg_donatable::Ptr{UInt8},
        num_results::Cint,
        op_results::Ptr{Ptr{PjRtBuffer}},
        futures::Ptr{UInt8},
        future_results::Ptr{Ptr{FutureType}},
    )::Cvoid
end

function XLAExecute(
    exec,
    op_args_len,
    op_args,
    is_arg_donatable,
    num_results,
    op_results,
    futures,
    future_results,
)
    @ccall Reactant_jll.libReactantExtra.XLAExecute(
        exec::Ptr{PjRtLoadedExecutable},
        op_args_len::Cint,
        op_args::Ptr{Ptr{PjRtBuffer}},
        is_arg_donatable::Ptr{UInt8},
        num_results::Cint,
        op_results::Ptr{Ptr{PjRtBuffer}},
        futures::Ptr{UInt8},
        future_results::Ptr{Ptr{FutureType}},
    )::Cvoid
end

function PjRtLoadedExecutableNumReplicas(exec)
    @ccall Reactant_jll.libReactantExtra.PjRtLoadedExecutableNumReplicas(
        exec::Ptr{PjRtLoadedExecutable}
    )::Cint
end

function PjRtLoadedExecutableNumPartitions(exec)
    @ccall Reactant_jll.libReactantExtra.PjRtLoadedExecutableNumPartitions(
        exec::Ptr{PjRtLoadedExecutable}
    )::Cint
end

function RegisterDialects(cctx)
    @ccall Reactant_jll.libReactantExtra.RegisterDialects(cctx::MlirContext)::Cvoid
end

function InitializePasses(creg)
    @ccall Reactant_jll.libReactantExtra.InitializePasses(creg::MlirDialectRegistry)::Cvoid
end

function InitializeRegistry(creg)
    @ccall Reactant_jll.libReactantExtra.InitializeRegistry(
        creg::MlirDialectRegistry
    )::Cvoid
end

function LinkInModule(prevModC, newModC, entryfn)
    @ccall Reactant_jll.libReactantExtra.LinkInModule(
        prevModC::MlirModule, newModC::MlirModule, entryfn::Cstring
    )::MlirOperation
end

function pjrt_client_dtor(client)
    @ccall Reactant_jll.libReactantExtra.pjrt_client_dtor(
        client::Ptr{HeldPjRtClient}
    )::Cvoid
end

function pjrt_client_num_devices(client)
    @ccall Reactant_jll.libReactantExtra.pjrt_client_num_devices(
        client::Ptr{HeldPjRtClient}
    )::Cint
end

function pjrt_client_num_addressable_devices(client)
    @ccall Reactant_jll.libReactantExtra.pjrt_client_num_addressable_devices(
        client::Ptr{HeldPjRtClient}
    )::Cint
end

function pjrt_client_pid(client)
    @ccall Reactant_jll.libReactantExtra.pjrt_client_pid(client::Ptr{HeldPjRtClient})::Cint
end

function pjrt_client_get_device(client, device_id)
    @ccall Reactant_jll.libReactantExtra.pjrt_client_get_device(
        client::Ptr{HeldPjRtClient}, device_id::Cint
    )::Ptr{PjRtDevice}
end

function pjrt_client_get_addressable_device(client, device_id)
    @ccall Reactant_jll.libReactantExtra.pjrt_client_get_addressable_device(
        client::Ptr{HeldPjRtClient}, device_id::Cint
    )::Ptr{PjRtDevice}
end

function pjrt_client_platform_name(client)
    @ccall Reactant_jll.libReactantExtra.pjrt_client_platform_name(
        client::Ptr{HeldPjRtClient}
    )::Cstring
end

function pjrt_buffer_from_host(client, data, ptype, dim, cshape, device)
    @ccall Reactant_jll.libReactantExtra.pjrt_buffer_from_host(
        client::Ptr{HeldPjRtClient},
        data::Ptr{Cvoid},
        ptype::UInt64,
        dim::Csize_t,
        cshape::Ptr{Int64},
        device::Ptr{PjRtDevice},
    )::Ptr{HeldPjRtBuffer}
end

function pjrt_buffer_dtor(buffer)
    @ccall Reactant_jll.libReactantExtra.pjrt_buffer_dtor(
        buffer::Ptr{HeldPjRtBuffer}
    )::Cvoid
end

function pjrt_buffer_unsafe_buffer_pointer(buffer)
    @ccall Reactant_jll.libReactantExtra.pjrt_buffer_unsafe_buffer_pointer(
        buffer::Ptr{HeldPjRtBuffer}
    )::Ptr{Cvoid}
end

function pjrt_buffer_is_on_cpu(buffer)
    @ccall Reactant_jll.libReactantExtra.pjrt_buffer_is_on_cpu(
        buffer::Ptr{HeldPjRtBuffer}
    )::Bool
end

function pjrt_buffer_copy_to_device(buffer, dst_device)
    @ccall Reactant_jll.libReactantExtra.pjrt_buffer_copy_to_device(
        buffer::Ptr{HeldPjRtBuffer}, dst_device::Ptr{PjRtDevice}
    )::Ptr{HeldPjRtBuffer}
end

function pjrt_buffer_to_host(buffer, data)
    @ccall Reactant_jll.libReactantExtra.pjrt_buffer_to_host(
        buffer::Ptr{HeldPjRtBuffer}, data::Ptr{Cvoid}
    )::Cvoid
end

function pjrt_buffer_print(buffer)
    @ccall Reactant_jll.libReactantExtra.pjrt_buffer_print(
        buffer::Ptr{HeldPjRtBuffer}
    )::Cvoid
end

function pjrt_buffer_get_device(buffer)
    @ccall Reactant_jll.libReactantExtra.pjrt_buffer_get_device(
        buffer::Ptr{HeldPjRtBuffer}
    )::Ptr{PjRtDevice}
end

function pjrt_buffer_get_client(buffer)
    @ccall Reactant_jll.libReactantExtra.pjrt_buffer_get_client(
        buffer::Ptr{HeldPjRtBuffer}
    )::Ptr{HeldPjRtClient}
end

function ifrt_client_dtor(client)
    @ccall Reactant_jll.libReactantExtra.ifrt_client_dtor(client::Ptr{Client})::Cvoid
end

function ifrt_client_make_array_from_host_buffer(
    client, data, dtype_kind, ndims, c_shape, sharding, c_semantics
)
    @ccall Reactant_jll.libReactantExtra.ifrt_client_make_array_from_host_buffer(
        client::Ptr{Client},
        data::Ptr{Cvoid},
        dtype_kind::Cint,
        ndims::Cint,
        c_shape::Ptr{Int64},
        sharding::Ptr{HeldIfrtConstSharding},
        c_semantics::Cint,
    )::Ptr{HeldIfrtArray}
end

function ifrt_client_make_single_shard_array_from_host_buffer(
    client, data, dtype_kind, ndims, c_shape, c_semantics, device, mem_kind
)
    @ccall Reactant_jll.libReactantExtra.ifrt_client_make_single_shard_array_from_host_buffer(
        client::Ptr{Client},
        data::Ptr{Cvoid},
        dtype_kind::Cint,
        ndims::Cint,
        c_shape::Ptr{Int64},
        c_semantics::Cint,
        device::Ptr{Device},
        mem_kind::Cstring,
    )::Ptr{HeldIfrtArray}
end

function ifrt_client_assemble_array_from_single_shards(
    client, ndims, c_shape, sharding, narrays, c_arrays, c_semantics
)
    @ccall Reactant_jll.libReactantExtra.ifrt_client_assemble_array_from_single_shards(
        client::Ptr{Client},
        ndims::Int32,
        c_shape::Ptr{Int64},
        sharding::Ptr{HeldIfrtConstSharding},
        narrays::Int32,
        c_arrays::Ptr{Ptr{HeldIfrtArray}},
        c_semantics::Int32,
    )::Ptr{HeldIfrtArray}
end

function ifrt_pjrt_array_create(client, buffer)
    @ccall Reactant_jll.libReactantExtra.ifrt_pjrt_array_create(
        client::Ptr{PjRtClient}, buffer::Ptr{HeldPjRtBuffer}
    )::Ptr{HeldIfrtArray}
end

function ifrt_compile(
    client,
    cmod,
    device_id,
    mesh_ids,
    num_mesh_ids,
    xla_gpu_cuda_data_dir,
    use_shardy_partitioner,
    num_replicas,
    num_partitions,
    use_spmd_partitioning,
    kernel_cache_enabled,
    kernel_cache_path,
    autotune_cache_enabled,
    autotune_cache_path,
    process_id,
    xla_enable_enzyme_comms_opt,
)
    @ccall Reactant_jll.libReactantExtra.ifrt_compile(
        client::Ptr{Client},
        cmod::MlirModule,
        device_id::Int64,
        mesh_ids::Ptr{Int64},
        num_mesh_ids::Int64,
        xla_gpu_cuda_data_dir::Cstring,
        use_shardy_partitioner::Bool,
        num_replicas::Int64,
        num_partitions::Int64,
        use_spmd_partitioning::Bool,
        kernel_cache_enabled::Bool,
        kernel_cache_path::Cstring,
        autotune_cache_enabled::Bool,
        autotune_cache_path::Cstring,
        process_id::Cint,
        xla_enable_enzyme_comms_opt::Bool,
    )::Ptr{HeldIfrtLoadedExecutable}
end

function ifrt_compile_with_proto(
    client, cmod, compile_options_proto, compile_options_proto_size
)
    @ccall Reactant_jll.libReactantExtra.ifrt_compile_with_proto(
        client::Ptr{Client},
        cmod::MlirModule,
        compile_options_proto::Ptr{UInt8},
        compile_options_proto_size::Csize_t,
    )::Ptr{HeldIfrtLoadedExecutable}
end

function ifrt_pjrt_loaded_executable_dtor(exec)
    @ccall Reactant_jll.libReactantExtra.ifrt_pjrt_loaded_executable_dtor(
        exec::Ptr{PjRtLoadedExecutable}
    )::Cvoid
end

function ifrt_array_dtor(array)
    @ccall Reactant_jll.libReactantExtra.ifrt_array_dtor(array::Ptr{HeldIfrtArray})::Cvoid
end

function ifrt_CopyArrayToHostBuffer(array, data, semantics)
    @ccall Reactant_jll.libReactantExtra.ifrt_CopyArrayToHostBuffer(
        array::Ptr{HeldIfrtArray}, data::Ptr{Cvoid}, semantics::Cint
    )::Ptr{FutureType}
end

function PjRtLoadedExecutableGetHloModules(exec, hlo_modules, nmodules)
    @ccall Reactant_jll.libReactantExtra.PjRtLoadedExecutableGetHloModules(
        exec::Ptr{PjRtLoadedExecutable}, hlo_modules::Ptr{Ptr{Cvoid}}, nmodules::Ptr{Int32}
    )::Cvoid
end

function HloModuleToString(hlo_module, print_options)
    @ccall Reactant_jll.libReactantExtra.HloModuleToString(
        hlo_module::Ptr{HeldHloModule}, print_options::Int32
    )::Cstring
end

function FreeHloModule(hlo_module)
    @ccall Reactant_jll.libReactantExtra.FreeHloModule(
        hlo_module::Ptr{HeldHloModule}
    )::Cvoid
end

function ifrt_proxy_grpc_server_dtor(server)
    @ccall Reactant_jll.libReactantExtra.ifrt_proxy_grpc_server_dtor(
        server::Ptr{GrpcServer}
    )::Cvoid
end

function ifrt_proxy_grpc_server_address(server)
    @ccall Reactant_jll.libReactantExtra.ifrt_proxy_grpc_server_address(
        server::Ptr{GrpcServer}
    )::Cstring
end

function ifrt_proxy_grpc_server_wait(server)
    @ccall Reactant_jll.libReactantExtra.ifrt_proxy_grpc_server_wait(
        server::Ptr{GrpcServer}
    )::Cvoid
end

function ifrt_proxy_create_client(c_proxy_server_address, connection_timeout_in_minutes)
    @ccall Reactant_jll.libReactantExtra.ifrt_proxy_create_client(
        c_proxy_server_address::Cstring, connection_timeout_in_minutes::Cint
    )::Ptr{Client}
end

function ifrt_pjrt_make_client_with_default_kv_store(
    pjrt_client, node_id, num_nodes, distributed_runtime_client, error, key_prefix
)
    @ccall Reactant_jll.libReactantExtra.ifrt_pjrt_make_client_with_default_kv_store(
        pjrt_client::Ptr{PjRtClient},
        node_id::Cint,
        num_nodes::Cint,
        distributed_runtime_client::Ptr{Cvoid},
        error::Ptr{Cstring},
        key_prefix::Cstring,
    )::Ptr{Client}
end

function ifrt_make_pjrt_cpu_client(
    asynchronous, node_id, num_nodes, distributed_runtime_client, error
)
    @ccall Reactant_jll.libReactantExtra.ifrt_make_pjrt_cpu_client(
        asynchronous::UInt8,
        node_id::Cint,
        num_nodes::Cint,
        distributed_runtime_client::Ptr{Cvoid},
        error::Ptr{Cstring},
    )::Ptr{Client}
end

function ifrt_make_pjrt_gpu_client(
    node_id,
    num_nodes,
    allowed_devices,
    num_allowed_devices,
    memory_fraction,
    preallocate,
    platform_name,
    error,
    distributed_runtime_client,
)
    @ccall Reactant_jll.libReactantExtra.ifrt_make_pjrt_gpu_client(
        node_id::Cint,
        num_nodes::Cint,
        allowed_devices::Ptr{Int64},
        num_allowed_devices::Int64,
        memory_fraction::Cdouble,
        preallocate::Bool,
        platform_name::Cstring,
        error::Ptr{Cstring},
        distributed_runtime_client::Ptr{Cvoid},
    )::Ptr{Client}
end

function ifrt_make_pjrt_tpu_client(
    tpu_path, error, node_id, num_nodes, distributed_runtime_client
)
    @ccall Reactant_jll.libReactantExtra.ifrt_make_pjrt_tpu_client(
        tpu_path::Cstring,
        error::Ptr{Cstring},
        node_id::Cint,
        num_nodes::Cint,
        distributed_runtime_client::Ptr{Cvoid},
    )::Ptr{Client}
end

function ifrt_FreeClient(client)
    @ccall Reactant_jll.libReactantExtra.ifrt_FreeClient(client::Ptr{Client})::Cvoid
end

function ifrt_client_device_count(client)
    @ccall Reactant_jll.libReactantExtra.ifrt_client_device_count(client::Ptr{Client})::Cint
end

function ifrt_client_addressable_device_count(client)
    @ccall Reactant_jll.libReactantExtra.ifrt_client_addressable_device_count(
        client::Ptr{Client}
    )::Cint
end

function ifrt_client_devices(client, out_devices)
    @ccall Reactant_jll.libReactantExtra.ifrt_client_devices(
        client::Ptr{Client}, out_devices::Ptr{Ptr{Device}}
    )::Cvoid
end

function ifrt_client_addressable_devices(client, out_devices)
    @ccall Reactant_jll.libReactantExtra.ifrt_client_addressable_devices(
        client::Ptr{Client}, out_devices::Ptr{Ptr{Device}}
    )::Cvoid
end

function ifrt_client_all_devices(client, out_devices)
    @ccall Reactant_jll.libReactantExtra.ifrt_client_all_devices(
        client::Ptr{Client}, out_devices::Ptr{Ptr{Device}}
    )::Cvoid
end

function ifrt_client_lookup_device(client, dev_id)
    @ccall Reactant_jll.libReactantExtra.ifrt_client_lookup_device(
        client::Ptr{Client}, dev_id::Cint
    )::Ptr{Device}
end

function ifrt_client_lookup_addressable_device(client, local_hw_id)
    @ccall Reactant_jll.libReactantExtra.ifrt_client_lookup_addressable_device(
        client::Ptr{Client}, local_hw_id::Cint
    )::Ptr{Device}
end

function ifrt_ClientProcessIndex(client)
    @ccall Reactant_jll.libReactantExtra.ifrt_ClientProcessIndex(client::Ptr{Client})::Cint
end

function ifrt_ClientGetPlatformName(client)
    @ccall Reactant_jll.libReactantExtra.ifrt_ClientGetPlatformName(
        client::Ptr{Client}
    )::Cstring
end

function ifrt_ClientGetDevice(client, idx)
    @ccall Reactant_jll.libReactantExtra.ifrt_ClientGetDevice(
        client::Ptr{Client}, idx::Cint
    )::Ptr{Device}
end

function ifrt_ClientGetAddressableDevice(client, idx)
    @ccall Reactant_jll.libReactantExtra.ifrt_ClientGetAddressableDevice(
        client::Ptr{Client}, idx::Cint
    )::Ptr{Device}
end

function ifrt_DeviceGetGlobalDeviceId(device)
    @ccall Reactant_jll.libReactantExtra.ifrt_DeviceGetGlobalDeviceId(
        device::Ptr{Device}
    )::Int64
end

function ifrt_DeviceGetKind(device)
    @ccall Reactant_jll.libReactantExtra.ifrt_DeviceGetKind(device::Ptr{Device})::Cstring
end

function ifrt_DeviceToClient(device)
    @ccall Reactant_jll.libReactantExtra.ifrt_DeviceToClient(
        device::Ptr{Device}
    )::Ptr{Client}
end

function ifrt_DeviceIsAddressable(device)
    @ccall Reactant_jll.libReactantExtra.ifrt_DeviceIsAddressable(device::Ptr{Device})::Bool
end

function ifrt_DeviceGetLocalHardwareId(device)
    @ccall Reactant_jll.libReactantExtra.ifrt_DeviceGetLocalHardwareId(
        device::Ptr{Device}
    )::Int64
end

function ifrt_DeviceGetDefaultMemory(device)
    @ccall Reactant_jll.libReactantExtra.ifrt_DeviceGetDefaultMemory(
        device::Ptr{Device}
    )::Ptr{Memory}
end

function ifrt_DeviceGetMemories(device, size)
    @ccall Reactant_jll.libReactantExtra.ifrt_DeviceGetMemories(
        device::Ptr{Device}, size::Ptr{Int32}
    )::Ptr{Ptr{Memory}}
end

function ifrt_MemoryGetMemoryKind(memory)
    @ccall Reactant_jll.libReactantExtra.ifrt_MemoryGetMemoryKind(
        memory::Ptr{Memory}
    )::Ptr{MemoryKind}
end

function ifrt_MemoryToString(memory)
    @ccall Reactant_jll.libReactantExtra.ifrt_MemoryToString(memory::Ptr{Memory})::Cstring
end

function ifrt_MemoryKindToString(memory_kind)
    @ccall Reactant_jll.libReactantExtra.ifrt_MemoryKindToString(
        memory_kind::Ptr{MemoryKind}
    )::Cstring
end

function ifrt_MemoryKindsAreEqual(a, b)
    @ccall Reactant_jll.libReactantExtra.ifrt_MemoryKindsAreEqual(
        a::Ptr{MemoryKind}, b::Ptr{MemoryKind}
    )::Bool
end

function free_op_sharding(op_sharding)
    @ccall Reactant_jll.libReactantExtra.free_op_sharding(
        op_sharding::Ptr{OpSharding}
    )::Cvoid
end

function op_sharding_to_op_sharding_type(op_sharding)
    @ccall Reactant_jll.libReactantExtra.op_sharding_to_op_sharding_type(
        op_sharding::Ptr{OpSharding}
    )::Int32
end

function op_sharding_to_shard_group_type(op_sharding)
    @ccall Reactant_jll.libReactantExtra.op_sharding_to_shard_group_type(
        op_sharding::Ptr{OpSharding}
    )::Int32
end

function op_sharding_to_shard_group_id(op_sharding)
    @ccall Reactant_jll.libReactantExtra.op_sharding_to_shard_group_id(
        op_sharding::Ptr{OpSharding}
    )::Int32
end

function op_sharding_is_shard_group(op_sharding)
    @ccall Reactant_jll.libReactantExtra.op_sharding_is_shard_group(
        op_sharding::Ptr{OpSharding}
    )::Bool
end

function op_sharding_replicate_on_last_tile_dim(op_sharding)
    @ccall Reactant_jll.libReactantExtra.op_sharding_replicate_on_last_tile_dim(
        op_sharding::Ptr{OpSharding}
    )::Bool
end

function op_sharding_has_last_tile_dims(op_sharding)
    @ccall Reactant_jll.libReactantExtra.op_sharding_has_last_tile_dims(
        op_sharding::Ptr{OpSharding}
    )::Bool
end

function op_sharding_last_tile_dims_size(op_sharding)
    @ccall Reactant_jll.libReactantExtra.op_sharding_last_tile_dims_size(
        op_sharding::Ptr{OpSharding}
    )::Int32
end

function op_sharding_last_tile_dims(op_sharding, last_tile_dims)
    @ccall Reactant_jll.libReactantExtra.op_sharding_last_tile_dims(
        op_sharding::Ptr{OpSharding}, last_tile_dims::Ptr{Int32}
    )::Cvoid
end

function op_sharding_has_iota_reshape_dims(op_sharding)
    @ccall Reactant_jll.libReactantExtra.op_sharding_has_iota_reshape_dims(
        op_sharding::Ptr{OpSharding}
    )::Bool
end

function op_sharding_iota_reshape_dims_size(op_sharding)
    @ccall Reactant_jll.libReactantExtra.op_sharding_iota_reshape_dims_size(
        op_sharding::Ptr{OpSharding}
    )::Int32
end

function op_sharding_iota_reshape_dims(op_sharding, iota_reshape_dims)
    @ccall Reactant_jll.libReactantExtra.op_sharding_iota_reshape_dims(
        op_sharding::Ptr{OpSharding}, iota_reshape_dims::Ptr{Int32}
    )::Cvoid
end

function op_sharding_has_iota_transpose_perm(op_sharding)
    @ccall Reactant_jll.libReactantExtra.op_sharding_has_iota_transpose_perm(
        op_sharding::Ptr{OpSharding}
    )::Bool
end

function op_sharding_iota_transpose_perm_size(op_sharding)
    @ccall Reactant_jll.libReactantExtra.op_sharding_iota_transpose_perm_size(
        op_sharding::Ptr{OpSharding}
    )::Int32
end

function op_sharding_iota_transpose_perm(op_sharding, iota_transpose_perm)
    @ccall Reactant_jll.libReactantExtra.op_sharding_iota_transpose_perm(
        op_sharding::Ptr{OpSharding}, iota_transpose_perm::Ptr{Int32}
    )::Cvoid
end

function op_sharding_has_tile_assignment_dimensions(op_sharding)
    @ccall Reactant_jll.libReactantExtra.op_sharding_has_tile_assignment_dimensions(
        op_sharding::Ptr{OpSharding}
    )::Bool
end

function op_sharding_tile_assignment_dimensions_size(op_sharding)
    @ccall Reactant_jll.libReactantExtra.op_sharding_tile_assignment_dimensions_size(
        op_sharding::Ptr{OpSharding}
    )::Int32
end

function op_sharding_tile_assignment_dimensions(op_sharding, tile_assignment_dimensions)
    @ccall Reactant_jll.libReactantExtra.op_sharding_tile_assignment_dimensions(
        op_sharding::Ptr{OpSharding}, tile_assignment_dimensions::Ptr{Int32}
    )::Cvoid
end

function op_sharding_has_tile_assignment_devices(op_sharding)
    @ccall Reactant_jll.libReactantExtra.op_sharding_has_tile_assignment_devices(
        op_sharding::Ptr{OpSharding}
    )::Bool
end

function op_sharding_tile_assignment_devices_size(op_sharding)
    @ccall Reactant_jll.libReactantExtra.op_sharding_tile_assignment_devices_size(
        op_sharding::Ptr{OpSharding}
    )::Int32
end

function op_sharding_tile_assignment_devices(op_sharding, tile_assignment_devices)
    @ccall Reactant_jll.libReactantExtra.op_sharding_tile_assignment_devices(
        op_sharding::Ptr{OpSharding}, tile_assignment_devices::Ptr{Int32}
    )::Cvoid
end

function free_hlo_sharding(hlo_sharding)
    @ccall Reactant_jll.libReactantExtra.free_hlo_sharding(
        hlo_sharding::Ptr{HloSharding}
    )::Cvoid
end

function hlo_sharding_from_op_sharding(op_sharding)
    @ccall Reactant_jll.libReactantExtra.hlo_sharding_from_op_sharding(
        op_sharding::Ptr{OpSharding}
    )::Ptr{HloSharding}
end

function hlo_sharding_to_op_sharding(hlo_sharding)
    @ccall Reactant_jll.libReactantExtra.hlo_sharding_to_op_sharding(
        hlo_sharding::Ptr{HloSharding}
    )::Ptr{OpSharding}
end

function hlo_sharding_to_string(hlo_sharding)
    @ccall Reactant_jll.libReactantExtra.hlo_sharding_to_string(
        hlo_sharding::Ptr{HloSharding}
    )::Cstring
end

function ifrt_memory_kind_from_string(c_str)
    @ccall Reactant_jll.libReactantExtra.ifrt_memory_kind_from_string(
        c_str::Cstring
    )::Ptr{MemoryKind}
end

function ifrt_memory_kind_with_optional_memory_space()
    @ccall Reactant_jll.libReactantExtra.ifrt_memory_kind_with_optional_memory_space()::Ptr{
        MemoryKind
    }
end

function ifrt_memory_kind_has_value(memory_kind)
    @ccall Reactant_jll.libReactantExtra.ifrt_memory_kind_has_value(
        memory_kind::Ptr{MemoryKind}
    )::Bool
end

function free_ifrt_sharding(sharding)
    @ccall Reactant_jll.libReactantExtra.free_ifrt_sharding(
        sharding::Ptr{HeldIfrtSharding}
    )::Cvoid
end

function ifrt_sharding_from_xla_hlo_sharding(
    client, device_list, num_devices, memory_kind, xla_hlo_sharding
)
    @ccall Reactant_jll.libReactantExtra.ifrt_sharding_from_xla_hlo_sharding(
        client::Ptr{Client},
        device_list::Ptr{Ptr{Device}},
        num_devices::Int32,
        memory_kind::Ptr{MemoryKind},
        xla_hlo_sharding::Ptr{HloSharding},
    )::Ptr{HeldIfrtSharding}
end

function ifrt_sharding_to_xla_hlo_sharding(sharding)
    @ccall Reactant_jll.libReactantExtra.ifrt_sharding_to_xla_hlo_sharding(
        sharding::Ptr{HeldIfrtSharding}
    )::Ptr{HloSharding}
end

function ifrt_sharding_is_single_device_sharding(sharding)
    @ccall Reactant_jll.libReactantExtra.ifrt_sharding_is_single_device_sharding(
        sharding::Ptr{HeldIfrtSharding}
    )::Bool
end

function ifrt_sharding_is_fully_replicated(sharding)
    @ccall Reactant_jll.libReactantExtra.ifrt_sharding_is_fully_replicated(
        sharding::Ptr{HeldIfrtSharding}
    )::Bool
end

function ifrt_sharding_to_string(sharding)
    @ccall Reactant_jll.libReactantExtra.ifrt_sharding_to_string(
        sharding::Ptr{HeldIfrtSharding}
    )::Cstring
end

function ifrt_sharding_devices_size(sharding)
    @ccall Reactant_jll.libReactantExtra.ifrt_sharding_devices_size(
        sharding::Ptr{HeldIfrtSharding}
    )::Int32
end

function ifrt_sharding_to_device_list(sharding, devices)
    @ccall Reactant_jll.libReactantExtra.ifrt_sharding_to_device_list(
        sharding::Ptr{HeldIfrtSharding}, devices::Ptr{Ptr{Device}}
    )::Cvoid
end

function ifrt_sharding_to_index_domains(
    sharding, array_size_list, array_size_len, index_domain_origins, index_domain_shapes
)
    @ccall Reactant_jll.libReactantExtra.ifrt_sharding_to_index_domains(
        sharding::Ptr{HeldIfrtSharding},
        array_size_list::Ptr{Int64},
        array_size_len::Int32,
        index_domain_origins::Ptr{Int64},
        index_domain_shapes::Ptr{Int64},
    )::Cvoid
end

function hlo_sharding_is_tuple(hloSharding)
    @ccall Reactant_jll.libReactantExtra.hlo_sharding_is_tuple(
        hloSharding::Ptr{HloSharding}
    )::Bool
end

function hlo_sharding_is_replicated(hloSharding)
    @ccall Reactant_jll.libReactantExtra.hlo_sharding_is_replicated(
        hloSharding::Ptr{HloSharding}
    )::Bool
end

function hlo_sharding_is_manual(hloSharding)
    @ccall Reactant_jll.libReactantExtra.hlo_sharding_is_manual(
        hloSharding::Ptr{HloSharding}
    )::Bool
end

function hlo_sharding_is_unknown(hloSharding)
    @ccall Reactant_jll.libReactantExtra.hlo_sharding_is_unknown(
        hloSharding::Ptr{HloSharding}
    )::Bool
end

function hlo_sharding_is_tiled(hloSharding)
    @ccall Reactant_jll.libReactantExtra.hlo_sharding_is_tiled(
        hloSharding::Ptr{HloSharding}
    )::Bool
end

function hlo_sharding_is_maximal(hloSharding)
    @ccall Reactant_jll.libReactantExtra.hlo_sharding_is_maximal(
        hloSharding::Ptr{HloSharding}
    )::Bool
end

function hlo_sharding_replicate_on_last_tile_dim(hloSharding)
    @ccall Reactant_jll.libReactantExtra.hlo_sharding_replicate_on_last_tile_dim(
        hloSharding::Ptr{HloSharding}
    )::Bool
end

function hlo_sharding_tile_assignment_dimensions_size(hloSharding)
    @ccall Reactant_jll.libReactantExtra.hlo_sharding_tile_assignment_dimensions_size(
        hloSharding::Ptr{HloSharding}
    )::Int32
end

function hlo_sharding_tile_assignment_devices_size(hloSharding)
    @ccall Reactant_jll.libReactantExtra.hlo_sharding_tile_assignment_devices_size(
        hloSharding::Ptr{HloSharding}
    )::Int32
end

function hlo_sharding_tile_assignment_dimensions(hloSharding, dims, size)
    @ccall Reactant_jll.libReactantExtra.hlo_sharding_tile_assignment_dimensions(
        hloSharding::Ptr{HloSharding}, dims::Ptr{Int64}, size::Int32
    )::Cvoid
end

function hlo_sharding_tile_assignment_devices(hloSharding, devices, size)
    @ccall Reactant_jll.libReactantExtra.hlo_sharding_tile_assignment_devices(
        hloSharding::Ptr{HloSharding}, devices::Ptr{Int64}, size::Int32
    )::Cvoid
end

function hlo_sharding_check_eq(hloSharding, other)
    @ccall Reactant_jll.libReactantExtra.hlo_sharding_check_eq(
        hloSharding::Ptr{HloSharding}, other::Ptr{HloSharding}
    )::Bool
end

function hlo_sharding_check_eq_ignoring_metadata(hloSharding, other)
    @ccall Reactant_jll.libReactantExtra.hlo_sharding_check_eq_ignoring_metadata(
        hloSharding::Ptr{HloSharding}, other::Ptr{HloSharding}
    )::Bool
end

function ifrt_free_future(Future)
    @ccall Reactant_jll.libReactantExtra.ifrt_free_future(
        Future::Ptr{IfRtFutureType}
    )::Cvoid
end

function ifrt_future_is_ready(Future)
    @ccall Reactant_jll.libReactantExtra.ifrt_future_is_ready(
        Future::Ptr{IfRtFutureType}
    )::UInt8
end

function ifrt_future_await(Future)
    @ccall Reactant_jll.libReactantExtra.ifrt_future_await(
        Future::Ptr{IfRtFutureType}
    )::Cvoid
end

function ifrt_free_array(array)
    @ccall Reactant_jll.libReactantExtra.ifrt_free_array(array::Ptr{HeldIfrtArray})::Cvoid
end

function ifrt_array_shape(array)
    @ccall Reactant_jll.libReactantExtra.ifrt_array_shape(
        array::Ptr{HeldIfrtArray}
    )::Ptr{Int64}
end

function ifrt_array_ndims(array)
    @ccall Reactant_jll.libReactantExtra.ifrt_array_ndims(array::Ptr{HeldIfrtArray})::Int64
end

function ifrt_array_eltype(array)
    @ccall Reactant_jll.libReactantExtra.ifrt_array_eltype(array::Ptr{HeldIfrtArray})::Cint
end

function ifrt_array_to_client(array)
    @ccall Reactant_jll.libReactantExtra.ifrt_array_to_client(
        array::Ptr{HeldIfrtArray}
    )::Ptr{Client}
end

function ifrt_array_to_sharding(array)
    @ccall Reactant_jll.libReactantExtra.ifrt_array_to_sharding(
        array::Ptr{HeldIfrtArray}
    )::Ptr{HeldIfrtConstSharding}
end

function ifrt_array_copy_to_host_buffer(array, data)
    @ccall Reactant_jll.libReactantExtra.ifrt_array_copy_to_host_buffer(
        array::Ptr{HeldIfrtArray}, data::Ptr{Cvoid}
    )::Cvoid
end

function ifrt_array_disassemble_into_single_device_arrays(
    array, c_semantics, c_single_device_shard_semantics, narrays
)
    @ccall Reactant_jll.libReactantExtra.ifrt_array_disassemble_into_single_device_arrays(
        array::Ptr{HeldIfrtArray},
        c_semantics::Int32,
        c_single_device_shard_semantics::Int32,
        narrays::Ptr{Int32},
    )::Ptr{Ptr{HeldIfrtArray}}
end

function GetDistributedRuntimeClientWithOptions(c_address, options)
    @ccall Reactant_jll.libReactantExtra.GetDistributedRuntimeClientWithOptions(
        c_address::Cstring, options::Ptr{DistributedRuntimeClientOptions}
    )::Ptr{HeldDistributedRuntimeClient}
end

function GetDistributedRuntimeClient(
    c_address,
    node_id,
    rpc_timeout_in_seconds,
    init_timeout,
    shutdown_timeout_in_minutes,
    heartbeat_timeout_in_seconds,
    use_compression,
)
    @ccall Reactant_jll.libReactantExtra.GetDistributedRuntimeClient(
        c_address::Cstring,
        node_id::Int32,
        rpc_timeout_in_seconds::Int32,
        init_timeout::Int32,
        shutdown_timeout_in_minutes::Int32,
        heartbeat_timeout_in_seconds::Int32,
        use_compression::Bool,
    )::Ptr{HeldDistributedRuntimeClient}
end

function free_distributed_runtime_client(client)
    @ccall Reactant_jll.libReactantExtra.free_distributed_runtime_client(
        client::Ptr{HeldDistributedRuntimeClient}
    )::Cvoid
end

function distributed_runtime_client_connect(client)
    @ccall Reactant_jll.libReactantExtra.distributed_runtime_client_connect(
        client::Ptr{HeldDistributedRuntimeClient}
    )::Cvoid
end

function distributed_runtime_client_shutdown(client)
    @ccall Reactant_jll.libReactantExtra.distributed_runtime_client_shutdown(
        client::Ptr{HeldDistributedRuntimeClient}
    )::Cvoid
end

function GetDistributedRuntimeServiceWithOptions(c_address, options)
    @ccall Reactant_jll.libReactantExtra.GetDistributedRuntimeServiceWithOptions(
        c_address::Cstring, options::Ptr{DistributedRuntimeServiceOptions}
    )::Ptr{DistributedRuntimeService}
end

function GetDistributedRuntimeService(
    c_address,
    num_nodes,
    heartbeat_timeout_in_seconds,
    cluster_register_timeout_in_minutes,
    shutdown_timeout_in_minutes,
)
    @ccall Reactant_jll.libReactantExtra.GetDistributedRuntimeService(
        c_address::Cstring,
        num_nodes::Cint,
        heartbeat_timeout_in_seconds::Int32,
        cluster_register_timeout_in_minutes::Int32,
        shutdown_timeout_in_minutes::Int32,
    )::Ptr{DistributedRuntimeService}
end

function free_distributed_runtime_service(service)
    @ccall Reactant_jll.libReactantExtra.free_distributed_runtime_service(
        service::Ptr{DistributedRuntimeService}
    )::Cvoid
end

function distributed_runtime_service_shutdown(service)
    @ccall Reactant_jll.libReactantExtra.distributed_runtime_service_shutdown(
        service::Ptr{DistributedRuntimeService}
    )::Cvoid
end

function hloShardingFromTensorShardingAttr(cattr, cmeshAttr)
    @ccall Reactant_jll.libReactantExtra.hloShardingFromTensorShardingAttr(
        cattr::MlirAttribute, cmeshAttr::MlirAttribute
    )::Ptr{HloSharding}
end

function hloShardingToTensorShardingAttr(
    cctx, hloSharding, cmeshName, cmeshAttr, rank, isClosed, priority
)
    @ccall Reactant_jll.libReactantExtra.hloShardingToTensorShardingAttr(
        cctx::MlirContext,
        hloSharding::Ptr{HloSharding},
        cmeshName::MlirAttribute,
        cmeshAttr::MlirAttribute,
        rank::Int64,
        isClosed::Ptr{Bool},
        priority::Ptr{Int64},
    )::MlirAttribute
end

function ifrt_loaded_executable_dtor(exec)
    @ccall Reactant_jll.libReactantExtra.ifrt_loaded_executable_dtor(
        exec::Ptr{HeldIfrtLoadedExecutable}
    )::Cvoid
end

function ifrt_loaded_executable_execute(
    exec, num_args, op_args, is_arg_donatable, num_results, op_results, futures, status
)
    @ccall Reactant_jll.libReactantExtra.ifrt_loaded_executable_execute(
        exec::Ptr{HeldIfrtLoadedExecutable},
        num_args::Cint,
        op_args::Ptr{Ptr{HeldIfrtArray}},
        is_arg_donatable::Ptr{UInt8},
        num_results::Cint,
        op_results::Ptr{Ptr{HeldIfrtArray}},
        futures::Ptr{UInt8},
        status::Ptr{Ptr{FutureType}},
    )::Cvoid
end

function ifrt_loaded_executable_client(exec)
    @ccall Reactant_jll.libReactantExtra.ifrt_loaded_executable_client(
        exec::Ptr{HeldIfrtLoadedExecutable}
    )::Ptr{Client}
end

function ifrt_loaded_executable_get_parameter_shardings(
    exec, op_shardings, num_op_shardings
)
    @ccall Reactant_jll.libReactantExtra.ifrt_loaded_executable_get_parameter_shardings(
        exec::Ptr{HeldIfrtLoadedExecutable},
        op_shardings::Ptr{Ptr{OpSharding}},
        num_op_shardings::Int32,
    )::Cvoid
end

function ifrt_loaded_executable_get_output_shardings(exec, op_shardings, num_op_shardings)
    @ccall Reactant_jll.libReactantExtra.ifrt_loaded_executable_get_output_shardings(
        exec::Ptr{HeldIfrtLoadedExecutable},
        op_shardings::Ptr{Ptr{OpSharding}},
        num_op_shardings::Int32,
    )::Cvoid
end

function ifrt_loaded_executable_get_hlo_modules(exec, hlo_modules, nmodules)
    @ccall Reactant_jll.libReactantExtra.ifrt_loaded_executable_get_hlo_modules(
        exec::Ptr{HeldIfrtLoadedExecutable},
        hlo_modules::Ptr{Ptr{Cvoid}},
        nmodules::Ptr{Int32},
    )::Cvoid
end

function ifrt_loaded_executable_num_devices(exec)
    @ccall Reactant_jll.libReactantExtra.ifrt_loaded_executable_num_devices(
        exec::Ptr{HeldIfrtLoadedExecutable}
    )::Int32
end

function pjrt_hlo_module_cost_analysis_properties(client, hlo_module, jlproperties)
    @ccall Reactant_jll.libReactantExtra.pjrt_hlo_module_cost_analysis_properties(
        client::Ptr{PjRtClient},
        hlo_module::Ptr{HeldHloModule},
        jlproperties::Ptr{JLHloCostAnalysisProperties},
    )::Cvoid
end

function ifrt_hlo_module_cost_analysis_properties(client, hlo_module, jlproperties)
    @ccall Reactant_jll.libReactantExtra.ifrt_hlo_module_cost_analysis_properties(
        client::Ptr{Client},
        hlo_module::Ptr{HeldHloModule},
        jlproperties::Ptr{JLHloCostAnalysisProperties},
    )::Cvoid
end

function pjrt_device_is_addressable(device)
    @ccall Reactant_jll.libReactantExtra.pjrt_device_is_addressable(
        device::Ptr{PjRtDevice}
    )::Bool
end

function mlirGetParentOfTypeFunctionOp(op)
    @ccall Reactant_jll.libReactantExtra.mlirGetParentOfTypeFunctionOp(
        op::Ptr{Operation}
    )::Ptr{Operation}
end

function ifrt_copy_arrays_to_device_with_sharding(
    client, arrays, num_arrays, dst_sharding, c_semantics
)
    @ccall Reactant_jll.libReactantExtra.ifrt_copy_arrays_to_device_with_sharding(
        client::Ptr{Client},
        arrays::Ptr{Ptr{HeldIfrtArray}},
        num_arrays::Int32,
        dst_sharding::Ptr{HeldIfrtConstSharding},
        c_semantics::Int32,
    )::Ptr{Ptr{HeldIfrtArray}}
end

function ifrt_make_array_from_host_buffer_shards(
    client,
    host_buffers,
    num_buffers,
    host_buffer_shapes,
    addressable_shard_indices,
    addressable_shard_indices_sizes,
    dtype_kind,
    ndims,
    final_buffer_shape,
    sharding,
    c_host_buffer_semantics,
)
    @ccall Reactant_jll.libReactantExtra.ifrt_make_array_from_host_buffer_shards(
        client::Ptr{Client},
        host_buffers::Ptr{Ptr{Cvoid}},
        num_buffers::Cint,
        host_buffer_shapes::Ptr{Ptr{Int64}},
        addressable_shard_indices::Ptr{Ptr{Int64}},
        addressable_shard_indices_sizes::Ptr{Int64},
        dtype_kind::Cint,
        ndims::Cint,
        final_buffer_shape::Ptr{Int64},
        sharding::Ptr{HeldIfrtConstSharding},
        c_host_buffer_semantics::Int32,
    )::Ptr{HeldIfrtArray}
end

function ifrt_copy_array(array)
    @ccall Reactant_jll.libReactantExtra.ifrt_copy_array(
        array::Ptr{HeldIfrtArray}
    )::Ptr{HeldIfrtArray}
end

function reactantXLAThrow(str)
    @ccall Reactant_jll.libReactantExtra.reactantXLAThrow(str::Cstring)::Cvoid
end

function reactantXLAInit(lrtP, backend)
    @ccall Reactant_jll.libReactantExtra.reactantXLAInit(
        lrtP::Ptr{Ptr{LinkableRuntime}}, backend::Cstring
    )::Cvoid
end

function reactantXLADeInit(lrt)
    @ccall Reactant_jll.libReactantExtra.reactantXLADeInit(
        lrt::Ptr{Ptr{LinkableRuntime}}
    )::Cvoid
end

function reactantXLAMemcpy(lrtP, dst, src, size, direction)
    @ccall Reactant_jll.libReactantExtra.reactantXLAMemcpy(
        lrtP::Ptr{Ptr{LinkableRuntime}},
        dst::Ptr{Cvoid},
        src::Ptr{Cvoid},
        size::Csize_t,
        direction::Int32,
    )::Cvoid
end

function reactantXLAMalloc(lrtP, ptype, shapeLen, shape)
    @ccall Reactant_jll.libReactantExtra.reactantXLAMalloc(
        lrtP::Ptr{Ptr{LinkableRuntime}}, ptype::UInt64, shapeLen::UInt64, shape::Ptr{UInt64}
    )::Ptr{Cvoid}
end

function reactantXLAFree(lrtP, buffer0)
    @ccall Reactant_jll.libReactantExtra.reactantXLAFree(
        lrtP::Ptr{Ptr{LinkableRuntime}}, buffer0::Ptr{Cvoid}
    )::Cvoid
end

function reactantXLAExec(lrtP, modstr, argcnt, args, constcnt, consts)
    @ccall Reactant_jll.libReactantExtra.reactantXLAExec(
        lrtP::Ptr{Ptr{LinkableRuntime}},
        modstr::Cstring,
        argcnt::Int64,
        args::Ptr{Ptr{Cvoid}},
        constcnt::Int64,
        consts::Ptr{Int64},
    )::Cvoid
end

function convertMlirModuleToHloModule(mod)
    @ccall Reactant_jll.libReactantExtra.convertMlirModuleToHloModule(
        mod::MlirModule
    )::Ptr{HeldHloModule}
end

function parseAndReturnUnverifiedHloModule(cstr)
    @ccall Reactant_jll.libReactantExtra.parseAndReturnUnverifiedHloModule(
        cstr::Cstring
    )::Ptr{HeldHloModule}
end

function hloModuleGetEntryComputation(hlo_module)
    @ccall Reactant_jll.libReactantExtra.hloModuleGetEntryComputation(
        hlo_module::Ptr{HeldHloModule}
    )::Ptr{HloComputation}
end

function freeHloComputation(hlo_computation)
    @ccall Reactant_jll.libReactantExtra.freeHloComputation(
        hlo_computation::Ptr{HloComputation}
    )::Cvoid
end

function hloComputationToString(hlo_computation, print_options)
    @ccall Reactant_jll.libReactantExtra.hloComputationToString(
        hlo_computation::Ptr{HloComputation}, print_options::Int32
    )::Cstring
end

function hloComputationInstructionCount(hlo_computation)
    @ccall Reactant_jll.libReactantExtra.hloComputationInstructionCount(
        hlo_computation::Ptr{HloComputation}
    )::Int64
end

function hloComputationGetInstructionsPostOrder(
    hlo_computation, num_instructions, hlo_instructions
)
    @ccall Reactant_jll.libReactantExtra.hloComputationGetInstructionsPostOrder(
        hlo_computation::Ptr{HloComputation},
        num_instructions::Int64,
        hlo_instructions::Ptr{Ptr{HloInstruction}},
    )::Cvoid
end

function freeHloInstruction(hlo_instruction)
    @ccall Reactant_jll.libReactantExtra.freeHloInstruction(
        hlo_instruction::Ptr{HloInstruction}
    )::Cvoid
end

function hloInstructionToString(hlo_instruction, print_options)
    @ccall Reactant_jll.libReactantExtra.hloInstructionToString(
        hlo_instruction::Ptr{HloInstruction}, print_options::Int32
    )::Cstring
end

function hloInstructionHasToApply(hlo_instruction)
    @ccall Reactant_jll.libReactantExtra.hloInstructionHasToApply(
        hlo_instruction::Ptr{HloInstruction}
    )::UInt8
end

function hloInstructionGetToApply(hlo_instruction)
    @ccall Reactant_jll.libReactantExtra.hloInstructionGetToApply(
        hlo_instruction::Ptr{HloInstruction}
    )::Ptr{HloComputation}
end

function hloInstructionGetOpcode(hlo_instruction)
    @ccall Reactant_jll.libReactantExtra.hloInstructionGetOpcode(
        hlo_instruction::Ptr{HloInstruction}
    )::UInt8
end

function hloOpcodeToString(opcode)
    @ccall Reactant_jll.libReactantExtra.hloOpcodeToString(opcode::UInt8)::Cstring
end

function hloInstructionIsFusion(hlo_instruction)
    @ccall Reactant_jll.libReactantExtra.hloInstructionIsFusion(
        hlo_instruction::Ptr{HloInstruction}
    )::UInt8
end

function hloInstructionGetFusionKind(hlo_instruction)
    @ccall Reactant_jll.libReactantExtra.hloInstructionGetFusionKind(
        hlo_instruction::Ptr{HloInstruction}
    )::UInt8
end

function hloFusionKindToString(kind)
    @ccall Reactant_jll.libReactantExtra.hloFusionKindToString(kind::UInt8)::Cstring
end

function hloInstructionFusedInstructionsComputation(hlo_instruction)
    @ccall Reactant_jll.libReactantExtra.hloInstructionFusedInstructionsComputation(
        hlo_instruction::Ptr{HloInstruction}
    )::Ptr{HloComputation}
end

function CreateGPUPerformanceModel(device_description)
    @ccall Reactant_jll.libReactantExtra.CreateGPUPerformanceModel(
        device_description::Ptr{DeviceDescription}
    )::Ptr{GPUPerformanceModel}
end

function RunAnalysisOnHloModule(gpu_performance_model, hlo_module)
    @ccall Reactant_jll.libReactantExtra.RunAnalysisOnHloModule(
        gpu_performance_model::Ptr{GPUPerformanceModel}, hlo_module::Ptr{HeldHloModule}
    )::Cvoid
end

function EstimateRunTimeForInstruction(gpu_performance_model, hlo_instruction, jldata)
    @ccall Reactant_jll.libReactantExtra.EstimateRunTimeForInstruction(
        gpu_performance_model::Ptr{GPUPerformanceModel},
        hlo_instruction::Ptr{HloInstruction},
        jldata::Ptr{JLEstimateRunTimeData},
    )::Cvoid
end

function InitializeXProfStubs(cstr_worker_service_address)
    @ccall Reactant_jll.libReactantExtra.InitializeXProfStubs(
        cstr_worker_service_address::Cstring
    )::Cvoid
end

function StartGrpcServer(port)
    @ccall Reactant_jll.libReactantExtra.StartGrpcServer(port::Cint)::Cvoid
end

function XSpaceToToolsData(
    xspace_paths,
    num_paths,
    tool_name,
    bool_keys,
    bool_values,
    bool_count,
    int_keys,
    int_values,
    int_count,
    str_keys,
    str_values,
    str_count,
    result_data,
    result_size,
    is_binary,
    error,
)
    @ccall Reactant_jll.libReactantExtra.XSpaceToToolsData(
        xspace_paths::Ptr{Cstring},
        num_paths::Int64,
        tool_name::Cstring,
        bool_keys::Ptr{Cstring},
        bool_values::Ptr{Bool},
        bool_count::Int64,
        int_keys::Ptr{Cstring},
        int_values::Ptr{Cint},
        int_count::Int64,
        str_keys::Ptr{Cstring},
        str_values::Ptr{Cstring},
        str_count::Int64,
        result_data::Ptr{Cstring},
        result_size::Ptr{Int64},
        is_binary::Ptr{Bool},
        error::Ptr{Cstring},
    )::Cint
end

function ReactantGetDebugOptions(size)
    @ccall Reactant_jll.libReactantExtra.ReactantGetDebugOptions(
        size::Ptr{Csize_t}
    )::Ptr{Cvoid}
end

function ReactantGetCompileOptions(size)
    @ccall Reactant_jll.libReactantExtra.ReactantGetCompileOptions(
        size::Ptr{Csize_t}
    )::Ptr{Cvoid}
end

function ReactantCompileMhloToLLVM(
    mhlo_text, mhlo_text_len, out_output_str, xla_runtime, pass_pipeline
)
    @ccall Reactant_jll.libReactantExtra.ReactantCompileMhloToLLVM(
        mhlo_text::Cstring,
        mhlo_text_len::Csize_t,
        out_output_str::Ptr{Cstring},
        xla_runtime::UInt8,
        pass_pipeline::Cstring,
    )::Ptr{LocalExecutable}
end

function ReactantFreeLocalExecutable(exec)
    @ccall Reactant_jll.libReactantExtra.ReactantFreeLocalExecutable(
        exec::Ptr{LocalExecutable}
    )::Cvoid
end

function ReactantCreateLLVMMod(
    fn_str,
    fn_len,
    source_str,
    source_len,
    out_shapes_data,
    out_shapes_sizes,
    num_out_shapes,
    out_names_data,
    num_out_names,
    in_shapes_data,
    in_shapes_sizes,
    num_in_shapes,
    in_names_data,
    num_in_names,
    argv_data,
    num_argv,
    mode_enum,
    lang_enum,
    xla_runtime,
    pass_pipeline,
    out_module,
    out_context,
    out_off,
    out_tmp_buf,
)
    @ccall Reactant_jll.libReactantExtra.ReactantCreateLLVMMod(
        fn_str::Cstring,
        fn_len::Csize_t,
        source_str::Cstring,
        source_len::Csize_t,
        out_shapes_data::Ptr{Int64},
        out_shapes_sizes::Ptr{Csize_t},
        num_out_shapes::Csize_t,
        out_names_data::Ptr{Cstring},
        num_out_names::Csize_t,
        in_shapes_data::Ptr{Int64},
        in_shapes_sizes::Ptr{Csize_t},
        num_in_shapes::Csize_t,
        in_names_data::Ptr{Cstring},
        num_in_names::Csize_t,
        argv_data::Ptr{Cstring},
        num_argv::Csize_t,
        mode_enum::Cint,
        lang_enum::Cint,
        xla_runtime::UInt8,
        pass_pipeline::Cstring,
        out_module::Ptr{Ptr{Module}},
        out_context::Ptr{Ptr{LLVMContext}},
        out_off::Ptr{Csize_t},
        out_tmp_buf::Ptr{Csize_t},
    )::Cvoid
end

function ReactantLexMLIR(
    ctx, input, input_len, token_kinds, token_offsets, token_lengths, max_tokens
)
    @ccall Reactant_jll.libReactantExtra.ReactantLexMLIR(
        ctx::MlirContext,
        input::Cstring,
        input_len::Int32,
        token_kinds::Ptr{Int32},
        token_offsets::Ptr{Int32},
        token_lengths::Ptr{Int32},
        max_tokens::Int32,
    )::Int32
end

function registerReactantXLAFFI()
    @ccall Reactant_jll.libReactantExtra.registerReactantXLAFFI()::Cvoid
end

const MLIR_CAPI_DWARF_ADDRESS_SPACE_NULL = -1

function addSdyPropagationPipeline(
    pm,
    keepShardingRules,
    conservativePropagation,
    debugShardingOrigins,
    debugPropagationEdgeSharding,
    skipConvertToReshard,
    skipInline,
    enableInsertExplicitCollectives,
)
    @ccall Reactant_jll.libReactantExtra.addSdyPropagationPipeline(
        pm::MlirOpPassManager,
        keepShardingRules::UInt8,
        conservativePropagation::UInt8,
        debugShardingOrigins::UInt8,
        debugPropagationEdgeSharding::UInt8,
        skipConvertToReshard::UInt8,
        skipInline::UInt8,
        enableInsertExplicitCollectives::UInt8,
    )::Cvoid
end

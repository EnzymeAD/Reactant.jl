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
    @ccall mlir_c.mlirStringRefCreate(str::Cstring, length::Csize_t)::MlirStringRef
end

"""
    mlirStringRefCreateFromCString(str)

Constructs a string reference from a null-terminated C string. Prefer [`mlirStringRefCreate`](@ref) if the length of the string is known.
"""
function mlirStringRefCreateFromCString(str)
    @ccall mlir_c.mlirStringRefCreateFromCString(str::Cstring)::MlirStringRef
end

"""
    mlirStringRefEqual(string, other)

Returns true if two string references are equal, false otherwise.
"""
function mlirStringRefEqual(string, other)
    @ccall mlir_c.mlirStringRefEqual(string::MlirStringRef, other::MlirStringRef)::Bool
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
    @ccall mlir_c.mlirLogicalResultIsSuccess(res::MlirLogicalResult)::Bool
end

"""
    mlirLogicalResultIsFailure(res)

Checks if the given logical result represents a failure.
"""
function mlirLogicalResultIsFailure(res)
    @ccall mlir_c.mlirLogicalResultIsFailure(res::MlirLogicalResult)::Bool
end

"""
    mlirLogicalResultSuccess()

Creates a logical result representing a success.
"""
function mlirLogicalResultSuccess()
    @ccall mlir_c.mlirLogicalResultSuccess()::MlirLogicalResult
end

"""
    mlirLogicalResultFailure()

Creates a logical result representing a failure.
"""
function mlirLogicalResultFailure()
    @ccall mlir_c.mlirLogicalResultFailure()::MlirLogicalResult
end

"""
    mlirLlvmThreadPoolCreate()

Create an LLVM thread pool. This is reexported here to avoid directly pulling in the LLVM headers directly.
"""
function mlirLlvmThreadPoolCreate()
    @ccall mlir_c.mlirLlvmThreadPoolCreate()::MlirLlvmThreadPool
end

"""
    mlirLlvmThreadPoolDestroy(pool)

Destroy an LLVM thread pool.
"""
function mlirLlvmThreadPoolDestroy(pool)
    @ccall mlir_c.mlirLlvmThreadPoolDestroy(pool::MlirLlvmThreadPool)::Cvoid
end

"""
    mlirTypeIDCreate(ptr)

`ptr` must be 8 byte aligned and unique to a type valid for the duration of the returned type id's usage
"""
function mlirTypeIDCreate(ptr)
    @ccall mlir_c.mlirTypeIDCreate(ptr::Ptr{Cvoid})::MlirTypeID
end

"""
    mlirTypeIDIsNull(typeID)

Checks whether a type id is null.
"""
function mlirTypeIDIsNull(typeID)
    @ccall mlir_c.mlirTypeIDIsNull(typeID::MlirTypeID)::Bool
end

"""
    mlirTypeIDEqual(typeID1, typeID2)

Checks if two type ids are equal.
"""
function mlirTypeIDEqual(typeID1, typeID2)
    @ccall mlir_c.mlirTypeIDEqual(typeID1::MlirTypeID, typeID2::MlirTypeID)::Bool
end

"""
    mlirTypeIDHashValue(typeID)

Returns the hash value of the type id.
"""
function mlirTypeIDHashValue(typeID)
    @ccall mlir_c.mlirTypeIDHashValue(typeID::MlirTypeID)::Csize_t
end

"""
    mlirTypeIDAllocatorCreate()

Creates a type id allocator for dynamic type id creation
"""
function mlirTypeIDAllocatorCreate()
    @ccall mlir_c.mlirTypeIDAllocatorCreate()::MlirTypeIDAllocator
end

"""
    mlirTypeIDAllocatorDestroy(allocator)

Deallocates the allocator and all allocated type ids
"""
function mlirTypeIDAllocatorDestroy(allocator)
    @ccall mlir_c.mlirTypeIDAllocatorDestroy(allocator::MlirTypeIDAllocator)::Cvoid
end

"""
    mlirTypeIDAllocatorAllocateTypeID(allocator)

Allocates a type id that is valid for the lifetime of the allocator
"""
function mlirTypeIDAllocatorAllocateTypeID(allocator)
    @ccall mlir_c.mlirTypeIDAllocatorAllocateTypeID(
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
    @ccall mlir_c.mlirContextCreate()::MlirContext
end

"""
    mlirContextCreateWithThreading(threadingEnabled)

Creates an MLIR context with an explicit setting of the multithreading setting and transfers its ownership to the caller.
"""
function mlirContextCreateWithThreading(threadingEnabled)
    @ccall mlir_c.mlirContextCreateWithThreading(threadingEnabled::Bool)::MlirContext
end

"""
    mlirContextCreateWithRegistry(registry, threadingEnabled)

Creates an MLIR context, setting the multithreading setting explicitly and pre-loading the dialects from the provided DialectRegistry.
"""
function mlirContextCreateWithRegistry(registry, threadingEnabled)
    @ccall mlir_c.mlirContextCreateWithRegistry(
        registry::MlirDialectRegistry, threadingEnabled::Bool
    )::MlirContext
end

"""
    mlirContextEqual(ctx1, ctx2)

Checks if two contexts are equal.
"""
function mlirContextEqual(ctx1, ctx2)
    @ccall mlir_c.mlirContextEqual(ctx1::MlirContext, ctx2::MlirContext)::Bool
end

"""
    mlirContextIsNull(context)

Checks whether a context is null.
"""
function mlirContextIsNull(context)
    @ccall mlir_c.mlirContextIsNull(context::MlirContext)::Bool
end

"""
    mlirContextDestroy(context)

Takes an MLIR context owned by the caller and destroys it.
"""
function mlirContextDestroy(context)
    @ccall mlir_c.mlirContextDestroy(context::MlirContext)::Cvoid
end

"""
    mlirContextSetAllowUnregisteredDialects(context, allow)

Sets whether unregistered dialects are allowed in this context.
"""
function mlirContextSetAllowUnregisteredDialects(context, allow)
    @ccall mlir_c.mlirContextSetAllowUnregisteredDialects(
        context::MlirContext, allow::Bool
    )::Cvoid
end

"""
    mlirContextGetAllowUnregisteredDialects(context)

Returns whether the context allows unregistered dialects.
"""
function mlirContextGetAllowUnregisteredDialects(context)
    @ccall mlir_c.mlirContextGetAllowUnregisteredDialects(context::MlirContext)::Bool
end

"""
    mlirContextGetNumRegisteredDialects(context)

Returns the number of dialects registered with the given context. A registered dialect will be loaded if needed by the parser.
"""
function mlirContextGetNumRegisteredDialects(context)
    @ccall mlir_c.mlirContextGetNumRegisteredDialects(context::MlirContext)::Cptrdiff_t
end

"""
    mlirContextAppendDialectRegistry(ctx, registry)

Append the contents of the given dialect registry to the registry associated with the context.
"""
function mlirContextAppendDialectRegistry(ctx, registry)
    @ccall mlir_c.mlirContextAppendDialectRegistry(
        ctx::MlirContext, registry::MlirDialectRegistry
    )::Cvoid
end

"""
    mlirContextGetNumLoadedDialects(context)

Returns the number of dialects loaded by the context.
"""
function mlirContextGetNumLoadedDialects(context)
    @ccall mlir_c.mlirContextGetNumLoadedDialects(context::MlirContext)::Cptrdiff_t
end

"""
    mlirContextGetOrLoadDialect(context, name)

Gets the dialect instance owned by the given context using the dialect namespace to identify it, loads (i.e., constructs the instance of) the dialect if necessary. If the dialect is not registered with the context, returns null. Use mlirContextLoad<Name>Dialect to load an unregistered dialect.
"""
function mlirContextGetOrLoadDialect(context, name)
    @ccall mlir_c.mlirContextGetOrLoadDialect(
        context::MlirContext, name::MlirStringRef
    )::MlirDialect
end

"""
    mlirContextEnableMultithreading(context, enable)

Set threading mode (must be set to false to mlir-print-ir-after-all).
"""
function mlirContextEnableMultithreading(context, enable)
    @ccall mlir_c.mlirContextEnableMultithreading(context::MlirContext, enable::Bool)::Cvoid
end

"""
    mlirContextLoadAllAvailableDialects(context)

Eagerly loads all available dialects registered with a context, making them available for use for IR construction.
"""
function mlirContextLoadAllAvailableDialects(context)
    @ccall mlir_c.mlirContextLoadAllAvailableDialects(context::MlirContext)::Cvoid
end

"""
    mlirContextIsRegisteredOperation(context, name)

Returns whether the given fully-qualified operation (i.e. 'dialect.operation') is registered with the context. This will return true if the dialect is loaded and the operation is registered within the dialect.
"""
function mlirContextIsRegisteredOperation(context, name)
    @ccall mlir_c.mlirContextIsRegisteredOperation(
        context::MlirContext, name::MlirStringRef
    )::Bool
end

"""
    mlirContextSetThreadPool(context, threadPool)

Sets the thread pool of the context explicitly, enabling multithreading in the process. This API should be used to avoid re-creating thread pools in long-running applications that perform multiple compilations, see the C++ documentation for MLIRContext for details.
"""
function mlirContextSetThreadPool(context, threadPool)
    @ccall mlir_c.mlirContextSetThreadPool(
        context::MlirContext, threadPool::MlirLlvmThreadPool
    )::Cvoid
end

"""
    mlirContextGetNumThreads(context)

Gets the number of threads of the thread pool of the context when multithreading is enabled. Returns 1 if no multithreading.
"""
function mlirContextGetNumThreads(context)
    @ccall mlir_c.mlirContextGetNumThreads(context::MlirContext)::Cuint
end

"""
    mlirContextGetThreadPool(context)

Gets the thread pool of the context when enabled multithreading, otherwise an assertion is raised.
"""
function mlirContextGetThreadPool(context)
    @ccall mlir_c.mlirContextGetThreadPool(context::MlirContext)::MlirLlvmThreadPool
end

"""
    mlirDialectGetContext(dialect)

Returns the context that owns the dialect.
"""
function mlirDialectGetContext(dialect)
    @ccall mlir_c.mlirDialectGetContext(dialect::MlirDialect)::MlirContext
end

"""
    mlirDialectIsNull(dialect)

Checks if the dialect is null.
"""
function mlirDialectIsNull(dialect)
    @ccall mlir_c.mlirDialectIsNull(dialect::MlirDialect)::Bool
end

"""
    mlirDialectEqual(dialect1, dialect2)

Checks if two dialects that belong to the same context are equal. Dialects from different contexts will not compare equal.
"""
function mlirDialectEqual(dialect1, dialect2)
    @ccall mlir_c.mlirDialectEqual(dialect1::MlirDialect, dialect2::MlirDialect)::Bool
end

"""
    mlirDialectGetNamespace(dialect)

Returns the namespace of the given dialect.
"""
function mlirDialectGetNamespace(dialect)
    @ccall mlir_c.mlirDialectGetNamespace(dialect::MlirDialect)::MlirStringRef
end

"""
    mlirDialectHandleGetNamespace(arg1)

Returns the namespace associated with the provided dialect handle.
"""
function mlirDialectHandleGetNamespace(arg1)
    @ccall mlir_c.mlirDialectHandleGetNamespace(arg1::MlirDialectHandle)::MlirStringRef
end

"""
    mlirDialectHandleInsertDialect(arg1, arg2)

Inserts the dialect associated with the provided dialect handle into the provided dialect registry
"""
function mlirDialectHandleInsertDialect(arg1, arg2)
    @ccall mlir_c.mlirDialectHandleInsertDialect(
        arg1::MlirDialectHandle, arg2::MlirDialectRegistry
    )::Cvoid
end

"""
    mlirDialectHandleRegisterDialect(arg1, arg2)

Registers the dialect associated with the provided dialect handle.
"""
function mlirDialectHandleRegisterDialect(arg1, arg2)
    @ccall mlir_c.mlirDialectHandleRegisterDialect(
        arg1::MlirDialectHandle, arg2::MlirContext
    )::Cvoid
end

"""
    mlirDialectHandleLoadDialect(arg1, arg2)

Loads the dialect associated with the provided dialect handle.
"""
function mlirDialectHandleLoadDialect(arg1, arg2)
    @ccall mlir_c.mlirDialectHandleLoadDialect(
        arg1::MlirDialectHandle, arg2::MlirContext
    )::MlirDialect
end

"""
    mlirDialectRegistryCreate()

Creates a dialect registry and transfers its ownership to the caller.
"""
function mlirDialectRegistryCreate()
    @ccall mlir_c.mlirDialectRegistryCreate()::MlirDialectRegistry
end

"""
    mlirDialectRegistryIsNull(registry)

Checks if the dialect registry is null.
"""
function mlirDialectRegistryIsNull(registry)
    @ccall mlir_c.mlirDialectRegistryIsNull(registry::MlirDialectRegistry)::Bool
end

"""
    mlirDialectRegistryDestroy(registry)

Takes a dialect registry owned by the caller and destroys it.
"""
function mlirDialectRegistryDestroy(registry)
    @ccall mlir_c.mlirDialectRegistryDestroy(registry::MlirDialectRegistry)::Cvoid
end

"""
    mlirLocationGetAttribute(location)

Returns the underlying location attribute of this location.
"""
function mlirLocationGetAttribute(location)
    @ccall mlir_c.mlirLocationGetAttribute(location::MlirLocation)::MlirAttribute
end

"""
    mlirLocationFromAttribute(attribute)

Creates a location from a location attribute.
"""
function mlirLocationFromAttribute(attribute)
    @ccall mlir_c.mlirLocationFromAttribute(attribute::MlirAttribute)::MlirLocation
end

"""
    mlirLocationFileLineColGet(context, filename, line, col)

Creates an File/Line/Column location owned by the given context.
"""
function mlirLocationFileLineColGet(context, filename, line, col)
    @ccall mlir_c.mlirLocationFileLineColGet(
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
    @ccall mlir_c.mlirLocationFileLineColRangeGet(
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
    @ccall mlir_c.mlirLocationFileLineColRangeGetFilename(
        location::MlirLocation
    )::MlirIdentifier
end

"""
    mlirLocationFileLineColRangeGetStartLine(location)

Getter for start\\_line of FileLineColRange.
"""
function mlirLocationFileLineColRangeGetStartLine(location)
    @ccall mlir_c.mlirLocationFileLineColRangeGetStartLine(location::MlirLocation)::Cint
end

"""
    mlirLocationFileLineColRangeGetStartColumn(location)

Getter for start\\_column of FileLineColRange.
"""
function mlirLocationFileLineColRangeGetStartColumn(location)
    @ccall mlir_c.mlirLocationFileLineColRangeGetStartColumn(location::MlirLocation)::Cint
end

"""
    mlirLocationFileLineColRangeGetEndLine(location)

Getter for end\\_line of FileLineColRange.
"""
function mlirLocationFileLineColRangeGetEndLine(location)
    @ccall mlir_c.mlirLocationFileLineColRangeGetEndLine(location::MlirLocation)::Cint
end

"""
    mlirLocationFileLineColRangeGetEndColumn(location)

Getter for end\\_column of FileLineColRange.
"""
function mlirLocationFileLineColRangeGetEndColumn(location)
    @ccall mlir_c.mlirLocationFileLineColRangeGetEndColumn(location::MlirLocation)::Cint
end

"""
    mlirLocationFileLineColRangeGetTypeID()

TypeID Getter for FileLineColRange.
"""
function mlirLocationFileLineColRangeGetTypeID()
    @ccall mlir_c.mlirLocationFileLineColRangeGetTypeID()::MlirTypeID
end

"""
    mlirLocationIsAFileLineColRange(location)

Checks whether the given location is an FileLineColRange.
"""
function mlirLocationIsAFileLineColRange(location)
    @ccall mlir_c.mlirLocationIsAFileLineColRange(location::MlirLocation)::Bool
end

"""
    mlirLocationCallSiteGet(callee, caller)

Creates a call site location with a callee and a caller.
"""
function mlirLocationCallSiteGet(callee, caller)
    @ccall mlir_c.mlirLocationCallSiteGet(
        callee::MlirLocation, caller::MlirLocation
    )::MlirLocation
end

"""
    mlirLocationCallSiteGetCallee(location)

Getter for callee of CallSite.
"""
function mlirLocationCallSiteGetCallee(location)
    @ccall mlir_c.mlirLocationCallSiteGetCallee(location::MlirLocation)::MlirLocation
end

"""
    mlirLocationCallSiteGetCaller(location)

Getter for caller of CallSite.
"""
function mlirLocationCallSiteGetCaller(location)
    @ccall mlir_c.mlirLocationCallSiteGetCaller(location::MlirLocation)::MlirLocation
end

"""
    mlirLocationCallSiteGetTypeID()

TypeID Getter for CallSite.
"""
function mlirLocationCallSiteGetTypeID()
    @ccall mlir_c.mlirLocationCallSiteGetTypeID()::MlirTypeID
end

"""
    mlirLocationIsACallSite(location)

Checks whether the given location is an CallSite.
"""
function mlirLocationIsACallSite(location)
    @ccall mlir_c.mlirLocationIsACallSite(location::MlirLocation)::Bool
end

"""
    mlirLocationFusedGet(ctx, nLocations, locations, metadata)

Creates a fused location with an array of locations and metadata.
"""
function mlirLocationFusedGet(ctx, nLocations, locations, metadata)
    @ccall mlir_c.mlirLocationFusedGet(
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
    @ccall mlir_c.mlirLocationFusedGetNumLocations(location::MlirLocation)::Cuint
end

"""
    mlirLocationFusedGetLocations(location, locationsCPtr)

Getter for locations of Fused. Requires pre-allocated memory of #fusedLocations X sizeof([`MlirLocation`](@ref)).
"""
function mlirLocationFusedGetLocations(location, locationsCPtr)
    @ccall mlir_c.mlirLocationFusedGetLocations(
        location::MlirLocation, locationsCPtr::Ptr{MlirLocation}
    )::Cvoid
end

"""
    mlirLocationFusedGetMetadata(location)

Getter for metadata of Fused.
"""
function mlirLocationFusedGetMetadata(location)
    @ccall mlir_c.mlirLocationFusedGetMetadata(location::MlirLocation)::MlirAttribute
end

"""
    mlirLocationFusedGetTypeID()

TypeID Getter for Fused.
"""
function mlirLocationFusedGetTypeID()
    @ccall mlir_c.mlirLocationFusedGetTypeID()::MlirTypeID
end

"""
    mlirLocationIsAFused(location)

Checks whether the given location is an Fused.
"""
function mlirLocationIsAFused(location)
    @ccall mlir_c.mlirLocationIsAFused(location::MlirLocation)::Bool
end

"""
    mlirLocationNameGet(context, name, childLoc)

Creates a name location owned by the given context. Providing null location for childLoc is allowed and if childLoc is null location, then the behavior is the same as having unknown child location.
"""
function mlirLocationNameGet(context, name, childLoc)
    @ccall mlir_c.mlirLocationNameGet(
        context::MlirContext, name::MlirStringRef, childLoc::MlirLocation
    )::MlirLocation
end

"""
    mlirLocationNameGetName(location)

Getter for name of Name.
"""
function mlirLocationNameGetName(location)
    @ccall mlir_c.mlirLocationNameGetName(location::MlirLocation)::MlirIdentifier
end

"""
    mlirLocationNameGetChildLoc(location)

Getter for childLoc of Name.
"""
function mlirLocationNameGetChildLoc(location)
    @ccall mlir_c.mlirLocationNameGetChildLoc(location::MlirLocation)::MlirLocation
end

"""
    mlirLocationNameGetTypeID()

TypeID Getter for Name.
"""
function mlirLocationNameGetTypeID()
    @ccall mlir_c.mlirLocationNameGetTypeID()::MlirTypeID
end

"""
    mlirLocationIsAName(location)

Checks whether the given location is an Name.
"""
function mlirLocationIsAName(location)
    @ccall mlir_c.mlirLocationIsAName(location::MlirLocation)::Bool
end

"""
    mlirLocationUnknownGet(context)

Creates a location with unknown position owned by the given context.
"""
function mlirLocationUnknownGet(context)
    @ccall mlir_c.mlirLocationUnknownGet(context::MlirContext)::MlirLocation
end

"""
    mlirLocationGetContext(location)

Gets the context that a location was created with.
"""
function mlirLocationGetContext(location)
    @ccall mlir_c.mlirLocationGetContext(location::MlirLocation)::MlirContext
end

"""
    mlirLocationIsNull(location)

Checks if the location is null.
"""
function mlirLocationIsNull(location)
    @ccall mlir_c.mlirLocationIsNull(location::MlirLocation)::Bool
end

"""
    mlirLocationEqual(l1, l2)

Checks if two locations are equal.
"""
function mlirLocationEqual(l1, l2)
    @ccall mlir_c.mlirLocationEqual(l1::MlirLocation, l2::MlirLocation)::Bool
end

"""
    mlirLocationPrint(location, callback, userData)

Prints a location by sending chunks of the string representation and forwarding `userData to `callback`. Note that the callback may be called several times with consecutive chunks of the string.
"""
function mlirLocationPrint(location, callback, userData)
    @ccall mlir_c.mlirLocationPrint(
        location::MlirLocation, callback::MlirStringCallback, userData::Ptr{Cvoid}
    )::Cvoid
end

"""
    mlirModuleCreateEmpty(location)

Creates a new, empty module and transfers ownership to the caller.
"""
function mlirModuleCreateEmpty(location)
    @ccall mlir_c.mlirModuleCreateEmpty(location::MlirLocation)::MlirModule
end

"""
    mlirModuleCreateParse(context, _module)

Parses a module from the string and transfers ownership to the caller.
"""
function mlirModuleCreateParse(context, _module)
    @ccall mlir_c.mlirModuleCreateParse(
        context::MlirContext, _module::MlirStringRef
    )::MlirModule
end

"""
    mlirModuleCreateParseFromFile(context, fileName)

Parses a module from file and transfers ownership to the caller.
"""
function mlirModuleCreateParseFromFile(context, fileName)
    @ccall mlir_c.mlirModuleCreateParseFromFile(
        context::MlirContext, fileName::MlirStringRef
    )::MlirModule
end

"""
    mlirModuleGetContext(_module)

Gets the context that a module was created with.
"""
function mlirModuleGetContext(_module)
    @ccall mlir_c.mlirModuleGetContext(_module::MlirModule)::MlirContext
end

"""
    mlirModuleGetBody(_module)

Gets the body of the module, i.e. the only block it contains.
"""
function mlirModuleGetBody(_module)
    @ccall mlir_c.mlirModuleGetBody(_module::MlirModule)::MlirBlock
end

"""
    mlirModuleIsNull(_module)

Checks whether a module is null.
"""
function mlirModuleIsNull(_module)
    @ccall mlir_c.mlirModuleIsNull(_module::MlirModule)::Bool
end

"""
    mlirModuleDestroy(_module)

Takes a module owned by the caller and deletes it.
"""
function mlirModuleDestroy(_module)
    @ccall mlir_c.mlirModuleDestroy(_module::MlirModule)::Cvoid
end

"""
    mlirModuleGetOperation(_module)

Views the module as a generic operation.
"""
function mlirModuleGetOperation(_module)
    @ccall mlir_c.mlirModuleGetOperation(_module::MlirModule)::MlirOperation
end

"""
    mlirModuleFromOperation(op)

Views the generic operation as a module. The returned module is null when the input operation was not a ModuleOp.
"""
function mlirModuleFromOperation(op)
    @ccall mlir_c.mlirModuleFromOperation(op::MlirOperation)::MlirModule
end

"""
    mlirModuleEqual(lhs, rhs)

Checks if two modules are equal.
"""
function mlirModuleEqual(lhs, rhs)
    @ccall mlir_c.mlirModuleEqual(lhs::MlirModule, rhs::MlirModule)::Bool
end

"""
    mlirModuleHashValue(mod)

Compute a hash for the given module.
"""
function mlirModuleHashValue(mod)
    @ccall mlir_c.mlirModuleHashValue(mod::MlirModule)::Csize_t
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
    @ccall mlir_c.mlirOperationStateGet(
        name::MlirStringRef, loc::MlirLocation
    )::MlirOperationState
end

"""
    mlirOperationStateAddResults(state, n, results)

Adds a list of components to the operation state.
"""
function mlirOperationStateAddResults(state, n, results)
    @ccall mlir_c.mlirOperationStateAddResults(
        state::Ptr{MlirOperationState}, n::Cptrdiff_t, results::Ptr{MlirType}
    )::Cvoid
end

function mlirOperationStateAddOperands(state, n, operands)
    @ccall mlir_c.mlirOperationStateAddOperands(
        state::Ptr{MlirOperationState}, n::Cptrdiff_t, operands::Ptr{MlirValue}
    )::Cvoid
end

function mlirOperationStateAddOwnedRegions(state, n, regions)
    @ccall mlir_c.mlirOperationStateAddOwnedRegions(
        state::Ptr{MlirOperationState}, n::Cptrdiff_t, regions::Ptr{MlirRegion}
    )::Cvoid
end

function mlirOperationStateAddSuccessors(state, n, successors)
    @ccall mlir_c.mlirOperationStateAddSuccessors(
        state::Ptr{MlirOperationState}, n::Cptrdiff_t, successors::Ptr{MlirBlock}
    )::Cvoid
end

function mlirOperationStateAddAttributes(state, n, attributes)
    @ccall mlir_c.mlirOperationStateAddAttributes(
        state::Ptr{MlirOperationState}, n::Cptrdiff_t, attributes::Ptr{MlirNamedAttribute}
    )::Cvoid
end

"""
    mlirOperationStateEnableResultTypeInference(state)

Enables result type inference for the operation under construction. If enabled, then the caller must not have called [`mlirOperationStateAddResults`](@ref)(). Note that if enabled, the [`mlirOperationCreate`](@ref)() call is failable: it will return a null operation on inference failure and will emit diagnostics.
"""
function mlirOperationStateEnableResultTypeInference(state)
    @ccall mlir_c.mlirOperationStateEnableResultTypeInference(
        state::Ptr{MlirOperationState}
    )::Cvoid
end

"""
    mlirAsmStateCreateForOperation(op, flags)

Creates new AsmState, as with AsmState the IR should not be mutated in-between using this state. Must be freed with a call to [`mlirAsmStateDestroy`](@ref)().
"""
function mlirAsmStateCreateForOperation(op, flags)
    @ccall mlir_c.mlirAsmStateCreateForOperation(
        op::MlirOperation, flags::MlirOpPrintingFlags
    )::MlirAsmState
end

"""
    mlirAsmStateCreateForValue(value, flags)

Creates new AsmState from value. Must be freed with a call to [`mlirAsmStateDestroy`](@ref)().
"""
function mlirAsmStateCreateForValue(value, flags)
    @ccall mlir_c.mlirAsmStateCreateForValue(
        value::MlirValue, flags::MlirOpPrintingFlags
    )::MlirAsmState
end

"""
    mlirAsmStateDestroy(state)

Destroys printing flags created with mlirAsmStateCreate.
"""
function mlirAsmStateDestroy(state)
    @ccall mlir_c.mlirAsmStateDestroy(state::MlirAsmState)::Cvoid
end

"""
    mlirOpPrintingFlagsCreate()

Creates new printing flags with defaults, intended for customization. Must be freed with a call to [`mlirOpPrintingFlagsDestroy`](@ref)().
"""
function mlirOpPrintingFlagsCreate()
    @ccall mlir_c.mlirOpPrintingFlagsCreate()::MlirOpPrintingFlags
end

"""
    mlirOpPrintingFlagsDestroy(flags)

Destroys printing flags created with [`mlirOpPrintingFlagsCreate`](@ref).
"""
function mlirOpPrintingFlagsDestroy(flags)
    @ccall mlir_c.mlirOpPrintingFlagsDestroy(flags::MlirOpPrintingFlags)::Cvoid
end

"""
    mlirOpPrintingFlagsElideLargeElementsAttrs(flags, largeElementLimit)

Enables the elision of large elements attributes by printing a lexically valid but otherwise meaningless form instead of the element data. The `largeElementLimit` is used to configure what is considered to be a "large" ElementsAttr by providing an upper limit to the number of elements.
"""
function mlirOpPrintingFlagsElideLargeElementsAttrs(flags, largeElementLimit)
    @ccall mlir_c.mlirOpPrintingFlagsElideLargeElementsAttrs(
        flags::MlirOpPrintingFlags, largeElementLimit::Cptrdiff_t
    )::Cvoid
end

"""
    mlirOpPrintingFlagsElideLargeResourceString(flags, largeResourceLimit)

Enables the elision of large resources strings by omitting them from the `dialect_resources` section. The `largeResourceLimit` is used to configure what is considered to be a "large" resource by providing an upper limit to the string size.
"""
function mlirOpPrintingFlagsElideLargeResourceString(flags, largeResourceLimit)
    @ccall mlir_c.mlirOpPrintingFlagsElideLargeResourceString(
        flags::MlirOpPrintingFlags, largeResourceLimit::Cptrdiff_t
    )::Cvoid
end

"""
    mlirOpPrintingFlagsEnableDebugInfo(flags, enable, prettyForm)

Enable or disable printing of debug information (based on `enable`). If 'prettyForm' is set to true, debug information is printed in a more readable 'pretty' form. Note: The IR generated with 'prettyForm' is not parsable.
"""
function mlirOpPrintingFlagsEnableDebugInfo(flags, enable, prettyForm)
    @ccall mlir_c.mlirOpPrintingFlagsEnableDebugInfo(
        flags::MlirOpPrintingFlags, enable::Bool, prettyForm::Bool
    )::Cvoid
end

"""
    mlirOpPrintingFlagsPrintGenericOpForm(flags)

Always print operations in the generic form.
"""
function mlirOpPrintingFlagsPrintGenericOpForm(flags)
    @ccall mlir_c.mlirOpPrintingFlagsPrintGenericOpForm(flags::MlirOpPrintingFlags)::Cvoid
end

"""
    mlirOpPrintingFlagsPrintNameLocAsPrefix(flags)

Print the name and location, if NamedLoc, as a prefix to the SSA ID.
"""
function mlirOpPrintingFlagsPrintNameLocAsPrefix(flags)
    @ccall mlir_c.mlirOpPrintingFlagsPrintNameLocAsPrefix(flags::MlirOpPrintingFlags)::Cvoid
end

"""
    mlirOpPrintingFlagsUseLocalScope(flags)

Use local scope when printing the operation. This allows for using the printer in a more localized and thread-safe setting, but may not necessarily be identical to what the IR will look like when dumping the full module.
"""
function mlirOpPrintingFlagsUseLocalScope(flags)
    @ccall mlir_c.mlirOpPrintingFlagsUseLocalScope(flags::MlirOpPrintingFlags)::Cvoid
end

"""
    mlirOpPrintingFlagsAssumeVerified(flags)

Do not verify the operation when using custom operation printers.
"""
function mlirOpPrintingFlagsAssumeVerified(flags)
    @ccall mlir_c.mlirOpPrintingFlagsAssumeVerified(flags::MlirOpPrintingFlags)::Cvoid
end

"""
    mlirOpPrintingFlagsSkipRegions(flags)

Skip printing regions.
"""
function mlirOpPrintingFlagsSkipRegions(flags)
    @ccall mlir_c.mlirOpPrintingFlagsSkipRegions(flags::MlirOpPrintingFlags)::Cvoid
end

"""
    mlirBytecodeWriterConfigCreate()

Creates new printing flags with defaults, intended for customization. Must be freed with a call to [`mlirBytecodeWriterConfigDestroy`](@ref)().
"""
function mlirBytecodeWriterConfigCreate()
    @ccall mlir_c.mlirBytecodeWriterConfigCreate()::MlirBytecodeWriterConfig
end

"""
    mlirBytecodeWriterConfigDestroy(config)

Destroys printing flags created with [`mlirBytecodeWriterConfigCreate`](@ref).
"""
function mlirBytecodeWriterConfigDestroy(config)
    @ccall mlir_c.mlirBytecodeWriterConfigDestroy(config::MlirBytecodeWriterConfig)::Cvoid
end

"""
    mlirBytecodeWriterConfigDesiredEmitVersion(flags, version)

Sets the version to emit in the writer config.
"""
function mlirBytecodeWriterConfigDesiredEmitVersion(flags, version)
    @ccall mlir_c.mlirBytecodeWriterConfigDesiredEmitVersion(
        flags::MlirBytecodeWriterConfig, version::Int64
    )::Cvoid
end

"""
    mlirOperationCreate(state)

Creates an operation and transfers ownership to the caller. Note that caller owned child objects are transferred in this call and must not be further used. Particularly, this applies to any regions added to the state (the implementation may invalidate any such pointers).

This call can fail under the following conditions, in which case, it will return a null operation and emit diagnostics: - Result type inference is enabled and cannot be performed.
"""
function mlirOperationCreate(state)
    @ccall mlir_c.mlirOperationCreate(state::Ptr{MlirOperationState})::MlirOperation
end

"""
    mlirOperationCreateParse(context, sourceStr, sourceName)

Parses an operation, giving ownership to the caller. If parsing fails a null operation will be returned, and an error diagnostic emitted.

`sourceStr` may be either the text assembly format, or binary bytecode format. `sourceName` is used as the file name of the source; any IR without locations will get a `FileLineColLoc` location with `sourceName` as the file name.
"""
function mlirOperationCreateParse(context, sourceStr, sourceName)
    @ccall mlir_c.mlirOperationCreateParse(
        context::MlirContext, sourceStr::MlirStringRef, sourceName::MlirStringRef
    )::MlirOperation
end

"""
    mlirOperationClone(op)

Creates a deep copy of an operation. The operation is not inserted and ownership is transferred to the caller.
"""
function mlirOperationClone(op)
    @ccall mlir_c.mlirOperationClone(op::MlirOperation)::MlirOperation
end

"""
    mlirOperationDestroy(op)

Takes an operation owned by the caller and destroys it.
"""
function mlirOperationDestroy(op)
    @ccall mlir_c.mlirOperationDestroy(op::MlirOperation)::Cvoid
end

"""
    mlirOperationRemoveFromParent(op)

Removes the given operation from its parent block. The operation is not destroyed. The ownership of the operation is transferred to the caller.
"""
function mlirOperationRemoveFromParent(op)
    @ccall mlir_c.mlirOperationRemoveFromParent(op::MlirOperation)::Cvoid
end

"""
    mlirOperationIsNull(op)

Checks whether the underlying operation is null.
"""
function mlirOperationIsNull(op)
    @ccall mlir_c.mlirOperationIsNull(op::MlirOperation)::Bool
end

"""
    mlirOperationEqual(op, other)

Checks whether two operation handles point to the same operation. This does not perform deep comparison.
"""
function mlirOperationEqual(op, other)
    @ccall mlir_c.mlirOperationEqual(op::MlirOperation, other::MlirOperation)::Bool
end

"""
    mlirOperationHashValue(op)

Compute a hash for the given operation.
"""
function mlirOperationHashValue(op)
    @ccall mlir_c.mlirOperationHashValue(op::MlirOperation)::Csize_t
end

"""
    mlirOperationGetContext(op)

Gets the context this operation is associated with
"""
function mlirOperationGetContext(op)
    @ccall mlir_c.mlirOperationGetContext(op::MlirOperation)::MlirContext
end

"""
    mlirOperationGetLocation(op)

Gets the location of the operation.
"""
function mlirOperationGetLocation(op)
    @ccall mlir_c.mlirOperationGetLocation(op::MlirOperation)::MlirLocation
end

"""
    mlirOperationSetLocation(op, loc)

Sets the location of the operation.
"""
function mlirOperationSetLocation(op, loc)
    @ccall mlir_c.mlirOperationSetLocation(op::MlirOperation, loc::MlirLocation)::Cvoid
end

"""
    mlirOperationGetTypeID(op)

Gets the type id of the operation. Returns null if the operation does not have a registered operation description.
"""
function mlirOperationGetTypeID(op)
    @ccall mlir_c.mlirOperationGetTypeID(op::MlirOperation)::MlirTypeID
end

"""
    mlirOperationGetName(op)

Gets the name of the operation as an identifier.
"""
function mlirOperationGetName(op)
    @ccall mlir_c.mlirOperationGetName(op::MlirOperation)::MlirIdentifier
end

"""
    mlirOperationGetBlock(op)

Gets the block that owns this operation, returning null if the operation is not owned.
"""
function mlirOperationGetBlock(op)
    @ccall mlir_c.mlirOperationGetBlock(op::MlirOperation)::MlirBlock
end

"""
    mlirOperationGetParentOperation(op)

Gets the operation that owns this operation, returning null if the operation is not owned.
"""
function mlirOperationGetParentOperation(op)
    @ccall mlir_c.mlirOperationGetParentOperation(op::MlirOperation)::MlirOperation
end

"""
    mlirOperationGetNumRegions(op)

Returns the number of regions attached to the given operation.
"""
function mlirOperationGetNumRegions(op)
    @ccall mlir_c.mlirOperationGetNumRegions(op::MlirOperation)::Cptrdiff_t
end

"""
    mlirOperationGetRegion(op, pos)

Returns `pos`-th region attached to the operation.
"""
function mlirOperationGetRegion(op, pos)
    @ccall mlir_c.mlirOperationGetRegion(op::MlirOperation, pos::Cptrdiff_t)::MlirRegion
end

"""
    mlirOperationGetNextInBlock(op)

Returns an operation immediately following the given operation it its enclosing block.
"""
function mlirOperationGetNextInBlock(op)
    @ccall mlir_c.mlirOperationGetNextInBlock(op::MlirOperation)::MlirOperation
end

"""
    mlirOperationGetNumOperands(op)

Returns the number of operands of the operation.
"""
function mlirOperationGetNumOperands(op)
    @ccall mlir_c.mlirOperationGetNumOperands(op::MlirOperation)::Cptrdiff_t
end

"""
    mlirOperationGetOperand(op, pos)

Returns `pos`-th operand of the operation.
"""
function mlirOperationGetOperand(op, pos)
    @ccall mlir_c.mlirOperationGetOperand(op::MlirOperation, pos::Cptrdiff_t)::MlirValue
end

"""
    mlirOperationSetOperand(op, pos, newValue)

Sets the `pos`-th operand of the operation.
"""
function mlirOperationSetOperand(op, pos, newValue)
    @ccall mlir_c.mlirOperationSetOperand(
        op::MlirOperation, pos::Cptrdiff_t, newValue::MlirValue
    )::Cvoid
end

"""
    mlirOperationSetOperands(op, nOperands, operands)

Replaces the operands of the operation.
"""
function mlirOperationSetOperands(op, nOperands, operands)
    @ccall mlir_c.mlirOperationSetOperands(
        op::MlirOperation, nOperands::Cptrdiff_t, operands::Ptr{MlirValue}
    )::Cvoid
end

"""
    mlirOperationGetNumResults(op)

Returns the number of results of the operation.
"""
function mlirOperationGetNumResults(op)
    @ccall mlir_c.mlirOperationGetNumResults(op::MlirOperation)::Cptrdiff_t
end

"""
    mlirOperationGetResult(op, pos)

Returns `pos`-th result of the operation.
"""
function mlirOperationGetResult(op, pos)
    @ccall mlir_c.mlirOperationGetResult(op::MlirOperation, pos::Cptrdiff_t)::MlirValue
end

"""
    mlirOperationGetNumSuccessors(op)

Returns the number of successor blocks of the operation.
"""
function mlirOperationGetNumSuccessors(op)
    @ccall mlir_c.mlirOperationGetNumSuccessors(op::MlirOperation)::Cptrdiff_t
end

"""
    mlirOperationGetSuccessor(op, pos)

Returns `pos`-th successor of the operation.
"""
function mlirOperationGetSuccessor(op, pos)
    @ccall mlir_c.mlirOperationGetSuccessor(op::MlirOperation, pos::Cptrdiff_t)::MlirBlock
end

"""
    mlirOperationSetSuccessor(op, pos, block)

Set `pos`-th successor of the operation.
"""
function mlirOperationSetSuccessor(op, pos, block)
    @ccall mlir_c.mlirOperationSetSuccessor(
        op::MlirOperation, pos::Cptrdiff_t, block::MlirBlock
    )::Cvoid
end

"""
    mlirOperationHasInherentAttributeByName(op, name)

Returns true if this operation defines an inherent attribute with this name. Note: the attribute can be optional, so [`mlirOperationGetInherentAttributeByName`](@ref) can still return a null attribute.
"""
function mlirOperationHasInherentAttributeByName(op, name)
    @ccall mlir_c.mlirOperationHasInherentAttributeByName(
        op::MlirOperation, name::MlirStringRef
    )::Bool
end

"""
    mlirOperationGetInherentAttributeByName(op, name)

Returns an inherent attribute attached to the operation given its name.
"""
function mlirOperationGetInherentAttributeByName(op, name)
    @ccall mlir_c.mlirOperationGetInherentAttributeByName(
        op::MlirOperation, name::MlirStringRef
    )::MlirAttribute
end

"""
    mlirOperationSetInherentAttributeByName(op, name, attr)

Sets an inherent attribute by name, replacing the existing if it exists. This has no effect if "name" does not match an inherent attribute.
"""
function mlirOperationSetInherentAttributeByName(op, name, attr)
    @ccall mlir_c.mlirOperationSetInherentAttributeByName(
        op::MlirOperation, name::MlirStringRef, attr::MlirAttribute
    )::Cvoid
end

"""
    mlirOperationGetNumDiscardableAttributes(op)

Returns the number of discardable attributes attached to the operation.
"""
function mlirOperationGetNumDiscardableAttributes(op)
    @ccall mlir_c.mlirOperationGetNumDiscardableAttributes(op::MlirOperation)::Cptrdiff_t
end

"""
    mlirOperationGetDiscardableAttribute(op, pos)

Return `pos`-th discardable attribute of the operation.
"""
function mlirOperationGetDiscardableAttribute(op, pos)
    @ccall mlir_c.mlirOperationGetDiscardableAttribute(
        op::MlirOperation, pos::Cptrdiff_t
    )::MlirNamedAttribute
end

"""
    mlirOperationGetDiscardableAttributeByName(op, name)

Returns a discardable attribute attached to the operation given its name.
"""
function mlirOperationGetDiscardableAttributeByName(op, name)
    @ccall mlir_c.mlirOperationGetDiscardableAttributeByName(
        op::MlirOperation, name::MlirStringRef
    )::MlirAttribute
end

"""
    mlirOperationSetDiscardableAttributeByName(op, name, attr)

Sets a discardable attribute by name, replacing the existing if it exists or adding a new one otherwise. The new `attr` Attribute is not allowed to be null, use [`mlirOperationRemoveDiscardableAttributeByName`](@ref) to remove an Attribute instead.
"""
function mlirOperationSetDiscardableAttributeByName(op, name, attr)
    @ccall mlir_c.mlirOperationSetDiscardableAttributeByName(
        op::MlirOperation, name::MlirStringRef, attr::MlirAttribute
    )::Cvoid
end

"""
    mlirOperationRemoveDiscardableAttributeByName(op, name)

Removes a discardable attribute by name. Returns false if the attribute was not found and true if removed.
"""
function mlirOperationRemoveDiscardableAttributeByName(op, name)
    @ccall mlir_c.mlirOperationRemoveDiscardableAttributeByName(
        op::MlirOperation, name::MlirStringRef
    )::Bool
end

"""
    mlirOperationGetNumAttributes(op)

Returns the number of attributes attached to the operation. Deprecated, please use `mlirOperationGetNumInherentAttributes` or [`mlirOperationGetNumDiscardableAttributes`](@ref).
"""
function mlirOperationGetNumAttributes(op)
    @ccall mlir_c.mlirOperationGetNumAttributes(op::MlirOperation)::Cptrdiff_t
end

"""
    mlirOperationGetAttribute(op, pos)

Return `pos`-th attribute of the operation. Deprecated, please use `mlirOperationGetInherentAttribute` or [`mlirOperationGetDiscardableAttribute`](@ref).
"""
function mlirOperationGetAttribute(op, pos)
    @ccall mlir_c.mlirOperationGetAttribute(
        op::MlirOperation, pos::Cptrdiff_t
    )::MlirNamedAttribute
end

"""
    mlirOperationGetAttributeByName(op, name)

Returns an attribute attached to the operation given its name. Deprecated, please use [`mlirOperationGetInherentAttributeByName`](@ref) or [`mlirOperationGetDiscardableAttributeByName`](@ref).
"""
function mlirOperationGetAttributeByName(op, name)
    @ccall mlir_c.mlirOperationGetAttributeByName(
        op::MlirOperation, name::MlirStringRef
    )::MlirAttribute
end

"""
    mlirOperationSetAttributeByName(op, name, attr)

Sets an attribute by name, replacing the existing if it exists or adding a new one otherwise. Deprecated, please use [`mlirOperationSetInherentAttributeByName`](@ref) or [`mlirOperationSetDiscardableAttributeByName`](@ref).
"""
function mlirOperationSetAttributeByName(op, name, attr)
    @ccall mlir_c.mlirOperationSetAttributeByName(
        op::MlirOperation, name::MlirStringRef, attr::MlirAttribute
    )::Cvoid
end

"""
    mlirOperationRemoveAttributeByName(op, name)

Removes an attribute by name. Returns false if the attribute was not found and true if removed. Deprecated, please use `mlirOperationRemoveInherentAttributeByName` or [`mlirOperationRemoveDiscardableAttributeByName`](@ref).
"""
function mlirOperationRemoveAttributeByName(op, name)
    @ccall mlir_c.mlirOperationRemoveAttributeByName(
        op::MlirOperation, name::MlirStringRef
    )::Bool
end

"""
    mlirOperationPrint(op, callback, userData)

Prints an operation by sending chunks of the string representation and forwarding `userData to `callback`. Note that the callback may be called several times with consecutive chunks of the string.
"""
function mlirOperationPrint(op, callback, userData)
    @ccall mlir_c.mlirOperationPrint(
        op::MlirOperation, callback::MlirStringCallback, userData::Ptr{Cvoid}
    )::Cvoid
end

"""
    mlirOperationPrintWithFlags(op, flags, callback, userData)

Same as [`mlirOperationPrint`](@ref) but accepts flags controlling the printing behavior.
"""
function mlirOperationPrintWithFlags(op, flags, callback, userData)
    @ccall mlir_c.mlirOperationPrintWithFlags(
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
    @ccall mlir_c.mlirOperationPrintWithState(
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
    @ccall mlir_c.mlirOperationWriteBytecode(
        op::MlirOperation, callback::MlirStringCallback, userData::Ptr{Cvoid}
    )::Cvoid
end

"""
    mlirOperationWriteBytecodeWithConfig(op, config, callback, userData)

Same as [`mlirOperationWriteBytecode`](@ref) but with writer config and returns failure only if desired bytecode could not be honored.
"""
function mlirOperationWriteBytecodeWithConfig(op, config, callback, userData)
    @ccall mlir_c.mlirOperationWriteBytecodeWithConfig(
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
    @ccall mlir_c.mlirOperationDump(op::MlirOperation)::Cvoid
end

"""
    mlirOperationVerify(op)

Verify the operation and return true if it passes, false if it fails.
"""
function mlirOperationVerify(op)
    @ccall mlir_c.mlirOperationVerify(op::MlirOperation)::Bool
end

"""
    mlirOperationMoveAfter(op, other)

Moves the given operation immediately after the other operation in its parent block. The given operation may be owned by the caller or by its current block. The other operation must belong to a block. In any case, the ownership is transferred to the block of the other operation.
"""
function mlirOperationMoveAfter(op, other)
    @ccall mlir_c.mlirOperationMoveAfter(op::MlirOperation, other::MlirOperation)::Cvoid
end

"""
    mlirOperationMoveBefore(op, other)

Moves the given operation immediately before the other operation in its parent block. The given operation may be owner by the caller or by its current block. The other operation must belong to a block. In any case, the ownership is transferred to the block of the other operation.
"""
function mlirOperationMoveBefore(op, other)
    @ccall mlir_c.mlirOperationMoveBefore(op::MlirOperation, other::MlirOperation)::Cvoid
end

"""
    mlirOperationIsBeforeInBlock(op, other)

Given an operation 'other' that is within the same parent block, return whether the current operation is before 'other' in the operation list of the parent block. Note: This function has an average complexity of O(1), but worst case may take O(N) where N is the number of operations within the parent block.
"""
function mlirOperationIsBeforeInBlock(op, other)
    @ccall mlir_c.mlirOperationIsBeforeInBlock(
        op::MlirOperation, other::MlirOperation
    )::Bool
end

"""
    MlirWalkResult

Operation walk result.
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
Operation walker type. The handler is passed an (opaque) reference to an operation and a pointer to a `userData`.
"""
const MlirOperationWalkCallback = Ptr{Cvoid}

"""
    mlirOperationWalk(op, callback, userData, walkOrder)

Walks operation `op` in `walkOrder` and calls `callback` on that operation. `*userData` is passed to the callback as well and can be used to tunnel some context or other data into the callback.
"""
function mlirOperationWalk(op, callback, userData, walkOrder)
    @ccall mlir_c.mlirOperationWalk(
        op::MlirOperation,
        callback::MlirOperationWalkCallback,
        userData::Ptr{Cvoid},
        walkOrder::MlirWalkOrder,
    )::Cvoid
end

"""
    mlirRegionCreate()

Creates a new empty region and transfers ownership to the caller.
"""
function mlirRegionCreate()
    @ccall mlir_c.mlirRegionCreate()::MlirRegion
end

"""
    mlirRegionDestroy(region)

Takes a region owned by the caller and destroys it.
"""
function mlirRegionDestroy(region)
    @ccall mlir_c.mlirRegionDestroy(region::MlirRegion)::Cvoid
end

"""
    mlirRegionIsNull(region)

Checks whether a region is null.
"""
function mlirRegionIsNull(region)
    @ccall mlir_c.mlirRegionIsNull(region::MlirRegion)::Bool
end

"""
    mlirRegionEqual(region, other)

Checks whether two region handles point to the same region. This does not perform deep comparison.
"""
function mlirRegionEqual(region, other)
    @ccall mlir_c.mlirRegionEqual(region::MlirRegion, other::MlirRegion)::Bool
end

"""
    mlirRegionGetFirstBlock(region)

Gets the first block in the region.
"""
function mlirRegionGetFirstBlock(region)
    @ccall mlir_c.mlirRegionGetFirstBlock(region::MlirRegion)::MlirBlock
end

"""
    mlirRegionAppendOwnedBlock(region, block)

Takes a block owned by the caller and appends it to the given region.
"""
function mlirRegionAppendOwnedBlock(region, block)
    @ccall mlir_c.mlirRegionAppendOwnedBlock(region::MlirRegion, block::MlirBlock)::Cvoid
end

"""
    mlirRegionInsertOwnedBlock(region, pos, block)

Takes a block owned by the caller and inserts it at `pos` to the given region. This is an expensive operation that linearly scans the region, prefer insertAfter/Before instead.
"""
function mlirRegionInsertOwnedBlock(region, pos, block)
    @ccall mlir_c.mlirRegionInsertOwnedBlock(
        region::MlirRegion, pos::Cptrdiff_t, block::MlirBlock
    )::Cvoid
end

"""
    mlirRegionInsertOwnedBlockAfter(region, reference, block)

Takes a block owned by the caller and inserts it after the (non-owned) reference block in the given region. The reference block must belong to the region. If the reference block is null, prepends the block to the region.
"""
function mlirRegionInsertOwnedBlockAfter(region, reference, block)
    @ccall mlir_c.mlirRegionInsertOwnedBlockAfter(
        region::MlirRegion, reference::MlirBlock, block::MlirBlock
    )::Cvoid
end

"""
    mlirRegionInsertOwnedBlockBefore(region, reference, block)

Takes a block owned by the caller and inserts it before the (non-owned) reference block in the given region. The reference block must belong to the region. If the reference block is null, appends the block to the region.
"""
function mlirRegionInsertOwnedBlockBefore(region, reference, block)
    @ccall mlir_c.mlirRegionInsertOwnedBlockBefore(
        region::MlirRegion, reference::MlirBlock, block::MlirBlock
    )::Cvoid
end

"""
    mlirOperationGetFirstRegion(op)

Returns first region attached to the operation.
"""
function mlirOperationGetFirstRegion(op)
    @ccall mlir_c.mlirOperationGetFirstRegion(op::MlirOperation)::MlirRegion
end

"""
    mlirRegionGetNextInOperation(region)

Returns the region immediately following the given region in its parent operation.
"""
function mlirRegionGetNextInOperation(region)
    @ccall mlir_c.mlirRegionGetNextInOperation(region::MlirRegion)::MlirRegion
end

"""
    mlirRegionTakeBody(target, source)

Moves the entire content of the source region to the target region.
"""
function mlirRegionTakeBody(target, source)
    @ccall mlir_c.mlirRegionTakeBody(target::MlirRegion, source::MlirRegion)::Cvoid
end

"""
    mlirBlockCreate(nArgs, args, locs)

Creates a new empty block with the given argument types and transfers ownership to the caller.
"""
function mlirBlockCreate(nArgs, args, locs)
    @ccall mlir_c.mlirBlockCreate(
        nArgs::Cptrdiff_t, args::Ptr{MlirType}, locs::Ptr{MlirLocation}
    )::MlirBlock
end

"""
    mlirBlockDestroy(block)

Takes a block owned by the caller and destroys it.
"""
function mlirBlockDestroy(block)
    @ccall mlir_c.mlirBlockDestroy(block::MlirBlock)::Cvoid
end

"""
    mlirBlockDetach(block)

Detach a block from the owning region and assume ownership.
"""
function mlirBlockDetach(block)
    @ccall mlir_c.mlirBlockDetach(block::MlirBlock)::Cvoid
end

"""
    mlirBlockIsNull(block)

Checks whether a block is null.
"""
function mlirBlockIsNull(block)
    @ccall mlir_c.mlirBlockIsNull(block::MlirBlock)::Bool
end

"""
    mlirBlockEqual(block, other)

Checks whether two blocks handles point to the same block. This does not perform deep comparison.
"""
function mlirBlockEqual(block, other)
    @ccall mlir_c.mlirBlockEqual(block::MlirBlock, other::MlirBlock)::Bool
end

"""
    mlirBlockGetParentOperation(arg1)

Returns the closest surrounding operation that contains this block.
"""
function mlirBlockGetParentOperation(arg1)
    @ccall mlir_c.mlirBlockGetParentOperation(arg1::MlirBlock)::MlirOperation
end

"""
    mlirBlockGetParentRegion(block)

Returns the region that contains this block.
"""
function mlirBlockGetParentRegion(block)
    @ccall mlir_c.mlirBlockGetParentRegion(block::MlirBlock)::MlirRegion
end

"""
    mlirBlockGetNextInRegion(block)

Returns the block immediately following the given block in its parent region.
"""
function mlirBlockGetNextInRegion(block)
    @ccall mlir_c.mlirBlockGetNextInRegion(block::MlirBlock)::MlirBlock
end

"""
    mlirBlockGetFirstOperation(block)

Returns the first operation in the block.
"""
function mlirBlockGetFirstOperation(block)
    @ccall mlir_c.mlirBlockGetFirstOperation(block::MlirBlock)::MlirOperation
end

"""
    mlirBlockGetTerminator(block)

Returns the terminator operation in the block or null if no terminator.
"""
function mlirBlockGetTerminator(block)
    @ccall mlir_c.mlirBlockGetTerminator(block::MlirBlock)::MlirOperation
end

"""
    mlirBlockAppendOwnedOperation(block, operation)

Takes an operation owned by the caller and appends it to the block.
"""
function mlirBlockAppendOwnedOperation(block, operation)
    @ccall mlir_c.mlirBlockAppendOwnedOperation(
        block::MlirBlock, operation::MlirOperation
    )::Cvoid
end

"""
    mlirBlockInsertOwnedOperation(block, pos, operation)

Takes an operation owned by the caller and inserts it as `pos` to the block. This is an expensive operation that scans the block linearly, prefer insertBefore/After instead.
"""
function mlirBlockInsertOwnedOperation(block, pos, operation)
    @ccall mlir_c.mlirBlockInsertOwnedOperation(
        block::MlirBlock, pos::Cptrdiff_t, operation::MlirOperation
    )::Cvoid
end

"""
    mlirBlockInsertOwnedOperationAfter(block, reference, operation)

Takes an operation owned by the caller and inserts it after the (non-owned) reference operation in the given block. If the reference is null, prepends the operation. Otherwise, the reference must belong to the block.
"""
function mlirBlockInsertOwnedOperationAfter(block, reference, operation)
    @ccall mlir_c.mlirBlockInsertOwnedOperationAfter(
        block::MlirBlock, reference::MlirOperation, operation::MlirOperation
    )::Cvoid
end

"""
    mlirBlockInsertOwnedOperationBefore(block, reference, operation)

Takes an operation owned by the caller and inserts it before the (non-owned) reference operation in the given block. If the reference is null, appends the operation. Otherwise, the reference must belong to the block.
"""
function mlirBlockInsertOwnedOperationBefore(block, reference, operation)
    @ccall mlir_c.mlirBlockInsertOwnedOperationBefore(
        block::MlirBlock, reference::MlirOperation, operation::MlirOperation
    )::Cvoid
end

"""
    mlirBlockGetNumArguments(block)

Returns the number of arguments of the block.
"""
function mlirBlockGetNumArguments(block)
    @ccall mlir_c.mlirBlockGetNumArguments(block::MlirBlock)::Cptrdiff_t
end

"""
    mlirBlockAddArgument(block, type, loc)

Appends an argument of the specified type to the block. Returns the newly added argument.
"""
function mlirBlockAddArgument(block, type, loc)
    @ccall mlir_c.mlirBlockAddArgument(
        block::MlirBlock, type::MlirType, loc::MlirLocation
    )::MlirValue
end

"""
    mlirBlockEraseArgument(block, index)

Erase the argument at 'index' and remove it from the argument list.
"""
function mlirBlockEraseArgument(block, index)
    @ccall mlir_c.mlirBlockEraseArgument(block::MlirBlock, index::Cuint)::Cvoid
end

"""
    mlirBlockInsertArgument(block, pos, type, loc)

Inserts an argument of the specified type at a specified index to the block. Returns the newly added argument.
"""
function mlirBlockInsertArgument(block, pos, type, loc)
    @ccall mlir_c.mlirBlockInsertArgument(
        block::MlirBlock, pos::Cptrdiff_t, type::MlirType, loc::MlirLocation
    )::MlirValue
end

"""
    mlirBlockGetArgument(block, pos)

Returns `pos`-th argument of the block.
"""
function mlirBlockGetArgument(block, pos)
    @ccall mlir_c.mlirBlockGetArgument(block::MlirBlock, pos::Cptrdiff_t)::MlirValue
end

"""
    mlirBlockPrint(block, callback, userData)

Prints a block by sending chunks of the string representation and forwarding `userData to `callback`. Note that the callback may be called several times with consecutive chunks of the string.
"""
function mlirBlockPrint(block, callback, userData)
    @ccall mlir_c.mlirBlockPrint(
        block::MlirBlock, callback::MlirStringCallback, userData::Ptr{Cvoid}
    )::Cvoid
end

"""
    mlirBlockGetNumSuccessors(block)

Returns the number of successor blocks of the block.
"""
function mlirBlockGetNumSuccessors(block)
    @ccall mlir_c.mlirBlockGetNumSuccessors(block::MlirBlock)::Cptrdiff_t
end

"""
    mlirBlockGetSuccessor(block, pos)

Returns `pos`-th successor of the block.
"""
function mlirBlockGetSuccessor(block, pos)
    @ccall mlir_c.mlirBlockGetSuccessor(block::MlirBlock, pos::Cptrdiff_t)::MlirBlock
end

"""
    mlirBlockGetNumPredecessors(block)

Returns the number of predecessor blocks of the block.
"""
function mlirBlockGetNumPredecessors(block)
    @ccall mlir_c.mlirBlockGetNumPredecessors(block::MlirBlock)::Cptrdiff_t
end

"""
    mlirBlockGetPredecessor(block, pos)

Returns `pos`-th predecessor of the block.

WARNING: This getter is more expensive than the others here because the impl actually iterates the use-def chain (of block operands) anew for each indexed access.
"""
function mlirBlockGetPredecessor(block, pos)
    @ccall mlir_c.mlirBlockGetPredecessor(block::MlirBlock, pos::Cptrdiff_t)::MlirBlock
end

"""
    mlirValueIsNull(value)

Returns whether the value is null.
"""
function mlirValueIsNull(value)
    @ccall mlir_c.mlirValueIsNull(value::MlirValue)::Bool
end

"""
    mlirValueEqual(value1, value2)

Returns 1 if two values are equal, 0 otherwise.
"""
function mlirValueEqual(value1, value2)
    @ccall mlir_c.mlirValueEqual(value1::MlirValue, value2::MlirValue)::Bool
end

"""
    mlirValueIsABlockArgument(value)

Returns 1 if the value is a block argument, 0 otherwise.
"""
function mlirValueIsABlockArgument(value)
    @ccall mlir_c.mlirValueIsABlockArgument(value::MlirValue)::Bool
end

"""
    mlirValueIsAOpResult(value)

Returns 1 if the value is an operation result, 0 otherwise.
"""
function mlirValueIsAOpResult(value)
    @ccall mlir_c.mlirValueIsAOpResult(value::MlirValue)::Bool
end

"""
    mlirBlockArgumentGetOwner(value)

Returns the block in which this value is defined as an argument. Asserts if the value is not a block argument.
"""
function mlirBlockArgumentGetOwner(value)
    @ccall mlir_c.mlirBlockArgumentGetOwner(value::MlirValue)::MlirBlock
end

"""
    mlirBlockArgumentGetArgNumber(value)

Returns the position of the value in the argument list of its block.
"""
function mlirBlockArgumentGetArgNumber(value)
    @ccall mlir_c.mlirBlockArgumentGetArgNumber(value::MlirValue)::Cptrdiff_t
end

"""
    mlirBlockArgumentSetType(value, type)

Sets the type of the block argument to the given type.
"""
function mlirBlockArgumentSetType(value, type)
    @ccall mlir_c.mlirBlockArgumentSetType(value::MlirValue, type::MlirType)::Cvoid
end

"""
    mlirBlockArgumentSetLocation(value, loc)

Sets the location of the block argument to the given location.
"""
function mlirBlockArgumentSetLocation(value, loc)
    @ccall mlir_c.mlirBlockArgumentSetLocation(value::MlirValue, loc::MlirLocation)::Cvoid
end

"""
    mlirOpResultGetOwner(value)

Returns an operation that produced this value as its result. Asserts if the value is not an op result.
"""
function mlirOpResultGetOwner(value)
    @ccall mlir_c.mlirOpResultGetOwner(value::MlirValue)::MlirOperation
end

"""
    mlirOpResultGetResultNumber(value)

Returns the position of the value in the list of results of the operation that produced it.
"""
function mlirOpResultGetResultNumber(value)
    @ccall mlir_c.mlirOpResultGetResultNumber(value::MlirValue)::Cptrdiff_t
end

"""
    mlirValueGetType(value)

Returns the type of the value.
"""
function mlirValueGetType(value)
    @ccall mlir_c.mlirValueGetType(value::MlirValue)::MlirType
end

"""
    mlirValueSetType(value, type)

Set the type of the value.
"""
function mlirValueSetType(value, type)
    @ccall mlir_c.mlirValueSetType(value::MlirValue, type::MlirType)::Cvoid
end

"""
    mlirValueDump(value)

Prints the value to the standard error stream.
"""
function mlirValueDump(value)
    @ccall mlir_c.mlirValueDump(value::MlirValue)::Cvoid
end

"""
    mlirValuePrint(value, callback, userData)

Prints a value by sending chunks of the string representation and forwarding `userData to `callback`. Note that the callback may be called several times with consecutive chunks of the string.
"""
function mlirValuePrint(value, callback, userData)
    @ccall mlir_c.mlirValuePrint(
        value::MlirValue, callback::MlirStringCallback, userData::Ptr{Cvoid}
    )::Cvoid
end

"""
    mlirValuePrintAsOperand(value, state, callback, userData)

Prints a value as an operand (i.e., the ValueID).
"""
function mlirValuePrintAsOperand(value, state, callback, userData)
    @ccall mlir_c.mlirValuePrintAsOperand(
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
    @ccall mlir_c.mlirValueGetFirstUse(value::MlirValue)::MlirOpOperand
end

"""
    mlirValueReplaceAllUsesOfWith(of, with)

Replace all uses of 'of' value with the 'with' value, updating anything in the IR that uses 'of' to use the other value instead. When this returns there are zero uses of 'of'.
"""
function mlirValueReplaceAllUsesOfWith(of, with)
    @ccall mlir_c.mlirValueReplaceAllUsesOfWith(of::MlirValue, with::MlirValue)::Cvoid
end

"""
    mlirValueReplaceAllUsesExcept(of, with, numExceptions, exceptions)

Replace all uses of 'of' value with 'with' value, updating anything in the IR that uses 'of' to use 'with' instead, except if the user is listed in 'exceptions'. The 'exceptions' parameter is an array of [`MlirOperation`](@ref) pointers with a length of 'numExceptions'.
"""
function mlirValueReplaceAllUsesExcept(of, with, numExceptions, exceptions)
    @ccall mlir_c.mlirValueReplaceAllUsesExcept(
        of::MlirValue,
        with::MlirValue,
        numExceptions::Cptrdiff_t,
        exceptions::Ptr{MlirOperation},
    )::Cvoid
end

"""
    mlirValueGetLocation(v)

Gets the location of the value.
"""
function mlirValueGetLocation(v)
    @ccall mlir_c.mlirValueGetLocation(v::MlirValue)::MlirLocation
end

"""
    mlirValueGetContext(v)

Gets the context that a value was created with.
"""
function mlirValueGetContext(v)
    @ccall mlir_c.mlirValueGetContext(v::MlirValue)::MlirContext
end

"""
    mlirOpOperandIsNull(opOperand)

Returns whether the op operand is null.
"""
function mlirOpOperandIsNull(opOperand)
    @ccall mlir_c.mlirOpOperandIsNull(opOperand::MlirOpOperand)::Bool
end

"""
    mlirOpOperandGetValue(opOperand)

Returns the value of an op operand.
"""
function mlirOpOperandGetValue(opOperand)
    @ccall mlir_c.mlirOpOperandGetValue(opOperand::MlirOpOperand)::MlirValue
end

"""
    mlirOpOperandGetOwner(opOperand)

Returns the owner operation of an op operand.
"""
function mlirOpOperandGetOwner(opOperand)
    @ccall mlir_c.mlirOpOperandGetOwner(opOperand::MlirOpOperand)::MlirOperation
end

"""
    mlirOpOperandGetOperandNumber(opOperand)

Returns the operand number of an op operand.
"""
function mlirOpOperandGetOperandNumber(opOperand)
    @ccall mlir_c.mlirOpOperandGetOperandNumber(opOperand::MlirOpOperand)::Cuint
end

"""
    mlirOpOperandGetNextUse(opOperand)

Returns an op operand representing the next use of the value, or a null op operand if there is no next use.
"""
function mlirOpOperandGetNextUse(opOperand)
    @ccall mlir_c.mlirOpOperandGetNextUse(opOperand::MlirOpOperand)::MlirOpOperand
end

"""
    mlirTypeParseGet(context, type)

Parses a type. The type is owned by the context.
"""
function mlirTypeParseGet(context, type)
    @ccall mlir_c.mlirTypeParseGet(context::MlirContext, type::MlirStringRef)::MlirType
end

"""
    mlirTypeGetContext(type)

Gets the context that a type was created with.
"""
function mlirTypeGetContext(type)
    @ccall mlir_c.mlirTypeGetContext(type::MlirType)::MlirContext
end

"""
    mlirTypeGetTypeID(type)

Gets the type ID of the type.
"""
function mlirTypeGetTypeID(type)
    @ccall mlir_c.mlirTypeGetTypeID(type::MlirType)::MlirTypeID
end

"""
    mlirTypeGetDialect(type)

Gets the dialect a type belongs to.
"""
function mlirTypeGetDialect(type)
    @ccall mlir_c.mlirTypeGetDialect(type::MlirType)::MlirDialect
end

"""
    mlirTypeIsNull(type)

Checks whether a type is null.
"""
function mlirTypeIsNull(type)
    @ccall mlir_c.mlirTypeIsNull(type::MlirType)::Bool
end

"""
    mlirTypeEqual(t1, t2)

Checks if two types are equal.
"""
function mlirTypeEqual(t1, t2)
    @ccall mlir_c.mlirTypeEqual(t1::MlirType, t2::MlirType)::Bool
end

"""
    mlirTypePrint(type, callback, userData)

Prints a location by sending chunks of the string representation and forwarding `userData to `callback`. Note that the callback may be called several times with consecutive chunks of the string.
"""
function mlirTypePrint(type, callback, userData)
    @ccall mlir_c.mlirTypePrint(
        type::MlirType, callback::MlirStringCallback, userData::Ptr{Cvoid}
    )::Cvoid
end

"""
    mlirTypeDump(type)

Prints the type to the standard error stream.
"""
function mlirTypeDump(type)
    @ccall mlir_c.mlirTypeDump(type::MlirType)::Cvoid
end

"""
    mlirAttributeParseGet(context, attr)

Parses an attribute. The attribute is owned by the context.
"""
function mlirAttributeParseGet(context, attr)
    @ccall mlir_c.mlirAttributeParseGet(
        context::MlirContext, attr::MlirStringRef
    )::MlirAttribute
end

"""
    mlirAttributeGetContext(attribute)

Gets the context that an attribute was created with.
"""
function mlirAttributeGetContext(attribute)
    @ccall mlir_c.mlirAttributeGetContext(attribute::MlirAttribute)::MlirContext
end

"""
    mlirAttributeGetType(attribute)

Gets the type of this attribute.
"""
function mlirAttributeGetType(attribute)
    @ccall mlir_c.mlirAttributeGetType(attribute::MlirAttribute)::MlirType
end

"""
    mlirAttributeGetTypeID(attribute)

Gets the type id of the attribute.
"""
function mlirAttributeGetTypeID(attribute)
    @ccall mlir_c.mlirAttributeGetTypeID(attribute::MlirAttribute)::MlirTypeID
end

"""
    mlirAttributeGetDialect(attribute)

Gets the dialect of the attribute.
"""
function mlirAttributeGetDialect(attribute)
    @ccall mlir_c.mlirAttributeGetDialect(attribute::MlirAttribute)::MlirDialect
end

"""
    mlirAttributeIsNull(attr)

Checks whether an attribute is null.
"""
function mlirAttributeIsNull(attr)
    @ccall mlir_c.mlirAttributeIsNull(attr::MlirAttribute)::Bool
end

"""
    mlirAttributeEqual(a1, a2)

Checks if two attributes are equal.
"""
function mlirAttributeEqual(a1, a2)
    @ccall mlir_c.mlirAttributeEqual(a1::MlirAttribute, a2::MlirAttribute)::Bool
end

"""
    mlirAttributePrint(attr, callback, userData)

Prints an attribute by sending chunks of the string representation and forwarding `userData to `callback`. Note that the callback may be called several times with consecutive chunks of the string.
"""
function mlirAttributePrint(attr, callback, userData)
    @ccall mlir_c.mlirAttributePrint(
        attr::MlirAttribute, callback::MlirStringCallback, userData::Ptr{Cvoid}
    )::Cvoid
end

"""
    mlirAttributeDump(attr)

Prints the attribute to the standard error stream.
"""
function mlirAttributeDump(attr)
    @ccall mlir_c.mlirAttributeDump(attr::MlirAttribute)::Cvoid
end

"""
    mlirNamedAttributeGet(name, attr)

Associates an attribute with the name. Takes ownership of neither.
"""
function mlirNamedAttributeGet(name, attr)
    @ccall mlir_c.mlirNamedAttributeGet(
        name::MlirIdentifier, attr::MlirAttribute
    )::MlirNamedAttribute
end

"""
    mlirIdentifierGet(context, str)

Gets an identifier with the given string value.
"""
function mlirIdentifierGet(context, str)
    @ccall mlir_c.mlirIdentifierGet(
        context::MlirContext, str::MlirStringRef
    )::MlirIdentifier
end

"""
    mlirIdentifierGetContext(arg1)

Returns the context associated with this identifier
"""
function mlirIdentifierGetContext(arg1)
    @ccall mlir_c.mlirIdentifierGetContext(arg1::MlirIdentifier)::MlirContext
end

"""
    mlirIdentifierEqual(ident, other)

Checks whether two identifiers are the same.
"""
function mlirIdentifierEqual(ident, other)
    @ccall mlir_c.mlirIdentifierEqual(ident::MlirIdentifier, other::MlirIdentifier)::Bool
end

"""
    mlirIdentifierStr(ident)

Gets the string value of the identifier.
"""
function mlirIdentifierStr(ident)
    @ccall mlir_c.mlirIdentifierStr(ident::MlirIdentifier)::MlirStringRef
end

"""
    mlirSymbolTableGetSymbolAttributeName()

Returns the name of the attribute used to store symbol names compatible with symbol tables.
"""
function mlirSymbolTableGetSymbolAttributeName()
    @ccall mlir_c.mlirSymbolTableGetSymbolAttributeName()::MlirStringRef
end

"""
    mlirSymbolTableGetVisibilityAttributeName()

Returns the name of the attribute used to store symbol visibility.
"""
function mlirSymbolTableGetVisibilityAttributeName()
    @ccall mlir_c.mlirSymbolTableGetVisibilityAttributeName()::MlirStringRef
end

"""
    mlirSymbolTableCreate(operation)

Creates a symbol table for the given operation. If the operation does not have the SymbolTable trait, returns a null symbol table.
"""
function mlirSymbolTableCreate(operation)
    @ccall mlir_c.mlirSymbolTableCreate(operation::MlirOperation)::MlirSymbolTable
end

"""
    mlirSymbolTableIsNull(symbolTable)

Returns true if the symbol table is null.
"""
function mlirSymbolTableIsNull(symbolTable)
    @ccall mlir_c.mlirSymbolTableIsNull(symbolTable::MlirSymbolTable)::Bool
end

"""
    mlirSymbolTableDestroy(symbolTable)

Destroys the symbol table created with [`mlirSymbolTableCreate`](@ref). This does not affect the operations in the table.
"""
function mlirSymbolTableDestroy(symbolTable)
    @ccall mlir_c.mlirSymbolTableDestroy(symbolTable::MlirSymbolTable)::Cvoid
end

"""
    mlirSymbolTableLookup(symbolTable, name)

Looks up a symbol with the given name in the given symbol table and returns the operation that corresponds to the symbol. If the symbol cannot be found, returns a null operation.
"""
function mlirSymbolTableLookup(symbolTable, name)
    @ccall mlir_c.mlirSymbolTableLookup(
        symbolTable::MlirSymbolTable, name::MlirStringRef
    )::MlirOperation
end

"""
    mlirSymbolTableInsert(symbolTable, operation)

Inserts the given operation into the given symbol table. The operation must have the symbol trait. If the symbol table already has a symbol with the same name, renames the symbol being inserted to ensure name uniqueness. Note that this does not move the operation itself into the block of the symbol table operation, this should be done separately. Returns the name of the symbol after insertion.
"""
function mlirSymbolTableInsert(symbolTable, operation)
    @ccall mlir_c.mlirSymbolTableInsert(
        symbolTable::MlirSymbolTable, operation::MlirOperation
    )::MlirAttribute
end

"""
    mlirSymbolTableErase(symbolTable, operation)

Removes the given operation from the symbol table and erases it.
"""
function mlirSymbolTableErase(symbolTable, operation)
    @ccall mlir_c.mlirSymbolTableErase(
        symbolTable::MlirSymbolTable, operation::MlirOperation
    )::Cvoid
end

"""
    mlirSymbolTableReplaceAllSymbolUses(oldSymbol, newSymbol, from)

Attempt to replace all uses that are nested within the given operation of the given symbol 'oldSymbol' with the provided 'newSymbol'. This does not traverse into nested symbol tables. Will fail atomically if there are any unknown operations that may be potential symbol tables.
"""
function mlirSymbolTableReplaceAllSymbolUses(oldSymbol, newSymbol, from)
    @ccall mlir_c.mlirSymbolTableReplaceAllSymbolUses(
        oldSymbol::MlirStringRef, newSymbol::MlirStringRef, from::MlirOperation
    )::MlirLogicalResult
end

"""
    mlirSymbolTableWalkSymbolTables(from, allSymUsesVisible, callback, userData)

Walks all symbol table operations nested within, and including, `op`. For each symbol table operation, the provided callback is invoked with the op and a boolean signifying if the symbols within that symbol table can be treated as if all uses within the IR are visible to the caller. `allSymUsesVisible` identifies whether all of the symbol uses of symbols within `op` are visible.
"""
function mlirSymbolTableWalkSymbolTables(from, allSymUsesVisible, callback, userData)
    @ccall mlir_c.mlirSymbolTableWalkSymbolTables(
        from::MlirOperation,
        allSymUsesVisible::Bool,
        callback::Ptr{Cvoid},
        userData::Ptr{Cvoid},
    )::Cvoid
end

struct MlirAffineExpr
    ptr::Ptr{Cvoid}
end

"""
    mlirAffineExprGetContext(affineExpr)

Gets the context that owns the affine expression.
"""
function mlirAffineExprGetContext(affineExpr)
    @ccall mlir_c.mlirAffineExprGetContext(affineExpr::MlirAffineExpr)::MlirContext
end

"""
    mlirAffineExprEqual(lhs, rhs)

Returns `true` if the two affine expressions are equal.
"""
function mlirAffineExprEqual(lhs, rhs)
    @ccall mlir_c.mlirAffineExprEqual(lhs::MlirAffineExpr, rhs::MlirAffineExpr)::Bool
end

"""
    mlirAffineExprIsNull(affineExpr)

Returns `true` if the given affine expression is a null expression. Note constant zero is not a null expression.
"""
function mlirAffineExprIsNull(affineExpr)
    @ccall mlir_c.mlirAffineExprIsNull(affineExpr::MlirAffineExpr)::Bool
end

"""
    mlirAffineExprPrint(affineExpr, callback, userData)

Prints an affine expression by sending chunks of the string representation and forwarding `userData to `callback`. Note that the callback may be called several times with consecutive chunks of the string.
"""
function mlirAffineExprPrint(affineExpr, callback, userData)
    @ccall mlir_c.mlirAffineExprPrint(
        affineExpr::MlirAffineExpr, callback::MlirStringCallback, userData::Ptr{Cvoid}
    )::Cvoid
end

"""
    mlirAffineExprDump(affineExpr)

Prints the affine expression to the standard error stream.
"""
function mlirAffineExprDump(affineExpr)
    @ccall mlir_c.mlirAffineExprDump(affineExpr::MlirAffineExpr)::Cvoid
end

"""
    mlirAffineExprIsSymbolicOrConstant(affineExpr)

Checks whether the given affine expression is made out of only symbols and constants.
"""
function mlirAffineExprIsSymbolicOrConstant(affineExpr)
    @ccall mlir_c.mlirAffineExprIsSymbolicOrConstant(affineExpr::MlirAffineExpr)::Bool
end

"""
    mlirAffineExprIsPureAffine(affineExpr)

Checks whether the given affine expression is a pure affine expression, i.e. mul, floordiv, ceildic, and mod is only allowed w.r.t constants.
"""
function mlirAffineExprIsPureAffine(affineExpr)
    @ccall mlir_c.mlirAffineExprIsPureAffine(affineExpr::MlirAffineExpr)::Bool
end

"""
    mlirAffineExprGetLargestKnownDivisor(affineExpr)

Returns the greatest known integral divisor of this affine expression. The result is always positive.
"""
function mlirAffineExprGetLargestKnownDivisor(affineExpr)
    @ccall mlir_c.mlirAffineExprGetLargestKnownDivisor(affineExpr::MlirAffineExpr)::Int64
end

"""
    mlirAffineExprIsMultipleOf(affineExpr, factor)

Checks whether the given affine expression is a multiple of 'factor'.
"""
function mlirAffineExprIsMultipleOf(affineExpr, factor)
    @ccall mlir_c.mlirAffineExprIsMultipleOf(
        affineExpr::MlirAffineExpr, factor::Int64
    )::Bool
end

"""
    mlirAffineExprIsFunctionOfDim(affineExpr, position)

Checks whether the given affine expression involves AffineDimExpr 'position'.
"""
function mlirAffineExprIsFunctionOfDim(affineExpr, position)
    @ccall mlir_c.mlirAffineExprIsFunctionOfDim(
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
    @ccall mlir_c.mlirAffineExprCompose(
        affineExpr::MlirAffineExpr, affineMap::MlirAffineMap
    )::MlirAffineExpr
end

"""
    mlirAffineExprShiftDims(affineExpr, numDims, shift, offset)

Replace dims[offset ... numDims) by dims[offset + shift ... shift + numDims).
"""
function mlirAffineExprShiftDims(affineExpr, numDims, shift, offset)
    @ccall mlir_c.mlirAffineExprShiftDims(
        affineExpr::MlirAffineExpr, numDims::UInt32, shift::UInt32, offset::UInt32
    )::MlirAffineExpr
end

"""
    mlirAffineExprShiftSymbols(affineExpr, numSymbols, shift, offset)

Replace symbols[offset ... numSymbols) by symbols[offset + shift ... shift + numSymbols).
"""
function mlirAffineExprShiftSymbols(affineExpr, numSymbols, shift, offset)
    @ccall mlir_c.mlirAffineExprShiftSymbols(
        affineExpr::MlirAffineExpr, numSymbols::UInt32, shift::UInt32, offset::UInt32
    )::MlirAffineExpr
end

"""
    mlirSimplifyAffineExpr(expr, numDims, numSymbols)

Simplify an affine expression by flattening and some amount of simple analysis. This has complexity linear in the number of nodes in 'expr'. Returns the simplified expression, which is the same as the input expression if it can't be simplified. When `expr` is semi-affine, a simplified semi-affine expression is constructed in the sorted order of dimension and symbol positions.
"""
function mlirSimplifyAffineExpr(expr, numDims, numSymbols)
    @ccall mlir_c.mlirSimplifyAffineExpr(
        expr::MlirAffineExpr, numDims::UInt32, numSymbols::UInt32
    )::MlirAffineExpr
end

"""
    mlirAffineExprIsADim(affineExpr)

Checks whether the given affine expression is a dimension expression.
"""
function mlirAffineExprIsADim(affineExpr)
    @ccall mlir_c.mlirAffineExprIsADim(affineExpr::MlirAffineExpr)::Bool
end

"""
    mlirAffineDimExprGet(ctx, position)

Creates an affine dimension expression with 'position' in the context.
"""
function mlirAffineDimExprGet(ctx, position)
    @ccall mlir_c.mlirAffineDimExprGet(
        ctx::MlirContext, position::Cptrdiff_t
    )::MlirAffineExpr
end

"""
    mlirAffineDimExprGetPosition(affineExpr)

Returns the position of the given affine dimension expression.
"""
function mlirAffineDimExprGetPosition(affineExpr)
    @ccall mlir_c.mlirAffineDimExprGetPosition(affineExpr::MlirAffineExpr)::Cptrdiff_t
end

"""
    mlirAffineExprIsASymbol(affineExpr)

Checks whether the given affine expression is a symbol expression.
"""
function mlirAffineExprIsASymbol(affineExpr)
    @ccall mlir_c.mlirAffineExprIsASymbol(affineExpr::MlirAffineExpr)::Bool
end

"""
    mlirAffineSymbolExprGet(ctx, position)

Creates an affine symbol expression with 'position' in the context.
"""
function mlirAffineSymbolExprGet(ctx, position)
    @ccall mlir_c.mlirAffineSymbolExprGet(
        ctx::MlirContext, position::Cptrdiff_t
    )::MlirAffineExpr
end

"""
    mlirAffineSymbolExprGetPosition(affineExpr)

Returns the position of the given affine symbol expression.
"""
function mlirAffineSymbolExprGetPosition(affineExpr)
    @ccall mlir_c.mlirAffineSymbolExprGetPosition(affineExpr::MlirAffineExpr)::Cptrdiff_t
end

"""
    mlirAffineExprIsAConstant(affineExpr)

Checks whether the given affine expression is a constant expression.
"""
function mlirAffineExprIsAConstant(affineExpr)
    @ccall mlir_c.mlirAffineExprIsAConstant(affineExpr::MlirAffineExpr)::Bool
end

"""
    mlirAffineConstantExprGet(ctx, constant)

Creates an affine constant expression with 'constant' in the context.
"""
function mlirAffineConstantExprGet(ctx, constant)
    @ccall mlir_c.mlirAffineConstantExprGet(
        ctx::MlirContext, constant::Int64
    )::MlirAffineExpr
end

"""
    mlirAffineConstantExprGetValue(affineExpr)

Returns the value of the given affine constant expression.
"""
function mlirAffineConstantExprGetValue(affineExpr)
    @ccall mlir_c.mlirAffineConstantExprGetValue(affineExpr::MlirAffineExpr)::Int64
end

"""
    mlirAffineExprIsAAdd(affineExpr)

Checks whether the given affine expression is an add expression.
"""
function mlirAffineExprIsAAdd(affineExpr)
    @ccall mlir_c.mlirAffineExprIsAAdd(affineExpr::MlirAffineExpr)::Bool
end

"""
    mlirAffineAddExprGet(lhs, rhs)

Creates an affine add expression with 'lhs' and 'rhs'.
"""
function mlirAffineAddExprGet(lhs, rhs)
    @ccall mlir_c.mlirAffineAddExprGet(
        lhs::MlirAffineExpr, rhs::MlirAffineExpr
    )::MlirAffineExpr
end

"""
    mlirAffineExprIsAMul(affineExpr)

Checks whether the given affine expression is an mul expression.
"""
function mlirAffineExprIsAMul(affineExpr)
    @ccall mlir_c.mlirAffineExprIsAMul(affineExpr::MlirAffineExpr)::Bool
end

"""
    mlirAffineMulExprGet(lhs, rhs)

Creates an affine mul expression with 'lhs' and 'rhs'.
"""
function mlirAffineMulExprGet(lhs, rhs)
    @ccall mlir_c.mlirAffineMulExprGet(
        lhs::MlirAffineExpr, rhs::MlirAffineExpr
    )::MlirAffineExpr
end

"""
    mlirAffineExprIsAMod(affineExpr)

Checks whether the given affine expression is an mod expression.
"""
function mlirAffineExprIsAMod(affineExpr)
    @ccall mlir_c.mlirAffineExprIsAMod(affineExpr::MlirAffineExpr)::Bool
end

"""
    mlirAffineModExprGet(lhs, rhs)

Creates an affine mod expression with 'lhs' and 'rhs'.
"""
function mlirAffineModExprGet(lhs, rhs)
    @ccall mlir_c.mlirAffineModExprGet(
        lhs::MlirAffineExpr, rhs::MlirAffineExpr
    )::MlirAffineExpr
end

"""
    mlirAffineExprIsAFloorDiv(affineExpr)

Checks whether the given affine expression is an floordiv expression.
"""
function mlirAffineExprIsAFloorDiv(affineExpr)
    @ccall mlir_c.mlirAffineExprIsAFloorDiv(affineExpr::MlirAffineExpr)::Bool
end

"""
    mlirAffineFloorDivExprGet(lhs, rhs)

Creates an affine floordiv expression with 'lhs' and 'rhs'.
"""
function mlirAffineFloorDivExprGet(lhs, rhs)
    @ccall mlir_c.mlirAffineFloorDivExprGet(
        lhs::MlirAffineExpr, rhs::MlirAffineExpr
    )::MlirAffineExpr
end

"""
    mlirAffineExprIsACeilDiv(affineExpr)

Checks whether the given affine expression is an ceildiv expression.
"""
function mlirAffineExprIsACeilDiv(affineExpr)
    @ccall mlir_c.mlirAffineExprIsACeilDiv(affineExpr::MlirAffineExpr)::Bool
end

"""
    mlirAffineCeilDivExprGet(lhs, rhs)

Creates an affine ceildiv expression with 'lhs' and 'rhs'.
"""
function mlirAffineCeilDivExprGet(lhs, rhs)
    @ccall mlir_c.mlirAffineCeilDivExprGet(
        lhs::MlirAffineExpr, rhs::MlirAffineExpr
    )::MlirAffineExpr
end

"""
    mlirAffineExprIsABinary(affineExpr)

Checks whether the given affine expression is binary.
"""
function mlirAffineExprIsABinary(affineExpr)
    @ccall mlir_c.mlirAffineExprIsABinary(affineExpr::MlirAffineExpr)::Bool
end

"""
    mlirAffineBinaryOpExprGetLHS(affineExpr)

Returns the left hand side affine expression of the given affine binary operation expression.
"""
function mlirAffineBinaryOpExprGetLHS(affineExpr)
    @ccall mlir_c.mlirAffineBinaryOpExprGetLHS(affineExpr::MlirAffineExpr)::MlirAffineExpr
end

"""
    mlirAffineBinaryOpExprGetRHS(affineExpr)

Returns the right hand side affine expression of the given affine binary operation expression.
"""
function mlirAffineBinaryOpExprGetRHS(affineExpr)
    @ccall mlir_c.mlirAffineBinaryOpExprGetRHS(affineExpr::MlirAffineExpr)::MlirAffineExpr
end

"""
    mlirAffineMapGetContext(affineMap)

Gets the context that the given affine map was created with
"""
function mlirAffineMapGetContext(affineMap)
    @ccall mlir_c.mlirAffineMapGetContext(affineMap::MlirAffineMap)::MlirContext
end

"""
    mlirAffineMapIsNull(affineMap)

Checks whether an affine map is null.
"""
function mlirAffineMapIsNull(affineMap)
    @ccall mlir_c.mlirAffineMapIsNull(affineMap::MlirAffineMap)::Bool
end

"""
    mlirAffineMapEqual(a1, a2)

Checks if two affine maps are equal.
"""
function mlirAffineMapEqual(a1, a2)
    @ccall mlir_c.mlirAffineMapEqual(a1::MlirAffineMap, a2::MlirAffineMap)::Bool
end

"""
    mlirAffineMapPrint(affineMap, callback, userData)

Prints an affine map by sending chunks of the string representation and forwarding `userData to `callback`. Note that the callback may be called several times with consecutive chunks of the string.
"""
function mlirAffineMapPrint(affineMap, callback, userData)
    @ccall mlir_c.mlirAffineMapPrint(
        affineMap::MlirAffineMap, callback::MlirStringCallback, userData::Ptr{Cvoid}
    )::Cvoid
end

"""
    mlirAffineMapDump(affineMap)

Prints the affine map to the standard error stream.
"""
function mlirAffineMapDump(affineMap)
    @ccall mlir_c.mlirAffineMapDump(affineMap::MlirAffineMap)::Cvoid
end

"""
    mlirAffineMapEmptyGet(ctx)

Creates a zero result affine map with no dimensions or symbols in the context. The affine map is owned by the context.
"""
function mlirAffineMapEmptyGet(ctx)
    @ccall mlir_c.mlirAffineMapEmptyGet(ctx::MlirContext)::MlirAffineMap
end

"""
    mlirAffineMapZeroResultGet(ctx, dimCount, symbolCount)

Creates a zero result affine map of the given dimensions and symbols in the context. The affine map is owned by the context.
"""
function mlirAffineMapZeroResultGet(ctx, dimCount, symbolCount)
    @ccall mlir_c.mlirAffineMapZeroResultGet(
        ctx::MlirContext, dimCount::Cptrdiff_t, symbolCount::Cptrdiff_t
    )::MlirAffineMap
end

"""
    mlirAffineMapGet(ctx, dimCount, symbolCount, nAffineExprs, affineExprs)

Creates an affine map with results defined by the given list of affine expressions. The map resulting map also has the requested number of input dimensions and symbols, regardless of them being used in the results.
"""
function mlirAffineMapGet(ctx, dimCount, symbolCount, nAffineExprs, affineExprs)
    @ccall mlir_c.mlirAffineMapGet(
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
    @ccall mlir_c.mlirAffineMapConstantGet(ctx::MlirContext, val::Int64)::MlirAffineMap
end

"""
    mlirAffineMapMultiDimIdentityGet(ctx, numDims)

Creates an affine map with 'numDims' identity in the context. The affine map is owned by the context.
"""
function mlirAffineMapMultiDimIdentityGet(ctx, numDims)
    @ccall mlir_c.mlirAffineMapMultiDimIdentityGet(
        ctx::MlirContext, numDims::Cptrdiff_t
    )::MlirAffineMap
end

"""
    mlirAffineMapMinorIdentityGet(ctx, dims, results)

Creates an identity affine map on the most minor dimensions in the context. The affine map is owned by the context. The function asserts that the number of dimensions is greater or equal to the number of results.
"""
function mlirAffineMapMinorIdentityGet(ctx, dims, results)
    @ccall mlir_c.mlirAffineMapMinorIdentityGet(
        ctx::MlirContext, dims::Cptrdiff_t, results::Cptrdiff_t
    )::MlirAffineMap
end

"""
    mlirAffineMapPermutationGet(ctx, size, permutation)

Creates an affine map with a permutation expression and its size in the context. The permutation expression is a non-empty vector of integers. The elements of the permutation vector must be continuous from 0 and cannot be repeated (i.e. `[1,2,0]` is a valid permutation. `[2,0]` or `[1,1,2]` is an invalid permutation.) The affine map is owned by the context.
"""
function mlirAffineMapPermutationGet(ctx, size, permutation)
    @ccall mlir_c.mlirAffineMapPermutationGet(
        ctx::MlirContext, size::Cptrdiff_t, permutation::Ptr{Cuint}
    )::MlirAffineMap
end

"""
    mlirAffineMapIsIdentity(affineMap)

Checks whether the given affine map is an identity affine map. The function asserts that the number of dimensions is greater or equal to the number of results.
"""
function mlirAffineMapIsIdentity(affineMap)
    @ccall mlir_c.mlirAffineMapIsIdentity(affineMap::MlirAffineMap)::Bool
end

"""
    mlirAffineMapIsMinorIdentity(affineMap)

Checks whether the given affine map is a minor identity affine map.
"""
function mlirAffineMapIsMinorIdentity(affineMap)
    @ccall mlir_c.mlirAffineMapIsMinorIdentity(affineMap::MlirAffineMap)::Bool
end

"""
    mlirAffineMapIsEmpty(affineMap)

Checks whether the given affine map is an empty affine map.
"""
function mlirAffineMapIsEmpty(affineMap)
    @ccall mlir_c.mlirAffineMapIsEmpty(affineMap::MlirAffineMap)::Bool
end

"""
    mlirAffineMapIsSingleConstant(affineMap)

Checks whether the given affine map is a single result constant affine map.
"""
function mlirAffineMapIsSingleConstant(affineMap)
    @ccall mlir_c.mlirAffineMapIsSingleConstant(affineMap::MlirAffineMap)::Bool
end

"""
    mlirAffineMapGetSingleConstantResult(affineMap)

Returns the constant result of the given affine map. The function asserts that the map has a single constant result.
"""
function mlirAffineMapGetSingleConstantResult(affineMap)
    @ccall mlir_c.mlirAffineMapGetSingleConstantResult(affineMap::MlirAffineMap)::Int64
end

"""
    mlirAffineMapGetNumDims(affineMap)

Returns the number of dimensions of the given affine map.
"""
function mlirAffineMapGetNumDims(affineMap)
    @ccall mlir_c.mlirAffineMapGetNumDims(affineMap::MlirAffineMap)::Cptrdiff_t
end

"""
    mlirAffineMapGetNumSymbols(affineMap)

Returns the number of symbols of the given affine map.
"""
function mlirAffineMapGetNumSymbols(affineMap)
    @ccall mlir_c.mlirAffineMapGetNumSymbols(affineMap::MlirAffineMap)::Cptrdiff_t
end

"""
    mlirAffineMapGetNumResults(affineMap)

Returns the number of results of the given affine map.
"""
function mlirAffineMapGetNumResults(affineMap)
    @ccall mlir_c.mlirAffineMapGetNumResults(affineMap::MlirAffineMap)::Cptrdiff_t
end

"""
    mlirAffineMapGetResult(affineMap, pos)

Returns the result at the given position.
"""
function mlirAffineMapGetResult(affineMap, pos)
    @ccall mlir_c.mlirAffineMapGetResult(
        affineMap::MlirAffineMap, pos::Cptrdiff_t
    )::MlirAffineExpr
end

"""
    mlirAffineMapGetNumInputs(affineMap)

Returns the number of inputs (dimensions + symbols) of the given affine map.
"""
function mlirAffineMapGetNumInputs(affineMap)
    @ccall mlir_c.mlirAffineMapGetNumInputs(affineMap::MlirAffineMap)::Cptrdiff_t
end

"""
    mlirAffineMapIsProjectedPermutation(affineMap)

Checks whether the given affine map represents a subset of a symbol-less permutation map.
"""
function mlirAffineMapIsProjectedPermutation(affineMap)
    @ccall mlir_c.mlirAffineMapIsProjectedPermutation(affineMap::MlirAffineMap)::Bool
end

"""
    mlirAffineMapIsPermutation(affineMap)

Checks whether the given affine map represents a symbol-less permutation map.
"""
function mlirAffineMapIsPermutation(affineMap)
    @ccall mlir_c.mlirAffineMapIsPermutation(affineMap::MlirAffineMap)::Bool
end

"""
    mlirAffineMapGetSubMap(affineMap, size, resultPos)

Returns the affine map consisting of the `resultPos` subset.
"""
function mlirAffineMapGetSubMap(affineMap, size, resultPos)
    @ccall mlir_c.mlirAffineMapGetSubMap(
        affineMap::MlirAffineMap, size::Cptrdiff_t, resultPos::Ptr{Cptrdiff_t}
    )::MlirAffineMap
end

"""
    mlirAffineMapGetMajorSubMap(affineMap, numResults)

Returns the affine map consisting of the most major `numResults` results. Returns the null AffineMap if the `numResults` is equal to zero. Returns the `affineMap` if `numResults` is greater or equals to number of results of the given affine map.
"""
function mlirAffineMapGetMajorSubMap(affineMap, numResults)
    @ccall mlir_c.mlirAffineMapGetMajorSubMap(
        affineMap::MlirAffineMap, numResults::Cptrdiff_t
    )::MlirAffineMap
end

"""
    mlirAffineMapGetMinorSubMap(affineMap, numResults)

Returns the affine map consisting of the most minor `numResults` results. Returns the null AffineMap if the `numResults` is equal to zero. Returns the `affineMap` if `numResults` is greater or equals to number of results of the given affine map.
"""
function mlirAffineMapGetMinorSubMap(affineMap, numResults)
    @ccall mlir_c.mlirAffineMapGetMinorSubMap(
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
    @ccall mlir_c.mlirAffineMapReplace(
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
    @ccall mlir_c.mlirAffineMapCompressUnusedSymbols(
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
    @ccall mlir_c.mlirIntegerSetGetContext(set::MlirIntegerSet)::MlirContext
end

"""
    mlirIntegerSetIsNull(set)

Checks whether an integer set is a null object.
"""
function mlirIntegerSetIsNull(set)
    @ccall mlir_c.mlirIntegerSetIsNull(set::MlirIntegerSet)::Bool
end

"""
    mlirIntegerSetEqual(s1, s2)

Checks if two integer set objects are equal. This is a "shallow" comparison of two objects. Only the sets with some small number of constraints are uniqued and compare equal here. Set objects that represent the same integer set with different constraints may be considered non-equal by this check. Set difference followed by an (expensive) emptiness check should be used to check equivalence of the underlying integer sets.
"""
function mlirIntegerSetEqual(s1, s2)
    @ccall mlir_c.mlirIntegerSetEqual(s1::MlirIntegerSet, s2::MlirIntegerSet)::Bool
end

"""
    mlirIntegerSetPrint(set, callback, userData)

Prints an integer set by sending chunks of the string representation and forwarding `userData to `callback`. Note that the callback may be called several times with consecutive chunks of the string.
"""
function mlirIntegerSetPrint(set, callback, userData)
    @ccall mlir_c.mlirIntegerSetPrint(
        set::MlirIntegerSet, callback::MlirStringCallback, userData::Ptr{Cvoid}
    )::Cvoid
end

"""
    mlirIntegerSetDump(set)

Prints an integer set to the standard error stream.
"""
function mlirIntegerSetDump(set)
    @ccall mlir_c.mlirIntegerSetDump(set::MlirIntegerSet)::Cvoid
end

"""
    mlirIntegerSetEmptyGet(context, numDims, numSymbols)

Gets or creates a new canonically empty integer set with the give number of dimensions and symbols in the given context.
"""
function mlirIntegerSetEmptyGet(context, numDims, numSymbols)
    @ccall mlir_c.mlirIntegerSetEmptyGet(
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
    @ccall mlir_c.mlirIntegerSetGet(
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
    @ccall mlir_c.mlirIntegerSetReplaceGet(
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
    @ccall mlir_c.mlirIntegerSetIsCanonicalEmpty(set::MlirIntegerSet)::Bool
end

"""
    mlirIntegerSetGetNumDims(set)

Returns the number of dimensions in the given set.
"""
function mlirIntegerSetGetNumDims(set)
    @ccall mlir_c.mlirIntegerSetGetNumDims(set::MlirIntegerSet)::Cptrdiff_t
end

"""
    mlirIntegerSetGetNumSymbols(set)

Returns the number of symbols in the given set.
"""
function mlirIntegerSetGetNumSymbols(set)
    @ccall mlir_c.mlirIntegerSetGetNumSymbols(set::MlirIntegerSet)::Cptrdiff_t
end

"""
    mlirIntegerSetGetNumInputs(set)

Returns the number of inputs (dimensions + symbols) in the given set.
"""
function mlirIntegerSetGetNumInputs(set)
    @ccall mlir_c.mlirIntegerSetGetNumInputs(set::MlirIntegerSet)::Cptrdiff_t
end

"""
    mlirIntegerSetGetNumConstraints(set)

Returns the number of constraints (equalities + inequalities) in the given set.
"""
function mlirIntegerSetGetNumConstraints(set)
    @ccall mlir_c.mlirIntegerSetGetNumConstraints(set::MlirIntegerSet)::Cptrdiff_t
end

"""
    mlirIntegerSetGetNumEqualities(set)

Returns the number of equalities in the given set.
"""
function mlirIntegerSetGetNumEqualities(set)
    @ccall mlir_c.mlirIntegerSetGetNumEqualities(set::MlirIntegerSet)::Cptrdiff_t
end

"""
    mlirIntegerSetGetNumInequalities(set)

Returns the number of inequalities in the given set.
"""
function mlirIntegerSetGetNumInequalities(set)
    @ccall mlir_c.mlirIntegerSetGetNumInequalities(set::MlirIntegerSet)::Cptrdiff_t
end

"""
    mlirIntegerSetGetConstraint(set, pos)

Returns `pos`-th constraint of the set.
"""
function mlirIntegerSetGetConstraint(set, pos)
    @ccall mlir_c.mlirIntegerSetGetConstraint(
        set::MlirIntegerSet, pos::Cptrdiff_t
    )::MlirAffineExpr
end

"""
    mlirIntegerSetIsConstraintEq(set, pos)

Returns `true` of the `pos`-th constraint of the set is an equality constraint, `false` otherwise.
"""
function mlirIntegerSetIsConstraintEq(set, pos)
    @ccall mlir_c.mlirIntegerSetIsConstraintEq(set::MlirIntegerSet, pos::Cptrdiff_t)::Bool
end

"""
    mlirAttributeGetNull()

Returns an empty attribute.
"""
function mlirAttributeGetNull()
    @ccall mlir_c.mlirAttributeGetNull()::MlirAttribute
end

function mlirAttributeIsALocation(attr)
    @ccall mlir_c.mlirAttributeIsALocation(attr::MlirAttribute)::Bool
end

"""
    mlirAttributeIsAAffineMap(attr)

Checks whether the given attribute is an affine map attribute.
"""
function mlirAttributeIsAAffineMap(attr)
    @ccall mlir_c.mlirAttributeIsAAffineMap(attr::MlirAttribute)::Bool
end

"""
    mlirAffineMapAttrGet(map)

Creates an affine map attribute wrapping the given map. The attribute belongs to the same context as the affine map.
"""
function mlirAffineMapAttrGet(map)
    @ccall mlir_c.mlirAffineMapAttrGet(map::MlirAffineMap)::MlirAttribute
end

"""
    mlirAffineMapAttrGetValue(attr)

Returns the affine map wrapped in the given affine map attribute.
"""
function mlirAffineMapAttrGetValue(attr)
    @ccall mlir_c.mlirAffineMapAttrGetValue(attr::MlirAttribute)::MlirAffineMap
end

"""
    mlirAffineMapAttrGetTypeID()

Returns the typeID of an AffineMap attribute.
"""
function mlirAffineMapAttrGetTypeID()
    @ccall mlir_c.mlirAffineMapAttrGetTypeID()::MlirTypeID
end

"""
    mlirAttributeIsAArray(attr)

Checks whether the given attribute is an array attribute.
"""
function mlirAttributeIsAArray(attr)
    @ccall mlir_c.mlirAttributeIsAArray(attr::MlirAttribute)::Bool
end

"""
    mlirArrayAttrGet(ctx, numElements, elements)

Creates an array element containing the given list of elements in the given context.
"""
function mlirArrayAttrGet(ctx, numElements, elements)
    @ccall mlir_c.mlirArrayAttrGet(
        ctx::MlirContext, numElements::Cptrdiff_t, elements::Ptr{MlirAttribute}
    )::MlirAttribute
end

"""
    mlirArrayAttrGetNumElements(attr)

Returns the number of elements stored in the given array attribute.
"""
function mlirArrayAttrGetNumElements(attr)
    @ccall mlir_c.mlirArrayAttrGetNumElements(attr::MlirAttribute)::Cptrdiff_t
end

"""
    mlirArrayAttrGetElement(attr, pos)

Returns pos-th element stored in the given array attribute.
"""
function mlirArrayAttrGetElement(attr, pos)
    @ccall mlir_c.mlirArrayAttrGetElement(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::MlirAttribute
end

"""
    mlirArrayAttrGetTypeID()

Returns the typeID of an Array attribute.
"""
function mlirArrayAttrGetTypeID()
    @ccall mlir_c.mlirArrayAttrGetTypeID()::MlirTypeID
end

"""
    mlirAttributeIsADictionary(attr)

Checks whether the given attribute is a dictionary attribute.
"""
function mlirAttributeIsADictionary(attr)
    @ccall mlir_c.mlirAttributeIsADictionary(attr::MlirAttribute)::Bool
end

"""
    mlirDictionaryAttrGet(ctx, numElements, elements)

Creates a dictionary attribute containing the given list of elements in the provided context.
"""
function mlirDictionaryAttrGet(ctx, numElements, elements)
    @ccall mlir_c.mlirDictionaryAttrGet(
        ctx::MlirContext, numElements::Cptrdiff_t, elements::Ptr{MlirNamedAttribute}
    )::MlirAttribute
end

"""
    mlirDictionaryAttrGetNumElements(attr)

Returns the number of attributes contained in a dictionary attribute.
"""
function mlirDictionaryAttrGetNumElements(attr)
    @ccall mlir_c.mlirDictionaryAttrGetNumElements(attr::MlirAttribute)::Cptrdiff_t
end

"""
    mlirDictionaryAttrGetElement(attr, pos)

Returns pos-th element of the given dictionary attribute.
"""
function mlirDictionaryAttrGetElement(attr, pos)
    @ccall mlir_c.mlirDictionaryAttrGetElement(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::MlirNamedAttribute
end

"""
    mlirDictionaryAttrGetElementByName(attr, name)

Returns the dictionary attribute element with the given name or NULL if the given name does not exist in the dictionary.
"""
function mlirDictionaryAttrGetElementByName(attr, name)
    @ccall mlir_c.mlirDictionaryAttrGetElementByName(
        attr::MlirAttribute, name::MlirStringRef
    )::MlirAttribute
end

"""
    mlirDictionaryAttrGetTypeID()

Returns the typeID of a Dictionary attribute.
"""
function mlirDictionaryAttrGetTypeID()
    @ccall mlir_c.mlirDictionaryAttrGetTypeID()::MlirTypeID
end

"""
    mlirAttributeIsAFloat(attr)

Checks whether the given attribute is a floating point attribute.
"""
function mlirAttributeIsAFloat(attr)
    @ccall mlir_c.mlirAttributeIsAFloat(attr::MlirAttribute)::Bool
end

"""
    mlirFloatAttrDoubleGet(ctx, type, value)

Creates a floating point attribute in the given context with the given double value and double-precision FP semantics.
"""
function mlirFloatAttrDoubleGet(ctx, type, value)
    @ccall mlir_c.mlirFloatAttrDoubleGet(
        ctx::MlirContext, type::MlirType, value::Cdouble
    )::MlirAttribute
end

"""
    mlirFloatAttrDoubleGetChecked(loc, type, value)

Same as "[`mlirFloatAttrDoubleGet`](@ref)", but if the type is not valid for a construction of a FloatAttr, returns a null [`MlirAttribute`](@ref).
"""
function mlirFloatAttrDoubleGetChecked(loc, type, value)
    @ccall mlir_c.mlirFloatAttrDoubleGetChecked(
        loc::MlirLocation, type::MlirType, value::Cdouble
    )::MlirAttribute
end

"""
    mlirFloatAttrGetValueDouble(attr)

Returns the value stored in the given floating point attribute, interpreting the value as double.
"""
function mlirFloatAttrGetValueDouble(attr)
    @ccall mlir_c.mlirFloatAttrGetValueDouble(attr::MlirAttribute)::Cdouble
end

"""
    mlirFloatAttrGetTypeID()

Returns the typeID of a Float attribute.
"""
function mlirFloatAttrGetTypeID()
    @ccall mlir_c.mlirFloatAttrGetTypeID()::MlirTypeID
end

"""
    mlirAttributeIsAInteger(attr)

Checks whether the given attribute is an integer attribute.
"""
function mlirAttributeIsAInteger(attr)
    @ccall mlir_c.mlirAttributeIsAInteger(attr::MlirAttribute)::Bool
end

"""
    mlirIntegerAttrGet(type, value)

Creates an integer attribute of the given type with the given integer value.
"""
function mlirIntegerAttrGet(type, value)
    @ccall mlir_c.mlirIntegerAttrGet(type::MlirType, value::Int64)::MlirAttribute
end

"""
    mlirIntegerAttrGetValueInt(attr)

Returns the value stored in the given integer attribute, assuming the value is of signless type and fits into a signed 64-bit integer.
"""
function mlirIntegerAttrGetValueInt(attr)
    @ccall mlir_c.mlirIntegerAttrGetValueInt(attr::MlirAttribute)::Int64
end

"""
    mlirIntegerAttrGetValueSInt(attr)

Returns the value stored in the given integer attribute, assuming the value is of signed type and fits into a signed 64-bit integer.
"""
function mlirIntegerAttrGetValueSInt(attr)
    @ccall mlir_c.mlirIntegerAttrGetValueSInt(attr::MlirAttribute)::Int64
end

"""
    mlirIntegerAttrGetValueUInt(attr)

Returns the value stored in the given integer attribute, assuming the value is of unsigned type and fits into an unsigned 64-bit integer.
"""
function mlirIntegerAttrGetValueUInt(attr)
    @ccall mlir_c.mlirIntegerAttrGetValueUInt(attr::MlirAttribute)::UInt64
end

"""
    mlirIntegerAttrGetTypeID()

Returns the typeID of an Integer attribute.
"""
function mlirIntegerAttrGetTypeID()
    @ccall mlir_c.mlirIntegerAttrGetTypeID()::MlirTypeID
end

"""
    mlirAttributeIsABool(attr)

Checks whether the given attribute is a bool attribute.
"""
function mlirAttributeIsABool(attr)
    @ccall mlir_c.mlirAttributeIsABool(attr::MlirAttribute)::Bool
end

"""
    mlirBoolAttrGet(ctx, value)

Creates a bool attribute in the given context with the given value.
"""
function mlirBoolAttrGet(ctx, value)
    @ccall mlir_c.mlirBoolAttrGet(ctx::MlirContext, value::Cint)::MlirAttribute
end

"""
    mlirBoolAttrGetValue(attr)

Returns the value stored in the given bool attribute.
"""
function mlirBoolAttrGetValue(attr)
    @ccall mlir_c.mlirBoolAttrGetValue(attr::MlirAttribute)::Bool
end

"""
    mlirAttributeIsAIntegerSet(attr)

Checks whether the given attribute is an integer set attribute.
"""
function mlirAttributeIsAIntegerSet(attr)
    @ccall mlir_c.mlirAttributeIsAIntegerSet(attr::MlirAttribute)::Bool
end

"""
    mlirIntegerSetAttrGet(set)

Creates an integer set attribute wrapping the given set. The attribute belongs to the same context as the integer set.
"""
function mlirIntegerSetAttrGet(set)
    @ccall mlir_c.mlirIntegerSetAttrGet(set::MlirIntegerSet)::MlirAttribute
end

"""
    mlirIntegerSetAttrGetValue(attr)

Returns the integer set wrapped in the given integer set attribute.
"""
function mlirIntegerSetAttrGetValue(attr)
    @ccall mlir_c.mlirIntegerSetAttrGetValue(attr::MlirAttribute)::MlirIntegerSet
end

"""
    mlirIntegerSetAttrGetTypeID()

Returns the typeID of an IntegerSet attribute.
"""
function mlirIntegerSetAttrGetTypeID()
    @ccall mlir_c.mlirIntegerSetAttrGetTypeID()::MlirTypeID
end

"""
    mlirAttributeIsAOpaque(attr)

Checks whether the given attribute is an opaque attribute.
"""
function mlirAttributeIsAOpaque(attr)
    @ccall mlir_c.mlirAttributeIsAOpaque(attr::MlirAttribute)::Bool
end

"""
    mlirOpaqueAttrGet(ctx, dialectNamespace, dataLength, data, type)

Creates an opaque attribute in the given context associated with the dialect identified by its namespace. The attribute contains opaque byte data of the specified length (data need not be null-terminated).
"""
function mlirOpaqueAttrGet(ctx, dialectNamespace, dataLength, data, type)
    @ccall mlir_c.mlirOpaqueAttrGet(
        ctx::MlirContext,
        dialectNamespace::MlirStringRef,
        dataLength::Cptrdiff_t,
        data::Cstring,
        type::MlirType,
    )::MlirAttribute
end

"""
    mlirOpaqueAttrGetDialectNamespace(attr)

Returns the namespace of the dialect with which the given opaque attribute is associated. The namespace string is owned by the context.
"""
function mlirOpaqueAttrGetDialectNamespace(attr)
    @ccall mlir_c.mlirOpaqueAttrGetDialectNamespace(attr::MlirAttribute)::MlirStringRef
end

"""
    mlirOpaqueAttrGetData(attr)

Returns the raw data as a string reference. The data remains live as long as the context in which the attribute lives.
"""
function mlirOpaqueAttrGetData(attr)
    @ccall mlir_c.mlirOpaqueAttrGetData(attr::MlirAttribute)::MlirStringRef
end

"""
    mlirOpaqueAttrGetTypeID()

Returns the typeID of an Opaque attribute.
"""
function mlirOpaqueAttrGetTypeID()
    @ccall mlir_c.mlirOpaqueAttrGetTypeID()::MlirTypeID
end

"""
    mlirAttributeIsAString(attr)

Checks whether the given attribute is a string attribute.
"""
function mlirAttributeIsAString(attr)
    @ccall mlir_c.mlirAttributeIsAString(attr::MlirAttribute)::Bool
end

"""
    mlirStringAttrGet(ctx, str)

Creates a string attribute in the given context containing the given string.
"""
function mlirStringAttrGet(ctx, str)
    @ccall mlir_c.mlirStringAttrGet(ctx::MlirContext, str::MlirStringRef)::MlirAttribute
end

"""
    mlirStringAttrTypedGet(type, str)

Creates a string attribute in the given context containing the given string. Additionally, the attribute has the given type.
"""
function mlirStringAttrTypedGet(type, str)
    @ccall mlir_c.mlirStringAttrTypedGet(type::MlirType, str::MlirStringRef)::MlirAttribute
end

"""
    mlirStringAttrGetValue(attr)

Returns the attribute values as a string reference. The data remains live as long as the context in which the attribute lives.
"""
function mlirStringAttrGetValue(attr)
    @ccall mlir_c.mlirStringAttrGetValue(attr::MlirAttribute)::MlirStringRef
end

"""
    mlirStringAttrGetTypeID()

Returns the typeID of a String attribute.
"""
function mlirStringAttrGetTypeID()
    @ccall mlir_c.mlirStringAttrGetTypeID()::MlirTypeID
end

"""
    mlirAttributeIsASymbolRef(attr)

Checks whether the given attribute is a symbol reference attribute.
"""
function mlirAttributeIsASymbolRef(attr)
    @ccall mlir_c.mlirAttributeIsASymbolRef(attr::MlirAttribute)::Bool
end

"""
    mlirSymbolRefAttrGet(ctx, symbol, numReferences, references)

Creates a symbol reference attribute in the given context referencing a symbol identified by the given string inside a list of nested references. Each of the references in the list must not be nested.
"""
function mlirSymbolRefAttrGet(ctx, symbol, numReferences, references)
    @ccall mlir_c.mlirSymbolRefAttrGet(
        ctx::MlirContext,
        symbol::MlirStringRef,
        numReferences::Cptrdiff_t,
        references::Ptr{MlirAttribute},
    )::MlirAttribute
end

"""
    mlirSymbolRefAttrGetRootReference(attr)

Returns the string reference to the root referenced symbol. The data remains live as long as the context in which the attribute lives.
"""
function mlirSymbolRefAttrGetRootReference(attr)
    @ccall mlir_c.mlirSymbolRefAttrGetRootReference(attr::MlirAttribute)::MlirStringRef
end

"""
    mlirSymbolRefAttrGetLeafReference(attr)

Returns the string reference to the leaf referenced symbol. The data remains live as long as the context in which the attribute lives.
"""
function mlirSymbolRefAttrGetLeafReference(attr)
    @ccall mlir_c.mlirSymbolRefAttrGetLeafReference(attr::MlirAttribute)::MlirStringRef
end

"""
    mlirSymbolRefAttrGetNumNestedReferences(attr)

Returns the number of references nested in the given symbol reference attribute.
"""
function mlirSymbolRefAttrGetNumNestedReferences(attr)
    @ccall mlir_c.mlirSymbolRefAttrGetNumNestedReferences(attr::MlirAttribute)::Cptrdiff_t
end

"""
    mlirSymbolRefAttrGetNestedReference(attr, pos)

Returns pos-th reference nested in the given symbol reference attribute.
"""
function mlirSymbolRefAttrGetNestedReference(attr, pos)
    @ccall mlir_c.mlirSymbolRefAttrGetNestedReference(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::MlirAttribute
end

"""
    mlirSymbolRefAttrGetTypeID()

Returns the typeID of an SymbolRef attribute.
"""
function mlirSymbolRefAttrGetTypeID()
    @ccall mlir_c.mlirSymbolRefAttrGetTypeID()::MlirTypeID
end

"""
    mlirDisctinctAttrCreate(referencedAttr)

Creates a DisctinctAttr with the referenced attribute.
"""
function mlirDisctinctAttrCreate(referencedAttr)
    @ccall mlir_c.mlirDisctinctAttrCreate(referencedAttr::MlirAttribute)::MlirAttribute
end

"""
    mlirAttributeIsAFlatSymbolRef(attr)

Checks whether the given attribute is a flat symbol reference attribute.
"""
function mlirAttributeIsAFlatSymbolRef(attr)
    @ccall mlir_c.mlirAttributeIsAFlatSymbolRef(attr::MlirAttribute)::Bool
end

"""
    mlirFlatSymbolRefAttrGet(ctx, symbol)

Creates a flat symbol reference attribute in the given context referencing a symbol identified by the given string.
"""
function mlirFlatSymbolRefAttrGet(ctx, symbol)
    @ccall mlir_c.mlirFlatSymbolRefAttrGet(
        ctx::MlirContext, symbol::MlirStringRef
    )::MlirAttribute
end

"""
    mlirFlatSymbolRefAttrGetValue(attr)

Returns the referenced symbol as a string reference. The data remains live as long as the context in which the attribute lives.
"""
function mlirFlatSymbolRefAttrGetValue(attr)
    @ccall mlir_c.mlirFlatSymbolRefAttrGetValue(attr::MlirAttribute)::MlirStringRef
end

"""
    mlirAttributeIsAType(attr)

Checks whether the given attribute is a type attribute.
"""
function mlirAttributeIsAType(attr)
    @ccall mlir_c.mlirAttributeIsAType(attr::MlirAttribute)::Bool
end

"""
    mlirTypeAttrGet(type)

Creates a type attribute wrapping the given type in the same context as the type.
"""
function mlirTypeAttrGet(type)
    @ccall mlir_c.mlirTypeAttrGet(type::MlirType)::MlirAttribute
end

"""
    mlirTypeAttrGetValue(attr)

Returns the type stored in the given type attribute.
"""
function mlirTypeAttrGetValue(attr)
    @ccall mlir_c.mlirTypeAttrGetValue(attr::MlirAttribute)::MlirType
end

"""
    mlirTypeAttrGetTypeID()

Returns the typeID of a Type attribute.
"""
function mlirTypeAttrGetTypeID()
    @ccall mlir_c.mlirTypeAttrGetTypeID()::MlirTypeID
end

"""
    mlirAttributeIsAUnit(attr)

Checks whether the given attribute is a unit attribute.
"""
function mlirAttributeIsAUnit(attr)
    @ccall mlir_c.mlirAttributeIsAUnit(attr::MlirAttribute)::Bool
end

"""
    mlirUnitAttrGet(ctx)

Creates a unit attribute in the given context.
"""
function mlirUnitAttrGet(ctx)
    @ccall mlir_c.mlirUnitAttrGet(ctx::MlirContext)::MlirAttribute
end

"""
    mlirUnitAttrGetTypeID()

Returns the typeID of a Unit attribute.
"""
function mlirUnitAttrGetTypeID()
    @ccall mlir_c.mlirUnitAttrGetTypeID()::MlirTypeID
end

"""
    mlirAttributeIsAElements(attr)

Checks whether the given attribute is an elements attribute.
"""
function mlirAttributeIsAElements(attr)
    @ccall mlir_c.mlirAttributeIsAElements(attr::MlirAttribute)::Bool
end

"""
    mlirElementsAttrGetValue(attr, rank, idxs)

Returns the element at the given rank-dimensional index.
"""
function mlirElementsAttrGetValue(attr, rank, idxs)
    @ccall mlir_c.mlirElementsAttrGetValue(
        attr::MlirAttribute, rank::Cptrdiff_t, idxs::Ptr{UInt64}
    )::MlirAttribute
end

"""
    mlirElementsAttrIsValidIndex(attr, rank, idxs)

Checks whether the given rank-dimensional index is valid in the given elements attribute.
"""
function mlirElementsAttrIsValidIndex(attr, rank, idxs)
    @ccall mlir_c.mlirElementsAttrIsValidIndex(
        attr::MlirAttribute, rank::Cptrdiff_t, idxs::Ptr{UInt64}
    )::Bool
end

"""
    mlirElementsAttrGetNumElements(attr)

Gets the total number of elements in the given elements attribute. In order to iterate over the attribute, obtain its type, which must be a statically shaped type and use its sizes to build a multi-dimensional index.
"""
function mlirElementsAttrGetNumElements(attr)
    @ccall mlir_c.mlirElementsAttrGetNumElements(attr::MlirAttribute)::Int64
end

function mlirDenseArrayAttrGetTypeID()
    @ccall mlir_c.mlirDenseArrayAttrGetTypeID()::MlirTypeID
end

"""
    mlirAttributeIsADenseBoolArray(attr)

Checks whether the given attribute is a dense array attribute.
"""
function mlirAttributeIsADenseBoolArray(attr)
    @ccall mlir_c.mlirAttributeIsADenseBoolArray(attr::MlirAttribute)::Bool
end

function mlirAttributeIsADenseI8Array(attr)
    @ccall mlir_c.mlirAttributeIsADenseI8Array(attr::MlirAttribute)::Bool
end

function mlirAttributeIsADenseI16Array(attr)
    @ccall mlir_c.mlirAttributeIsADenseI16Array(attr::MlirAttribute)::Bool
end

function mlirAttributeIsADenseI32Array(attr)
    @ccall mlir_c.mlirAttributeIsADenseI32Array(attr::MlirAttribute)::Bool
end

function mlirAttributeIsADenseI64Array(attr)
    @ccall mlir_c.mlirAttributeIsADenseI64Array(attr::MlirAttribute)::Bool
end

function mlirAttributeIsADenseF32Array(attr)
    @ccall mlir_c.mlirAttributeIsADenseF32Array(attr::MlirAttribute)::Bool
end

function mlirAttributeIsADenseF64Array(attr)
    @ccall mlir_c.mlirAttributeIsADenseF64Array(attr::MlirAttribute)::Bool
end

"""
    mlirDenseBoolArrayGet(ctx, size, values)

Create a dense array attribute with the given elements.
"""
function mlirDenseBoolArrayGet(ctx, size, values)
    @ccall mlir_c.mlirDenseBoolArrayGet(
        ctx::MlirContext, size::Cptrdiff_t, values::Ptr{Cint}
    )::MlirAttribute
end

function mlirDenseI8ArrayGet(ctx, size, values)
    @ccall mlir_c.mlirDenseI8ArrayGet(
        ctx::MlirContext, size::Cptrdiff_t, values::Ptr{Int8}
    )::MlirAttribute
end

function mlirDenseI16ArrayGet(ctx, size, values)
    @ccall mlir_c.mlirDenseI16ArrayGet(
        ctx::MlirContext, size::Cptrdiff_t, values::Ptr{Int16}
    )::MlirAttribute
end

function mlirDenseI32ArrayGet(ctx, size, values)
    @ccall mlir_c.mlirDenseI32ArrayGet(
        ctx::MlirContext, size::Cptrdiff_t, values::Ptr{Int32}
    )::MlirAttribute
end

function mlirDenseI64ArrayGet(ctx, size, values)
    @ccall mlir_c.mlirDenseI64ArrayGet(
        ctx::MlirContext, size::Cptrdiff_t, values::Ptr{Int64}
    )::MlirAttribute
end

function mlirDenseF32ArrayGet(ctx, size, values)
    @ccall mlir_c.mlirDenseF32ArrayGet(
        ctx::MlirContext, size::Cptrdiff_t, values::Ptr{Cfloat}
    )::MlirAttribute
end

function mlirDenseF64ArrayGet(ctx, size, values)
    @ccall mlir_c.mlirDenseF64ArrayGet(
        ctx::MlirContext, size::Cptrdiff_t, values::Ptr{Cdouble}
    )::MlirAttribute
end

"""
    mlirDenseArrayGetNumElements(attr)

Get the size of a dense array.
"""
function mlirDenseArrayGetNumElements(attr)
    @ccall mlir_c.mlirDenseArrayGetNumElements(attr::MlirAttribute)::Cptrdiff_t
end

"""
    mlirDenseBoolArrayGetElement(attr, pos)

Get an element of a dense array.
"""
function mlirDenseBoolArrayGetElement(attr, pos)
    @ccall mlir_c.mlirDenseBoolArrayGetElement(attr::MlirAttribute, pos::Cptrdiff_t)::Bool
end

function mlirDenseI8ArrayGetElement(attr, pos)
    @ccall mlir_c.mlirDenseI8ArrayGetElement(attr::MlirAttribute, pos::Cptrdiff_t)::Int8
end

function mlirDenseI16ArrayGetElement(attr, pos)
    @ccall mlir_c.mlirDenseI16ArrayGetElement(attr::MlirAttribute, pos::Cptrdiff_t)::Int16
end

function mlirDenseI32ArrayGetElement(attr, pos)
    @ccall mlir_c.mlirDenseI32ArrayGetElement(attr::MlirAttribute, pos::Cptrdiff_t)::Int32
end

function mlirDenseI64ArrayGetElement(attr, pos)
    @ccall mlir_c.mlirDenseI64ArrayGetElement(attr::MlirAttribute, pos::Cptrdiff_t)::Int64
end

function mlirDenseF32ArrayGetElement(attr, pos)
    @ccall mlir_c.mlirDenseF32ArrayGetElement(attr::MlirAttribute, pos::Cptrdiff_t)::Cfloat
end

function mlirDenseF64ArrayGetElement(attr, pos)
    @ccall mlir_c.mlirDenseF64ArrayGetElement(attr::MlirAttribute, pos::Cptrdiff_t)::Cdouble
end

"""
    mlirAttributeIsADenseElements(attr)

Checks whether the given attribute is a dense elements attribute.
"""
function mlirAttributeIsADenseElements(attr)
    @ccall mlir_c.mlirAttributeIsADenseElements(attr::MlirAttribute)::Bool
end

function mlirAttributeIsADenseIntElements(attr)
    @ccall mlir_c.mlirAttributeIsADenseIntElements(attr::MlirAttribute)::Bool
end

function mlirAttributeIsADenseFPElements(attr)
    @ccall mlir_c.mlirAttributeIsADenseFPElements(attr::MlirAttribute)::Bool
end

"""
    mlirDenseIntOrFPElementsAttrGetTypeID()

Returns the typeID of an DenseIntOrFPElements attribute.
"""
function mlirDenseIntOrFPElementsAttrGetTypeID()
    @ccall mlir_c.mlirDenseIntOrFPElementsAttrGetTypeID()::MlirTypeID
end

"""
    mlirDenseElementsAttrGet(shapedType, numElements, elements)

Creates a dense elements attribute with the given Shaped type and elements in the same context as the type.
"""
function mlirDenseElementsAttrGet(shapedType, numElements, elements)
    @ccall mlir_c.mlirDenseElementsAttrGet(
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
    @ccall mlir_c.mlirDenseElementsAttrRawBufferGet(
        shapedType::MlirType, rawBufferSize::Csize_t, rawBuffer::Ptr{Cvoid}
    )::MlirAttribute
end

"""
    mlirDenseElementsAttrSplatGet(shapedType, element)

Creates a dense elements attribute with the given Shaped type containing a single replicated element (splat).
"""
function mlirDenseElementsAttrSplatGet(shapedType, element)
    @ccall mlir_c.mlirDenseElementsAttrSplatGet(
        shapedType::MlirType, element::MlirAttribute
    )::MlirAttribute
end

function mlirDenseElementsAttrBoolSplatGet(shapedType, element)
    @ccall mlir_c.mlirDenseElementsAttrBoolSplatGet(
        shapedType::MlirType, element::Bool
    )::MlirAttribute
end

function mlirDenseElementsAttrUInt8SplatGet(shapedType, element)
    @ccall mlir_c.mlirDenseElementsAttrUInt8SplatGet(
        shapedType::MlirType, element::UInt8
    )::MlirAttribute
end

function mlirDenseElementsAttrInt8SplatGet(shapedType, element)
    @ccall mlir_c.mlirDenseElementsAttrInt8SplatGet(
        shapedType::MlirType, element::Int8
    )::MlirAttribute
end

function mlirDenseElementsAttrUInt32SplatGet(shapedType, element)
    @ccall mlir_c.mlirDenseElementsAttrUInt32SplatGet(
        shapedType::MlirType, element::UInt32
    )::MlirAttribute
end

function mlirDenseElementsAttrInt32SplatGet(shapedType, element)
    @ccall mlir_c.mlirDenseElementsAttrInt32SplatGet(
        shapedType::MlirType, element::Int32
    )::MlirAttribute
end

function mlirDenseElementsAttrUInt64SplatGet(shapedType, element)
    @ccall mlir_c.mlirDenseElementsAttrUInt64SplatGet(
        shapedType::MlirType, element::UInt64
    )::MlirAttribute
end

function mlirDenseElementsAttrInt64SplatGet(shapedType, element)
    @ccall mlir_c.mlirDenseElementsAttrInt64SplatGet(
        shapedType::MlirType, element::Int64
    )::MlirAttribute
end

function mlirDenseElementsAttrFloatSplatGet(shapedType, element)
    @ccall mlir_c.mlirDenseElementsAttrFloatSplatGet(
        shapedType::MlirType, element::Cfloat
    )::MlirAttribute
end

function mlirDenseElementsAttrDoubleSplatGet(shapedType, element)
    @ccall mlir_c.mlirDenseElementsAttrDoubleSplatGet(
        shapedType::MlirType, element::Cdouble
    )::MlirAttribute
end

"""
    mlirDenseElementsAttrBoolGet(shapedType, numElements, elements)

Creates a dense elements attribute with the given shaped type from elements of a specific type. Expects the element type of the shaped type to match the data element type.
"""
function mlirDenseElementsAttrBoolGet(shapedType, numElements, elements)
    @ccall mlir_c.mlirDenseElementsAttrBoolGet(
        shapedType::MlirType, numElements::Cptrdiff_t, elements::Ptr{Cint}
    )::MlirAttribute
end

function mlirDenseElementsAttrUInt8Get(shapedType, numElements, elements)
    @ccall mlir_c.mlirDenseElementsAttrUInt8Get(
        shapedType::MlirType, numElements::Cptrdiff_t, elements::Ptr{UInt8}
    )::MlirAttribute
end

function mlirDenseElementsAttrInt8Get(shapedType, numElements, elements)
    @ccall mlir_c.mlirDenseElementsAttrInt8Get(
        shapedType::MlirType, numElements::Cptrdiff_t, elements::Ptr{Int8}
    )::MlirAttribute
end

function mlirDenseElementsAttrUInt16Get(shapedType, numElements, elements)
    @ccall mlir_c.mlirDenseElementsAttrUInt16Get(
        shapedType::MlirType, numElements::Cptrdiff_t, elements::Ptr{UInt16}
    )::MlirAttribute
end

function mlirDenseElementsAttrInt16Get(shapedType, numElements, elements)
    @ccall mlir_c.mlirDenseElementsAttrInt16Get(
        shapedType::MlirType, numElements::Cptrdiff_t, elements::Ptr{Int16}
    )::MlirAttribute
end

function mlirDenseElementsAttrUInt32Get(shapedType, numElements, elements)
    @ccall mlir_c.mlirDenseElementsAttrUInt32Get(
        shapedType::MlirType, numElements::Cptrdiff_t, elements::Ptr{UInt32}
    )::MlirAttribute
end

function mlirDenseElementsAttrInt32Get(shapedType, numElements, elements)
    @ccall mlir_c.mlirDenseElementsAttrInt32Get(
        shapedType::MlirType, numElements::Cptrdiff_t, elements::Ptr{Int32}
    )::MlirAttribute
end

function mlirDenseElementsAttrUInt64Get(shapedType, numElements, elements)
    @ccall mlir_c.mlirDenseElementsAttrUInt64Get(
        shapedType::MlirType, numElements::Cptrdiff_t, elements::Ptr{UInt64}
    )::MlirAttribute
end

function mlirDenseElementsAttrInt64Get(shapedType, numElements, elements)
    @ccall mlir_c.mlirDenseElementsAttrInt64Get(
        shapedType::MlirType, numElements::Cptrdiff_t, elements::Ptr{Int64}
    )::MlirAttribute
end

function mlirDenseElementsAttrFloatGet(shapedType, numElements, elements)
    @ccall mlir_c.mlirDenseElementsAttrFloatGet(
        shapedType::MlirType, numElements::Cptrdiff_t, elements::Ptr{Cfloat}
    )::MlirAttribute
end

function mlirDenseElementsAttrDoubleGet(shapedType, numElements, elements)
    @ccall mlir_c.mlirDenseElementsAttrDoubleGet(
        shapedType::MlirType, numElements::Cptrdiff_t, elements::Ptr{Cdouble}
    )::MlirAttribute
end

function mlirDenseElementsAttrBFloat16Get(shapedType, numElements, elements)
    @ccall mlir_c.mlirDenseElementsAttrBFloat16Get(
        shapedType::MlirType, numElements::Cptrdiff_t, elements::Ptr{UInt16}
    )::MlirAttribute
end

function mlirDenseElementsAttrFloat16Get(shapedType, numElements, elements)
    @ccall mlir_c.mlirDenseElementsAttrFloat16Get(
        shapedType::MlirType, numElements::Cptrdiff_t, elements::Ptr{UInt16}
    )::MlirAttribute
end

"""
    mlirDenseElementsAttrStringGet(shapedType, numElements, strs)

Creates a dense elements attribute with the given shaped type from string elements.
"""
function mlirDenseElementsAttrStringGet(shapedType, numElements, strs)
    @ccall mlir_c.mlirDenseElementsAttrStringGet(
        shapedType::MlirType, numElements::Cptrdiff_t, strs::Ptr{MlirStringRef}
    )::MlirAttribute
end

"""
    mlirDenseElementsAttrReshapeGet(attr, shapedType)

Creates a dense elements attribute that has the same data as the given dense elements attribute and a different shaped type. The new type must have the same total number of elements.
"""
function mlirDenseElementsAttrReshapeGet(attr, shapedType)
    @ccall mlir_c.mlirDenseElementsAttrReshapeGet(
        attr::MlirAttribute, shapedType::MlirType
    )::MlirAttribute
end

"""
    mlirDenseElementsAttrIsSplat(attr)

Checks whether the given dense elements attribute contains a single replicated value (splat).
"""
function mlirDenseElementsAttrIsSplat(attr)
    @ccall mlir_c.mlirDenseElementsAttrIsSplat(attr::MlirAttribute)::Bool
end

"""
    mlirDenseElementsAttrGetSplatValue(attr)

Returns the single replicated value (splat) of a specific type contained by the given dense elements attribute.
"""
function mlirDenseElementsAttrGetSplatValue(attr)
    @ccall mlir_c.mlirDenseElementsAttrGetSplatValue(attr::MlirAttribute)::MlirAttribute
end

function mlirDenseElementsAttrGetBoolSplatValue(attr)
    @ccall mlir_c.mlirDenseElementsAttrGetBoolSplatValue(attr::MlirAttribute)::Cint
end

function mlirDenseElementsAttrGetInt8SplatValue(attr)
    @ccall mlir_c.mlirDenseElementsAttrGetInt8SplatValue(attr::MlirAttribute)::Int8
end

function mlirDenseElementsAttrGetUInt8SplatValue(attr)
    @ccall mlir_c.mlirDenseElementsAttrGetUInt8SplatValue(attr::MlirAttribute)::UInt8
end

function mlirDenseElementsAttrGetInt32SplatValue(attr)
    @ccall mlir_c.mlirDenseElementsAttrGetInt32SplatValue(attr::MlirAttribute)::Int32
end

function mlirDenseElementsAttrGetUInt32SplatValue(attr)
    @ccall mlir_c.mlirDenseElementsAttrGetUInt32SplatValue(attr::MlirAttribute)::UInt32
end

function mlirDenseElementsAttrGetInt64SplatValue(attr)
    @ccall mlir_c.mlirDenseElementsAttrGetInt64SplatValue(attr::MlirAttribute)::Int64
end

function mlirDenseElementsAttrGetUInt64SplatValue(attr)
    @ccall mlir_c.mlirDenseElementsAttrGetUInt64SplatValue(attr::MlirAttribute)::UInt64
end

function mlirDenseElementsAttrGetFloatSplatValue(attr)
    @ccall mlir_c.mlirDenseElementsAttrGetFloatSplatValue(attr::MlirAttribute)::Cfloat
end

function mlirDenseElementsAttrGetDoubleSplatValue(attr)
    @ccall mlir_c.mlirDenseElementsAttrGetDoubleSplatValue(attr::MlirAttribute)::Cdouble
end

function mlirDenseElementsAttrGetStringSplatValue(attr)
    @ccall mlir_c.mlirDenseElementsAttrGetStringSplatValue(
        attr::MlirAttribute
    )::MlirStringRef
end

"""
    mlirDenseElementsAttrGetBoolValue(attr, pos)

Returns the pos-th value (flat contiguous indexing) of a specific type contained by the given dense elements attribute.
"""
function mlirDenseElementsAttrGetBoolValue(attr, pos)
    @ccall mlir_c.mlirDenseElementsAttrGetBoolValue(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Bool
end

function mlirDenseElementsAttrGetInt8Value(attr, pos)
    @ccall mlir_c.mlirDenseElementsAttrGetInt8Value(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int8
end

function mlirDenseElementsAttrGetUInt8Value(attr, pos)
    @ccall mlir_c.mlirDenseElementsAttrGetUInt8Value(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::UInt8
end

function mlirDenseElementsAttrGetInt16Value(attr, pos)
    @ccall mlir_c.mlirDenseElementsAttrGetInt16Value(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int16
end

function mlirDenseElementsAttrGetUInt16Value(attr, pos)
    @ccall mlir_c.mlirDenseElementsAttrGetUInt16Value(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::UInt16
end

function mlirDenseElementsAttrGetInt32Value(attr, pos)
    @ccall mlir_c.mlirDenseElementsAttrGetInt32Value(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int32
end

function mlirDenseElementsAttrGetUInt32Value(attr, pos)
    @ccall mlir_c.mlirDenseElementsAttrGetUInt32Value(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::UInt32
end

function mlirDenseElementsAttrGetInt64Value(attr, pos)
    @ccall mlir_c.mlirDenseElementsAttrGetInt64Value(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function mlirDenseElementsAttrGetUInt64Value(attr, pos)
    @ccall mlir_c.mlirDenseElementsAttrGetUInt64Value(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::UInt64
end

function mlirDenseElementsAttrGetIndexValue(attr, pos)
    @ccall mlir_c.mlirDenseElementsAttrGetIndexValue(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::UInt64
end

function mlirDenseElementsAttrGetFloatValue(attr, pos)
    @ccall mlir_c.mlirDenseElementsAttrGetFloatValue(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Cfloat
end

function mlirDenseElementsAttrGetDoubleValue(attr, pos)
    @ccall mlir_c.mlirDenseElementsAttrGetDoubleValue(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Cdouble
end

function mlirDenseElementsAttrGetStringValue(attr, pos)
    @ccall mlir_c.mlirDenseElementsAttrGetStringValue(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::MlirStringRef
end

"""
    mlirDenseElementsAttrGetRawData(attr)

Returns the raw data of the given dense elements attribute.
"""
function mlirDenseElementsAttrGetRawData(attr)
    @ccall mlir_c.mlirDenseElementsAttrGetRawData(attr::MlirAttribute)::Ptr{Cvoid}
end

function mlirAttributeIsADenseResourceElements(attr)
    @ccall mlir_c.mlirAttributeIsADenseResourceElements(attr::MlirAttribute)::Bool
end

"""
    mlirUnmanagedDenseResourceElementsAttrGet(shapedType, name, data, dataLength, dataAlignment, dataIsMutable, deleter, userData)

Unlike the typed accessors below, constructs the attribute with a raw data buffer and no type/alignment checking. Use a more strongly typed accessor if possible. If dataIsMutable is false, then an immutable AsmResourceBlob will be created and that passed data contents will be treated as const. If the deleter is non NULL, then it will be called when the data buffer can no longer be accessed (passing userData to it).
"""
function mlirUnmanagedDenseResourceElementsAttrGet(
    shapedType, name, data, dataLength, dataAlignment, dataIsMutable, deleter, userData
)
    @ccall mlir_c.mlirUnmanagedDenseResourceElementsAttrGet(
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

function mlirUnmanagedDenseBoolResourceElementsAttrGet(
    shapedType, name, numElements, elements
)
    @ccall mlir_c.mlirUnmanagedDenseBoolResourceElementsAttrGet(
        shapedType::MlirType,
        name::MlirStringRef,
        numElements::Cptrdiff_t,
        elements::Ptr{Cint},
    )::MlirAttribute
end

function mlirUnmanagedDenseUInt8ResourceElementsAttrGet(
    shapedType, name, numElements, elements
)
    @ccall mlir_c.mlirUnmanagedDenseUInt8ResourceElementsAttrGet(
        shapedType::MlirType,
        name::MlirStringRef,
        numElements::Cptrdiff_t,
        elements::Ptr{UInt8},
    )::MlirAttribute
end

function mlirUnmanagedDenseInt8ResourceElementsAttrGet(
    shapedType, name, numElements, elements
)
    @ccall mlir_c.mlirUnmanagedDenseInt8ResourceElementsAttrGet(
        shapedType::MlirType,
        name::MlirStringRef,
        numElements::Cptrdiff_t,
        elements::Ptr{Int8},
    )::MlirAttribute
end

function mlirUnmanagedDenseUInt16ResourceElementsAttrGet(
    shapedType, name, numElements, elements
)
    @ccall mlir_c.mlirUnmanagedDenseUInt16ResourceElementsAttrGet(
        shapedType::MlirType,
        name::MlirStringRef,
        numElements::Cptrdiff_t,
        elements::Ptr{UInt16},
    )::MlirAttribute
end

function mlirUnmanagedDenseInt16ResourceElementsAttrGet(
    shapedType, name, numElements, elements
)
    @ccall mlir_c.mlirUnmanagedDenseInt16ResourceElementsAttrGet(
        shapedType::MlirType,
        name::MlirStringRef,
        numElements::Cptrdiff_t,
        elements::Ptr{Int16},
    )::MlirAttribute
end

function mlirUnmanagedDenseUInt32ResourceElementsAttrGet(
    shapedType, name, numElements, elements
)
    @ccall mlir_c.mlirUnmanagedDenseUInt32ResourceElementsAttrGet(
        shapedType::MlirType,
        name::MlirStringRef,
        numElements::Cptrdiff_t,
        elements::Ptr{UInt32},
    )::MlirAttribute
end

function mlirUnmanagedDenseInt32ResourceElementsAttrGet(
    shapedType, name, numElements, elements
)
    @ccall mlir_c.mlirUnmanagedDenseInt32ResourceElementsAttrGet(
        shapedType::MlirType,
        name::MlirStringRef,
        numElements::Cptrdiff_t,
        elements::Ptr{Int32},
    )::MlirAttribute
end

function mlirUnmanagedDenseUInt64ResourceElementsAttrGet(
    shapedType, name, numElements, elements
)
    @ccall mlir_c.mlirUnmanagedDenseUInt64ResourceElementsAttrGet(
        shapedType::MlirType,
        name::MlirStringRef,
        numElements::Cptrdiff_t,
        elements::Ptr{UInt64},
    )::MlirAttribute
end

function mlirUnmanagedDenseInt64ResourceElementsAttrGet(
    shapedType, name, numElements, elements
)
    @ccall mlir_c.mlirUnmanagedDenseInt64ResourceElementsAttrGet(
        shapedType::MlirType,
        name::MlirStringRef,
        numElements::Cptrdiff_t,
        elements::Ptr{Int64},
    )::MlirAttribute
end

function mlirUnmanagedDenseFloatResourceElementsAttrGet(
    shapedType, name, numElements, elements
)
    @ccall mlir_c.mlirUnmanagedDenseFloatResourceElementsAttrGet(
        shapedType::MlirType,
        name::MlirStringRef,
        numElements::Cptrdiff_t,
        elements::Ptr{Cfloat},
    )::MlirAttribute
end

function mlirUnmanagedDenseDoubleResourceElementsAttrGet(
    shapedType, name, numElements, elements
)
    @ccall mlir_c.mlirUnmanagedDenseDoubleResourceElementsAttrGet(
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
    @ccall mlir_c.mlirDenseBoolResourceElementsAttrGetValue(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Bool
end

function mlirDenseInt8ResourceElementsAttrGetValue(attr, pos)
    @ccall mlir_c.mlirDenseInt8ResourceElementsAttrGetValue(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int8
end

function mlirDenseUInt8ResourceElementsAttrGetValue(attr, pos)
    @ccall mlir_c.mlirDenseUInt8ResourceElementsAttrGetValue(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::UInt8
end

function mlirDenseInt16ResourceElementsAttrGetValue(attr, pos)
    @ccall mlir_c.mlirDenseInt16ResourceElementsAttrGetValue(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int16
end

function mlirDenseUInt16ResourceElementsAttrGetValue(attr, pos)
    @ccall mlir_c.mlirDenseUInt16ResourceElementsAttrGetValue(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::UInt16
end

function mlirDenseInt32ResourceElementsAttrGetValue(attr, pos)
    @ccall mlir_c.mlirDenseInt32ResourceElementsAttrGetValue(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int32
end

function mlirDenseUInt32ResourceElementsAttrGetValue(attr, pos)
    @ccall mlir_c.mlirDenseUInt32ResourceElementsAttrGetValue(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::UInt32
end

function mlirDenseInt64ResourceElementsAttrGetValue(attr, pos)
    @ccall mlir_c.mlirDenseInt64ResourceElementsAttrGetValue(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function mlirDenseUInt64ResourceElementsAttrGetValue(attr, pos)
    @ccall mlir_c.mlirDenseUInt64ResourceElementsAttrGetValue(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::UInt64
end

function mlirDenseFloatResourceElementsAttrGetValue(attr, pos)
    @ccall mlir_c.mlirDenseFloatResourceElementsAttrGetValue(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Cfloat
end

function mlirDenseDoubleResourceElementsAttrGetValue(attr, pos)
    @ccall mlir_c.mlirDenseDoubleResourceElementsAttrGetValue(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Cdouble
end

"""
    mlirAttributeIsASparseElements(attr)

Checks whether the given attribute is a sparse elements attribute.
"""
function mlirAttributeIsASparseElements(attr)
    @ccall mlir_c.mlirAttributeIsASparseElements(attr::MlirAttribute)::Bool
end

"""
    mlirSparseElementsAttribute(shapedType, denseIndices, denseValues)

Creates a sparse elements attribute of the given shape from a list of indices and a list of associated values. Both lists are expected to be dense elements attributes with the same number of elements. The list of indices is expected to contain 64-bit integers. The attribute is created in the same context as the type.
"""
function mlirSparseElementsAttribute(shapedType, denseIndices, denseValues)
    @ccall mlir_c.mlirSparseElementsAttribute(
        shapedType::MlirType, denseIndices::MlirAttribute, denseValues::MlirAttribute
    )::MlirAttribute
end

"""
    mlirSparseElementsAttrGetIndices(attr)

Returns the dense elements attribute containing 64-bit integer indices of non-null elements in the given sparse elements attribute.
"""
function mlirSparseElementsAttrGetIndices(attr)
    @ccall mlir_c.mlirSparseElementsAttrGetIndices(attr::MlirAttribute)::MlirAttribute
end

"""
    mlirSparseElementsAttrGetValues(attr)

Returns the dense elements attribute containing the non-null elements in the given sparse elements attribute.
"""
function mlirSparseElementsAttrGetValues(attr)
    @ccall mlir_c.mlirSparseElementsAttrGetValues(attr::MlirAttribute)::MlirAttribute
end

"""
    mlirSparseElementsAttrGetTypeID()

Returns the typeID of a SparseElements attribute.
"""
function mlirSparseElementsAttrGetTypeID()
    @ccall mlir_c.mlirSparseElementsAttrGetTypeID()::MlirTypeID
end

function mlirAttributeIsAStridedLayout(attr)
    @ccall mlir_c.mlirAttributeIsAStridedLayout(attr::MlirAttribute)::Bool
end

function mlirStridedLayoutAttrGet(ctx, offset, numStrides, strides)
    @ccall mlir_c.mlirStridedLayoutAttrGet(
        ctx::MlirContext, offset::Int64, numStrides::Cptrdiff_t, strides::Ptr{Int64}
    )::MlirAttribute
end

function mlirStridedLayoutAttrGetOffset(attr)
    @ccall mlir_c.mlirStridedLayoutAttrGetOffset(attr::MlirAttribute)::Int64
end

function mlirStridedLayoutAttrGetNumStrides(attr)
    @ccall mlir_c.mlirStridedLayoutAttrGetNumStrides(attr::MlirAttribute)::Cptrdiff_t
end

function mlirStridedLayoutAttrGetStride(attr, pos)
    @ccall mlir_c.mlirStridedLayoutAttrGetStride(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

"""
    mlirStridedLayoutAttrGetTypeID()

Returns the typeID of a StridedLayout attribute.
"""
function mlirStridedLayoutAttrGetTypeID()
    @ccall mlir_c.mlirStridedLayoutAttrGetTypeID()::MlirTypeID
end

"""
    mlirIntegerTypeGetTypeID()

Returns the typeID of an Integer type.
"""
function mlirIntegerTypeGetTypeID()
    @ccall mlir_c.mlirIntegerTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAInteger(type)

Checks whether the given type is an integer type.
"""
function mlirTypeIsAInteger(type)
    @ccall mlir_c.mlirTypeIsAInteger(type::MlirType)::Bool
end

"""
    mlirIntegerTypeGet(ctx, bitwidth)

Creates a signless integer type of the given bitwidth in the context. The type is owned by the context.
"""
function mlirIntegerTypeGet(ctx, bitwidth)
    @ccall mlir_c.mlirIntegerTypeGet(ctx::MlirContext, bitwidth::Cuint)::MlirType
end

"""
    mlirIntegerTypeSignedGet(ctx, bitwidth)

Creates a signed integer type of the given bitwidth in the context. The type is owned by the context.
"""
function mlirIntegerTypeSignedGet(ctx, bitwidth)
    @ccall mlir_c.mlirIntegerTypeSignedGet(ctx::MlirContext, bitwidth::Cuint)::MlirType
end

"""
    mlirIntegerTypeUnsignedGet(ctx, bitwidth)

Creates an unsigned integer type of the given bitwidth in the context. The type is owned by the context.
"""
function mlirIntegerTypeUnsignedGet(ctx, bitwidth)
    @ccall mlir_c.mlirIntegerTypeUnsignedGet(ctx::MlirContext, bitwidth::Cuint)::MlirType
end

"""
    mlirIntegerTypeGetWidth(type)

Returns the bitwidth of an integer type.
"""
function mlirIntegerTypeGetWidth(type)
    @ccall mlir_c.mlirIntegerTypeGetWidth(type::MlirType)::Cuint
end

"""
    mlirIntegerTypeIsSignless(type)

Checks whether the given integer type is signless.
"""
function mlirIntegerTypeIsSignless(type)
    @ccall mlir_c.mlirIntegerTypeIsSignless(type::MlirType)::Bool
end

"""
    mlirIntegerTypeIsSigned(type)

Checks whether the given integer type is signed.
"""
function mlirIntegerTypeIsSigned(type)
    @ccall mlir_c.mlirIntegerTypeIsSigned(type::MlirType)::Bool
end

"""
    mlirIntegerTypeIsUnsigned(type)

Checks whether the given integer type is unsigned.
"""
function mlirIntegerTypeIsUnsigned(type)
    @ccall mlir_c.mlirIntegerTypeIsUnsigned(type::MlirType)::Bool
end

"""
    mlirIndexTypeGetTypeID()

Returns the typeID of an Index type.
"""
function mlirIndexTypeGetTypeID()
    @ccall mlir_c.mlirIndexTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAIndex(type)

Checks whether the given type is an index type.
"""
function mlirTypeIsAIndex(type)
    @ccall mlir_c.mlirTypeIsAIndex(type::MlirType)::Bool
end

"""
    mlirIndexTypeGet(ctx)

Creates an index type in the given context. The type is owned by the context.
"""
function mlirIndexTypeGet(ctx)
    @ccall mlir_c.mlirIndexTypeGet(ctx::MlirContext)::MlirType
end

"""
    mlirTypeIsAFloat(type)

Checks whether the given type is a floating-point type.
"""
function mlirTypeIsAFloat(type)
    @ccall mlir_c.mlirTypeIsAFloat(type::MlirType)::Bool
end

"""
    mlirFloatTypeGetWidth(type)

Returns the bitwidth of a floating-point type.
"""
function mlirFloatTypeGetWidth(type)
    @ccall mlir_c.mlirFloatTypeGetWidth(type::MlirType)::Cuint
end

"""
    mlirFloat4E2M1FNTypeGetTypeID()

Returns the typeID of an Float4E2M1FN type.
"""
function mlirFloat4E2M1FNTypeGetTypeID()
    @ccall mlir_c.mlirFloat4E2M1FNTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAFloat4E2M1FN(type)

Checks whether the given type is an f4E2M1FN type.
"""
function mlirTypeIsAFloat4E2M1FN(type)
    @ccall mlir_c.mlirTypeIsAFloat4E2M1FN(type::MlirType)::Bool
end

"""
    mlirFloat4E2M1FNTypeGet(ctx)

Creates an f4E2M1FN type in the given context. The type is owned by the context.
"""
function mlirFloat4E2M1FNTypeGet(ctx)
    @ccall mlir_c.mlirFloat4E2M1FNTypeGet(ctx::MlirContext)::MlirType
end

"""
    mlirFloat6E2M3FNTypeGetTypeID()

Returns the typeID of an Float6E2M3FN type.
"""
function mlirFloat6E2M3FNTypeGetTypeID()
    @ccall mlir_c.mlirFloat6E2M3FNTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAFloat6E2M3FN(type)

Checks whether the given type is an f6E2M3FN type.
"""
function mlirTypeIsAFloat6E2M3FN(type)
    @ccall mlir_c.mlirTypeIsAFloat6E2M3FN(type::MlirType)::Bool
end

"""
    mlirFloat6E2M3FNTypeGet(ctx)

Creates an f6E2M3FN type in the given context. The type is owned by the context.
"""
function mlirFloat6E2M3FNTypeGet(ctx)
    @ccall mlir_c.mlirFloat6E2M3FNTypeGet(ctx::MlirContext)::MlirType
end

"""
    mlirFloat6E3M2FNTypeGetTypeID()

Returns the typeID of an Float6E3M2FN type.
"""
function mlirFloat6E3M2FNTypeGetTypeID()
    @ccall mlir_c.mlirFloat6E3M2FNTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAFloat6E3M2FN(type)

Checks whether the given type is an f6E3M2FN type.
"""
function mlirTypeIsAFloat6E3M2FN(type)
    @ccall mlir_c.mlirTypeIsAFloat6E3M2FN(type::MlirType)::Bool
end

"""
    mlirFloat6E3M2FNTypeGet(ctx)

Creates an f6E3M2FN type in the given context. The type is owned by the context.
"""
function mlirFloat6E3M2FNTypeGet(ctx)
    @ccall mlir_c.mlirFloat6E3M2FNTypeGet(ctx::MlirContext)::MlirType
end

"""
    mlirFloat8E5M2TypeGetTypeID()

Returns the typeID of an Float8E5M2 type.
"""
function mlirFloat8E5M2TypeGetTypeID()
    @ccall mlir_c.mlirFloat8E5M2TypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAFloat8E5M2(type)

Checks whether the given type is an f8E5M2 type.
"""
function mlirTypeIsAFloat8E5M2(type)
    @ccall mlir_c.mlirTypeIsAFloat8E5M2(type::MlirType)::Bool
end

"""
    mlirFloat8E5M2TypeGet(ctx)

Creates an f8E5M2 type in the given context. The type is owned by the context.
"""
function mlirFloat8E5M2TypeGet(ctx)
    @ccall mlir_c.mlirFloat8E5M2TypeGet(ctx::MlirContext)::MlirType
end

"""
    mlirFloat8E4M3TypeGetTypeID()

Returns the typeID of an Float8E4M3 type.
"""
function mlirFloat8E4M3TypeGetTypeID()
    @ccall mlir_c.mlirFloat8E4M3TypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAFloat8E4M3(type)

Checks whether the given type is an f8E4M3 type.
"""
function mlirTypeIsAFloat8E4M3(type)
    @ccall mlir_c.mlirTypeIsAFloat8E4M3(type::MlirType)::Bool
end

"""
    mlirFloat8E4M3TypeGet(ctx)

Creates an f8E4M3 type in the given context. The type is owned by the context.
"""
function mlirFloat8E4M3TypeGet(ctx)
    @ccall mlir_c.mlirFloat8E4M3TypeGet(ctx::MlirContext)::MlirType
end

"""
    mlirFloat8E4M3FNTypeGetTypeID()

Returns the typeID of an Float8E4M3FN type.
"""
function mlirFloat8E4M3FNTypeGetTypeID()
    @ccall mlir_c.mlirFloat8E4M3FNTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAFloat8E4M3FN(type)

Checks whether the given type is an f8E4M3FN type.
"""
function mlirTypeIsAFloat8E4M3FN(type)
    @ccall mlir_c.mlirTypeIsAFloat8E4M3FN(type::MlirType)::Bool
end

"""
    mlirFloat8E4M3FNTypeGet(ctx)

Creates an f8E4M3FN type in the given context. The type is owned by the context.
"""
function mlirFloat8E4M3FNTypeGet(ctx)
    @ccall mlir_c.mlirFloat8E4M3FNTypeGet(ctx::MlirContext)::MlirType
end

"""
    mlirFloat8E5M2FNUZTypeGetTypeID()

Returns the typeID of an Float8E5M2FNUZ type.
"""
function mlirFloat8E5M2FNUZTypeGetTypeID()
    @ccall mlir_c.mlirFloat8E5M2FNUZTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAFloat8E5M2FNUZ(type)

Checks whether the given type is an f8E5M2FNUZ type.
"""
function mlirTypeIsAFloat8E5M2FNUZ(type)
    @ccall mlir_c.mlirTypeIsAFloat8E5M2FNUZ(type::MlirType)::Bool
end

"""
    mlirFloat8E5M2FNUZTypeGet(ctx)

Creates an f8E5M2FNUZ type in the given context. The type is owned by the context.
"""
function mlirFloat8E5M2FNUZTypeGet(ctx)
    @ccall mlir_c.mlirFloat8E5M2FNUZTypeGet(ctx::MlirContext)::MlirType
end

"""
    mlirFloat8E4M3FNUZTypeGetTypeID()

Returns the typeID of an Float8E4M3FNUZ type.
"""
function mlirFloat8E4M3FNUZTypeGetTypeID()
    @ccall mlir_c.mlirFloat8E4M3FNUZTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAFloat8E4M3FNUZ(type)

Checks whether the given type is an f8E4M3FNUZ type.
"""
function mlirTypeIsAFloat8E4M3FNUZ(type)
    @ccall mlir_c.mlirTypeIsAFloat8E4M3FNUZ(type::MlirType)::Bool
end

"""
    mlirFloat8E4M3FNUZTypeGet(ctx)

Creates an f8E4M3FNUZ type in the given context. The type is owned by the context.
"""
function mlirFloat8E4M3FNUZTypeGet(ctx)
    @ccall mlir_c.mlirFloat8E4M3FNUZTypeGet(ctx::MlirContext)::MlirType
end

"""
    mlirFloat8E4M3B11FNUZTypeGetTypeID()

Returns the typeID of an Float8E4M3B11FNUZ type.
"""
function mlirFloat8E4M3B11FNUZTypeGetTypeID()
    @ccall mlir_c.mlirFloat8E4M3B11FNUZTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAFloat8E4M3B11FNUZ(type)

Checks whether the given type is an f8E4M3B11FNUZ type.
"""
function mlirTypeIsAFloat8E4M3B11FNUZ(type)
    @ccall mlir_c.mlirTypeIsAFloat8E4M3B11FNUZ(type::MlirType)::Bool
end

"""
    mlirFloat8E4M3B11FNUZTypeGet(ctx)

Creates an f8E4M3B11FNUZ type in the given context. The type is owned by the context.
"""
function mlirFloat8E4M3B11FNUZTypeGet(ctx)
    @ccall mlir_c.mlirFloat8E4M3B11FNUZTypeGet(ctx::MlirContext)::MlirType
end

"""
    mlirFloat8E3M4TypeGetTypeID()

Returns the typeID of an Float8E3M4 type.
"""
function mlirFloat8E3M4TypeGetTypeID()
    @ccall mlir_c.mlirFloat8E3M4TypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAFloat8E3M4(type)

Checks whether the given type is an f8E3M4 type.
"""
function mlirTypeIsAFloat8E3M4(type)
    @ccall mlir_c.mlirTypeIsAFloat8E3M4(type::MlirType)::Bool
end

"""
    mlirFloat8E3M4TypeGet(ctx)

Creates an f8E3M4 type in the given context. The type is owned by the context.
"""
function mlirFloat8E3M4TypeGet(ctx)
    @ccall mlir_c.mlirFloat8E3M4TypeGet(ctx::MlirContext)::MlirType
end

"""
    mlirFloat8E8M0FNUTypeGetTypeID()

Returns the typeID of an Float8E8M0FNU type.
"""
function mlirFloat8E8M0FNUTypeGetTypeID()
    @ccall mlir_c.mlirFloat8E8M0FNUTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAFloat8E8M0FNU(type)

Checks whether the given type is an f8E8M0FNU type.
"""
function mlirTypeIsAFloat8E8M0FNU(type)
    @ccall mlir_c.mlirTypeIsAFloat8E8M0FNU(type::MlirType)::Bool
end

"""
    mlirFloat8E8M0FNUTypeGet(ctx)

Creates an f8E8M0FNU type in the given context. The type is owned by the context.
"""
function mlirFloat8E8M0FNUTypeGet(ctx)
    @ccall mlir_c.mlirFloat8E8M0FNUTypeGet(ctx::MlirContext)::MlirType
end

"""
    mlirBFloat16TypeGetTypeID()

Returns the typeID of an BFloat16 type.
"""
function mlirBFloat16TypeGetTypeID()
    @ccall mlir_c.mlirBFloat16TypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsABF16(type)

Checks whether the given type is a bf16 type.
"""
function mlirTypeIsABF16(type)
    @ccall mlir_c.mlirTypeIsABF16(type::MlirType)::Bool
end

"""
    mlirBF16TypeGet(ctx)

Creates a bf16 type in the given context. The type is owned by the context.
"""
function mlirBF16TypeGet(ctx)
    @ccall mlir_c.mlirBF16TypeGet(ctx::MlirContext)::MlirType
end

"""
    mlirFloat16TypeGetTypeID()

Returns the typeID of an Float16 type.
"""
function mlirFloat16TypeGetTypeID()
    @ccall mlir_c.mlirFloat16TypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAF16(type)

Checks whether the given type is an f16 type.
"""
function mlirTypeIsAF16(type)
    @ccall mlir_c.mlirTypeIsAF16(type::MlirType)::Bool
end

"""
    mlirF16TypeGet(ctx)

Creates an f16 type in the given context. The type is owned by the context.
"""
function mlirF16TypeGet(ctx)
    @ccall mlir_c.mlirF16TypeGet(ctx::MlirContext)::MlirType
end

"""
    mlirFloat32TypeGetTypeID()

Returns the typeID of an Float32 type.
"""
function mlirFloat32TypeGetTypeID()
    @ccall mlir_c.mlirFloat32TypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAF32(type)

Checks whether the given type is an f32 type.
"""
function mlirTypeIsAF32(type)
    @ccall mlir_c.mlirTypeIsAF32(type::MlirType)::Bool
end

"""
    mlirF32TypeGet(ctx)

Creates an f32 type in the given context. The type is owned by the context.
"""
function mlirF32TypeGet(ctx)
    @ccall mlir_c.mlirF32TypeGet(ctx::MlirContext)::MlirType
end

"""
    mlirFloat64TypeGetTypeID()

Returns the typeID of an Float64 type.
"""
function mlirFloat64TypeGetTypeID()
    @ccall mlir_c.mlirFloat64TypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAF64(type)

Checks whether the given type is an f64 type.
"""
function mlirTypeIsAF64(type)
    @ccall mlir_c.mlirTypeIsAF64(type::MlirType)::Bool
end

"""
    mlirF64TypeGet(ctx)

Creates a f64 type in the given context. The type is owned by the context.
"""
function mlirF64TypeGet(ctx)
    @ccall mlir_c.mlirF64TypeGet(ctx::MlirContext)::MlirType
end

"""
    mlirFloatTF32TypeGetTypeID()

Returns the typeID of a TF32 type.
"""
function mlirFloatTF32TypeGetTypeID()
    @ccall mlir_c.mlirFloatTF32TypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsATF32(type)

Checks whether the given type is an TF32 type.
"""
function mlirTypeIsATF32(type)
    @ccall mlir_c.mlirTypeIsATF32(type::MlirType)::Bool
end

"""
    mlirTF32TypeGet(ctx)

Creates a TF32 type in the given context. The type is owned by the context.
"""
function mlirTF32TypeGet(ctx)
    @ccall mlir_c.mlirTF32TypeGet(ctx::MlirContext)::MlirType
end

"""
    mlirNoneTypeGetTypeID()

Returns the typeID of an None type.
"""
function mlirNoneTypeGetTypeID()
    @ccall mlir_c.mlirNoneTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsANone(type)

Checks whether the given type is a None type.
"""
function mlirTypeIsANone(type)
    @ccall mlir_c.mlirTypeIsANone(type::MlirType)::Bool
end

"""
    mlirNoneTypeGet(ctx)

Creates a None type in the given context. The type is owned by the context.
"""
function mlirNoneTypeGet(ctx)
    @ccall mlir_c.mlirNoneTypeGet(ctx::MlirContext)::MlirType
end

"""
    mlirComplexTypeGetTypeID()

Returns the typeID of an Complex type.
"""
function mlirComplexTypeGetTypeID()
    @ccall mlir_c.mlirComplexTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAComplex(type)

Checks whether the given type is a Complex type.
"""
function mlirTypeIsAComplex(type)
    @ccall mlir_c.mlirTypeIsAComplex(type::MlirType)::Bool
end

"""
    mlirComplexTypeGet(elementType)

Creates a complex type with the given element type in the same context as the element type. The type is owned by the context.
"""
function mlirComplexTypeGet(elementType)
    @ccall mlir_c.mlirComplexTypeGet(elementType::MlirType)::MlirType
end

"""
    mlirComplexTypeGetElementType(type)

Returns the element type of the given complex type.
"""
function mlirComplexTypeGetElementType(type)
    @ccall mlir_c.mlirComplexTypeGetElementType(type::MlirType)::MlirType
end

"""
    mlirTypeIsAShaped(type)

Checks whether the given type is a Shaped type.
"""
function mlirTypeIsAShaped(type)
    @ccall mlir_c.mlirTypeIsAShaped(type::MlirType)::Bool
end

"""
    mlirShapedTypeGetElementType(type)

Returns the element type of the shaped type.
"""
function mlirShapedTypeGetElementType(type)
    @ccall mlir_c.mlirShapedTypeGetElementType(type::MlirType)::MlirType
end

"""
    mlirShapedTypeHasRank(type)

Checks whether the given shaped type is ranked.
"""
function mlirShapedTypeHasRank(type)
    @ccall mlir_c.mlirShapedTypeHasRank(type::MlirType)::Bool
end

"""
    mlirShapedTypeGetRank(type)

Returns the rank of the given ranked shaped type.
"""
function mlirShapedTypeGetRank(type)
    @ccall mlir_c.mlirShapedTypeGetRank(type::MlirType)::Int64
end

"""
    mlirShapedTypeHasStaticShape(type)

Checks whether the given shaped type has a static shape.
"""
function mlirShapedTypeHasStaticShape(type)
    @ccall mlir_c.mlirShapedTypeHasStaticShape(type::MlirType)::Bool
end

"""
    mlirShapedTypeIsDynamicDim(type, dim)

Checks whether the dim-th dimension of the given shaped type is dynamic.
"""
function mlirShapedTypeIsDynamicDim(type, dim)
    @ccall mlir_c.mlirShapedTypeIsDynamicDim(type::MlirType, dim::Cptrdiff_t)::Bool
end

"""
    mlirShapedTypeIsStaticDim(type, dim)

Checks whether the dim-th dimension of the given shaped type is static.
"""
function mlirShapedTypeIsStaticDim(type, dim)
    @ccall mlir_c.mlirShapedTypeIsStaticDim(type::MlirType, dim::Cptrdiff_t)::Bool
end

"""
    mlirShapedTypeGetDimSize(type, dim)

Returns the dim-th dimension of the given ranked shaped type.
"""
function mlirShapedTypeGetDimSize(type, dim)
    @ccall mlir_c.mlirShapedTypeGetDimSize(type::MlirType, dim::Cptrdiff_t)::Int64
end

"""
    mlirShapedTypeIsDynamicSize(size)

Checks whether the given value is used as a placeholder for dynamic sizes in shaped types.
"""
function mlirShapedTypeIsDynamicSize(size)
    @ccall mlir_c.mlirShapedTypeIsDynamicSize(size::Int64)::Bool
end

"""
    mlirShapedTypeIsStaticSize(size)

Checks whether the given shaped type dimension value is statically-sized.
"""
function mlirShapedTypeIsStaticSize(size)
    @ccall mlir_c.mlirShapedTypeIsStaticSize(size::Int64)::Bool
end

"""
    mlirShapedTypeGetDynamicSize()

Returns the value indicating a dynamic size in a shaped type. Prefer [`mlirShapedTypeIsDynamicSize`](@ref) and [`mlirShapedTypeIsStaticSize`](@ref) to direct comparisons with this value.
"""
function mlirShapedTypeGetDynamicSize()
    @ccall mlir_c.mlirShapedTypeGetDynamicSize()::Int64
end

"""
    mlirShapedTypeIsDynamicStrideOrOffset(val)

Checks whether the given value is used as a placeholder for dynamic strides and offsets in shaped types.
"""
function mlirShapedTypeIsDynamicStrideOrOffset(val)
    @ccall mlir_c.mlirShapedTypeIsDynamicStrideOrOffset(val::Int64)::Bool
end

"""
    mlirShapedTypeIsStaticStrideOrOffset(val)

Checks whether the given dimension value of a stride or an offset is statically-sized.
"""
function mlirShapedTypeIsStaticStrideOrOffset(val)
    @ccall mlir_c.mlirShapedTypeIsStaticStrideOrOffset(val::Int64)::Bool
end

"""
    mlirShapedTypeGetDynamicStrideOrOffset()

Returns the value indicating a dynamic stride or offset in a shaped type. Prefer [`mlirShapedTypeIsDynamicStrideOrOffset`](@ref) and [`mlirShapedTypeIsStaticStrideOrOffset`](@ref) to direct comparisons with this value.
"""
function mlirShapedTypeGetDynamicStrideOrOffset()
    @ccall mlir_c.mlirShapedTypeGetDynamicStrideOrOffset()::Int64
end

"""
    mlirVectorTypeGetTypeID()

Returns the typeID of an Vector type.
"""
function mlirVectorTypeGetTypeID()
    @ccall mlir_c.mlirVectorTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAVector(type)

Checks whether the given type is a Vector type.
"""
function mlirTypeIsAVector(type)
    @ccall mlir_c.mlirTypeIsAVector(type::MlirType)::Bool
end

"""
    mlirVectorTypeGet(rank, shape, elementType)

Creates a vector type of the shape identified by its rank and dimensions, with the given element type in the same context as the element type. The type is owned by the context.
"""
function mlirVectorTypeGet(rank, shape, elementType)
    @ccall mlir_c.mlirVectorTypeGet(
        rank::Cptrdiff_t, shape::Ptr{Int64}, elementType::MlirType
    )::MlirType
end

"""
    mlirVectorTypeGetChecked(loc, rank, shape, elementType)

Same as "[`mlirVectorTypeGet`](@ref)" but returns a nullptr wrapping [`MlirType`](@ref) on illegal arguments, emitting appropriate diagnostics.
"""
function mlirVectorTypeGetChecked(loc, rank, shape, elementType)
    @ccall mlir_c.mlirVectorTypeGetChecked(
        loc::MlirLocation, rank::Cptrdiff_t, shape::Ptr{Int64}, elementType::MlirType
    )::MlirType
end

"""
    mlirVectorTypeGetScalable(rank, shape, scalable, elementType)

Creates a scalable vector type with the shape identified by its rank and dimensions. A subset of dimensions may be marked as scalable via the corresponding flag list, which is expected to have as many entries as the rank of the vector. The vector is created in the same context as the element type.
"""
function mlirVectorTypeGetScalable(rank, shape, scalable, elementType)
    @ccall mlir_c.mlirVectorTypeGetScalable(
        rank::Cptrdiff_t, shape::Ptr{Int64}, scalable::Ptr{Bool}, elementType::MlirType
    )::MlirType
end

"""
    mlirVectorTypeGetScalableChecked(loc, rank, shape, scalable, elementType)

Same as "[`mlirVectorTypeGetScalable`](@ref)" but returns a nullptr wrapping [`MlirType`](@ref) on illegal arguments, emitting appropriate diagnostics.
"""
function mlirVectorTypeGetScalableChecked(loc, rank, shape, scalable, elementType)
    @ccall mlir_c.mlirVectorTypeGetScalableChecked(
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
    @ccall mlir_c.mlirVectorTypeIsScalable(type::MlirType)::Bool
end

"""
    mlirVectorTypeIsDimScalable(type, dim)

Checks whether the "dim"-th dimension of the given vector is scalable.
"""
function mlirVectorTypeIsDimScalable(type, dim)
    @ccall mlir_c.mlirVectorTypeIsDimScalable(type::MlirType, dim::Cptrdiff_t)::Bool
end

"""
    mlirTypeIsATensor(type)

Checks whether the given type is a Tensor type.
"""
function mlirTypeIsATensor(type)
    @ccall mlir_c.mlirTypeIsATensor(type::MlirType)::Bool
end

"""
    mlirRankedTensorTypeGetTypeID()

Returns the typeID of an RankedTensor type.
"""
function mlirRankedTensorTypeGetTypeID()
    @ccall mlir_c.mlirRankedTensorTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsARankedTensor(type)

Checks whether the given type is a ranked tensor type.
"""
function mlirTypeIsARankedTensor(type)
    @ccall mlir_c.mlirTypeIsARankedTensor(type::MlirType)::Bool
end

"""
    mlirUnrankedTensorTypeGetTypeID()

Returns the typeID of an UnrankedTensor type.
"""
function mlirUnrankedTensorTypeGetTypeID()
    @ccall mlir_c.mlirUnrankedTensorTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAUnrankedTensor(type)

Checks whether the given type is an unranked tensor type.
"""
function mlirTypeIsAUnrankedTensor(type)
    @ccall mlir_c.mlirTypeIsAUnrankedTensor(type::MlirType)::Bool
end

"""
    mlirRankedTensorTypeGet(rank, shape, elementType, encoding)

Creates a tensor type of a fixed rank with the given shape, element type, and optional encoding in the same context as the element type. The type is owned by the context. Tensor types without any specific encoding field should assign [`mlirAttributeGetNull`](@ref)() to this parameter.
"""
function mlirRankedTensorTypeGet(rank, shape, elementType, encoding)
    @ccall mlir_c.mlirRankedTensorTypeGet(
        rank::Cptrdiff_t, shape::Ptr{Int64}, elementType::MlirType, encoding::MlirAttribute
    )::MlirType
end

"""
    mlirRankedTensorTypeGetChecked(loc, rank, shape, elementType, encoding)

Same as "[`mlirRankedTensorTypeGet`](@ref)" but returns a nullptr wrapping [`MlirType`](@ref) on illegal arguments, emitting appropriate diagnostics.
"""
function mlirRankedTensorTypeGetChecked(loc, rank, shape, elementType, encoding)
    @ccall mlir_c.mlirRankedTensorTypeGetChecked(
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
    @ccall mlir_c.mlirRankedTensorTypeGetEncoding(type::MlirType)::MlirAttribute
end

"""
    mlirUnrankedTensorTypeGet(elementType)

Creates an unranked tensor type with the given element type in the same context as the element type. The type is owned by the context.
"""
function mlirUnrankedTensorTypeGet(elementType)
    @ccall mlir_c.mlirUnrankedTensorTypeGet(elementType::MlirType)::MlirType
end

"""
    mlirUnrankedTensorTypeGetChecked(loc, elementType)

Same as "[`mlirUnrankedTensorTypeGet`](@ref)" but returns a nullptr wrapping [`MlirType`](@ref) on illegal arguments, emitting appropriate diagnostics.
"""
function mlirUnrankedTensorTypeGetChecked(loc, elementType)
    @ccall mlir_c.mlirUnrankedTensorTypeGetChecked(
        loc::MlirLocation, elementType::MlirType
    )::MlirType
end

"""
    mlirMemRefTypeGetTypeID()

Returns the typeID of an MemRef type.
"""
function mlirMemRefTypeGetTypeID()
    @ccall mlir_c.mlirMemRefTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAMemRef(type)

Checks whether the given type is a MemRef type.
"""
function mlirTypeIsAMemRef(type)
    @ccall mlir_c.mlirTypeIsAMemRef(type::MlirType)::Bool
end

"""
    mlirUnrankedMemRefTypeGetTypeID()

Returns the typeID of an UnrankedMemRef type.
"""
function mlirUnrankedMemRefTypeGetTypeID()
    @ccall mlir_c.mlirUnrankedMemRefTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAUnrankedMemRef(type)

Checks whether the given type is an UnrankedMemRef type.
"""
function mlirTypeIsAUnrankedMemRef(type)
    @ccall mlir_c.mlirTypeIsAUnrankedMemRef(type::MlirType)::Bool
end

"""
    mlirMemRefTypeGet(elementType, rank, shape, layout, memorySpace)

Creates a MemRef type with the given rank and shape, a potentially empty list of affine layout maps, the given memory space and element type, in the same context as element type. The type is owned by the context.
"""
function mlirMemRefTypeGet(elementType, rank, shape, layout, memorySpace)
    @ccall mlir_c.mlirMemRefTypeGet(
        elementType::MlirType,
        rank::Cptrdiff_t,
        shape::Ptr{Int64},
        layout::MlirAttribute,
        memorySpace::MlirAttribute,
    )::MlirType
end

"""
    mlirMemRefTypeGetChecked(loc, elementType, rank, shape, layout, memorySpace)

Same as "[`mlirMemRefTypeGet`](@ref)" but returns a nullptr-wrapping [`MlirType`](@ref) o illegal arguments, emitting appropriate diagnostics.
"""
function mlirMemRefTypeGetChecked(loc, elementType, rank, shape, layout, memorySpace)
    @ccall mlir_c.mlirMemRefTypeGetChecked(
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
    @ccall mlir_c.mlirMemRefTypeContiguousGet(
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
    @ccall mlir_c.mlirMemRefTypeContiguousGetChecked(
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
    @ccall mlir_c.mlirUnrankedMemRefTypeGet(
        elementType::MlirType, memorySpace::MlirAttribute
    )::MlirType
end

"""
    mlirUnrankedMemRefTypeGetChecked(loc, elementType, memorySpace)

Same as "[`mlirUnrankedMemRefTypeGet`](@ref)" but returns a nullptr wrapping [`MlirType`](@ref) on illegal arguments, emitting appropriate diagnostics.
"""
function mlirUnrankedMemRefTypeGetChecked(loc, elementType, memorySpace)
    @ccall mlir_c.mlirUnrankedMemRefTypeGetChecked(
        loc::MlirLocation, elementType::MlirType, memorySpace::MlirAttribute
    )::MlirType
end

"""
    mlirMemRefTypeGetLayout(type)

Returns the layout of the given MemRef type.
"""
function mlirMemRefTypeGetLayout(type)
    @ccall mlir_c.mlirMemRefTypeGetLayout(type::MlirType)::MlirAttribute
end

"""
    mlirMemRefTypeGetAffineMap(type)

Returns the affine map of the given MemRef type.
"""
function mlirMemRefTypeGetAffineMap(type)
    @ccall mlir_c.mlirMemRefTypeGetAffineMap(type::MlirType)::MlirAffineMap
end

"""
    mlirMemRefTypeGetMemorySpace(type)

Returns the memory space of the given MemRef type.
"""
function mlirMemRefTypeGetMemorySpace(type)
    @ccall mlir_c.mlirMemRefTypeGetMemorySpace(type::MlirType)::MlirAttribute
end

"""
    mlirMemRefTypeGetStridesAndOffset(type, strides, offset)

Returns the strides of the MemRef if the layout map is in strided form. Both strides and offset are out params. strides must point to pre-allocated memory of length equal to the rank of the memref.
"""
function mlirMemRefTypeGetStridesAndOffset(type, strides, offset)
    @ccall mlir_c.mlirMemRefTypeGetStridesAndOffset(
        type::MlirType, strides::Ptr{Int64}, offset::Ptr{Int64}
    )::MlirLogicalResult
end

"""
    mlirUnrankedMemrefGetMemorySpace(type)

Returns the memory spcae of the given Unranked MemRef type.
"""
function mlirUnrankedMemrefGetMemorySpace(type)
    @ccall mlir_c.mlirUnrankedMemrefGetMemorySpace(type::MlirType)::MlirAttribute
end

"""
    mlirTupleTypeGetTypeID()

Returns the typeID of an Tuple type.
"""
function mlirTupleTypeGetTypeID()
    @ccall mlir_c.mlirTupleTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsATuple(type)

Checks whether the given type is a tuple type.
"""
function mlirTypeIsATuple(type)
    @ccall mlir_c.mlirTypeIsATuple(type::MlirType)::Bool
end

"""
    mlirTupleTypeGet(ctx, numElements, elements)

Creates a tuple type that consists of the given list of elemental types. The type is owned by the context.
"""
function mlirTupleTypeGet(ctx, numElements, elements)
    @ccall mlir_c.mlirTupleTypeGet(
        ctx::MlirContext, numElements::Cptrdiff_t, elements::Ptr{MlirType}
    )::MlirType
end

"""
    mlirTupleTypeGetNumTypes(type)

Returns the number of types contained in a tuple.
"""
function mlirTupleTypeGetNumTypes(type)
    @ccall mlir_c.mlirTupleTypeGetNumTypes(type::MlirType)::Cptrdiff_t
end

"""
    mlirTupleTypeGetType(type, pos)

Returns the pos-th type in the tuple type.
"""
function mlirTupleTypeGetType(type, pos)
    @ccall mlir_c.mlirTupleTypeGetType(type::MlirType, pos::Cptrdiff_t)::MlirType
end

"""
    mlirFunctionTypeGetTypeID()

Returns the typeID of an Function type.
"""
function mlirFunctionTypeGetTypeID()
    @ccall mlir_c.mlirFunctionTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAFunction(type)

Checks whether the given type is a function type.
"""
function mlirTypeIsAFunction(type)
    @ccall mlir_c.mlirTypeIsAFunction(type::MlirType)::Bool
end

"""
    mlirFunctionTypeGet(ctx, numInputs, inputs, numResults, results)

Creates a function type, mapping a list of input types to result types.
"""
function mlirFunctionTypeGet(ctx, numInputs, inputs, numResults, results)
    @ccall mlir_c.mlirFunctionTypeGet(
        ctx::MlirContext,
        numInputs::Cptrdiff_t,
        inputs::Ptr{MlirType},
        numResults::Cptrdiff_t,
        results::Ptr{MlirType},
    )::MlirType
end

"""
    mlirFunctionTypeGetNumInputs(type)

Returns the number of input types.
"""
function mlirFunctionTypeGetNumInputs(type)
    @ccall mlir_c.mlirFunctionTypeGetNumInputs(type::MlirType)::Cptrdiff_t
end

"""
    mlirFunctionTypeGetNumResults(type)

Returns the number of result types.
"""
function mlirFunctionTypeGetNumResults(type)
    @ccall mlir_c.mlirFunctionTypeGetNumResults(type::MlirType)::Cptrdiff_t
end

"""
    mlirFunctionTypeGetInput(type, pos)

Returns the pos-th input type.
"""
function mlirFunctionTypeGetInput(type, pos)
    @ccall mlir_c.mlirFunctionTypeGetInput(type::MlirType, pos::Cptrdiff_t)::MlirType
end

"""
    mlirFunctionTypeGetResult(type, pos)

Returns the pos-th result type.
"""
function mlirFunctionTypeGetResult(type, pos)
    @ccall mlir_c.mlirFunctionTypeGetResult(type::MlirType, pos::Cptrdiff_t)::MlirType
end

"""
    mlirOpaqueTypeGetTypeID()

Returns the typeID of an Opaque type.
"""
function mlirOpaqueTypeGetTypeID()
    @ccall mlir_c.mlirOpaqueTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsAOpaque(type)

Checks whether the given type is an opaque type.
"""
function mlirTypeIsAOpaque(type)
    @ccall mlir_c.mlirTypeIsAOpaque(type::MlirType)::Bool
end

"""
    mlirOpaqueTypeGet(ctx, dialectNamespace, typeData)

Creates an opaque type in the given context associated with the dialect identified by its namespace. The type contains opaque byte data of the specified length (data need not be null-terminated).
"""
function mlirOpaqueTypeGet(ctx, dialectNamespace, typeData)
    @ccall mlir_c.mlirOpaqueTypeGet(
        ctx::MlirContext, dialectNamespace::MlirStringRef, typeData::MlirStringRef
    )::MlirType
end

"""
    mlirOpaqueTypeGetDialectNamespace(type)

Returns the namespace of the dialect with which the given opaque type is associated. The namespace string is owned by the context.
"""
function mlirOpaqueTypeGetDialectNamespace(type)
    @ccall mlir_c.mlirOpaqueTypeGetDialectNamespace(type::MlirType)::MlirStringRef
end

"""
    mlirOpaqueTypeGetData(type)

Returns the raw data as a string reference. The data remains live as long as the context in which the type lives.
"""
function mlirOpaqueTypeGetData(type)
    @ccall mlir_c.mlirOpaqueTypeGetData(type::MlirType)::MlirStringRef
end

"""
    mlirEnableGlobalDebug(enable)

Sets the global debugging flag.
"""
function mlirEnableGlobalDebug(enable)
    @ccall mlir_c.mlirEnableGlobalDebug(enable::Bool)::Cvoid
end

"""
    mlirIsGlobalDebugEnabled()

Retuns `true` if the global debugging flag is set, false otherwise.
"""
function mlirIsGlobalDebugEnabled()
    @ccall mlir_c.mlirIsGlobalDebugEnabled()::Bool
end

"""
    mlirSetGlobalDebugType(type)

Sets the current debug type, similarly to `-debug-only=type` in the command-line tools. Note that global debug should be enabled for any output to be produced.
"""
function mlirSetGlobalDebugType(type)
    @ccall mlir_c.mlirSetGlobalDebugType(type::Cstring)::Cvoid
end

"""
    mlirSetGlobalDebugTypes(types, n)

Sets multiple current debug types, similarly to `-debug-only=type1,type2" in the command-line tools. Note that global debug should be enabled for any output to be produced.
"""
function mlirSetGlobalDebugTypes(types, n)
    @ccall mlir_c.mlirSetGlobalDebugTypes(types::Ptr{Cstring}, n::Cptrdiff_t)::Cvoid
end

"""
    mlirIsCurrentDebugType(type)

Checks if `type` is set as the current debug type.
"""
function mlirIsCurrentDebugType(type)
    @ccall mlir_c.mlirIsCurrentDebugType(type::Cstring)::Bool
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
    @ccall mlir_c.mlirDiagnosticPrint(
        diagnostic::MlirDiagnostic, callback::MlirStringCallback, userData::Ptr{Cvoid}
    )::Cvoid
end

"""
    mlirDiagnosticGetLocation(diagnostic)

Returns the location at which the diagnostic is reported.
"""
function mlirDiagnosticGetLocation(diagnostic)
    @ccall mlir_c.mlirDiagnosticGetLocation(diagnostic::MlirDiagnostic)::MlirLocation
end

"""
    mlirDiagnosticGetSeverity(diagnostic)

Returns the severity of the diagnostic.
"""
function mlirDiagnosticGetSeverity(diagnostic)
    @ccall mlir_c.mlirDiagnosticGetSeverity(
        diagnostic::MlirDiagnostic
    )::MlirDiagnosticSeverity
end

"""
    mlirDiagnosticGetNumNotes(diagnostic)

Returns the number of notes attached to the diagnostic.
"""
function mlirDiagnosticGetNumNotes(diagnostic)
    @ccall mlir_c.mlirDiagnosticGetNumNotes(diagnostic::MlirDiagnostic)::Cptrdiff_t
end

"""
    mlirDiagnosticGetNote(diagnostic, pos)

Returns `pos`-th note attached to the diagnostic. Expects `pos` to be a valid zero-based index into the list of notes.
"""
function mlirDiagnosticGetNote(diagnostic, pos)
    @ccall mlir_c.mlirDiagnosticGetNote(
        diagnostic::MlirDiagnostic, pos::Cptrdiff_t
    )::MlirDiagnostic
end

"""
    mlirContextAttachDiagnosticHandler(context, handler, userData, deleteUserData)

Attaches the diagnostic handler to the context. Handlers are invoked in the reverse order of attachment until one of them processes the diagnostic completely. When a handler is invoked it is passed the `userData` that was provided when it was attached. If non-NULL, `deleteUserData` is called once the system no longer needs to call the handler (for instance after the handler is detached or the context is destroyed). Returns an identifier that can be used to detach the handler.
"""
function mlirContextAttachDiagnosticHandler(context, handler, userData, deleteUserData)
    @ccall mlir_c.mlirContextAttachDiagnosticHandler(
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
    @ccall mlir_c.mlirContextDetachDiagnosticHandler(
        context::MlirContext, id::MlirDiagnosticHandlerID
    )::Cvoid
end

"""
    mlirEmitError(location, message)

Emits an error at the given location through the diagnostics engine. Used for testing purposes.
"""
function mlirEmitError(location, message)
    @ccall mlir_c.mlirEmitError(location::MlirLocation, message::Cstring)::Cvoid
end

function mlirGetDialectHandle__amdgpu__()
    @ccall mlir_c.mlirGetDialectHandle__amdgpu__()::MlirDialectHandle
end

function mlirGetDialectHandle__arith__()
    @ccall mlir_c.mlirGetDialectHandle__arith__()::MlirDialectHandle
end

function mlirGetDialectHandle__async__()
    @ccall mlir_c.mlirGetDialectHandle__async__()::MlirDialectHandle
end

function mlirGetDialectHandle__cf__()
    @ccall mlir_c.mlirGetDialectHandle__cf__()::MlirDialectHandle
end

function mlirGetDialectHandle__emitc__()
    @ccall mlir_c.mlirGetDialectHandle__emitc__()::MlirDialectHandle
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
    @ccall mlir_c.mlirTypeIsAEmitCArrayType(type::MlirType)::Bool
end

function mlirEmitCArrayTypeGetTypeID()
    @ccall mlir_c.mlirEmitCArrayTypeGetTypeID()::MlirTypeID
end

function mlirEmitCArrayTypeGet(nDims, shape, elementType)
    @ccall mlir_c.mlirEmitCArrayTypeGet(
        nDims::Cptrdiff_t, shape::Ptr{Int64}, elementType::MlirType
    )::MlirType
end

function mlirTypeIsAEmitCLValueType(type)
    @ccall mlir_c.mlirTypeIsAEmitCLValueType(type::MlirType)::Bool
end

function mlirEmitCLValueTypeGetTypeID()
    @ccall mlir_c.mlirEmitCLValueTypeGetTypeID()::MlirTypeID
end

function mlirEmitCLValueTypeGet(valueType)
    @ccall mlir_c.mlirEmitCLValueTypeGet(valueType::MlirType)::MlirType
end

function mlirTypeIsAEmitCOpaqueType(type)
    @ccall mlir_c.mlirTypeIsAEmitCOpaqueType(type::MlirType)::Bool
end

function mlirEmitCOpaqueTypeGetTypeID()
    @ccall mlir_c.mlirEmitCOpaqueTypeGetTypeID()::MlirTypeID
end

function mlirEmitCOpaqueTypeGet(ctx, value)
    @ccall mlir_c.mlirEmitCOpaqueTypeGet(ctx::MlirContext, value::MlirStringRef)::MlirType
end

function mlirTypeIsAEmitCPointerType(type)
    @ccall mlir_c.mlirTypeIsAEmitCPointerType(type::MlirType)::Bool
end

function mlirEmitCPointerTypeGetTypeID()
    @ccall mlir_c.mlirEmitCPointerTypeGetTypeID()::MlirTypeID
end

function mlirEmitCPointerTypeGet(pointee)
    @ccall mlir_c.mlirEmitCPointerTypeGet(pointee::MlirType)::MlirType
end

function mlirTypeIsAEmitCPtrDiffTType(type)
    @ccall mlir_c.mlirTypeIsAEmitCPtrDiffTType(type::MlirType)::Bool
end

function mlirEmitCPtrDiffTTypeGetTypeID()
    @ccall mlir_c.mlirEmitCPtrDiffTTypeGetTypeID()::MlirTypeID
end

function mlirEmitCPtrDiffTTypeGet(ctx)
    @ccall mlir_c.mlirEmitCPtrDiffTTypeGet(ctx::MlirContext)::MlirType
end

function mlirTypeIsAEmitCSignedSizeTType(type)
    @ccall mlir_c.mlirTypeIsAEmitCSignedSizeTType(type::MlirType)::Bool
end

function mlirEmitCSignedSizeTTypeGetTypeID()
    @ccall mlir_c.mlirEmitCSignedSizeTTypeGetTypeID()::MlirTypeID
end

function mlirEmitCSignedSizeTTypeGet(ctx)
    @ccall mlir_c.mlirEmitCSignedSizeTTypeGet(ctx::MlirContext)::MlirType
end

function mlirTypeIsAEmitCSizeTType(type)
    @ccall mlir_c.mlirTypeIsAEmitCSizeTType(type::MlirType)::Bool
end

function mlirEmitCSizeTTypeGetTypeID()
    @ccall mlir_c.mlirEmitCSizeTTypeGetTypeID()::MlirTypeID
end

function mlirEmitCSizeTTypeGet(ctx)
    @ccall mlir_c.mlirEmitCSizeTTypeGet(ctx::MlirContext)::MlirType
end

function mlirAttributeIsAEmitCCmpPredicate(attr)
    @ccall mlir_c.mlirAttributeIsAEmitCCmpPredicate(attr::MlirAttribute)::Bool
end

function mlirEmitCCmpPredicateAttrGet(ctx, val)
    @ccall mlir_c.mlirEmitCCmpPredicateAttrGet(
        ctx::MlirContext, val::MlirEmitCCmpPredicate
    )::MlirAttribute
end

function mlirEmitCCmpPredicateAttrGetValue(attr)
    @ccall mlir_c.mlirEmitCCmpPredicateAttrGetValue(
        attr::MlirAttribute
    )::MlirEmitCCmpPredicate
end

function mlirEmitCCmpPredicateAttrGetTypeID()
    @ccall mlir_c.mlirEmitCCmpPredicateAttrGetTypeID()::MlirTypeID
end

function mlirAttributeIsAEmitCOpaque(attr)
    @ccall mlir_c.mlirAttributeIsAEmitCOpaque(attr::MlirAttribute)::Bool
end

function mlirEmitCOpaqueAttrGet(ctx, value)
    @ccall mlir_c.mlirEmitCOpaqueAttrGet(
        ctx::MlirContext, value::MlirStringRef
    )::MlirAttribute
end

function mlirEmitCOpaqueAttrGetValue(attr)
    @ccall mlir_c.mlirEmitCOpaqueAttrGetValue(attr::MlirAttribute)::MlirStringRef
end

function mlirEmitCOpaqueAttrGetTypeID()
    @ccall mlir_c.mlirEmitCOpaqueAttrGetTypeID()::MlirTypeID
end

function mlirGetDialectHandle__func__()
    @ccall mlir_c.mlirGetDialectHandle__func__()::MlirDialectHandle
end

"""
    mlirFuncSetArgAttr(op, pos, name, attr)

Sets the argument attribute 'name' of an argument at index 'pos'. Asserts that the operation is a FuncOp.
"""
function mlirFuncSetArgAttr(op, pos, name, attr)
    @ccall mlir_c.mlirFuncSetArgAttr(
        op::MlirOperation, pos::Cptrdiff_t, name::MlirStringRef, attr::MlirAttribute
    )::Cvoid
end

function mlirFuncSetResultAttr(op, pos, name, attr)
    @ccall mlir_c.mlirFuncSetResultAttr(
        op::MlirOperation, pos::Cptrdiff_t, name::MlirStringRef, attr::MlirAttribute
    )::Cvoid
end

function mlirGetDialectHandle__gpu__()
    @ccall mlir_c.mlirGetDialectHandle__gpu__()::MlirDialectHandle
end

function mlirTypeIsAGPUAsyncTokenType(type)
    @ccall mlir_c.mlirTypeIsAGPUAsyncTokenType(type::MlirType)::Bool
end

function mlirGPUAsyncTokenTypeGet(ctx)
    @ccall mlir_c.mlirGPUAsyncTokenTypeGet(ctx::MlirContext)::MlirType
end

function mlirAttributeIsAGPUObjectAttr(attr)
    @ccall mlir_c.mlirAttributeIsAGPUObjectAttr(attr::MlirAttribute)::Bool
end

function mlirGPUObjectAttrGet(mlirCtx, target, format, objectStrRef, mlirObjectProps)
    @ccall mlir_c.mlirGPUObjectAttrGet(
        mlirCtx::MlirContext,
        target::MlirAttribute,
        format::UInt32,
        objectStrRef::MlirStringRef,
        mlirObjectProps::MlirAttribute,
    )::MlirAttribute
end

function mlirGPUObjectAttrGetWithKernels(
    mlirCtx, target, format, objectStrRef, mlirObjectProps, mlirKernelsAttr
)
    @ccall mlir_c.mlirGPUObjectAttrGetWithKernels(
        mlirCtx::MlirContext,
        target::MlirAttribute,
        format::UInt32,
        objectStrRef::MlirStringRef,
        mlirObjectProps::MlirAttribute,
        mlirKernelsAttr::MlirAttribute,
    )::MlirAttribute
end

function mlirGPUObjectAttrGetTarget(mlirObjectAttr)
    @ccall mlir_c.mlirGPUObjectAttrGetTarget(mlirObjectAttr::MlirAttribute)::MlirAttribute
end

function mlirGPUObjectAttrGetFormat(mlirObjectAttr)
    @ccall mlir_c.mlirGPUObjectAttrGetFormat(mlirObjectAttr::MlirAttribute)::UInt32
end

function mlirGPUObjectAttrGetObject(mlirObjectAttr)
    @ccall mlir_c.mlirGPUObjectAttrGetObject(mlirObjectAttr::MlirAttribute)::MlirStringRef
end

function mlirGPUObjectAttrHasProperties(mlirObjectAttr)
    @ccall mlir_c.mlirGPUObjectAttrHasProperties(mlirObjectAttr::MlirAttribute)::Bool
end

function mlirGPUObjectAttrGetProperties(mlirObjectAttr)
    @ccall mlir_c.mlirGPUObjectAttrGetProperties(
        mlirObjectAttr::MlirAttribute
    )::MlirAttribute
end

function mlirGPUObjectAttrHasKernels(mlirObjectAttr)
    @ccall mlir_c.mlirGPUObjectAttrHasKernels(mlirObjectAttr::MlirAttribute)::Bool
end

function mlirGPUObjectAttrGetKernels(mlirObjectAttr)
    @ccall mlir_c.mlirGPUObjectAttrGetKernels(mlirObjectAttr::MlirAttribute)::MlirAttribute
end

function mlirGetDialectHandle__irdl__()
    @ccall mlir_c.mlirGetDialectHandle__irdl__()::MlirDialectHandle
end

"""
    mlirLoadIRDLDialects(_module)

Loads all IRDL dialects in the provided module, registering the dialects in the module's associated context.
"""
function mlirLoadIRDLDialects(_module)
    @ccall mlir_c.mlirLoadIRDLDialects(_module::MlirModule)::MlirLogicalResult
end

function mlirIRDLVariadicityAttrGet(ctx, value)
    @ccall mlir_c.mlirIRDLVariadicityAttrGet(
        ctx::MlirContext, value::MlirStringRef
    )::MlirAttribute
end

function mlirIRDLVariadicityArrayAttrGet(ctx, nValues, values)
    @ccall mlir_c.mlirIRDLVariadicityArrayAttrGet(
        ctx::MlirContext, nValues::Cptrdiff_t, values::Ptr{MlirAttribute}
    )::MlirAttribute
end

function mlirGetDialectHandle__index__()
    @ccall mlir_c.mlirGetDialectHandle__index__()::MlirDialectHandle
end

function mlirGetDialectHandle__llvm__()
    @ccall mlir_c.mlirGetDialectHandle__llvm__()::MlirDialectHandle
end

"""
    mlirLLVMPointerTypeGet(ctx, addressSpace)

Creates an llvm.ptr type.
"""
function mlirLLVMPointerTypeGet(ctx, addressSpace)
    @ccall mlir_c.mlirLLVMPointerTypeGet(ctx::MlirContext, addressSpace::Cuint)::MlirType
end

function mlirLLVMPointerTypeGetTypeID()
    @ccall mlir_c.mlirLLVMPointerTypeGetTypeID()::MlirTypeID
end

"""
    mlirTypeIsALLVMPointerType(type)

Returns `true` if the type is an LLVM dialect pointer type.
"""
function mlirTypeIsALLVMPointerType(type)
    @ccall mlir_c.mlirTypeIsALLVMPointerType(type::MlirType)::Bool
end

"""
    mlirLLVMPointerTypeGetAddressSpace(pointerType)

Returns address space of llvm.ptr
"""
function mlirLLVMPointerTypeGetAddressSpace(pointerType)
    @ccall mlir_c.mlirLLVMPointerTypeGetAddressSpace(pointerType::MlirType)::Cuint
end

"""
    mlirLLVMVoidTypeGet(ctx)

Creates an llmv.void type.
"""
function mlirLLVMVoidTypeGet(ctx)
    @ccall mlir_c.mlirLLVMVoidTypeGet(ctx::MlirContext)::MlirType
end

"""
    mlirLLVMArrayTypeGet(elementType, numElements)

Creates an llvm.array type.
"""
function mlirLLVMArrayTypeGet(elementType, numElements)
    @ccall mlir_c.mlirLLVMArrayTypeGet(elementType::MlirType, numElements::Cuint)::MlirType
end

"""
    mlirLLVMArrayTypeGetElementType(type)

Returns the element type of the llvm.array type.
"""
function mlirLLVMArrayTypeGetElementType(type)
    @ccall mlir_c.mlirLLVMArrayTypeGetElementType(type::MlirType)::MlirType
end

"""
    mlirLLVMFunctionTypeGet(resultType, nArgumentTypes, argumentTypes, isVarArg)

Creates an llvm.func type.
"""
function mlirLLVMFunctionTypeGet(resultType, nArgumentTypes, argumentTypes, isVarArg)
    @ccall mlir_c.mlirLLVMFunctionTypeGet(
        resultType::MlirType,
        nArgumentTypes::Cptrdiff_t,
        argumentTypes::Ptr{MlirType},
        isVarArg::Bool,
    )::MlirType
end

"""
    mlirLLVMFunctionTypeGetNumInputs(type)

Returns the number of input types.
"""
function mlirLLVMFunctionTypeGetNumInputs(type)
    @ccall mlir_c.mlirLLVMFunctionTypeGetNumInputs(type::MlirType)::Cptrdiff_t
end

"""
    mlirLLVMFunctionTypeGetInput(type, pos)

Returns the pos-th input type.
"""
function mlirLLVMFunctionTypeGetInput(type, pos)
    @ccall mlir_c.mlirLLVMFunctionTypeGetInput(type::MlirType, pos::Cptrdiff_t)::MlirType
end

"""
    mlirLLVMFunctionTypeGetReturnType(type)

Returns the return type of the function type.
"""
function mlirLLVMFunctionTypeGetReturnType(type)
    @ccall mlir_c.mlirLLVMFunctionTypeGetReturnType(type::MlirType)::MlirType
end

"""
    mlirTypeIsALLVMStructType(type)

Returns `true` if the type is an LLVM dialect struct type.
"""
function mlirTypeIsALLVMStructType(type)
    @ccall mlir_c.mlirTypeIsALLVMStructType(type::MlirType)::Bool
end

function mlirLLVMStructTypeGetTypeID()
    @ccall mlir_c.mlirLLVMStructTypeGetTypeID()::MlirTypeID
end

"""
    mlirLLVMStructTypeIsLiteral(type)

Returns `true` if the type is a literal (unnamed) LLVM struct type.
"""
function mlirLLVMStructTypeIsLiteral(type)
    @ccall mlir_c.mlirLLVMStructTypeIsLiteral(type::MlirType)::Bool
end

"""
    mlirLLVMStructTypeGetNumElementTypes(type)

Returns the number of fields in the struct. Asserts if the struct is opaque or not yet initialized.
"""
function mlirLLVMStructTypeGetNumElementTypes(type)
    @ccall mlir_c.mlirLLVMStructTypeGetNumElementTypes(type::MlirType)::Cptrdiff_t
end

"""
    mlirLLVMStructTypeGetElementType(type, position)

Returns the `positions`-th field of the struct. Asserts if the struct is opaque, not yet initialized or if the position is out of range.
"""
function mlirLLVMStructTypeGetElementType(type, position)
    @ccall mlir_c.mlirLLVMStructTypeGetElementType(
        type::MlirType, position::Cptrdiff_t
    )::MlirType
end

"""
    mlirLLVMStructTypeIsPacked(type)

Returns `true` if the struct is packed.
"""
function mlirLLVMStructTypeIsPacked(type)
    @ccall mlir_c.mlirLLVMStructTypeIsPacked(type::MlirType)::Bool
end

"""
    mlirLLVMStructTypeGetIdentifier(type)

Returns the identifier of the identified struct. Asserts that the struct is identified, i.e., not literal.
"""
function mlirLLVMStructTypeGetIdentifier(type)
    @ccall mlir_c.mlirLLVMStructTypeGetIdentifier(type::MlirType)::MlirStringRef
end

"""
    mlirLLVMStructTypeIsOpaque(type)

Returns `true` is the struct is explicitly opaque (will not have a body) or uninitialized (will eventually have a body).
"""
function mlirLLVMStructTypeIsOpaque(type)
    @ccall mlir_c.mlirLLVMStructTypeIsOpaque(type::MlirType)::Bool
end

"""
    mlirLLVMStructTypeLiteralGet(ctx, nFieldTypes, fieldTypes, isPacked)

Creates an LLVM literal (unnamed) struct type. This may assert if the fields have types not compatible with the LLVM dialect. For a graceful failure, use the checked version.
"""
function mlirLLVMStructTypeLiteralGet(ctx, nFieldTypes, fieldTypes, isPacked)
    @ccall mlir_c.mlirLLVMStructTypeLiteralGet(
        ctx::MlirContext, nFieldTypes::Cptrdiff_t, fieldTypes::Ptr{MlirType}, isPacked::Bool
    )::MlirType
end

"""
    mlirLLVMStructTypeLiteralGetChecked(loc, nFieldTypes, fieldTypes, isPacked)

Creates an LLVM literal (unnamed) struct type if possible. Emits a diagnostic at the given location and returns null otherwise.
"""
function mlirLLVMStructTypeLiteralGetChecked(loc, nFieldTypes, fieldTypes, isPacked)
    @ccall mlir_c.mlirLLVMStructTypeLiteralGetChecked(
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
    @ccall mlir_c.mlirLLVMStructTypeIdentifiedGet(
        ctx::MlirContext, name::MlirStringRef
    )::MlirType
end

"""
    mlirLLVMStructTypeIdentifiedNewGet(ctx, name, nFieldTypes, fieldTypes, isPacked)

Creates an LLVM identified struct type with no body and a name starting with the given prefix. If a struct with the exact name as the given prefix already exists, appends an unspecified suffix to the name so that the name is unique in context.
"""
function mlirLLVMStructTypeIdentifiedNewGet(ctx, name, nFieldTypes, fieldTypes, isPacked)
    @ccall mlir_c.mlirLLVMStructTypeIdentifiedNewGet(
        ctx::MlirContext,
        name::MlirStringRef,
        nFieldTypes::Cptrdiff_t,
        fieldTypes::Ptr{MlirType},
        isPacked::Bool,
    )::MlirType
end

function mlirLLVMStructTypeOpaqueGet(ctx, name)
    @ccall mlir_c.mlirLLVMStructTypeOpaqueGet(
        ctx::MlirContext, name::MlirStringRef
    )::MlirType
end

"""
    mlirLLVMStructTypeSetBody(structType, nFieldTypes, fieldTypes, isPacked)

Sets the body of the identified struct if it hasn't been set yet. Returns whether the operation was successful.
"""
function mlirLLVMStructTypeSetBody(structType, nFieldTypes, fieldTypes, isPacked)
    @ccall mlir_c.mlirLLVMStructTypeSetBody(
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
    @ccall mlir_c.mlirLLVMCConvAttrGet(
        ctx::MlirContext, cconv::MlirLLVMCConv
    )::MlirAttribute
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
    @ccall mlir_c.mlirLLVMComdatAttrGet(
        ctx::MlirContext, comdat::MlirLLVMComdat
    )::MlirAttribute
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
    @ccall mlir_c.mlirLLVMLinkageAttrGet(
        ctx::MlirContext, linkage::MlirLLVMLinkage
    )::MlirAttribute
end

"""
    mlirLLVMDINullTypeAttrGet(ctx)

Creates a LLVM DINullType attribute.
"""
function mlirLLVMDINullTypeAttrGet(ctx)
    @ccall mlir_c.mlirLLVMDINullTypeAttrGet(ctx::MlirContext)::MlirAttribute
end

"""
    mlirLLVMDIExpressionElemAttrGet(ctx, opcode, nArguments, arguments)

Creates a LLVM DIExpressionElem attribute.
"""
function mlirLLVMDIExpressionElemAttrGet(ctx, opcode, nArguments, arguments)
    @ccall mlir_c.mlirLLVMDIExpressionElemAttrGet(
        ctx::MlirContext, opcode::Cuint, nArguments::Cptrdiff_t, arguments::Ptr{UInt64}
    )::MlirAttribute
end

"""
    mlirLLVMDIExpressionAttrGet(ctx, nOperations, operations)

Creates a LLVM DIExpression attribute.
"""
function mlirLLVMDIExpressionAttrGet(ctx, nOperations, operations)
    @ccall mlir_c.mlirLLVMDIExpressionAttrGet(
        ctx::MlirContext, nOperations::Cptrdiff_t, operations::Ptr{MlirAttribute}
    )::MlirAttribute
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
    @ccall mlir_c.mlirLLVMDIBasicTypeAttrGet(
        ctx::MlirContext,
        tag::Cuint,
        name::MlirAttribute,
        sizeInBits::UInt64,
        encoding::MlirLLVMTypeEncoding,
    )::MlirAttribute
end

"""
    mlirLLVMDICompositeTypeAttrGetRecSelf(recId)

Creates a self-referencing LLVM DICompositeType attribute.
"""
function mlirLLVMDICompositeTypeAttrGetRecSelf(recId)
    @ccall mlir_c.mlirLLVMDICompositeTypeAttrGetRecSelf(recId::MlirAttribute)::MlirAttribute
end

"""
    mlirLLVMDICompositeTypeAttrGet(ctx, recId, isRecSelf, tag, name, file, line, scope, baseType, flags, sizeInBits, alignInBits, nElements, elements, dataLocation, rank, allocated, associated)

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
)
    @ccall mlir_c.mlirLLVMDICompositeTypeAttrGet(
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
    )::MlirAttribute
end

"""
    mlirLLVMDIDerivedTypeAttrGet(ctx, tag, name, baseType, sizeInBits, alignInBits, offsetInBits, dwarfAddressSpace, extraData)

Creates a LLVM DIDerivedType attribute. Note that `dwarfAddressSpace` is an optional field, where [`MLIR_CAPI_DWARF_ADDRESS_SPACE_NULL`](@ref) indicates null and non-negative values indicate a value present.
"""
function mlirLLVMDIDerivedTypeAttrGet(
    ctx,
    tag,
    name,
    baseType,
    sizeInBits,
    alignInBits,
    offsetInBits,
    dwarfAddressSpace,
    extraData,
)
    @ccall mlir_c.mlirLLVMDIDerivedTypeAttrGet(
        ctx::MlirContext,
        tag::Cuint,
        name::MlirAttribute,
        baseType::MlirAttribute,
        sizeInBits::UInt64,
        alignInBits::UInt32,
        offsetInBits::UInt64,
        dwarfAddressSpace::Int64,
        extraData::MlirAttribute,
    )::MlirAttribute
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
    @ccall mlir_c.mlirLLVMDIStringTypeAttrGet(
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

"""
    mlirLLVMDIDerivedTypeAttrGetBaseType(diDerivedType)

Gets the base type from a LLVM DIDerivedType attribute.
"""
function mlirLLVMDIDerivedTypeAttrGetBaseType(diDerivedType)
    @ccall mlir_c.mlirLLVMDIDerivedTypeAttrGetBaseType(
        diDerivedType::MlirAttribute
    )::MlirAttribute
end

"""
    mlirLLVMDIFileAttrGet(ctx, name, directory)

Creates a LLVM DIFileAttr attribute.
"""
function mlirLLVMDIFileAttrGet(ctx, name, directory)
    @ccall mlir_c.mlirLLVMDIFileAttrGet(
        ctx::MlirContext, name::MlirAttribute, directory::MlirAttribute
    )::MlirAttribute
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
    mlirLLVMDICompileUnitAttrGet(ctx, id, sourceLanguage, file, producer, isOptimized, emissionKind, nameTableKind, splitDebugFilename)

Creates a LLVM DICompileUnit attribute.
"""
function mlirLLVMDICompileUnitAttrGet(
    ctx,
    id,
    sourceLanguage,
    file,
    producer,
    isOptimized,
    emissionKind,
    nameTableKind,
    splitDebugFilename,
)
    @ccall mlir_c.mlirLLVMDICompileUnitAttrGet(
        ctx::MlirContext,
        id::MlirAttribute,
        sourceLanguage::Cuint,
        file::MlirAttribute,
        producer::MlirAttribute,
        isOptimized::Bool,
        emissionKind::MlirLLVMDIEmissionKind,
        nameTableKind::MlirLLVMDINameTableKind,
        splitDebugFilename::MlirAttribute,
    )::MlirAttribute
end

"""
    mlirLLVMDIFlagsAttrGet(ctx, value)

Creates a LLVM DIFlags attribute.
"""
function mlirLLVMDIFlagsAttrGet(ctx, value)
    @ccall mlir_c.mlirLLVMDIFlagsAttrGet(ctx::MlirContext, value::UInt64)::MlirAttribute
end

"""
    mlirLLVMDILexicalBlockAttrGet(ctx, scope, file, line, column)

Creates a LLVM DILexicalBlock attribute.
"""
function mlirLLVMDILexicalBlockAttrGet(ctx, scope, file, line, column)
    @ccall mlir_c.mlirLLVMDILexicalBlockAttrGet(
        ctx::MlirContext,
        scope::MlirAttribute,
        file::MlirAttribute,
        line::Cuint,
        column::Cuint,
    )::MlirAttribute
end

"""
    mlirLLVMDILexicalBlockFileAttrGet(ctx, scope, file, discriminator)

Creates a LLVM DILexicalBlockFile attribute.
"""
function mlirLLVMDILexicalBlockFileAttrGet(ctx, scope, file, discriminator)
    @ccall mlir_c.mlirLLVMDILexicalBlockFileAttrGet(
        ctx::MlirContext, scope::MlirAttribute, file::MlirAttribute, discriminator::Cuint
    )::MlirAttribute
end

"""
    mlirLLVMDILocalVariableAttrGet(ctx, scope, name, diFile, line, arg, alignInBits, diType, flags)

Creates a LLVM DILocalVariableAttr attribute.
"""
function mlirLLVMDILocalVariableAttrGet(
    ctx, scope, name, diFile, line, arg, alignInBits, diType, flags
)
    @ccall mlir_c.mlirLLVMDILocalVariableAttrGet(
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

"""
    mlirLLVMDISubprogramAttrGetRecSelf(recId)

Creates a self-referencing LLVM DISubprogramAttr attribute.
"""
function mlirLLVMDISubprogramAttrGetRecSelf(recId)
    @ccall mlir_c.mlirLLVMDISubprogramAttrGetRecSelf(recId::MlirAttribute)::MlirAttribute
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
    @ccall mlir_c.mlirLLVMDISubprogramAttrGet(
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

"""
    mlirLLVMDIAnnotationAttrGet(ctx, name, value)

Creates a LLVM DIAnnotation attribute.
"""
function mlirLLVMDIAnnotationAttrGet(ctx, name, value)
    @ccall mlir_c.mlirLLVMDIAnnotationAttrGet(
        ctx::MlirContext, name::MlirAttribute, value::MlirAttribute
    )::MlirAttribute
end

"""
    mlirLLVMDISubprogramAttrGetScope(diSubprogram)

Gets the scope from this DISubprogramAttr.
"""
function mlirLLVMDISubprogramAttrGetScope(diSubprogram)
    @ccall mlir_c.mlirLLVMDISubprogramAttrGetScope(
        diSubprogram::MlirAttribute
    )::MlirAttribute
end

"""
    mlirLLVMDISubprogramAttrGetLine(diSubprogram)

Gets the line from this DISubprogramAttr.
"""
function mlirLLVMDISubprogramAttrGetLine(diSubprogram)
    @ccall mlir_c.mlirLLVMDISubprogramAttrGetLine(diSubprogram::MlirAttribute)::Cuint
end

"""
    mlirLLVMDISubprogramAttrGetScopeLine(diSubprogram)

Gets the scope line from this DISubprogram.
"""
function mlirLLVMDISubprogramAttrGetScopeLine(diSubprogram)
    @ccall mlir_c.mlirLLVMDISubprogramAttrGetScopeLine(diSubprogram::MlirAttribute)::Cuint
end

"""
    mlirLLVMDISubprogramAttrGetCompileUnit(diSubprogram)

Gets the compile unit from this DISubprogram.
"""
function mlirLLVMDISubprogramAttrGetCompileUnit(diSubprogram)
    @ccall mlir_c.mlirLLVMDISubprogramAttrGetCompileUnit(
        diSubprogram::MlirAttribute
    )::MlirAttribute
end

"""
    mlirLLVMDISubprogramAttrGetFile(diSubprogram)

Gets the file from this DISubprogramAttr.
"""
function mlirLLVMDISubprogramAttrGetFile(diSubprogram)
    @ccall mlir_c.mlirLLVMDISubprogramAttrGetFile(
        diSubprogram::MlirAttribute
    )::MlirAttribute
end

"""
    mlirLLVMDISubprogramAttrGetType(diSubprogram)

Gets the type from this DISubprogramAttr.
"""
function mlirLLVMDISubprogramAttrGetType(diSubprogram)
    @ccall mlir_c.mlirLLVMDISubprogramAttrGetType(
        diSubprogram::MlirAttribute
    )::MlirAttribute
end

"""
    mlirLLVMDISubroutineTypeAttrGet(ctx, callingConvention, nTypes, types)

Creates a LLVM DISubroutineTypeAttr attribute.
"""
function mlirLLVMDISubroutineTypeAttrGet(ctx, callingConvention, nTypes, types)
    @ccall mlir_c.mlirLLVMDISubroutineTypeAttrGet(
        ctx::MlirContext,
        callingConvention::Cuint,
        nTypes::Cptrdiff_t,
        types::Ptr{MlirAttribute},
    )::MlirAttribute
end

"""
    mlirLLVMDIModuleAttrGet(ctx, file, scope, name, configMacros, includePath, apinotes, line, isDecl)

Creates a LLVM DIModuleAttr attribute.
"""
function mlirLLVMDIModuleAttrGet(
    ctx, file, scope, name, configMacros, includePath, apinotes, line, isDecl
)
    @ccall mlir_c.mlirLLVMDIModuleAttrGet(
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

"""
    mlirLLVMDIImportedEntityAttrGet(ctx, tag, scope, entity, file, line, name, nElements, elements)

Creates a LLVM DIImportedEntityAttr attribute.
"""
function mlirLLVMDIImportedEntityAttrGet(
    ctx, tag, scope, entity, file, line, name, nElements, elements
)
    @ccall mlir_c.mlirLLVMDIImportedEntityAttrGet(
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

"""
    mlirLLVMDIModuleAttrGetScope(diModule)

Gets the scope of this DIModuleAttr.
"""
function mlirLLVMDIModuleAttrGetScope(diModule)
    @ccall mlir_c.mlirLLVMDIModuleAttrGetScope(diModule::MlirAttribute)::MlirAttribute
end

"""
    mlirLinalgFillBuiltinNamedOpRegion(mlirOp)

Apply the special region builder for the builtin named Linalg op. Assert that `mlirOp` is a builtin named Linalg op.
"""
function mlirLinalgFillBuiltinNamedOpRegion(mlirOp)
    @ccall mlir_c.mlirLinalgFillBuiltinNamedOpRegion(mlirOp::MlirOperation)::Cvoid
end

function mlirLinalgIsAContractionOp(op)
    @ccall mlir_c.mlirLinalgIsAContractionOp(op::MlirOperation)::Bool
end

struct MlirLinalgContractionDimensions
    batch::MlirAttribute
    m::MlirAttribute
    n::MlirAttribute
    k::MlirAttribute
end

function mlirLinalgInferContractionDimensions(op)
    @ccall mlir_c.mlirLinalgInferContractionDimensions(
        op::MlirOperation
    )::MlirLinalgContractionDimensions
end

function mlirLinalgInferContractionDimensionsFromMaps(indexingMaps, numMaps)
    @ccall mlir_c.mlirLinalgInferContractionDimensionsFromMaps(
        indexingMaps::Ptr{MlirAffineMap}, numMaps::Csize_t
    )::MlirLinalgContractionDimensions
end

function mlirLinalgIsAConvolutionOp(op)
    @ccall mlir_c.mlirLinalgIsAConvolutionOp(op::MlirOperation)::Bool
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
    @ccall mlir_c.mlirLinalgInferConvolutionDimensions(
        op::MlirOperation
    )::MlirLinalgConvolutionDimensions
end

function mlirLinalgGetIndexingMapsAttribute(op)
    @ccall mlir_c.mlirLinalgGetIndexingMapsAttribute(op::MlirOperation)::MlirAttribute
end

function mlirGetDialectHandle__linalg__()
    @ccall mlir_c.mlirGetDialectHandle__linalg__()::MlirDialectHandle
end

function mlirGetDialectHandle__ml_program__()
    @ccall mlir_c.mlirGetDialectHandle__ml_program__()::MlirDialectHandle
end

function mlirGetDialectHandle__math__()
    @ccall mlir_c.mlirGetDialectHandle__math__()::MlirDialectHandle
end

function mlirGetDialectHandle__memref__()
    @ccall mlir_c.mlirGetDialectHandle__memref__()::MlirDialectHandle
end

function mlirGetDialectHandle__nvgpu__()
    @ccall mlir_c.mlirGetDialectHandle__nvgpu__()::MlirDialectHandle
end

function mlirTypeIsANVGPUTensorMapDescriptorType(type)
    @ccall mlir_c.mlirTypeIsANVGPUTensorMapDescriptorType(type::MlirType)::Bool
end

function mlirNVGPUTensorMapDescriptorTypeGet(
    ctx, tensorMemrefType, swizzle, l2promo, oobFill, interleave
)
    @ccall mlir_c.mlirNVGPUTensorMapDescriptorTypeGet(
        ctx::MlirContext,
        tensorMemrefType::MlirType,
        swizzle::Cint,
        l2promo::Cint,
        oobFill::Cint,
        interleave::Cint,
    )::MlirType
end

function mlirGetDialectHandle__nvvm__()
    @ccall mlir_c.mlirGetDialectHandle__nvvm__()::MlirDialectHandle
end

function mlirGetDialectHandle__omp__()
    @ccall mlir_c.mlirGetDialectHandle__omp__()::MlirDialectHandle
end

function mlirGetDialectHandle__pdl__()
    @ccall mlir_c.mlirGetDialectHandle__pdl__()::MlirDialectHandle
end

function mlirTypeIsAPDLType(type)
    @ccall mlir_c.mlirTypeIsAPDLType(type::MlirType)::Bool
end

function mlirTypeIsAPDLAttributeType(type)
    @ccall mlir_c.mlirTypeIsAPDLAttributeType(type::MlirType)::Bool
end

function mlirPDLAttributeTypeGet(ctx)
    @ccall mlir_c.mlirPDLAttributeTypeGet(ctx::MlirContext)::MlirType
end

function mlirTypeIsAPDLOperationType(type)
    @ccall mlir_c.mlirTypeIsAPDLOperationType(type::MlirType)::Bool
end

function mlirPDLOperationTypeGet(ctx)
    @ccall mlir_c.mlirPDLOperationTypeGet(ctx::MlirContext)::MlirType
end

function mlirTypeIsAPDLRangeType(type)
    @ccall mlir_c.mlirTypeIsAPDLRangeType(type::MlirType)::Bool
end

function mlirPDLRangeTypeGet(elementType)
    @ccall mlir_c.mlirPDLRangeTypeGet(elementType::MlirType)::MlirType
end

function mlirPDLRangeTypeGetElementType(type)
    @ccall mlir_c.mlirPDLRangeTypeGetElementType(type::MlirType)::MlirType
end

function mlirTypeIsAPDLTypeType(type)
    @ccall mlir_c.mlirTypeIsAPDLTypeType(type::MlirType)::Bool
end

function mlirPDLTypeTypeGet(ctx)
    @ccall mlir_c.mlirPDLTypeTypeGet(ctx::MlirContext)::MlirType
end

function mlirTypeIsAPDLValueType(type)
    @ccall mlir_c.mlirTypeIsAPDLValueType(type::MlirType)::Bool
end

function mlirPDLValueTypeGet(ctx)
    @ccall mlir_c.mlirPDLValueTypeGet(ctx::MlirContext)::MlirType
end

function mlirGetDialectHandle__quant__()
    @ccall mlir_c.mlirGetDialectHandle__quant__()::MlirDialectHandle
end

"""
    mlirTypeIsAQuantizedType(type)

Returns `true` if the given type is a quantization dialect type.
"""
function mlirTypeIsAQuantizedType(type)
    @ccall mlir_c.mlirTypeIsAQuantizedType(type::MlirType)::Bool
end

"""
    mlirQuantizedTypeGetSignedFlag()

Returns the bit flag used to indicate signedness of a quantized type.
"""
function mlirQuantizedTypeGetSignedFlag()
    @ccall mlir_c.mlirQuantizedTypeGetSignedFlag()::Cuint
end

"""
    mlirQuantizedTypeGetDefaultMinimumForInteger(isSigned, integralWidth)

Returns the minimum possible value stored by a quantized type.
"""
function mlirQuantizedTypeGetDefaultMinimumForInteger(isSigned, integralWidth)
    @ccall mlir_c.mlirQuantizedTypeGetDefaultMinimumForInteger(
        isSigned::Bool, integralWidth::Cuint
    )::Int64
end

"""
    mlirQuantizedTypeGetDefaultMaximumForInteger(isSigned, integralWidth)

Returns the maximum possible value stored by a quantized type.
"""
function mlirQuantizedTypeGetDefaultMaximumForInteger(isSigned, integralWidth)
    @ccall mlir_c.mlirQuantizedTypeGetDefaultMaximumForInteger(
        isSigned::Bool, integralWidth::Cuint
    )::Int64
end

"""
    mlirQuantizedTypeGetExpressedType(type)

Gets the original type approximated by the given quantized type.
"""
function mlirQuantizedTypeGetExpressedType(type)
    @ccall mlir_c.mlirQuantizedTypeGetExpressedType(type::MlirType)::MlirType
end

"""
    mlirQuantizedTypeGetFlags(type)

Gets the flags associated with the given quantized type.
"""
function mlirQuantizedTypeGetFlags(type)
    @ccall mlir_c.mlirQuantizedTypeGetFlags(type::MlirType)::Cuint
end

"""
    mlirQuantizedTypeIsSigned(type)

Returns `true` if the given type is signed, `false` otherwise.
"""
function mlirQuantizedTypeIsSigned(type)
    @ccall mlir_c.mlirQuantizedTypeIsSigned(type::MlirType)::Bool
end

"""
    mlirQuantizedTypeGetStorageType(type)

Returns the underlying type used to store the values.
"""
function mlirQuantizedTypeGetStorageType(type)
    @ccall mlir_c.mlirQuantizedTypeGetStorageType(type::MlirType)::MlirType
end

"""
    mlirQuantizedTypeGetStorageTypeMin(type)

Returns the minimum value that the storage type of the given quantized type can take.
"""
function mlirQuantizedTypeGetStorageTypeMin(type)
    @ccall mlir_c.mlirQuantizedTypeGetStorageTypeMin(type::MlirType)::Int64
end

"""
    mlirQuantizedTypeGetStorageTypeMax(type)

Returns the maximum value that the storage type of the given quantized type can take.
"""
function mlirQuantizedTypeGetStorageTypeMax(type)
    @ccall mlir_c.mlirQuantizedTypeGetStorageTypeMax(type::MlirType)::Int64
end

"""
    mlirQuantizedTypeGetStorageTypeIntegralWidth(type)

Returns the integral bitwidth that the storage type of the given quantized type can represent exactly.
"""
function mlirQuantizedTypeGetStorageTypeIntegralWidth(type)
    @ccall mlir_c.mlirQuantizedTypeGetStorageTypeIntegralWidth(type::MlirType)::Cuint
end

"""
    mlirQuantizedTypeIsCompatibleExpressedType(type, candidate)

Returns `true` if the `candidate` type is compatible with the given quantized `type`.
"""
function mlirQuantizedTypeIsCompatibleExpressedType(type, candidate)
    @ccall mlir_c.mlirQuantizedTypeIsCompatibleExpressedType(
        type::MlirType, candidate::MlirType
    )::Bool
end

"""
    mlirQuantizedTypeGetQuantizedElementType(type)

Returns the element type of the given quantized type as another quantized type.
"""
function mlirQuantizedTypeGetQuantizedElementType(type)
    @ccall mlir_c.mlirQuantizedTypeGetQuantizedElementType(type::MlirType)::MlirType
end

"""
    mlirQuantizedTypeCastFromStorageType(type, candidate)

Casts from a type based on the storage type of the given type to a corresponding type based on the given type. Returns a null type if the cast is not valid.
"""
function mlirQuantizedTypeCastFromStorageType(type, candidate)
    @ccall mlir_c.mlirQuantizedTypeCastFromStorageType(
        type::MlirType, candidate::MlirType
    )::MlirType
end

"""
    mlirQuantizedTypeCastToStorageType(type)

Casts from a type based on a quantized type to a corresponding typed based on the storage type. Returns a null type if the cast is not valid.
"""
function mlirQuantizedTypeCastToStorageType(type)
    @ccall mlir_c.mlirQuantizedTypeCastToStorageType(type::MlirType)::MlirType
end

"""
    mlirQuantizedTypeCastFromExpressedType(type, candidate)

Casts from a type based on the expressed type of the given type to a corresponding type based on the given type. Returns a null type if the cast is not valid.
"""
function mlirQuantizedTypeCastFromExpressedType(type, candidate)
    @ccall mlir_c.mlirQuantizedTypeCastFromExpressedType(
        type::MlirType, candidate::MlirType
    )::MlirType
end

"""
    mlirQuantizedTypeCastToExpressedType(type)

Casts from a type based on a quantized type to a corresponding typed based on the expressed type. Returns a null type if the cast is not valid.
"""
function mlirQuantizedTypeCastToExpressedType(type)
    @ccall mlir_c.mlirQuantizedTypeCastToExpressedType(type::MlirType)::MlirType
end

"""
    mlirQuantizedTypeCastExpressedToStorageType(type, candidate)

Casts from a type based on the expressed type of the given quantized type to equivalent type based on storage type of the same quantized type.
"""
function mlirQuantizedTypeCastExpressedToStorageType(type, candidate)
    @ccall mlir_c.mlirQuantizedTypeCastExpressedToStorageType(
        type::MlirType, candidate::MlirType
    )::MlirType
end

"""
    mlirTypeIsAAnyQuantizedType(type)

Returns `true` if the given type is an AnyQuantizedType.
"""
function mlirTypeIsAAnyQuantizedType(type)
    @ccall mlir_c.mlirTypeIsAAnyQuantizedType(type::MlirType)::Bool
end

"""
    mlirAnyQuantizedTypeGet(flags, storageType, expressedType, storageTypeMin, storageTypeMax)

Creates an instance of AnyQuantizedType with the given parameters in the same context as `storageType` and returns it. The instance is owned by the context.
"""
function mlirAnyQuantizedTypeGet(
    flags, storageType, expressedType, storageTypeMin, storageTypeMax
)
    @ccall mlir_c.mlirAnyQuantizedTypeGet(
        flags::Cuint,
        storageType::MlirType,
        expressedType::MlirType,
        storageTypeMin::Int64,
        storageTypeMax::Int64,
    )::MlirType
end

"""
    mlirTypeIsAUniformQuantizedType(type)

Returns `true` if the given type is a UniformQuantizedType.
"""
function mlirTypeIsAUniformQuantizedType(type)
    @ccall mlir_c.mlirTypeIsAUniformQuantizedType(type::MlirType)::Bool
end

"""
    mlirUniformQuantizedTypeGet(flags, storageType, expressedType, scale, zeroPoint, storageTypeMin, storageTypeMax)

Creates an instance of UniformQuantizedType with the given parameters in the same context as `storageType` and returns it. The instance is owned by the context.
"""
function mlirUniformQuantizedTypeGet(
    flags, storageType, expressedType, scale, zeroPoint, storageTypeMin, storageTypeMax
)
    @ccall mlir_c.mlirUniformQuantizedTypeGet(
        flags::Cuint,
        storageType::MlirType,
        expressedType::MlirType,
        scale::Cdouble,
        zeroPoint::Int64,
        storageTypeMin::Int64,
        storageTypeMax::Int64,
    )::MlirType
end

"""
    mlirUniformQuantizedTypeGetScale(type)

Returns the scale of the given uniform quantized type.
"""
function mlirUniformQuantizedTypeGetScale(type)
    @ccall mlir_c.mlirUniformQuantizedTypeGetScale(type::MlirType)::Cdouble
end

"""
    mlirUniformQuantizedTypeGetZeroPoint(type)

Returns the zero point of the given uniform quantized type.
"""
function mlirUniformQuantizedTypeGetZeroPoint(type)
    @ccall mlir_c.mlirUniformQuantizedTypeGetZeroPoint(type::MlirType)::Int64
end

"""
    mlirUniformQuantizedTypeIsFixedPoint(type)

Returns `true` if the given uniform quantized type is fixed-point.
"""
function mlirUniformQuantizedTypeIsFixedPoint(type)
    @ccall mlir_c.mlirUniformQuantizedTypeIsFixedPoint(type::MlirType)::Bool
end

"""
    mlirTypeIsAUniformQuantizedPerAxisType(type)

Returns `true` if the given type is a UniformQuantizedPerAxisType.
"""
function mlirTypeIsAUniformQuantizedPerAxisType(type)
    @ccall mlir_c.mlirTypeIsAUniformQuantizedPerAxisType(type::MlirType)::Bool
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
    @ccall mlir_c.mlirUniformQuantizedPerAxisTypeGet(
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

"""
    mlirUniformQuantizedPerAxisTypeGetNumDims(type)

Returns the number of axes in the given quantized per-axis type.
"""
function mlirUniformQuantizedPerAxisTypeGetNumDims(type)
    @ccall mlir_c.mlirUniformQuantizedPerAxisTypeGetNumDims(type::MlirType)::Cptrdiff_t
end

"""
    mlirUniformQuantizedPerAxisTypeGetScale(type, pos)

Returns `pos`-th scale of the given quantized per-axis type.
"""
function mlirUniformQuantizedPerAxisTypeGetScale(type, pos)
    @ccall mlir_c.mlirUniformQuantizedPerAxisTypeGetScale(
        type::MlirType, pos::Cptrdiff_t
    )::Cdouble
end

"""
    mlirUniformQuantizedPerAxisTypeGetZeroPoint(type, pos)

Returns `pos`-th zero point of the given quantized per-axis type.
"""
function mlirUniformQuantizedPerAxisTypeGetZeroPoint(type, pos)
    @ccall mlir_c.mlirUniformQuantizedPerAxisTypeGetZeroPoint(
        type::MlirType, pos::Cptrdiff_t
    )::Int64
end

"""
    mlirUniformQuantizedPerAxisTypeGetQuantizedDimension(type)

Returns the index of the quantized dimension in the given quantized per-axis type.
"""
function mlirUniformQuantizedPerAxisTypeGetQuantizedDimension(type)
    @ccall mlir_c.mlirUniformQuantizedPerAxisTypeGetQuantizedDimension(
        type::MlirType
    )::Int32
end

"""
    mlirUniformQuantizedPerAxisTypeIsFixedPoint(type)

Returns `true` if the given uniform quantized per-axis type is fixed-point.
"""
function mlirUniformQuantizedPerAxisTypeIsFixedPoint(type)
    @ccall mlir_c.mlirUniformQuantizedPerAxisTypeIsFixedPoint(type::MlirType)::Bool
end

"""
    mlirTypeIsAUniformQuantizedSubChannelType(type)

Returns `true` if the given type is a UniformQuantizedSubChannel.
"""
function mlirTypeIsAUniformQuantizedSubChannelType(type)
    @ccall mlir_c.mlirTypeIsAUniformQuantizedSubChannelType(type::MlirType)::Bool
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
    @ccall mlir_c.mlirUniformQuantizedSubChannelTypeGet(
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

"""
    mlirUniformQuantizedSubChannelTypeGetNumBlockSizes(type)

Returns the number of block sizes provided in type.
"""
function mlirUniformQuantizedSubChannelTypeGetNumBlockSizes(type)
    @ccall mlir_c.mlirUniformQuantizedSubChannelTypeGetNumBlockSizes(
        type::MlirType
    )::Cptrdiff_t
end

"""
    mlirUniformQuantizedSubChannelTypeGetQuantizedDimension(type, pos)

Returns the quantized dimension at the given position.
"""
function mlirUniformQuantizedSubChannelTypeGetQuantizedDimension(type, pos)
    @ccall mlir_c.mlirUniformQuantizedSubChannelTypeGetQuantizedDimension(
        type::MlirType, pos::Cptrdiff_t
    )::Int32
end

"""
    mlirUniformQuantizedSubChannelTypeGetBlockSize(type, pos)

Returns the block size at the given position.
"""
function mlirUniformQuantizedSubChannelTypeGetBlockSize(type, pos)
    @ccall mlir_c.mlirUniformQuantizedSubChannelTypeGetBlockSize(
        type::MlirType, pos::Cptrdiff_t
    )::Int64
end

"""
    mlirUniformQuantizedSubChannelTypeGetScales(type)

Returns the scales of the quantized type.
"""
function mlirUniformQuantizedSubChannelTypeGetScales(type)
    @ccall mlir_c.mlirUniformQuantizedSubChannelTypeGetScales(type::MlirType)::MlirAttribute
end

"""
    mlirUniformQuantizedSubChannelTypeGetZeroPoints(type)

Returns the zero-points of the quantized type.
"""
function mlirUniformQuantizedSubChannelTypeGetZeroPoints(type)
    @ccall mlir_c.mlirUniformQuantizedSubChannelTypeGetZeroPoints(
        type::MlirType
    )::MlirAttribute
end

"""
    mlirTypeIsACalibratedQuantizedType(type)

Returns `true` if the given type is a CalibratedQuantizedType.
"""
function mlirTypeIsACalibratedQuantizedType(type)
    @ccall mlir_c.mlirTypeIsACalibratedQuantizedType(type::MlirType)::Bool
end

"""
    mlirCalibratedQuantizedTypeGet(expressedType, min, max)

Creates an instance of CalibratedQuantizedType with the given parameters in the same context as `expressedType` and returns it. The instance is owned by the context.
"""
function mlirCalibratedQuantizedTypeGet(expressedType, min, max)
    @ccall mlir_c.mlirCalibratedQuantizedTypeGet(
        expressedType::MlirType, min::Cdouble, max::Cdouble
    )::MlirType
end

"""
    mlirCalibratedQuantizedTypeGetMin(type)

Returns the min value of the given calibrated quantized type.
"""
function mlirCalibratedQuantizedTypeGetMin(type)
    @ccall mlir_c.mlirCalibratedQuantizedTypeGetMin(type::MlirType)::Cdouble
end

"""
    mlirCalibratedQuantizedTypeGetMax(type)

Returns the max value of the given calibrated quantized type.
"""
function mlirCalibratedQuantizedTypeGetMax(type)
    @ccall mlir_c.mlirCalibratedQuantizedTypeGetMax(type::MlirType)::Cdouble
end

function mlirGetDialectHandle__rocdl__()
    @ccall mlir_c.mlirGetDialectHandle__rocdl__()::MlirDialectHandle
end

function mlirGetDialectHandle__scf__()
    @ccall mlir_c.mlirGetDialectHandle__scf__()::MlirDialectHandle
end

function mlirGetDialectHandle__smt__()
    @ccall mlir_c.mlirGetDialectHandle__smt__()::MlirDialectHandle
end

"""
    mlirSMTTypeIsAnyNonFuncSMTValueType(type)

Checks if the given type is any non-func SMT value type.
"""
function mlirSMTTypeIsAnyNonFuncSMTValueType(type)
    @ccall mlir_c.mlirSMTTypeIsAnyNonFuncSMTValueType(type::MlirType)::Bool
end

"""
    mlirSMTTypeIsAnySMTValueType(type)

Checks if the given type is any SMT value type.
"""
function mlirSMTTypeIsAnySMTValueType(type)
    @ccall mlir_c.mlirSMTTypeIsAnySMTValueType(type::MlirType)::Bool
end

"""
    mlirSMTTypeIsAArray(type)

Checks if the given type is a smt::ArrayType.
"""
function mlirSMTTypeIsAArray(type)
    @ccall mlir_c.mlirSMTTypeIsAArray(type::MlirType)::Bool
end

"""
    mlirSMTTypeGetArray(ctx, domainType, rangeType)

Creates an array type with the given domain and range types.
"""
function mlirSMTTypeGetArray(ctx, domainType, rangeType)
    @ccall mlir_c.mlirSMTTypeGetArray(
        ctx::MlirContext, domainType::MlirType, rangeType::MlirType
    )::MlirType
end

"""
    mlirSMTTypeIsABitVector(type)

Checks if the given type is a smt::BitVectorType.
"""
function mlirSMTTypeIsABitVector(type)
    @ccall mlir_c.mlirSMTTypeIsABitVector(type::MlirType)::Bool
end

"""
    mlirSMTTypeGetBitVector(ctx, width)

Creates a smt::BitVectorType with the given width.
"""
function mlirSMTTypeGetBitVector(ctx, width)
    @ccall mlir_c.mlirSMTTypeGetBitVector(ctx::MlirContext, width::Int32)::MlirType
end

"""
    mlirSMTTypeIsABool(type)

Checks if the given type is a smt::BoolType.
"""
function mlirSMTTypeIsABool(type)
    @ccall mlir_c.mlirSMTTypeIsABool(type::MlirType)::Bool
end

"""
    mlirSMTTypeGetBool(ctx)

Creates a smt::BoolType.
"""
function mlirSMTTypeGetBool(ctx)
    @ccall mlir_c.mlirSMTTypeGetBool(ctx::MlirContext)::MlirType
end

"""
    mlirSMTTypeIsAInt(type)

Checks if the given type is a smt::IntType.
"""
function mlirSMTTypeIsAInt(type)
    @ccall mlir_c.mlirSMTTypeIsAInt(type::MlirType)::Bool
end

"""
    mlirSMTTypeGetInt(ctx)

Creates a smt::IntType.
"""
function mlirSMTTypeGetInt(ctx)
    @ccall mlir_c.mlirSMTTypeGetInt(ctx::MlirContext)::MlirType
end

"""
    mlirSMTTypeIsASMTFunc(type)

Checks if the given type is a smt::FuncType.
"""
function mlirSMTTypeIsASMTFunc(type)
    @ccall mlir_c.mlirSMTTypeIsASMTFunc(type::MlirType)::Bool
end

"""
    mlirSMTTypeGetSMTFunc(ctx, numberOfDomainTypes, domainTypes, rangeType)

Creates a smt::FuncType with the given domain and range types.
"""
function mlirSMTTypeGetSMTFunc(ctx, numberOfDomainTypes, domainTypes, rangeType)
    @ccall mlir_c.mlirSMTTypeGetSMTFunc(
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
    @ccall mlir_c.mlirSMTTypeIsASort(type::MlirType)::Bool
end

"""
    mlirSMTTypeGetSort(ctx, identifier, numberOfSortParams, sortParams)

Creates a smt::SortType with the given identifier and sort parameters.
"""
function mlirSMTTypeGetSort(ctx, identifier, numberOfSortParams, sortParams)
    @ccall mlir_c.mlirSMTTypeGetSort(
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
    @ccall mlir_c.mlirSMTAttrCheckBVCmpPredicate(ctx::MlirContext, str::MlirStringRef)::Bool
end

"""
    mlirSMTAttrCheckIntPredicate(ctx, str)

Checks if the given string is a valid smt::IntPredicate.
"""
function mlirSMTAttrCheckIntPredicate(ctx, str)
    @ccall mlir_c.mlirSMTAttrCheckIntPredicate(ctx::MlirContext, str::MlirStringRef)::Bool
end

"""
    mlirSMTAttrIsASMTAttribute(attr)

Checks if the given attribute is a smt::SMTAttribute.
"""
function mlirSMTAttrIsASMTAttribute(attr)
    @ccall mlir_c.mlirSMTAttrIsASMTAttribute(attr::MlirAttribute)::Bool
end

"""
    mlirSMTAttrGetBitVector(ctx, value, width)

Creates a smt::BitVectorAttr with the given value and width.
"""
function mlirSMTAttrGetBitVector(ctx, value, width)
    @ccall mlir_c.mlirSMTAttrGetBitVector(
        ctx::MlirContext, value::UInt64, width::Cuint
    )::MlirAttribute
end

"""
    mlirSMTAttrGetBVCmpPredicate(ctx, str)

Creates a smt::BVCmpPredicateAttr with the given string.
"""
function mlirSMTAttrGetBVCmpPredicate(ctx, str)
    @ccall mlir_c.mlirSMTAttrGetBVCmpPredicate(
        ctx::MlirContext, str::MlirStringRef
    )::MlirAttribute
end

"""
    mlirSMTAttrGetIntPredicate(ctx, str)

Creates a smt::IntPredicateAttr with the given string.
"""
function mlirSMTAttrGetIntPredicate(ctx, str)
    @ccall mlir_c.mlirSMTAttrGetIntPredicate(
        ctx::MlirContext, str::MlirStringRef
    )::MlirAttribute
end

function mlirGetDialectHandle__spirv__()
    @ccall mlir_c.mlirGetDialectHandle__spirv__()::MlirDialectHandle
end

function mlirGetDialectHandle__shape__()
    @ccall mlir_c.mlirGetDialectHandle__shape__()::MlirDialectHandle
end

function mlirGetDialectHandle__sparse_tensor__()
    @ccall mlir_c.mlirGetDialectHandle__sparse_tensor__()::MlirDialectHandle
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
    @ccall mlir_c.mlirAttributeIsASparseTensorEncodingAttr(attr::MlirAttribute)::Bool
end

"""
    mlirSparseTensorEncodingAttrGet(ctx, lvlRank, lvlTypes, dimToLvl, lvlTodim, posWidth, crdWidth, explicitVal, implicitVal)

Creates a `sparse\\_tensor.encoding` attribute with the given parameters.
"""
function mlirSparseTensorEncodingAttrGet(
    ctx, lvlRank, lvlTypes, dimToLvl, lvlTodim, posWidth, crdWidth, explicitVal, implicitVal
)
    @ccall mlir_c.mlirSparseTensorEncodingAttrGet(
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

"""
    mlirSparseTensorEncodingGetLvlRank(attr)

Returns the level-rank of the `sparse\\_tensor.encoding` attribute.
"""
function mlirSparseTensorEncodingGetLvlRank(attr)
    @ccall mlir_c.mlirSparseTensorEncodingGetLvlRank(attr::MlirAttribute)::Cptrdiff_t
end

"""
    mlirSparseTensorEncodingAttrGetLvlType(attr, lvl)

Returns a specified level-type of the `sparse\\_tensor.encoding` attribute.
"""
function mlirSparseTensorEncodingAttrGetLvlType(attr, lvl)
    @ccall mlir_c.mlirSparseTensorEncodingAttrGetLvlType(
        attr::MlirAttribute, lvl::Cptrdiff_t
    )::MlirSparseTensorLevelType
end

"""
    mlirSparseTensorEncodingAttrGetLvlFmt(attr, lvl)

Returns a specified level-format of the `sparse\\_tensor.encoding` attribute.
"""
function mlirSparseTensorEncodingAttrGetLvlFmt(attr, lvl)
    @ccall mlir_c.mlirSparseTensorEncodingAttrGetLvlFmt(
        attr::MlirAttribute, lvl::Cptrdiff_t
    )::MlirSparseTensorLevelFormat
end

"""
    mlirSparseTensorEncodingAttrGetDimToLvl(attr)

Returns the dimension-to-level mapping of the `sparse\\_tensor.encoding` attribute.
"""
function mlirSparseTensorEncodingAttrGetDimToLvl(attr)
    @ccall mlir_c.mlirSparseTensorEncodingAttrGetDimToLvl(
        attr::MlirAttribute
    )::MlirAffineMap
end

"""
    mlirSparseTensorEncodingAttrGetLvlToDim(attr)

Returns the level-to-dimension mapping of the `sparse\\_tensor.encoding` attribute.
"""
function mlirSparseTensorEncodingAttrGetLvlToDim(attr)
    @ccall mlir_c.mlirSparseTensorEncodingAttrGetLvlToDim(
        attr::MlirAttribute
    )::MlirAffineMap
end

"""
    mlirSparseTensorEncodingAttrGetPosWidth(attr)

Returns the position bitwidth of the `sparse\\_tensor.encoding` attribute.
"""
function mlirSparseTensorEncodingAttrGetPosWidth(attr)
    @ccall mlir_c.mlirSparseTensorEncodingAttrGetPosWidth(attr::MlirAttribute)::Cint
end

"""
    mlirSparseTensorEncodingAttrGetCrdWidth(attr)

Returns the coordinate bitwidth of the `sparse\\_tensor.encoding` attribute.
"""
function mlirSparseTensorEncodingAttrGetCrdWidth(attr)
    @ccall mlir_c.mlirSparseTensorEncodingAttrGetCrdWidth(attr::MlirAttribute)::Cint
end

"""
    mlirSparseTensorEncodingAttrGetExplicitVal(attr)

Returns the explicit value of the `sparse\\_tensor.encoding` attribute.
"""
function mlirSparseTensorEncodingAttrGetExplicitVal(attr)
    @ccall mlir_c.mlirSparseTensorEncodingAttrGetExplicitVal(
        attr::MlirAttribute
    )::MlirAttribute
end

"""
    mlirSparseTensorEncodingAttrGetImplicitVal(attr)

Returns the implicit value of the `sparse\\_tensor.encoding` attribute.
"""
function mlirSparseTensorEncodingAttrGetImplicitVal(attr)
    @ccall mlir_c.mlirSparseTensorEncodingAttrGetImplicitVal(
        attr::MlirAttribute
    )::MlirAttribute
end

function mlirSparseTensorEncodingAttrGetStructuredN(lvlType)
    @ccall mlir_c.mlirSparseTensorEncodingAttrGetStructuredN(
        lvlType::MlirSparseTensorLevelType
    )::Cuint
end

function mlirSparseTensorEncodingAttrGetStructuredM(lvlType)
    @ccall mlir_c.mlirSparseTensorEncodingAttrGetStructuredM(
        lvlType::MlirSparseTensorLevelType
    )::Cuint
end

function mlirSparseTensorEncodingAttrBuildLvlType(lvlFmt, properties, propSize, n, m)
    @ccall mlir_c.mlirSparseTensorEncodingAttrBuildLvlType(
        lvlFmt::MlirSparseTensorLevelFormat,
        properties::Ptr{MlirSparseTensorLevelPropertyNondefault},
        propSize::Cuint,
        n::Cuint,
        m::Cuint,
    )::MlirSparseTensorLevelType
end

function mlirGetDialectHandle__tensor__()
    @ccall mlir_c.mlirGetDialectHandle__tensor__()::MlirDialectHandle
end

function mlirGetDialectHandle__transform__()
    @ccall mlir_c.mlirGetDialectHandle__transform__()::MlirDialectHandle
end

function mlirTypeIsATransformAnyOpType(type)
    @ccall mlir_c.mlirTypeIsATransformAnyOpType(type::MlirType)::Bool
end

function mlirTransformAnyOpTypeGetTypeID()
    @ccall mlir_c.mlirTransformAnyOpTypeGetTypeID()::MlirTypeID
end

function mlirTransformAnyOpTypeGet(ctx)
    @ccall mlir_c.mlirTransformAnyOpTypeGet(ctx::MlirContext)::MlirType
end

function mlirTypeIsATransformAnyParamType(type)
    @ccall mlir_c.mlirTypeIsATransformAnyParamType(type::MlirType)::Bool
end

function mlirTransformAnyParamTypeGetTypeID()
    @ccall mlir_c.mlirTransformAnyParamTypeGetTypeID()::MlirTypeID
end

function mlirTransformAnyParamTypeGet(ctx)
    @ccall mlir_c.mlirTransformAnyParamTypeGet(ctx::MlirContext)::MlirType
end

function mlirTypeIsATransformAnyValueType(type)
    @ccall mlir_c.mlirTypeIsATransformAnyValueType(type::MlirType)::Bool
end

function mlirTransformAnyValueTypeGetTypeID()
    @ccall mlir_c.mlirTransformAnyValueTypeGetTypeID()::MlirTypeID
end

function mlirTransformAnyValueTypeGet(ctx)
    @ccall mlir_c.mlirTransformAnyValueTypeGet(ctx::MlirContext)::MlirType
end

function mlirTypeIsATransformOperationType(type)
    @ccall mlir_c.mlirTypeIsATransformOperationType(type::MlirType)::Bool
end

function mlirTransformOperationTypeGetTypeID()
    @ccall mlir_c.mlirTransformOperationTypeGetTypeID()::MlirTypeID
end

function mlirTransformOperationTypeGet(ctx, operationName)
    @ccall mlir_c.mlirTransformOperationTypeGet(
        ctx::MlirContext, operationName::MlirStringRef
    )::MlirType
end

function mlirTransformOperationTypeGetOperationName(type)
    @ccall mlir_c.mlirTransformOperationTypeGetOperationName(type::MlirType)::MlirStringRef
end

function mlirTypeIsATransformParamType(type)
    @ccall mlir_c.mlirTypeIsATransformParamType(type::MlirType)::Bool
end

function mlirTransformParamTypeGetTypeID()
    @ccall mlir_c.mlirTransformParamTypeGetTypeID()::MlirTypeID
end

function mlirTransformParamTypeGet(ctx, type)
    @ccall mlir_c.mlirTransformParamTypeGet(ctx::MlirContext, type::MlirType)::MlirType
end

function mlirTransformParamTypeGetType(type)
    @ccall mlir_c.mlirTransformParamTypeGetType(type::MlirType)::MlirType
end

struct MlirTransformOptions
    ptr::Ptr{Cvoid}
end

"""
    mlirTransformOptionsCreate()

Creates a default-initialized transform options object.
"""
function mlirTransformOptionsCreate()
    @ccall mlir_c.mlirTransformOptionsCreate()::MlirTransformOptions
end

"""
    mlirTransformOptionsEnableExpensiveChecks(transformOptions, enable)

Enables or disables expensive checks in transform options.
"""
function mlirTransformOptionsEnableExpensiveChecks(transformOptions, enable)
    @ccall mlir_c.mlirTransformOptionsEnableExpensiveChecks(
        transformOptions::MlirTransformOptions, enable::Bool
    )::Cvoid
end

"""
    mlirTransformOptionsGetExpensiveChecksEnabled(transformOptions)

Returns true if expensive checks are enabled in transform options.
"""
function mlirTransformOptionsGetExpensiveChecksEnabled(transformOptions)
    @ccall mlir_c.mlirTransformOptionsGetExpensiveChecksEnabled(
        transformOptions::MlirTransformOptions
    )::Bool
end

"""
    mlirTransformOptionsEnforceSingleTopLevelTransformOp(transformOptions, enable)

Enables or disables the enforcement of the top-level transform op being single in transform options.
"""
function mlirTransformOptionsEnforceSingleTopLevelTransformOp(transformOptions, enable)
    @ccall mlir_c.mlirTransformOptionsEnforceSingleTopLevelTransformOp(
        transformOptions::MlirTransformOptions, enable::Bool
    )::Cvoid
end

"""
    mlirTransformOptionsGetEnforceSingleTopLevelTransformOp(transformOptions)

Returns true if the enforcement of the top-level transform op being single is enabled in transform options.
"""
function mlirTransformOptionsGetEnforceSingleTopLevelTransformOp(transformOptions)
    @ccall mlir_c.mlirTransformOptionsGetEnforceSingleTopLevelTransformOp(
        transformOptions::MlirTransformOptions
    )::Bool
end

"""
    mlirTransformOptionsDestroy(transformOptions)

Destroys a transform options object previously created by [`mlirTransformOptionsCreate`](@ref).
"""
function mlirTransformOptionsDestroy(transformOptions)
    @ccall mlir_c.mlirTransformOptionsDestroy(transformOptions::MlirTransformOptions)::Cvoid
end

"""
    mlirTransformApplyNamedSequence(payload, transformRoot, transformModule, transformOptions)

Applies the transformation script starting at the given transform root operation to the given payload operation. The module containing the transform root as well as the transform options should be provided. The transform operation must implement TransformOpInterface and the module must be a ModuleOp. Returns the status of the application.
"""
function mlirTransformApplyNamedSequence(
    payload, transformRoot, transformModule, transformOptions
)
    @ccall mlir_c.mlirTransformApplyNamedSequence(
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
    @ccall mlir_c.mlirMergeSymbolsIntoFromClone(
        target::MlirOperation, other::MlirOperation
    )::MlirLogicalResult
end

function mlirGetDialectHandle__vector__()
    @ccall mlir_c.mlirGetDialectHandle__vector__()::MlirDialectHandle
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
    @ccall mlir_c.mlirExecutionEngineCreate(
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
    @ccall mlir_c.mlirExecutionEngineInitialize(jit::MlirExecutionEngine)::Cvoid
end

"""
    mlirExecutionEngineDestroy(jit)

Destroy an ExecutionEngine instance.
"""
function mlirExecutionEngineDestroy(jit)
    @ccall mlir_c.mlirExecutionEngineDestroy(jit::MlirExecutionEngine)::Cvoid
end

"""
    mlirExecutionEngineIsNull(jit)

Checks whether an execution engine is null.
"""
function mlirExecutionEngineIsNull(jit)
    @ccall mlir_c.mlirExecutionEngineIsNull(jit::MlirExecutionEngine)::Bool
end

"""
    mlirExecutionEngineInvokePacked(jit, name, arguments)

Invoke a native function in the execution engine by name with the arguments and result of the invoked function passed as an array of pointers. The function must have been tagged with the `llvm.emit\\_c\\_interface` attribute. Returns a failure if the execution fails for any reason (the function name can't be resolved for instance).
"""
function mlirExecutionEngineInvokePacked(jit, name, arguments)
    @ccall mlir_c.mlirExecutionEngineInvokePacked(
        jit::MlirExecutionEngine, name::MlirStringRef, arguments::Ptr{Ptr{Cvoid}}
    )::MlirLogicalResult
end

"""
    mlirExecutionEngineLookupPacked(jit, name)

Lookup the wrapper of the native function in the execution engine with the given name, returns nullptr if the function can't be looked-up.
"""
function mlirExecutionEngineLookupPacked(jit, name)
    @ccall mlir_c.mlirExecutionEngineLookupPacked(
        jit::MlirExecutionEngine, name::MlirStringRef
    )::Ptr{Cvoid}
end

"""
    mlirExecutionEngineLookup(jit, name)

Lookup a native function in the execution engine by name, returns nullptr if the name can't be looked-up.
"""
function mlirExecutionEngineLookup(jit, name)
    @ccall mlir_c.mlirExecutionEngineLookup(
        jit::MlirExecutionEngine, name::MlirStringRef
    )::Ptr{Cvoid}
end

"""
    mlirExecutionEngineRegisterSymbol(jit, name, sym)

Register a symbol with the jit: this symbol will be accessible to the jitted code.
"""
function mlirExecutionEngineRegisterSymbol(jit, name, sym)
    @ccall mlir_c.mlirExecutionEngineRegisterSymbol(
        jit::MlirExecutionEngine, name::MlirStringRef, sym::Ptr{Cvoid}
    )::Cvoid
end

"""
    mlirExecutionEngineDumpToObjectFile(jit, fileName)

Dump as an object in `fileName`.
"""
function mlirExecutionEngineDumpToObjectFile(jit, fileName)
    @ccall mlir_c.mlirExecutionEngineDumpToObjectFile(
        jit::MlirExecutionEngine, fileName::MlirStringRef
    )::Cvoid
end

"""
    mlirOperationImplementsInterface(operation, interfaceTypeID)

Returns `true` if the given operation implements an interface identified by its TypeID.
"""
function mlirOperationImplementsInterface(operation, interfaceTypeID)
    @ccall mlir_c.mlirOperationImplementsInterface(
        operation::MlirOperation, interfaceTypeID::MlirTypeID
    )::Bool
end

"""
    mlirOperationImplementsInterfaceStatic(operationName, context, interfaceTypeID)

Returns `true` if the operation identified by its canonical string name implements the interface identified by its TypeID in the given context. Note that interfaces may be attached to operations in some contexts and not others.
"""
function mlirOperationImplementsInterfaceStatic(operationName, context, interfaceTypeID)
    @ccall mlir_c.mlirOperationImplementsInterfaceStatic(
        operationName::MlirStringRef, context::MlirContext, interfaceTypeID::MlirTypeID
    )::Bool
end

"""
    mlirInferTypeOpInterfaceTypeID()

Returns the interface TypeID of the InferTypeOpInterface.
"""
function mlirInferTypeOpInterfaceTypeID()
    @ccall mlir_c.mlirInferTypeOpInterfaceTypeID()::MlirTypeID
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
    @ccall mlir_c.mlirInferTypeOpInterfaceInferReturnTypes(
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
    @ccall mlir_c.mlirInferShapedTypeOpInterfaceTypeID()::MlirTypeID
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
    @ccall mlir_c.mlirInferShapedTypeOpInterfaceInferReturnTypes(
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
    @ccall mlir_c.mlirPassManagerCreate(ctx::MlirContext)::MlirPassManager
end

"""
    mlirPassManagerCreateOnOperation(ctx, anchorOp)

Create a new top-level PassManager anchored on `anchorOp`.
"""
function mlirPassManagerCreateOnOperation(ctx, anchorOp)
    @ccall mlir_c.mlirPassManagerCreateOnOperation(
        ctx::MlirContext, anchorOp::MlirStringRef
    )::MlirPassManager
end

"""
    mlirPassManagerDestroy(passManager)

Destroy the provided PassManager.
"""
function mlirPassManagerDestroy(passManager)
    @ccall mlir_c.mlirPassManagerDestroy(passManager::MlirPassManager)::Cvoid
end

"""
    mlirPassManagerIsNull(passManager)

Checks if a PassManager is null.
"""
function mlirPassManagerIsNull(passManager)
    @ccall mlir_c.mlirPassManagerIsNull(passManager::MlirPassManager)::Bool
end

"""
    mlirPassManagerGetAsOpPassManager(passManager)

Cast a top-level PassManager to a generic OpPassManager.
"""
function mlirPassManagerGetAsOpPassManager(passManager)
    @ccall mlir_c.mlirPassManagerGetAsOpPassManager(
        passManager::MlirPassManager
    )::MlirOpPassManager
end

"""
    mlirPassManagerRunOnOp(passManager, op)

Run the provided `passManager` on the given `op`.
"""
function mlirPassManagerRunOnOp(passManager, op)
    @ccall mlir_c.mlirPassManagerRunOnOp(
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
    @ccall mlir_c.mlirPassManagerEnableIRPrinting(
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
    @ccall mlir_c.mlirPassManagerEnableVerifier(
        passManager::MlirPassManager, enable::Bool
    )::Cvoid
end

"""
    mlirPassManagerEnableTiming(passManager)

Enable pass timing.
"""
function mlirPassManagerEnableTiming(passManager)
    @ccall mlir_c.mlirPassManagerEnableTiming(passManager::MlirPassManager)::Cvoid
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
    @ccall mlir_c.mlirPassManagerEnableStatistics(
        passManager::MlirPassManager, displayMode::MlirPassDisplayMode
    )::Cvoid
end

"""
    mlirPassManagerGetNestedUnder(passManager, operationName)

Nest an OpPassManager under the top-level PassManager, the nested passmanager will only run on operations matching the provided name. The returned OpPassManager will be destroyed when the parent is destroyed. To further nest more OpPassManager under the newly returned one, see `mlirOpPassManagerNest` below.
"""
function mlirPassManagerGetNestedUnder(passManager, operationName)
    @ccall mlir_c.mlirPassManagerGetNestedUnder(
        passManager::MlirPassManager, operationName::MlirStringRef
    )::MlirOpPassManager
end

"""
    mlirOpPassManagerGetNestedUnder(passManager, operationName)

Nest an OpPassManager under the provided OpPassManager, the nested passmanager will only run on operations matching the provided name. The returned OpPassManager will be destroyed when the parent is destroyed.
"""
function mlirOpPassManagerGetNestedUnder(passManager, operationName)
    @ccall mlir_c.mlirOpPassManagerGetNestedUnder(
        passManager::MlirOpPassManager, operationName::MlirStringRef
    )::MlirOpPassManager
end

"""
    mlirPassManagerAddOwnedPass(passManager, pass)

Add a pass and transfer ownership to the provided top-level mlirPassManager. If the pass is not a generic operation pass or a ModulePass, a new OpPassManager is implicitly nested under the provided PassManager.
"""
function mlirPassManagerAddOwnedPass(passManager, pass)
    @ccall mlir_c.mlirPassManagerAddOwnedPass(
        passManager::MlirPassManager, pass::MlirPass
    )::Cvoid
end

"""
    mlirOpPassManagerAddOwnedPass(passManager, pass)

Add a pass and transfer ownership to the provided mlirOpPassManager. If the pass is not a generic operation pass or matching the type of the provided PassManager, a new OpPassManager is implicitly nested under the provided PassManager.
"""
function mlirOpPassManagerAddOwnedPass(passManager, pass)
    @ccall mlir_c.mlirOpPassManagerAddOwnedPass(
        passManager::MlirOpPassManager, pass::MlirPass
    )::Cvoid
end

"""
    mlirOpPassManagerAddPipeline(passManager, pipelineElements, callback, userData)

Parse a sequence of textual MLIR pass pipeline elements and add them to the provided OpPassManager. If parsing fails an error message is reported using the provided callback.
"""
function mlirOpPassManagerAddPipeline(passManager, pipelineElements, callback, userData)
    @ccall mlir_c.mlirOpPassManagerAddPipeline(
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
    @ccall mlir_c.mlirPrintPassPipeline(
        passManager::MlirOpPassManager, callback::MlirStringCallback, userData::Ptr{Cvoid}
    )::Cvoid
end

"""
    mlirParsePassPipeline(passManager, pipeline, callback, userData)

Parse a textual MLIR pass pipeline and assign it to the provided OpPassManager. If parsing fails an error message is reported using the provided callback.
"""
function mlirParsePassPipeline(passManager, pipeline, callback, userData)
    @ccall mlir_c.mlirParsePassPipeline(
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
    @ccall mlir_c.mlirCreateExternalPass(
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
    @ccall mlir_c.mlirExternalPassSignalFailure(pass::MlirExternalPass)::Cvoid
end

"""
    mlirRegisterAllDialects(registry)

Appends all upstream dialects and extensions to the dialect registry.
"""
function mlirRegisterAllDialects(registry)
    @ccall mlir_c.mlirRegisterAllDialects(registry::MlirDialectRegistry)::Cvoid
end

"""
    mlirRegisterAllLLVMTranslations(context)

Register all translations to LLVM IR for dialects that can support it.
"""
function mlirRegisterAllLLVMTranslations(context)
    @ccall mlir_c.mlirRegisterAllLLVMTranslations(context::MlirContext)::Cvoid
end

"""
    mlirRegisterAllPasses()

Register all compiler passes of MLIR.
"""
function mlirRegisterAllPasses()
    @ccall mlir_c.mlirRegisterAllPasses()::Cvoid
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

struct MlirRewritePatternSet
    ptr::Ptr{Cvoid}
end

struct MlirPatternRewriter
    ptr::Ptr{Cvoid}
end

struct MlirRewritePattern
    ptr::Ptr{Cvoid}
end

"""
    mlirRewriterBaseGetContext(rewriter)

Get the MLIR context referenced by the rewriter.
"""
function mlirRewriterBaseGetContext(rewriter)
    @ccall mlir_c.mlirRewriterBaseGetContext(rewriter::MlirRewriterBase)::MlirContext
end

"""
    mlirRewriterBaseClearInsertionPoint(rewriter)

Reset the insertion point to no location. Creating an operation without a set insertion point is an error, but this can still be useful when the current insertion point a builder refers to is being removed.
"""
function mlirRewriterBaseClearInsertionPoint(rewriter)
    @ccall mlir_c.mlirRewriterBaseClearInsertionPoint(rewriter::MlirRewriterBase)::Cvoid
end

"""
    mlirRewriterBaseSetInsertionPointBefore(rewriter, op)

Sets the insertion point to the specified operation, which will cause subsequent insertions to go right before it.
"""
function mlirRewriterBaseSetInsertionPointBefore(rewriter, op)
    @ccall mlir_c.mlirRewriterBaseSetInsertionPointBefore(
        rewriter::MlirRewriterBase, op::MlirOperation
    )::Cvoid
end

"""
    mlirRewriterBaseSetInsertionPointAfter(rewriter, op)

Sets the insertion point to the node after the specified operation, which will cause subsequent insertions to go right after it.
"""
function mlirRewriterBaseSetInsertionPointAfter(rewriter, op)
    @ccall mlir_c.mlirRewriterBaseSetInsertionPointAfter(
        rewriter::MlirRewriterBase, op::MlirOperation
    )::Cvoid
end

"""
    mlirRewriterBaseSetInsertionPointAfterValue(rewriter, value)

Sets the insertion point to the node after the specified value. If value has a defining operation, sets the insertion point to the node after such defining operation. This will cause subsequent insertions to go right after it. Otherwise, value is a BlockArgument. Sets the insertion point to the start of its block.
"""
function mlirRewriterBaseSetInsertionPointAfterValue(rewriter, value)
    @ccall mlir_c.mlirRewriterBaseSetInsertionPointAfterValue(
        rewriter::MlirRewriterBase, value::MlirValue
    )::Cvoid
end

"""
    mlirRewriterBaseSetInsertionPointToStart(rewriter, block)

Sets the insertion point to the start of the specified block.
"""
function mlirRewriterBaseSetInsertionPointToStart(rewriter, block)
    @ccall mlir_c.mlirRewriterBaseSetInsertionPointToStart(
        rewriter::MlirRewriterBase, block::MlirBlock
    )::Cvoid
end

"""
    mlirRewriterBaseSetInsertionPointToEnd(rewriter, block)

Sets the insertion point to the end of the specified block.
"""
function mlirRewriterBaseSetInsertionPointToEnd(rewriter, block)
    @ccall mlir_c.mlirRewriterBaseSetInsertionPointToEnd(
        rewriter::MlirRewriterBase, block::MlirBlock
    )::Cvoid
end

"""
    mlirRewriterBaseGetInsertionBlock(rewriter)

Return the block the current insertion point belongs to. Note that the insertion point is not necessarily the end of the block.
"""
function mlirRewriterBaseGetInsertionBlock(rewriter)
    @ccall mlir_c.mlirRewriterBaseGetInsertionBlock(rewriter::MlirRewriterBase)::MlirBlock
end

"""
    mlirRewriterBaseGetBlock(rewriter)

Returns the current block of the rewriter.
"""
function mlirRewriterBaseGetBlock(rewriter)
    @ccall mlir_c.mlirRewriterBaseGetBlock(rewriter::MlirRewriterBase)::MlirBlock
end

"""
    mlirRewriterBaseGetOperationAfterInsertion(rewriter)

Returns the operation right after the current insertion point of the rewriter. A null [`MlirOperation`](@ref) will be returned
"""
function mlirRewriterBaseGetOperationAfterInsertion(rewriter)
    @ccall mlir_c.mlirRewriterBaseGetOperationAfterInsertion(
        rewriter::MlirRewriterBase
    )::MlirOperation
end

"""
    mlirRewriterBaseCreateBlockBefore(rewriter, insertBefore, nArgTypes, argTypes, locations)

Add new block with 'argTypes' arguments and set the insertion point to the end of it. The block is placed before 'insertBefore'. `locs` contains the locations of the inserted arguments, and should match the size of `argTypes`.
"""
function mlirRewriterBaseCreateBlockBefore(
    rewriter, insertBefore, nArgTypes, argTypes, locations
)
    @ccall mlir_c.mlirRewriterBaseCreateBlockBefore(
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
    @ccall mlir_c.mlirRewriterBaseInsert(
        rewriter::MlirRewriterBase, op::MlirOperation
    )::MlirOperation
end

"""
    mlirRewriterBaseClone(rewriter, op)

Creates a deep copy of the specified operation.
"""
function mlirRewriterBaseClone(rewriter, op)
    @ccall mlir_c.mlirRewriterBaseClone(
        rewriter::MlirRewriterBase, op::MlirOperation
    )::MlirOperation
end

"""
    mlirRewriterBaseCloneWithoutRegions(rewriter, op)

Creates a deep copy of this operation but keep the operation regions empty.
"""
function mlirRewriterBaseCloneWithoutRegions(rewriter, op)
    @ccall mlir_c.mlirRewriterBaseCloneWithoutRegions(
        rewriter::MlirRewriterBase, op::MlirOperation
    )::MlirOperation
end

"""
    mlirRewriterBaseCloneRegionBefore(rewriter, region, before)

Clone the blocks that belong to "region" before the given position in another region "parent".
"""
function mlirRewriterBaseCloneRegionBefore(rewriter, region, before)
    @ccall mlir_c.mlirRewriterBaseCloneRegionBefore(
        rewriter::MlirRewriterBase, region::MlirRegion, before::MlirBlock
    )::Cvoid
end

"""
    mlirRewriterBaseInlineRegionBefore(rewriter, region, before)

Move the blocks that belong to "region" before the given position in another region "parent". The two regions must be different. The caller is responsible for creating or updating the operation transferring flow of control to the region and passing it the correct block arguments.
"""
function mlirRewriterBaseInlineRegionBefore(rewriter, region, before)
    @ccall mlir_c.mlirRewriterBaseInlineRegionBefore(
        rewriter::MlirRewriterBase, region::MlirRegion, before::MlirBlock
    )::Cvoid
end

"""
    mlirRewriterBaseReplaceOpWithValues(rewriter, op, nValues, values)

Replace the results of the given (original) operation with the specified list of values (replacements). The result types of the given op and the replacements must match. The original op is erased.
"""
function mlirRewriterBaseReplaceOpWithValues(rewriter, op, nValues, values)
    @ccall mlir_c.mlirRewriterBaseReplaceOpWithValues(
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
    @ccall mlir_c.mlirRewriterBaseReplaceOpWithOperation(
        rewriter::MlirRewriterBase, op::MlirOperation, newOp::MlirOperation
    )::Cvoid
end

"""
    mlirRewriterBaseEraseOp(rewriter, op)

Erases an operation that is known to have no uses.
"""
function mlirRewriterBaseEraseOp(rewriter, op)
    @ccall mlir_c.mlirRewriterBaseEraseOp(
        rewriter::MlirRewriterBase, op::MlirOperation
    )::Cvoid
end

"""
    mlirRewriterBaseEraseBlock(rewriter, block)

Erases a block along with all operations inside it.
"""
function mlirRewriterBaseEraseBlock(rewriter, block)
    @ccall mlir_c.mlirRewriterBaseEraseBlock(
        rewriter::MlirRewriterBase, block::MlirBlock
    )::Cvoid
end

"""
    mlirRewriterBaseInlineBlockBefore(rewriter, source, op, nArgValues, argValues)

Inline the operations of block 'source' before the operation 'op'. The source block will be deleted and must have no uses. 'argValues' is used to replace the block arguments of 'source'

The source block must have no successors. Otherwise, the resulting IR would have unreachable operations.
"""
function mlirRewriterBaseInlineBlockBefore(rewriter, source, op, nArgValues, argValues)
    @ccall mlir_c.mlirRewriterBaseInlineBlockBefore(
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
    @ccall mlir_c.mlirRewriterBaseMergeBlocks(
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
    @ccall mlir_c.mlirRewriterBaseMoveOpBefore(
        rewriter::MlirRewriterBase, op::MlirOperation, existingOp::MlirOperation
    )::Cvoid
end

"""
    mlirRewriterBaseMoveOpAfter(rewriter, op, existingOp)

Unlink this operation from its current block and insert it right after `existingOp` which may be in the same or another block in the same function.
"""
function mlirRewriterBaseMoveOpAfter(rewriter, op, existingOp)
    @ccall mlir_c.mlirRewriterBaseMoveOpAfter(
        rewriter::MlirRewriterBase, op::MlirOperation, existingOp::MlirOperation
    )::Cvoid
end

"""
    mlirRewriterBaseMoveBlockBefore(rewriter, block, existingBlock)

Unlink this block and insert it right before `existingBlock`.
"""
function mlirRewriterBaseMoveBlockBefore(rewriter, block, existingBlock)
    @ccall mlir_c.mlirRewriterBaseMoveBlockBefore(
        rewriter::MlirRewriterBase, block::MlirBlock, existingBlock::MlirBlock
    )::Cvoid
end

"""
    mlirRewriterBaseStartOpModification(rewriter, op)

This method is used to notify the rewriter that an in-place operation modification is about to happen. A call to this function *must* be followed by a call to either `finalizeOpModification` or `cancelOpModification`. This is a minor efficiency win (it avoids creating a new operation and removing the old one) but also often allows simpler code in the client.
"""
function mlirRewriterBaseStartOpModification(rewriter, op)
    @ccall mlir_c.mlirRewriterBaseStartOpModification(
        rewriter::MlirRewriterBase, op::MlirOperation
    )::Cvoid
end

"""
    mlirRewriterBaseFinalizeOpModification(rewriter, op)

This method is used to signal the end of an in-place modification of the given operation. This can only be called on operations that were provided to a call to `startOpModification`.
"""
function mlirRewriterBaseFinalizeOpModification(rewriter, op)
    @ccall mlir_c.mlirRewriterBaseFinalizeOpModification(
        rewriter::MlirRewriterBase, op::MlirOperation
    )::Cvoid
end

"""
    mlirRewriterBaseCancelOpModification(rewriter, op)

This method cancels a pending in-place modification. This can only be called on operations that were provided to a call to `startOpModification`.
"""
function mlirRewriterBaseCancelOpModification(rewriter, op)
    @ccall mlir_c.mlirRewriterBaseCancelOpModification(
        rewriter::MlirRewriterBase, op::MlirOperation
    )::Cvoid
end

"""
    mlirRewriterBaseReplaceAllUsesWith(rewriter, from, to)

Find uses of `from` and replace them with `to`. Also notify the listener about every in-place op modification (for every use that was replaced).
"""
function mlirRewriterBaseReplaceAllUsesWith(rewriter, from, to)
    @ccall mlir_c.mlirRewriterBaseReplaceAllUsesWith(
        rewriter::MlirRewriterBase, from::MlirValue, to::MlirValue
    )::Cvoid
end

"""
    mlirRewriterBaseReplaceAllValueRangeUsesWith(rewriter, nValues, from, to)

Find uses of `from` and replace them with `to`. Also notify the listener about every in-place op modification (for every use that was replaced).
"""
function mlirRewriterBaseReplaceAllValueRangeUsesWith(rewriter, nValues, from, to)
    @ccall mlir_c.mlirRewriterBaseReplaceAllValueRangeUsesWith(
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
    @ccall mlir_c.mlirRewriterBaseReplaceAllOpUsesWithValueRange(
        rewriter::MlirRewriterBase, from::MlirOperation, nTo::Cptrdiff_t, to::Ptr{MlirValue}
    )::Cvoid
end

"""
    mlirRewriterBaseReplaceAllOpUsesWithOperation(rewriter, from, to)

Find uses of `from` and replace them with `to`. Also notify the listener about every in-place op modification (for every use that was replaced) and that the `from` operation is about to be replaced.
"""
function mlirRewriterBaseReplaceAllOpUsesWithOperation(rewriter, from, to)
    @ccall mlir_c.mlirRewriterBaseReplaceAllOpUsesWithOperation(
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
    @ccall mlir_c.mlirRewriterBaseReplaceOpUsesWithinBlock(
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
    @ccall mlir_c.mlirRewriterBaseReplaceAllUsesExcept(
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
    @ccall mlir_c.mlirIRRewriterCreate(context::MlirContext)::MlirRewriterBase
end

"""
    mlirIRRewriterCreateFromOp(op)

Create an IRRewriter and transfer ownership to the caller. Additionally set the insertion point before the operation.
"""
function mlirIRRewriterCreateFromOp(op)
    @ccall mlir_c.mlirIRRewriterCreateFromOp(op::MlirOperation)::MlirRewriterBase
end

"""
    mlirIRRewriterDestroy(rewriter)

Takes an IRRewriter owned by the caller and destroys it. It is the responsibility of the user to only pass an IRRewriter class.
"""
function mlirIRRewriterDestroy(rewriter)
    @ccall mlir_c.mlirIRRewriterDestroy(rewriter::MlirRewriterBase)::Cvoid
end

"""
    mlirFreezeRewritePattern(set)

Freeze the given [`MlirRewritePatternSet`](@ref) to a [`MlirFrozenRewritePatternSet`](@ref). Note that the ownership of the input set is transferred into the frozen set after this call.
"""
function mlirFreezeRewritePattern(set)
    @ccall mlir_c.mlirFreezeRewritePattern(
        set::MlirRewritePatternSet
    )::MlirFrozenRewritePatternSet
end

"""
    mlirFrozenRewritePatternSetDestroy(set)

Destroy the given [`MlirFrozenRewritePatternSet`](@ref).
"""
function mlirFrozenRewritePatternSetDestroy(set)
    @ccall mlir_c.mlirFrozenRewritePatternSetDestroy(
        set::MlirFrozenRewritePatternSet
    )::Cvoid
end

function mlirApplyPatternsAndFoldGreedilyWithOp(op, patterns, arg3)
    @ccall mlir_c.mlirApplyPatternsAndFoldGreedilyWithOp(
        op::MlirOperation,
        patterns::MlirFrozenRewritePatternSet,
        arg3::MlirGreedyRewriteDriverConfig,
    )::MlirLogicalResult
end

function mlirApplyPatternsAndFoldGreedily(op, patterns, arg3)
    @ccall mlir_c.mlirApplyPatternsAndFoldGreedily(
        op::MlirModule,
        patterns::MlirFrozenRewritePatternSet,
        arg3::MlirGreedyRewriteDriverConfig,
    )::MlirLogicalResult
end

"""
    mlirPatternRewriterAsBase(rewriter)

Cast the PatternRewriter to a RewriterBase
"""
function mlirPatternRewriterAsBase(rewriter)
    @ccall mlir_c.mlirPatternRewriterAsBase(rewriter::MlirPatternRewriter)::MlirRewriterBase
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
    @ccall mlir_c.mlirOpRewritePatternCreate(
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
    @ccall mlir_c.mlirRewritePatternSetCreate(context::MlirContext)::MlirRewritePatternSet
end

"""
    mlirRewritePatternSetDestroy(set)

Destruct the given [`MlirRewritePatternSet`](@ref).
"""
function mlirRewritePatternSetDestroy(set)
    @ccall mlir_c.mlirRewritePatternSetDestroy(set::MlirRewritePatternSet)::Cvoid
end

"""
    mlirRewritePatternSetAdd(set, pattern)

Add the given [`MlirRewritePattern`](@ref) into a [`MlirRewritePatternSet`](@ref). Note that the ownership of the pattern is transferred to the set after this call.
"""
function mlirRewritePatternSetAdd(set, pattern)
    @ccall mlir_c.mlirRewritePatternSetAdd(
        set::MlirRewritePatternSet, pattern::MlirRewritePattern
    )::Cvoid
end

"""
    mlirTranslateModuleToSMTLIB(arg1, arg2, userData, inlineSingleUseValues, indentLetBody)

Emits SMTLIB for the specified module using the provided callback and user data
"""
function mlirTranslateModuleToSMTLIB(
    arg1, arg2, userData, inlineSingleUseValues, indentLetBody
)
    @ccall mlir_c.mlirTranslateModuleToSMTLIB(
        arg1::MlirModule,
        arg2::MlirStringCallback,
        userData::Ptr{Cvoid},
        inlineSingleUseValues::Bool,
        indentLetBody::Bool,
    )::MlirLogicalResult
end

function mlirTranslateOperationToSMTLIB(
    arg1, arg2, userData, inlineSingleUseValues, indentLetBody
)
    @ccall mlir_c.mlirTranslateOperationToSMTLIB(
        arg1::MlirOperation,
        arg2::MlirStringCallback,
        userData::Ptr{Cvoid},
        inlineSingleUseValues::Bool,
        indentLetBody::Bool,
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
The top-level container for all LLVM global data. See the LLVMContext class.
"""
const LLVMContextRef = Ptr{LLVMOpaqueContext}

mutable struct LLVMOpaqueModule end

"""
The top-level container for all other LLVM Intermediate Representation (IR) objects.

# See also
llvm::Module
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
Interface used to provide a module to JIT or interpreter. This is now just a synonym for llvm::Module, but we have to keep using the different type to keep binary compatibility.
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
llvm::Module::ModuleFlagEntry
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
    @ccall mlir_c.LLVMParseCommandLineOptions(
        argc::Cint, argv::Ptr{Cstring}, Overview::Cstring
    )::Cint
end

function LLVMSearchForAddressOfSymbol(symbolName)
    @ccall mlir_c.LLVMSearchForAddressOfSymbol(symbolName::Cstring)::Ptr{Cint}
end

function LLVMAddSymbol(symbolName, symbolValue)
    @ccall mlir_c.LLVMAddSymbol(symbolName::Cstring, symbolValue::Ptr{Cvoid})::Cint
end

"""
    mlirTranslateModuleToLLVMIR(_module, context)

Translate operation that satisfies LLVM dialect module requirements into an LLVM IR module living in the given context. This translates operations from any dilalect that has a registered implementation of LLVMTranslationDialectInterface.

# Returns
the generated LLVM IR Module from the translated MLIR module, it is owned by the caller.
"""
function mlirTranslateModuleToLLVMIR(_module, context)
    @ccall mlir_c.mlirTranslateModuleToLLVMIR(
        _module::MlirOperation, context::LLVMContextRef
    )::LLVMModuleRef
end

function mlirTranslateModuleToLLVMIRToString(_module)
    @ccall mlir_c.mlirTranslateModuleToLLVMIRToString(_module::MlirOperation)::Cstring
end

struct MlirTypeFromLLVMIRTranslator
    ptr::Ptr{Cvoid}
end

"""
    mlirTypeFromLLVMIRTranslatorCreate(ctx)

Create an LLVM::TypeFromLLVMIRTranslator and transfer ownership to the caller.
"""
function mlirTypeFromLLVMIRTranslatorCreate(ctx)
    @ccall mlir_c.mlirTypeFromLLVMIRTranslatorCreate(
        ctx::MlirContext
    )::MlirTypeFromLLVMIRTranslator
end

"""
    mlirTypeFromLLVMIRTranslatorDestroy(translator)

Takes an LLVM::TypeFromLLVMIRTranslator owned by the caller and destroys it. It is the responsibility of the user to only pass an LLVM::TypeFromLLVMIRTranslator class.
"""
function mlirTypeFromLLVMIRTranslatorDestroy(translator)
    @ccall mlir_c.mlirTypeFromLLVMIRTranslatorDestroy(
        translator::MlirTypeFromLLVMIRTranslator
    )::Cvoid
end

"""
    mlirTypeFromLLVMIRTranslatorTranslateType(translator, llvmType)

Translates the given LLVM IR type to the MLIR LLVM dialect.
"""
function mlirTypeFromLLVMIRTranslatorTranslateType(translator, llvmType)
    @ccall mlir_c.mlirTypeFromLLVMIRTranslatorTranslateType(
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
    @ccall mlir_c.mlirTypeToLLVMIRTranslatorCreate(
        ctx::LLVMContextRef
    )::MlirTypeToLLVMIRTranslator
end

"""
    mlirTypeToLLVMIRTranslatorDestroy(translator)

Takes an LLVM::TypeToLLVMIRTranslator owned by the caller and destroys it. It is the responsibility of the user to only pass an LLVM::TypeToLLVMIRTranslator class.
"""
function mlirTypeToLLVMIRTranslatorDestroy(translator)
    @ccall mlir_c.mlirTypeToLLVMIRTranslatorDestroy(
        translator::MlirTypeToLLVMIRTranslator
    )::Cvoid
end

"""
    mlirTypeToLLVMIRTranslatorTranslateType(translator, mlirType)

Translates the given MLIR LLVM dialect to the LLVM IR type.
"""
function mlirTypeToLLVMIRTranslatorTranslateType(translator, mlirType)
    @ccall mlir_c.mlirTypeToLLVMIRTranslatorTranslateType(
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
    @ccall mlir_c.stablehloScatterDimensionNumbersGet(
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
    @ccall mlir_c.stablehloAttributeIsAScatterDimensionNumbers(attr::MlirAttribute)::Bool
end

function stablehloScatterDimensionNumbersGetUpdateWindowDimsSize(attr)
    @ccall mlir_c.stablehloScatterDimensionNumbersGetUpdateWindowDimsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function stablehloScatterDimensionNumbersGetUpdateWindowDimsElem(attr, pos)
    @ccall mlir_c.stablehloScatterDimensionNumbersGetUpdateWindowDimsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function stablehloScatterDimensionNumbersGetInsertedWindowDimsSize(attr)
    @ccall mlir_c.stablehloScatterDimensionNumbersGetInsertedWindowDimsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function stablehloScatterDimensionNumbersGetInsertedWindowDimsElem(attr, pos)
    @ccall mlir_c.stablehloScatterDimensionNumbersGetInsertedWindowDimsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function stablehloScatterDimensionNumbersGetInputBatchingDimsSize(attr)
    @ccall mlir_c.stablehloScatterDimensionNumbersGetInputBatchingDimsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function stablehloScatterDimensionNumbersGetInputBatchingDimsElem(attr, pos)
    @ccall mlir_c.stablehloScatterDimensionNumbersGetInputBatchingDimsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function stablehloScatterDimensionNumbersGetScatterIndicesBatchingDimsSize(attr)
    @ccall mlir_c.stablehloScatterDimensionNumbersGetScatterIndicesBatchingDimsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function stablehloScatterDimensionNumbersGetScatterIndicesBatchingDimsElem(attr, pos)
    @ccall mlir_c.stablehloScatterDimensionNumbersGetScatterIndicesBatchingDimsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function stablehloScatterDimensionNumbersGetScatteredDimsToOperandDimsSize(attr)
    @ccall mlir_c.stablehloScatterDimensionNumbersGetScatteredDimsToOperandDimsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function stablehloScatterDimensionNumbersGetScatteredDimsToOperandDimsElem(attr, pos)
    @ccall mlir_c.stablehloScatterDimensionNumbersGetScatteredDimsToOperandDimsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function stablehloDimensionNumbersGetIndexVectorDim(attr)
    @ccall mlir_c.stablehloDimensionNumbersGetIndexVectorDim(attr::MlirAttribute)::Int64
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
    @ccall mlir_c.stablehloGatherDimensionNumbersGet(
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
    @ccall mlir_c.stablehloAttributeIsAGatherDimensionNumbers(attr::MlirAttribute)::Bool
end

function stablehloGatherDimensionNumbersGetOffsetDimsSize(attr)
    @ccall mlir_c.stablehloGatherDimensionNumbersGetOffsetDimsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function stablehloGatherDimensionNumbersGetOffsetDimsElem(attr, pos)
    @ccall mlir_c.stablehloGatherDimensionNumbersGetOffsetDimsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function stablehloGatherDimensionNumbersGetCollapsedSliceDimsSize(attr)
    @ccall mlir_c.stablehloGatherDimensionNumbersGetCollapsedSliceDimsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function stablehloGatherDimensionNumbersGetCollapsedSliceDimsElem(attr, pos)
    @ccall mlir_c.stablehloGatherDimensionNumbersGetCollapsedSliceDimsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function stablehloGatherDimensionNumbersGetOperandBatchingDimsSize(attr)
    @ccall mlir_c.stablehloGatherDimensionNumbersGetOperandBatchingDimsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function stablehloGatherDimensionNumbersGetOperandBatchingDimsElem(attr, pos)
    @ccall mlir_c.stablehloGatherDimensionNumbersGetOperandBatchingDimsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function stablehloGatherDimensionNumbersGetStartIndicesBatchingDimsSize(attr)
    @ccall mlir_c.stablehloGatherDimensionNumbersGetStartIndicesBatchingDimsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function stablehloGatherDimensionNumbersGetStartIndicesBatchingDimsElem(attr, pos)
    @ccall mlir_c.stablehloGatherDimensionNumbersGetStartIndicesBatchingDimsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function stablehloGatherDimensionNumbersGetStartIndexMapSize(attr)
    @ccall mlir_c.stablehloGatherDimensionNumbersGetStartIndexMapSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function stablehloGatherDimensionNumbersGetStartIndexMapElem(attr, pos)
    @ccall mlir_c.stablehloGatherDimensionNumbersGetStartIndexMapElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function stablehloGatherDimensionNumbersGetIndexVectorDim(attr)
    @ccall mlir_c.stablehloGatherDimensionNumbersGetIndexVectorDim(
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
    @ccall mlir_c.stablehloDotAlgorithmGet(
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
    @ccall mlir_c.stablehloAttributeIsADotAlgorithm(attr::MlirAttribute)::Bool
end

function stablehloDotAlgorithmGetLhsPrecisionType(attr)
    @ccall mlir_c.stablehloDotAlgorithmGetLhsPrecisionType(attr::MlirAttribute)::MlirType
end

function stablehloDotAlgorithmGetRhsPrecisionType(attr)
    @ccall mlir_c.stablehloDotAlgorithmGetRhsPrecisionType(attr::MlirAttribute)::MlirType
end

function stablehloDotAlgorithmGetAccumulationType(attr)
    @ccall mlir_c.stablehloDotAlgorithmGetAccumulationType(attr::MlirAttribute)::MlirType
end

function stablehloDotAlgorithmGetLhsComponentCount(attr)
    @ccall mlir_c.stablehloDotAlgorithmGetLhsComponentCount(attr::MlirAttribute)::Int64
end

function stablehloDotAlgorithmGetRhsComponentCount(attr)
    @ccall mlir_c.stablehloDotAlgorithmGetRhsComponentCount(attr::MlirAttribute)::Int64
end

function stablehloDotAlgorithmGetNumPrimitiveOperations(attr)
    @ccall mlir_c.stablehloDotAlgorithmGetNumPrimitiveOperations(attr::MlirAttribute)::Int64
end

function stablehloDotAlgorithmGetAllowImpreciseAccumulation(attr)
    @ccall mlir_c.stablehloDotAlgorithmGetAllowImpreciseAccumulation(
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
    @ccall mlir_c.stablehloDotDimensionNumbersGet(
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
    @ccall mlir_c.stablehloAttributeIsADotDimensionNumbers(attr::MlirAttribute)::Bool
end

function stablehloDotDimensionNumbersGetLhsBatchingDimensionsSize(attr)
    @ccall mlir_c.stablehloDotDimensionNumbersGetLhsBatchingDimensionsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function stablehloDotDimensionNumbersGetLhsBatchingDimensionsElem(attr, pos)
    @ccall mlir_c.stablehloDotDimensionNumbersGetLhsBatchingDimensionsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function stablehloDotDimensionNumbersGetRhsBatchingDimensionsSize(attr)
    @ccall mlir_c.stablehloDotDimensionNumbersGetRhsBatchingDimensionsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function stablehloDotDimensionNumbersGetRhsBatchingDimensionsElem(attr, pos)
    @ccall mlir_c.stablehloDotDimensionNumbersGetRhsBatchingDimensionsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function stablehloDotDimensionNumbersGetLhsContractingDimensionsSize(attr)
    @ccall mlir_c.stablehloDotDimensionNumbersGetLhsContractingDimensionsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function stablehloDotDimensionNumbersGetLhsContractingDimensionsElem(attr, pos)
    @ccall mlir_c.stablehloDotDimensionNumbersGetLhsContractingDimensionsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function stablehloDotDimensionNumbersGetRhsContractingDimensionsSize(attr)
    @ccall mlir_c.stablehloDotDimensionNumbersGetRhsContractingDimensionsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function stablehloDotDimensionNumbersGetRhsContractingDimensionsElem(attr, pos)
    @ccall mlir_c.stablehloDotDimensionNumbersGetRhsContractingDimensionsElem(
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
    @ccall mlir_c.stablehloConvDimensionNumbersGet(
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
    @ccall mlir_c.stablehloAttributeIsAConvDimensionNumbers(attr::MlirAttribute)::Bool
end

function stablehloConvDimensionNumbersGetInputBatchDimension(attr)
    @ccall mlir_c.stablehloConvDimensionNumbersGetInputBatchDimension(
        attr::MlirAttribute
    )::Int64
end

function stablehloConvDimensionNumbersGetInputFeatureDimension(attr)
    @ccall mlir_c.stablehloConvDimensionNumbersGetInputFeatureDimension(
        attr::MlirAttribute
    )::Int64
end

function stablehloConvDimensionNumbersGetInputSpatialDimensionsSize(attr)
    @ccall mlir_c.stablehloConvDimensionNumbersGetInputSpatialDimensionsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function stablehloConvDimensionNumbersGetInputSpatialDimensionsElem(attr, pos)
    @ccall mlir_c.stablehloConvDimensionNumbersGetInputSpatialDimensionsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function stablehloConvDimensionNumbersGetKernelInputFeatureDimension(attr)
    @ccall mlir_c.stablehloConvDimensionNumbersGetKernelInputFeatureDimension(
        attr::MlirAttribute
    )::Int64
end

function stablehloConvDimensionNumbersGetKernelOutputFeatureDimension(attr)
    @ccall mlir_c.stablehloConvDimensionNumbersGetKernelOutputFeatureDimension(
        attr::MlirAttribute
    )::Int64
end

function stablehloConvDimensionNumbersGetKernelSpatialDimensionsSize(attr)
    @ccall mlir_c.stablehloConvDimensionNumbersGetKernelSpatialDimensionsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function stablehloConvDimensionNumbersGetKernelSpatialDimensionsElem(attr, pos)
    @ccall mlir_c.stablehloConvDimensionNumbersGetKernelSpatialDimensionsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function stablehloConvDimensionNumbersGetOutputBatchDimension(attr)
    @ccall mlir_c.stablehloConvDimensionNumbersGetOutputBatchDimension(
        attr::MlirAttribute
    )::Int64
end

function stablehloConvDimensionNumbersGetOutputFeatureDimension(attr)
    @ccall mlir_c.stablehloConvDimensionNumbersGetOutputFeatureDimension(
        attr::MlirAttribute
    )::Int64
end

function stablehloConvDimensionNumbersGetOutputSpatialDimensionsSize(attr)
    @ccall mlir_c.stablehloConvDimensionNumbersGetOutputSpatialDimensionsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function stablehloConvDimensionNumbersGetOutputSpatialDimensionsElem(attr, pos)
    @ccall mlir_c.stablehloConvDimensionNumbersGetOutputSpatialDimensionsElem(
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
    @ccall mlir_c.stablehloOutputOperandAliasGet(
        ctx::MlirContext,
        nOutputTupleIndices::Cptrdiff_t,
        outputTupleIndices::Ptr{Int64},
        operandIndex::Int64,
        nOperandTupleIndices::Cptrdiff_t,
        operandTupleIndices::Ptr{Int64},
    )::MlirAttribute
end

function stablehloAttributeIsAOutputOperandAlias(attr)
    @ccall mlir_c.stablehloAttributeIsAOutputOperandAlias(attr::MlirAttribute)::Bool
end

function stablehloOutputOperandAliasGetOutputTupleIndicesSize(attr)
    @ccall mlir_c.stablehloOutputOperandAliasGetOutputTupleIndicesSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function stablehloOutputOperandAliasGetOutputTupleIndicesElem(attr, pos)
    @ccall mlir_c.stablehloOutputOperandAliasGetOutputTupleIndicesElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function stablehloOutputOperandAliasGetOperandIndex(attr)
    @ccall mlir_c.stablehloOutputOperandAliasGetOperandIndex(attr::MlirAttribute)::Int64
end

function stablehloOutputOperandAliasGetOperandTupleIndicesSize(attr)
    @ccall mlir_c.stablehloOutputOperandAliasGetOperandTupleIndicesSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function stablehloOutputOperandAliasGetOperandTupleIndicesElem(attr, pos)
    @ccall mlir_c.stablehloOutputOperandAliasGetOperandTupleIndicesElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function stablehloComparisonDirectionAttrGet(ctx, value)
    @ccall mlir_c.stablehloComparisonDirectionAttrGet(
        ctx::MlirContext, value::MlirStringRef
    )::MlirAttribute
end

function stablehloAttributeIsAComparisonDirectionAttr(attr)
    @ccall mlir_c.stablehloAttributeIsAComparisonDirectionAttr(attr::MlirAttribute)::Bool
end

function stablehloComparisonDirectionAttrGetValue(attr)
    @ccall mlir_c.stablehloComparisonDirectionAttrGetValue(
        attr::MlirAttribute
    )::MlirStringRef
end

function stablehloComparisonTypeAttrGet(ctx, value)
    @ccall mlir_c.stablehloComparisonTypeAttrGet(
        ctx::MlirContext, value::MlirStringRef
    )::MlirAttribute
end

function stablehloAttributeIsAComparisonTypeAttr(attr)
    @ccall mlir_c.stablehloAttributeIsAComparisonTypeAttr(attr::MlirAttribute)::Bool
end

function stablehloComparisonTypeAttrGetValue(attr)
    @ccall mlir_c.stablehloComparisonTypeAttrGetValue(attr::MlirAttribute)::MlirStringRef
end

function stablehloPrecisionAttrGet(ctx, value)
    @ccall mlir_c.stablehloPrecisionAttrGet(
        ctx::MlirContext, value::MlirStringRef
    )::MlirAttribute
end

function stablehloAttributeIsAPrecisionAttr(attr)
    @ccall mlir_c.stablehloAttributeIsAPrecisionAttr(attr::MlirAttribute)::Bool
end

function stablehloPrecisionAttrGetValue(attr)
    @ccall mlir_c.stablehloPrecisionAttrGetValue(attr::MlirAttribute)::MlirStringRef
end

function stablehloFftTypeAttrGet(ctx, value)
    @ccall mlir_c.stablehloFftTypeAttrGet(
        ctx::MlirContext, value::MlirStringRef
    )::MlirAttribute
end

function stablehloAttributeIsAFftTypeAttr(attr)
    @ccall mlir_c.stablehloAttributeIsAFftTypeAttr(attr::MlirAttribute)::Bool
end

function stablehloFftTypeAttrGetValue(attr)
    @ccall mlir_c.stablehloFftTypeAttrGetValue(attr::MlirAttribute)::MlirStringRef
end

function stablehloTransposeAttrGet(ctx, value)
    @ccall mlir_c.stablehloTransposeAttrGet(
        ctx::MlirContext, value::MlirStringRef
    )::MlirAttribute
end

function stablehloAttributeIsATransposeAttr(attr)
    @ccall mlir_c.stablehloAttributeIsATransposeAttr(attr::MlirAttribute)::Bool
end

function stablehloTransposeAttrGetValue(attr)
    @ccall mlir_c.stablehloTransposeAttrGetValue(attr::MlirAttribute)::MlirStringRef
end

function stablehloRngDistributionAttrGet(ctx, value)
    @ccall mlir_c.stablehloRngDistributionAttrGet(
        ctx::MlirContext, value::MlirStringRef
    )::MlirAttribute
end

function stablehloAttributeIsARngDistributionAttr(attr)
    @ccall mlir_c.stablehloAttributeIsARngDistributionAttr(attr::MlirAttribute)::Bool
end

function stablehloRngDistributionAttrGetValue(attr)
    @ccall mlir_c.stablehloRngDistributionAttrGetValue(attr::MlirAttribute)::MlirStringRef
end

function stablehloRngAlgorithmAttrGet(ctx, value)
    @ccall mlir_c.stablehloRngAlgorithmAttrGet(
        ctx::MlirContext, value::MlirStringRef
    )::MlirAttribute
end

function stablehloAttributeIsARngAlgorithmAttr(attr)
    @ccall mlir_c.stablehloAttributeIsARngAlgorithmAttr(attr::MlirAttribute)::Bool
end

function stablehloRngAlgorithmAttrGetValue(attr)
    @ccall mlir_c.stablehloRngAlgorithmAttrGetValue(attr::MlirAttribute)::MlirStringRef
end

function stablehloChannelHandleGet(ctx, handle, type)
    @ccall mlir_c.stablehloChannelHandleGet(
        ctx::MlirContext, handle::Int64, type::Int64
    )::MlirAttribute
end

function stablehloAttributeIsChannelHandle(attr)
    @ccall mlir_c.stablehloAttributeIsChannelHandle(attr::MlirAttribute)::Bool
end

function stablehloChannelHandleGetHandle(attr)
    @ccall mlir_c.stablehloChannelHandleGetHandle(attr::MlirAttribute)::Int64
end

function stablehloChannelHandleGetType(attr)
    @ccall mlir_c.stablehloChannelHandleGetType(attr::MlirAttribute)::Int64
end

function stablehloTypeExtensionsGet(ctx, nBounds, bounds)
    @ccall mlir_c.stablehloTypeExtensionsGet(
        ctx::MlirContext, nBounds::Cptrdiff_t, bounds::Ptr{Int64}
    )::MlirAttribute
end

function stablehloAttributeIsTypeExtensions(attr)
    @ccall mlir_c.stablehloAttributeIsTypeExtensions(attr::MlirAttribute)::Bool
end

function stablehloTypeExtensionsGetBoundsSize(attr)
    @ccall mlir_c.stablehloTypeExtensionsGetBoundsSize(attr::MlirAttribute)::Cptrdiff_t
end

function stablehloTypeExtensionsGetBoundsElem(attr, pos)
    @ccall mlir_c.stablehloTypeExtensionsGetBoundsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function stablehloResultAccuracyModeAttrGet(ctx, value)
    @ccall mlir_c.stablehloResultAccuracyModeAttrGet(
        ctx::MlirContext, value::MlirStringRef
    )::MlirAttribute
end

function stablehloAttributeIsAResultAccuracyModeAttr(attr)
    @ccall mlir_c.stablehloAttributeIsAResultAccuracyModeAttr(attr::MlirAttribute)::Bool
end

function stablehloResultAccuracyModeAttrGetValue(attr)
    @ccall mlir_c.stablehloResultAccuracyModeAttrGetValue(
        attr::MlirAttribute
    )::MlirStringRef
end

function stablehloResultAccuracyAttrGet(ctx, atol, rtol, ulps, value)
    @ccall mlir_c.stablehloResultAccuracyAttrGet(
        ctx::MlirContext, atol::Cdouble, rtol::Cdouble, ulps::Int64, value::MlirStringRef
    )::MlirAttribute
end

function stablehloAttributeIsAResultAccuracyAttr(attr)
    @ccall mlir_c.stablehloAttributeIsAResultAccuracyAttr(attr::MlirAttribute)::Bool
end

function stablehloResultAccuracyAttrGetAtol(attr)
    @ccall mlir_c.stablehloResultAccuracyAttrGetAtol(attr::MlirAttribute)::Cdouble
end

function stablehloResultAccuracyAttrGetRtol(attr)
    @ccall mlir_c.stablehloResultAccuracyAttrGetRtol(attr::MlirAttribute)::Cdouble
end

function stablehloResultAccuracyAttrGetUlps(attr)
    @ccall mlir_c.stablehloResultAccuracyAttrGetUlps(attr::MlirAttribute)::Int64
end

function stablehloResultAccuracyAttrGetMode(attr)
    @ccall mlir_c.stablehloResultAccuracyAttrGetMode(attr::MlirAttribute)::MlirAttribute
end

function mlirGetDialectHandle__stablehlo__()
    @ccall mlir_c.mlirGetDialectHandle__stablehlo__()::MlirDialectHandle
end

function stablehloGetApiVersion()
    @ccall mlir_c.stablehloGetApiVersion()::Cint
end

@cenum MlirStablehloCompatibilityRequirement::UInt32 begin
    NONE = 0x0000000000000000
    WEEK_4 = 0x0000000000000001
    WEEK_12 = 0x0000000000000002
    MAX = 0x0000000000000003
end

function stablehloVersionFromCompatibilityRequirement(requirement, callback, userData)
    @ccall mlir_c.stablehloVersionFromCompatibilityRequirement(
        requirement::MlirStablehloCompatibilityRequirement,
        callback::MlirStringCallback,
        userData::Ptr{Cvoid},
    )::Cvoid
end

function stablehloGetCurrentVersion(callback, userData)
    @ccall mlir_c.stablehloGetCurrentVersion(
        callback::MlirStringCallback, userData::Ptr{Cvoid}
    )::Cvoid
end

function stablehloGetMinimumVersion(callback, userData)
    @ccall mlir_c.stablehloGetMinimumVersion(
        callback::MlirStringCallback, userData::Ptr{Cvoid}
    )::Cvoid
end

function stablehloGetSmallerVersion(version1, version2, callback, userData)
    @ccall mlir_c.stablehloGetSmallerVersion(
        version1::MlirStringRef,
        version2::MlirStringRef,
        callback::MlirStringCallback,
        userData::Ptr{Cvoid},
    )::MlirLogicalResult
end

function stablehloSerializePortableArtifactFromStringRef(
    moduleStr, targetVersion, callback, userData
)
    @ccall mlir_c.stablehloSerializePortableArtifactFromStringRef(
        moduleStr::MlirStringRef,
        targetVersion::MlirStringRef,
        callback::MlirStringCallback,
        userData::Ptr{Cvoid},
    )::MlirLogicalResult
end

function stablehloSerializePortableArtifactFromModule(
    moduleStr, targetVersion, callback, userData, allowOtherDialects
)
    @ccall mlir_c.stablehloSerializePortableArtifactFromModule(
        moduleStr::MlirModule,
        targetVersion::MlirStringRef,
        callback::MlirStringCallback,
        userData::Ptr{Cvoid},
        allowOtherDialects::Bool,
    )::MlirLogicalResult
end

function stablehloDeserializePortableArtifact(artifactStr, callback, userData)
    @ccall mlir_c.stablehloDeserializePortableArtifact(
        artifactStr::MlirStringRef, callback::MlirStringCallback, userData::Ptr{Cvoid}
    )::MlirLogicalResult
end

function stablehloDeserializePortableArtifactNoError(artifactStr, ctx)
    @ccall mlir_c.stablehloDeserializePortableArtifactNoError(
        artifactStr::MlirStringRef, ctx::MlirContext
    )::MlirModule
end

function stablehloTokenTypeGet(ctx)
    @ccall mlir_c.stablehloTokenTypeGet(ctx::MlirContext)::MlirType
end

function stablehloTypeIsAToken(type)
    @ccall mlir_c.stablehloTypeIsAToken(type::MlirType)::Bool
end

function sdyAttributeIsAMeshAxisAttr(attr)
    @ccall mlir_c.sdyAttributeIsAMeshAxisAttr(attr::MlirAttribute)::Bool
end

function sdyMeshAxisAttrGet(ctx, name, size)
    @ccall mlir_c.sdyMeshAxisAttrGet(
        ctx::MlirContext, name::MlirStringRef, size::Int64
    )::MlirAttribute
end

function sdyMeshAxisAttrGetName(attr)
    @ccall mlir_c.sdyMeshAxisAttrGetName(attr::MlirAttribute)::MlirStringRef
end

function sdyMeshAxisAttrGetSize(attr)
    @ccall mlir_c.sdyMeshAxisAttrGetSize(attr::MlirAttribute)::Int64
end

function sdyAttributeIsAMeshAttr(attr)
    @ccall mlir_c.sdyAttributeIsAMeshAttr(attr::MlirAttribute)::Bool
end

function sdyMeshAttrGet(ctx, nAxes, axes, nDeviceIds, deviceIds)
    @ccall mlir_c.sdyMeshAttrGet(
        ctx::MlirContext,
        nAxes::Cptrdiff_t,
        axes::Ptr{MlirAttribute},
        nDeviceIds::Cptrdiff_t,
        deviceIds::Ptr{Int64},
    )::MlirAttribute
end

function sdyMeshAttrGetDeviceIdsSize(attr)
    @ccall mlir_c.sdyMeshAttrGetDeviceIdsSize(attr::MlirAttribute)::Int64
end

function sdyMeshAttrGetDeviceIdsElem(attr, pos)
    @ccall mlir_c.sdyMeshAttrGetDeviceIdsElem(attr::MlirAttribute, pos::Int64)::Int64
end

function sdyMeshAttrGetAxesSize(attr)
    @ccall mlir_c.sdyMeshAttrGetAxesSize(attr::MlirAttribute)::Cptrdiff_t
end

function sdyMeshAttrGetAxesElem(attr, pos)
    @ccall mlir_c.sdyMeshAttrGetAxesElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::MlirAttribute
end

function sdyAttributeIsASubAxisInfoAttr(attr)
    @ccall mlir_c.sdyAttributeIsASubAxisInfoAttr(attr::MlirAttribute)::Bool
end

function sdySubAxisInfoAttrGet(ctx, preSize, size)
    @ccall mlir_c.sdySubAxisInfoAttrGet(
        ctx::MlirContext, preSize::Int64, size::Int64
    )::MlirAttribute
end

function sdySubAxisInfoAttrGetPreSize(attr)
    @ccall mlir_c.sdySubAxisInfoAttrGetPreSize(attr::MlirAttribute)::Int64
end

function sdySubAxisInfoAttrGetSize(attr)
    @ccall mlir_c.sdySubAxisInfoAttrGetSize(attr::MlirAttribute)::Int64
end

function sdyAttributeIsAnAxisRefAttr(attr)
    @ccall mlir_c.sdyAttributeIsAnAxisRefAttr(attr::MlirAttribute)::Bool
end

function sdyAxisRefAttrGet(ctx, name, subAxisInfo)
    @ccall mlir_c.sdyAxisRefAttrGet(
        ctx::MlirContext, name::MlirStringRef, subAxisInfo::MlirAttribute
    )::MlirAttribute
end

function sdyAxisRefAttrGetName(attr)
    @ccall mlir_c.sdyAxisRefAttrGetName(attr::MlirAttribute)::MlirStringRef
end

function sdyAxisRefAttrGetSubAxisInfo(attr)
    @ccall mlir_c.sdyAxisRefAttrGetSubAxisInfo(attr::MlirAttribute)::MlirAttribute
end

function sdyAttributeIsADimensionShardingAttr(attr)
    @ccall mlir_c.sdyAttributeIsADimensionShardingAttr(attr::MlirAttribute)::Bool
end

function sdyDimensionShardingAttrGet(ctx, nAxes, axes, isClosed, priority)
    @ccall mlir_c.sdyDimensionShardingAttrGet(
        ctx::MlirContext,
        nAxes::Cptrdiff_t,
        axes::Ptr{MlirAttribute},
        isClosed::Bool,
        priority::Int64,
    )::MlirAttribute
end

function sdyDimensionShardingAttrGetAxesSize(attr)
    @ccall mlir_c.sdyDimensionShardingAttrGetAxesSize(attr::MlirAttribute)::Cptrdiff_t
end

function sdyDimensionShardingAttrGetAxesElem(attr, pos)
    @ccall mlir_c.sdyDimensionShardingAttrGetAxesElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::MlirAttribute
end

function sdyDimensionShardingAttrGetIsClosed(attr)
    @ccall mlir_c.sdyDimensionShardingAttrGetIsClosed(attr::MlirAttribute)::Bool
end

function sdyDimensionShardingAttrGetPriority(attr)
    @ccall mlir_c.sdyDimensionShardingAttrGetPriority(attr::MlirAttribute)::Int64
end

function sdyAttributeIsATensorShardingAttr(attr)
    @ccall mlir_c.sdyAttributeIsATensorShardingAttr(attr::MlirAttribute)::Bool
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
)
    @ccall mlir_c.sdyTensorShardingAttrGet(
        ctx::MlirContext,
        meshOrRef::MlirAttribute,
        nDimShardings::Cptrdiff_t,
        dimShardings::Ptr{MlirAttribute},
        nReplicatedAxes::Cptrdiff_t,
        replicatedAxes::Ptr{MlirAttribute},
        nUnreducedAxes::Cptrdiff_t,
        unreducedAxes::Ptr{MlirAttribute},
    )::MlirAttribute
end

function sdyTensorShardingAttrGetMeshOrRef(attr)
    @ccall mlir_c.sdyTensorShardingAttrGetMeshOrRef(attr::MlirAttribute)::MlirAttribute
end

function sdyTensorShardingAttrGetDimShardingsSize(attr)
    @ccall mlir_c.sdyTensorShardingAttrGetDimShardingsSize(attr::MlirAttribute)::Cptrdiff_t
end

function sdyTensorShardingAttrGetDimShardingsElem(attr, pos)
    @ccall mlir_c.sdyTensorShardingAttrGetDimShardingsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::MlirAttribute
end

function sdyTensorShardingAttrGetReplicatedAxesSize(attr)
    @ccall mlir_c.sdyTensorShardingAttrGetReplicatedAxesSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function sdyTensorShardingAttrGetReplicatedAxesElem(attr, pos)
    @ccall mlir_c.sdyTensorShardingAttrGetReplicatedAxesElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::MlirAttribute
end

function sdyTensorShardingAttrGetUnreducedAxesSize(attr)
    @ccall mlir_c.sdyTensorShardingAttrGetUnreducedAxesSize(attr::MlirAttribute)::Cptrdiff_t
end

function sdyTensorShardingAttrGetUnreducedAxesElem(attr, pos)
    @ccall mlir_c.sdyTensorShardingAttrGetUnreducedAxesElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::MlirAttribute
end

function sdyAttributeIsATensorShardingPerValueAttr(attr)
    @ccall mlir_c.sdyAttributeIsATensorShardingPerValueAttr(attr::MlirAttribute)::Bool
end

function sdyTensorShardingPerValueAttrGet(ctx, nShardings, shardings)
    @ccall mlir_c.sdyTensorShardingPerValueAttrGet(
        ctx::MlirContext, nShardings::Cptrdiff_t, shardings::Ptr{MlirAttribute}
    )::MlirAttribute
end

function sdyTensorShardingPerValueAttrGetShardingsSize(attr)
    @ccall mlir_c.sdyTensorShardingPerValueAttrGetShardingsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function sdyTensorShardingPerValueAttrGetShardingsElem(attr, pos)
    @ccall mlir_c.sdyTensorShardingPerValueAttrGetShardingsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::MlirAttribute
end

function sdyAttributeIsADimMappingAttr(attr)
    @ccall mlir_c.sdyAttributeIsADimMappingAttr(attr::MlirAttribute)::Bool
end

function sdyDimMappingAttrGet(ctx, nFactorIndices, factorIndices)
    @ccall mlir_c.sdyDimMappingAttrGet(
        ctx::MlirContext, nFactorIndices::Cptrdiff_t, factorIndices::Ptr{Int64}
    )::MlirAttribute
end

function sdyDimMappingAttrGetFactorIndicesSize(attr)
    @ccall mlir_c.sdyDimMappingAttrGetFactorIndicesSize(attr::MlirAttribute)::Cptrdiff_t
end

function sdyDimMappingAttrGetFactorIndicesElem(attr, pos)
    @ccall mlir_c.sdyDimMappingAttrGetFactorIndicesElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function sdyAttributeIsATensorMappingAttr(attr)
    @ccall mlir_c.sdyAttributeIsATensorMappingAttr(attr::MlirAttribute)::Bool
end

function sdyTensorMappingAttrGet(ctx, nMappings, mappings)
    @ccall mlir_c.sdyTensorMappingAttrGet(
        ctx::MlirContext, nMappings::Cptrdiff_t, mappings::Ptr{MlirAttribute}
    )::MlirAttribute
end

function sdyTensorMappingAttrGetRank(attr)
    @ccall mlir_c.sdyTensorMappingAttrGetRank(attr::MlirAttribute)::Cptrdiff_t
end

function sdyTensorMappingAttrGetDimMappingsSize(attr)
    @ccall mlir_c.sdyTensorMappingAttrGetDimMappingsSize(attr::MlirAttribute)::Cptrdiff_t
end

function sdyTensorMappingAttrGetDimMappingsElem(attr, pos)
    @ccall mlir_c.sdyTensorMappingAttrGetDimMappingsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::MlirAttribute
end

function sdyAttributeIsAOpShardingRuleAttr(attr)
    @ccall mlir_c.sdyAttributeIsAOpShardingRuleAttr(attr::MlirAttribute)::Bool
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
    @ccall mlir_c.sdyOpShardingRuleAttrGet(
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
    @ccall mlir_c.sdyOpShardingRuleAttrGetIsCustom(attr::MlirAttribute)::Bool
end

function sdyOpShardingRuleAttrGetFactorSizesSize(attr)
    @ccall mlir_c.sdyOpShardingRuleAttrGetFactorSizesSize(attr::MlirAttribute)::Cptrdiff_t
end

function sdyOpShardingRuleAttrGetFactorSizesElem(attr, pos)
    @ccall mlir_c.sdyOpShardingRuleAttrGetFactorSizesElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function sdyOpShardingRuleAttrGetOperandMappingsSize(attr)
    @ccall mlir_c.sdyOpShardingRuleAttrGetOperandMappingsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function sdyOpShardingRuleAttrGetOperandMappingsElem(attr, pos)
    @ccall mlir_c.sdyOpShardingRuleAttrGetOperandMappingsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::MlirAttribute
end

function sdyOpShardingRuleAttrGetResultMappingsSize(attr)
    @ccall mlir_c.sdyOpShardingRuleAttrGetResultMappingsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function sdyOpShardingRuleAttrGetResultMappingsElem(attr, pos)
    @ccall mlir_c.sdyOpShardingRuleAttrGetResultMappingsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::MlirAttribute
end

function sdyOpShardingRuleAttrGetReductionFactorsSize(attr)
    @ccall mlir_c.sdyOpShardingRuleAttrGetReductionFactorsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function sdyOpShardingRuleAttrGetReductionFactorsElem(attr, pos)
    @ccall mlir_c.sdyOpShardingRuleAttrGetReductionFactorsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function sdyOpShardingRuleAttrGetNeedReplicationFactorsSize(attr)
    @ccall mlir_c.sdyOpShardingRuleAttrGetNeedReplicationFactorsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function sdyOpShardingRuleAttrGetNeedReplicationFactorsElem(attr, pos)
    @ccall mlir_c.sdyOpShardingRuleAttrGetNeedReplicationFactorsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function sdyOpShardingRuleAttrGetPermutationFactorsSize(attr)
    @ccall mlir_c.sdyOpShardingRuleAttrGetPermutationFactorsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function sdyOpShardingRuleAttrGetPermutationFactorsElem(attr, pos)
    @ccall mlir_c.sdyOpShardingRuleAttrGetPermutationFactorsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function sdyOpShardingRuleAttrGetBlockedPropagationFactorsSize(attr)
    @ccall mlir_c.sdyOpShardingRuleAttrGetBlockedPropagationFactorsSize(
        attr::MlirAttribute
    )::Cptrdiff_t
end

function sdyOpShardingRuleAttrGetBlockedPropagationFactorsElem(attr, pos)
    @ccall mlir_c.sdyOpShardingRuleAttrGetBlockedPropagationFactorsElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::Int64
end

function sdyAttributeIsAManualAxesAttr(attr)
    @ccall mlir_c.sdyAttributeIsAManualAxesAttr(attr::MlirAttribute)::Bool
end

function sdyManualAxesAttrGet(ctx, nAxes, axes)
    @ccall mlir_c.sdyManualAxesAttrGet(
        ctx::MlirContext, nAxes::Cptrdiff_t, axes::Ptr{MlirAttribute}
    )::MlirAttribute
end

function sdyManualAxesAttrGetAxesSize(attr)
    @ccall mlir_c.sdyManualAxesAttrGetAxesSize(attr::MlirAttribute)::Cptrdiff_t
end

function sdyManualAxesAttrGetAxesElem(attr, pos)
    @ccall mlir_c.sdyManualAxesAttrGetAxesElem(
        attr::MlirAttribute, pos::Cptrdiff_t
    )::MlirStringRef
end

function mlirGetDialectHandle__triton__()
    @ccall mlir_c.mlirGetDialectHandle__triton__()::MlirDialectHandle
end

function mlirTritonPointerTypeGet(pointeeType, addressSpace)
    @ccall mlir_c.mlirTritonPointerTypeGet(
        pointeeType::MlirType, addressSpace::Cint
    )::MlirType
end

function mlirTritonIsAPointer(type)
    @ccall mlir_c.mlirTritonIsAPointer(type::MlirType)::Bool
end

function mlirTritonPointerTypeGetPointeeType(pointerType)
    @ccall mlir_c.mlirTritonPointerTypeGetPointeeType(pointerType::MlirType)::MlirType
end

function mlirTritonPointerTypeGetAddressSpace(pointerType)
    @ccall mlir_c.mlirTritonPointerTypeGetAddressSpace(pointerType::MlirType)::Cint
end

function mlirTritonInferReduceOpEncoding(operandEncoding, axis)
    @ccall mlir_c.mlirTritonInferReduceOpEncoding(
        operandEncoding::MlirAttribute, axis::Cint
    )::MlirAttribute
end

function mlirGetDialectHandle__tpu__()
    @ccall mlir_c.mlirGetDialectHandle__tpu__()::MlirDialectHandle
end

function mlirTPUAnalyzePotentialCommunication(op, has_communication, has_custom_barrier)
    @ccall mlir_c.mlirTPUAnalyzePotentialCommunication(
        op::MlirOperation, has_communication::Ptr{Bool}, has_custom_barrier::Ptr{Bool}
    )::Cvoid
end

function mlirTpuRegisterMosaicSerdePass()
    @ccall mlir_c.mlirTpuRegisterMosaicSerdePass()::Cvoid
end

function mlirTpuFloat8EXMYTypeGetUnderlyingType(exmy_type)
    @ccall mlir_c.mlirTpuFloat8EXMYTypeGetUnderlyingType(exmy_type::MlirType)::MlirType
end

function mlirTpuIsAFloat8EXMYType(type)
    @ccall mlir_c.mlirTpuIsAFloat8EXMYType(type::MlirType)::Bool
end

function mlirTpuFloat8EXMYTypeGet(ctx, exmy_type)
    @ccall mlir_c.mlirTpuFloat8EXMYTypeGet(ctx::MlirContext, exmy_type::MlirType)::MlirType
end

function mlirMosaicGpuIsATileTransformAttr(attr)
    @ccall mlir_c.mlirMosaicGpuIsATileTransformAttr(attr::MlirAttribute)::Bool
end

function mlirMosaicGpuTileTransformAttrGet(ctx, tiling, tiling_size)
    @ccall mlir_c.mlirMosaicGpuTileTransformAttrGet(
        ctx::MlirContext, tiling::Ptr{Int32}, tiling_size::Int32
    )::MlirAttribute
end

function mlirMosaicGpuTileTransformAttrGetTilingSize(attr)
    @ccall mlir_c.mlirMosaicGpuTileTransformAttrGetTilingSize(attr::MlirAttribute)::Int32
end

function mlirMosaicGpuTileTransformAttrGetTiling(attr, index)
    @ccall mlir_c.mlirMosaicGpuTileTransformAttrGetTiling(
        attr::MlirAttribute, index::Int32
    )::Int32
end

function mlirMosaicGpuIsATransposeTransformAttr(attr)
    @ccall mlir_c.mlirMosaicGpuIsATransposeTransformAttr(attr::MlirAttribute)::Bool
end

function mlirMosaicGpuTransposeTransformAttrGet(ctx, permutation, permutation_size)
    @ccall mlir_c.mlirMosaicGpuTransposeTransformAttrGet(
        ctx::MlirContext, permutation::Ptr{Int32}, permutation_size::Int32
    )::MlirAttribute
end

function mlirMosaicGpuTransposeTransformAttrGetPermutationSize(attr)
    @ccall mlir_c.mlirMosaicGpuTransposeTransformAttrGetPermutationSize(
        attr::MlirAttribute
    )::Int32
end

function mlirMosaicGpuTransposeTransformAttrGetPermutation(attr, index)
    @ccall mlir_c.mlirMosaicGpuTransposeTransformAttrGetPermutation(
        attr::MlirAttribute, index::Int32
    )::Int32
end

function mlirMosaicGpuIsASwizzleTransformAttr(attr)
    @ccall mlir_c.mlirMosaicGpuIsASwizzleTransformAttr(attr::MlirAttribute)::Bool
end

function mlirMosaicGpuSwizzleTransformAttrGet(ctx, swizzle)
    @ccall mlir_c.mlirMosaicGpuSwizzleTransformAttrGet(
        ctx::MlirContext, swizzle::Int32
    )::MlirAttribute
end

function mlirMosaicGpuSwizzleTransformAttrGetSwizzle(attr)
    @ccall mlir_c.mlirMosaicGpuSwizzleTransformAttrGetSwizzle(attr::MlirAttribute)::Int32
end

function mlirGetDialectHandle__mosaic_gpu__()
    @ccall mlir_c.mlirGetDialectHandle__mosaic_gpu__()::MlirDialectHandle
end

function mlirDialectRegistryInsertMosaicGpuInlinerExtensions(registry)
    @ccall mlir_c.mlirDialectRegistryInsertMosaicGpuInlinerExtensions(
        registry::MlirDialectRegistry
    )::Cvoid
end

function enzymexlaLapackLayoutAttrGet(ctx, col_major)
    @ccall mlir_c.enzymexlaLapackLayoutAttrGet(
        ctx::MlirContext, col_major::UInt8
    )::MlirAttribute
end

function enzymexlaLapackTransposeAttrGet(ctx, mode)
    @ccall mlir_c.enzymexlaLapackTransposeAttrGet(
        ctx::MlirContext, mode::Int32
    )::MlirAttribute
end

function enzymexlaLapackSideAttrGet(ctx, left_side)
    @ccall mlir_c.enzymexlaLapackSideAttrGet(
        ctx::MlirContext, left_side::UInt8
    )::MlirAttribute
end

function enzymexlaLapackUploAttrGet(ctx, mode)
    @ccall mlir_c.enzymexlaLapackUploAttrGet(ctx::MlirContext, mode::Int32)::MlirAttribute
end

function enzymexlaQRAlgorithmAttrGet(ctx, mode)
    @ccall mlir_c.enzymexlaQRAlgorithmAttrGet(ctx::MlirContext, mode::Int32)::MlirAttribute
end

function enzymexlaSVDAlgorithmAttrGet(ctx, mode)
    @ccall mlir_c.enzymexlaSVDAlgorithmAttrGet(ctx::MlirContext, mode::Int32)::MlirAttribute
end

function enzymexlaGeluApproximationAttrGet(ctx, mode)
    @ccall mlir_c.enzymexlaGeluApproximationAttrGet(
        ctx::MlirContext, mode::Int32
    )::MlirAttribute
end

function enzymexlaMPIDatatypeAttrGet(ctx, mode)
    @ccall mlir_c.enzymexlaMPIDatatypeAttrGet(ctx::MlirContext, mode::Int32)::MlirAttribute
end

function enzymexlaMPIOpAttrGet(ctx, mode)
    @ccall mlir_c.enzymexlaMPIOpAttrGet(ctx::MlirContext, mode::Int32)::MlirAttribute
end

function enzymexlaGuaranteedAnalysisResultAttrGet(ctx, mode)
    @ccall mlir_c.enzymexlaGuaranteedAnalysisResultAttrGet(
        ctx::MlirContext, mode::Int32
    )::MlirAttribute
end

const MLIR_CAPI_DWARF_ADDRESS_SPACE_NULL = -1

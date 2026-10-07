"""
    Diagnostic

Represents an MLIR diagnostic. Only valid during the execution of a diagnostic
handler callback — must not be stored or used outside of it.
"""
struct Diagnostic
    ref::API.MlirDiagnostic
end

Base.cconvert(::Core.Type{API.MlirDiagnostic}, d::Diagnostic) = d
Base.unsafe_convert(::Core.Type{API.MlirDiagnostic}, d::Diagnostic) = d.ref

"""
    location(diagnostic)

Returns the location at which the diagnostic is reported.
"""
location(diagnostic::Diagnostic) = Location(API.mlirDiagnosticGetLocation(diagnostic))

"""
    severity(diagnostic)

Returns the severity of the diagnostic (one of `MlirDiagnosticError`,
`MlirDiagnosticWarning`, `MlirDiagnosticNote`, or `MlirDiagnosticRemark`).
"""
severity(diagnostic::Diagnostic) = API.mlirDiagnosticGetSeverity(diagnostic)

"""
    nnotes(diagnostic)

Returns the number of notes attached to the diagnostic.
"""
nnotes(diagnostic::Diagnostic) = Int(API.mlirDiagnosticGetNumNotes(diagnostic))

"""
    note(diagnostic, pos)

Returns the `pos`-th note (1-based) attached to the diagnostic.
"""
note(diagnostic::Diagnostic, pos) =
    Diagnostic(API.mlirDiagnosticGetNote(diagnostic, pos - one(pos)))

function Base.show(io::IO, diagnostic::Diagnostic)
    c_print_callback = @cfunction(print_callback, Cvoid, (API.MlirStringRef, Any))
    ref = Ref(io)
    return API.mlirDiagnosticPrint(diagnostic, c_print_callback, ref)
end

# Global registry: maps a UInt64 key → Julia callback.
# Keeping the callback here (rather than via a raw pointer) ensures the GC
# never collects it while the MLIR handler is registered.
const _handler_registry = Dict{UInt64,Any}()
const _handler_registry_key = Threads.Atomic{UInt64}(0)

# Internal trampoline called by the C diagnostic engine.
# `userdata` is a UInt64 key reinterpreted as Ptr{Cvoid}; we look the real
# callback up in _handler_registry to avoid any raw-pointer / GC interaction.
function _diagnostic_handler_trampoline(
    raw::API.MlirDiagnostic, userdata::Ptr{Cvoid}
)::API.MlirLogicalResult
    key = reinterpret(UInt64, userdata)
    callback = get(_handler_registry, key, nothing)
    callback === nothing && return failure().ref
    try
        result = callback(Diagnostic(raw))
        return (result === nothing || result === true) ? success().ref : failure().ref
    catch ex
        @error "Diagnostic handler threw an exception" exception = (ex, catch_backtrace())
        return failure().ref
    end
end

"""
    DiagnosticHandler

Represents an attached diagnostic handler. Use [`detach_diagnostic_handler!`](@ref)
to remove it, or let it be cleaned up when the context is destroyed.
"""
mutable struct DiagnosticHandler
    id::API.MlirDiagnosticHandlerID
    context::Context
    key::UInt64  # key into _handler_registry
end

"""
    attach_diagnostic_handler!(f; context=current_context()) -> DiagnosticHandler

Attaches `f` as a diagnostic handler to `context`. `f` is called as
`f(diagnostic::Diagnostic)` and should return `true` (or `nothing`) if it
handled the diagnostic completely, or `false` to let other handlers try.

Handlers are invoked in reverse order of attachment. Returns a
[`DiagnosticHandler`](@ref) token that can be passed to
[`detach_diagnostic_handler!`](@ref).
"""
function attach_diagnostic_handler!(f; context::Context=current_context())
    key = Threads.atomic_add!(_handler_registry_key, UInt64(1))
    _handler_registry[key] = f
    userdata = reinterpret(Ptr{Cvoid}, key)
    # @cfunction must be created here (not at module scope) so that the function
    # pointer is compiled fresh at call time and is valid after precompilation.
    c_trampoline = @cfunction(
        _diagnostic_handler_trampoline,
        API.MlirLogicalResult,
        (API.MlirDiagnostic, Ptr{Cvoid})
    )
    id = API.mlirContextAttachDiagnosticHandler(context, c_trampoline, userdata, C_NULL)
    return DiagnosticHandler(id, context, key)
end

"""
    detach_diagnostic_handler!(handler::DiagnosticHandler)

Detaches a previously attached diagnostic handler from its context.
"""
function detach_diagnostic_handler!(handler::DiagnosticHandler)
    delete!(_handler_registry, handler.key)
    API.mlirContextDetachDiagnosticHandler(handler.context, handler.id)
    return nothing
end

"""
    emit_error(location, message)

Emits an error diagnostic at the given location through the diagnostics engine.
"""
emit_error(location::Location, message::AbstractString) =
    API.mlirEmitError(location, message)

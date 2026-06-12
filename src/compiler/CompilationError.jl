using ..Reactant: MLIR

# ── Location helpers ─────────────────────────────────────────────────────────

function _sprint_diagnostic(diagnostic::MLIR.IR.Diagnostic)
    io = IOBuffer()
    show(io, diagnostic)
    for i in 1:MLIR.IR.nnotes(diagnostic)
        print(io, "\n  note: ")
        show(io, MLIR.IR.note(diagnostic, i))
    end
    return String(take!(io))
end

function _sprint_leaf_location(loc::MLIR.IR.Location)
    MLIR.IR.mlirIsNull(loc.ref) && return "unknown"
    if MLIR.API.mlirLocationIsAFileLineColRange(loc)
        filename = String(MLIR.API.mlirLocationFileLineColRangeGetFilename(loc))
        line = Int(MLIR.API.mlirLocationFileLineColRangeGetStartLine(loc))
        col = Int(MLIR.API.mlirLocationFileLineColRangeGetStartColumn(loc))
        return "$filename:$line:$col"
    elseif MLIR.API.mlirLocationIsAName(loc)
        name = String(MLIR.API.mlirLocationNameGetName(loc))
        child = MLIR.IR.Location(MLIR.API.mlirLocationNameGetChildLoc(loc))
        child_str = _sprint_leaf_location(child)
        return if (MLIR.IR.mlirIsNull(child.ref) || MLIR.API.mlirLocationIsAUnknown(child))
            "\"$name\""
        else
            "\"$name\" at $child_str"
        end
    elseif MLIR.API.mlirLocationIsAUnknown(loc)
        return "unknown"
    else
        io = IOBuffer()
        c_print_callback = @cfunction(
            MLIR.IR.print_callback, Cvoid, (MLIR.API.MlirStringRef, Any)
        )
        MLIR.API.mlirLocationPrint(loc, c_print_callback, Ref(io))
        return String(take!(io))
    end
end

# A single frame in a diagnostic stack trace.
# `name` comes from a NameLoc (e.g. a Julia function name); may be empty.
# `location` is the file:line:col string; may be empty for name-only frames.
struct StackFrame
    name::String
    location::String
end

# NameLoc("fname", child) → StackFrame("fname", sprint(child))
# Everything else         → StackFrame("", sprint(loc))
function _loc_to_frame(loc::MLIR.IR.Location)
    MLIR.IR.mlirIsNull(loc.ref) && return StackFrame("", "unknown")
    if MLIR.API.mlirLocationIsAName(loc)
        name = String(MLIR.API.mlirLocationNameGetName(loc))
        child = MLIR.IR.Location(MLIR.API.mlirLocationNameGetChildLoc(loc))
        loc_str =
            if (!MLIR.IR.mlirIsNull(child.ref) && !MLIR.API.mlirLocationIsAUnknown(child))
                _sprint_leaf_location(child)
            else
                ""
            end
        return StackFrame(name, loc_str)
    else
        return StackFrame("", _sprint_leaf_location(loc))
    end
end

# Walk a location tree and collect one StackFrame per frame, innermost first.
# CallSiteLoc encodes a call stack: callee = inner frame, caller = outer frame.
# FusedLoc holds multiple locations for the same op (e.g. inlined sites).
function _collect_frames!(frames::Vector{StackFrame}, loc::MLIR.IR.Location)
    MLIR.IR.mlirIsNull(loc.ref) && return frames
    if MLIR.API.mlirLocationIsACallSite(loc)
        callee = MLIR.IR.Location(MLIR.API.mlirLocationCallSiteGetCallee(loc))
        caller = MLIR.IR.Location(MLIR.API.mlirLocationCallSiteGetCaller(loc))
        _collect_frames!(frames, callee)
        _collect_frames!(frames, caller)
    elseif MLIR.API.mlirLocationIsAFused(loc)
        n = Int(MLIR.API.mlirLocationFusedGetNumLocations(loc))
        raw_locs = Vector{MLIR.API.MlirLocation}(undef, n)
        MLIR.API.mlirLocationFusedGetLocations(loc, pointer(raw_locs))
        for raw in raw_locs
            _collect_frames!(frames, MLIR.IR.Location(raw))
        end
    elseif !MLIR.API.mlirLocationIsAUnknown(loc)
        push!(frames, _loc_to_frame(loc))
    end
    return frames
end

function _collect_location_frames(loc::MLIR.IR.Location)
    return _collect_frames!(StackFrame[], loc)
end

# ── CompilationError ─────────────────────────────────────────────────────────

struct DiagnosticMessage
    severity::MLIR.API.MlirDiagnosticSeverity
    message::String
    frames::Vector{StackFrame}
end

struct CompilationError <: Exception
    key::String
    diagnostics::Vector{DiagnosticMessage}
end

function _severity_label(sev::MLIR.API.MlirDiagnosticSeverity)
    sev == MLIR.API.MlirDiagnosticError && return "error"
    sev == MLIR.API.MlirDiagnosticWarning && return "warning"
    sev == MLIR.API.MlirDiagnosticNote && return "note"
    return "remark"
end

function _severity_color(sev::MLIR.API.MlirDiagnosticSeverity)
    sev == MLIR.API.MlirDiagnosticError && return :red
    sev == MLIR.API.MlirDiagnosticWarning && return :yellow
    sev == MLIR.API.MlirDiagnosticNote && return :cyan
    return :light_black
end

function _print_frame(io::IO, j::Int, frame::StackFrame)
    print(io, "  ")
    printstyled(io, "[$j]"; color=:light_black)
    if isempty(frame.name)
        print(io, " ")
        printstyled(io, "@"; color=:cyan)
        print(io, " ")
        printstyled(io, frame.location; color=:light_black)
        println(io)
    elseif isempty(frame.location)
        print(io, " ")
        printstyled(io, frame.name; bold=true)
        println(io)
    else
        print(io, " ")
        printstyled(io, frame.name; bold=true)
        println(io)
        print(io, "     ")
        printstyled(io, "@"; color=:cyan)
        print(io, " ")
        printstyled(io, frame.location; color=:light_black)
        println(io)
    end
end

function _print_diagnostics(io::IO, diagnostics::Vector{DiagnosticMessage})
    for (i, diag) in enumerate(diagnostics)
        i > 1 && println(io)
        label = _severity_label(diag.severity)
        printstyled(io, label; color=_severity_color(diag.severity), bold=true)
        println(io, ": $(diag.message)")
        if !isempty(diag.frames)
            printstyled(io, "Stacktrace:"; bold=true)
            println(io)
            for (j, frame) in enumerate(diag.frames)
                _print_frame(io, j, frame)
            end
        end
    end
end

function Base.showerror(io::IO, err::CompilationError)
    if isempty(err.key)
        print(io, "CompilationError: MLIR pass pipeline failed")
    else
        print(io, "CompilationError: MLIR pass pipeline \"$(err.key)\" failed")
    end
    if !isempty(err.diagnostics)
        println(io, "\n")
        _print_diagnostics(io, err.diagnostics)
    end
end

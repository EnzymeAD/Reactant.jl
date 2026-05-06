import ..ReactantCore: correct_maybe_bcast_call
using ..Reactant: Reactant, MLIR

const COMMON_COMPILE_OPTIONS_DOCS = """
  - `compile_options`: If provided, then all other compilation options will be ignored.
    This should be an object of type [`CompileOptions`](@ref).
  - `optimize`: This option maps to the `optimization_passes` field of
    [`CompileOptions`](@ref). See the documentation of `CompileOptions` for more details.
  - `client`: XLA Client used for compilation. If not specified, the default client is used.

For details about other compilation options see the documentation of
[`CompileOptions`](@ref).
"""

const SYNC_DOCS = """
  - `sync`: Reactant computations are asynchronous by default. If `true`, the computation
    will be executed synchronously, blocking till the computation is complete. This is
    recommended when benchmarking.
"""

struct TextualModule
    ir::String

    function TextualModule(mod::MLIR.IR.Module; debug=false)
        io = IOBuffer()
        show(IOContext(io, :debug => debug), mod)
        return new(String(take!(io)))
    end
end

Base.show(io::IO, tm::TextualModule) = print(io, MLIR.Highlight.highlight(tm.ir))
Base.String(tm::TextualModule) = tm.ir

function Base.convert(::Type{MLIR.IR.Module}, tm::TextualModule)
    return parse(MLIR.IR.Module, tm.ir)
end

"""
    code_hlo(ctx, f, args; fn_kwargs = NamedTuple(), kwargs...)

Compile the function `f` with arguments `args` and return the compiled MLIR module.

See also: [`@code_hlo`](@ref).
"""
function code_hlo(ctx, f, args; fn_kwargs=NamedTuple(), kwargs...)
    if f isa Thunk
        FTy = thunk_fn_type(f)
        error(
            "`@code_hlo` expects the original function, not a compiled `Thunk`. " *
            "Pass the original function directly (of type `$FTy`), e.g. `@code_hlo my_function(args...)`.",
        )
    end
    options = Dict(
        k => v isa QuoteNode ? v.value : v for (k, v) in get_common_compile_options()
    )
    options[:shardy_passes] = :none
    merge!(options, pairs(kwargs))
    return first(compile_mlir(ctx, f, args; fn_kwargs, options...))
end

"""
    @code_hlo [optimize = ...] [no_nan = <true/false>] f(args...)

Prints the compiled MLIR module for the function `f` with arguments `args`.

## Options

$(COMMON_COMPILE_OPTIONS_DOCS)

See also [`@code_xla`](@ref), [`@code_mhlo`](@ref).
"""
macro code_hlo(args...)
    (; f, args, kwargs, options) = parse_call_expr(
        merge(
            get_common_compile_options(),
            Dict{Symbol,Any}(
                :shardy_passes => :(:none), :debug => false, :strip => :(:none)
            ),
        ),
        args...,
    )
    debug = get(() -> Expr(:kw, :debug, false), options, something(findfirst(opt -> opt.args[1] === :debug, options), -1)).args[2]
    options = filter(opt -> opt.args[1] !== :debug, options)
    return quote
        $MLIR.IR.@dispose ctx = $Reactant.ReactantContext() begin
            debug = $(esc(debug))
            mod = $code_hlo(
                ctx,
                $(esc(f)),
                $(esc(args));
                fn_kwargs=(; $(esc.(kwargs)...)),
                $(esc.(options)...),
            )
            try
                $TextualModule(mod; debug)
            finally
                $MLIR.IR.dispose(mod)
            end
        end
    end
end

"""
    code_mhlo(ctx, f, args; fn_kwargs = NamedTuple(), kwargs...)

Compile the function `f` with arguments `args` and return the compiled MLIR module.

See also: [`@code_mhlo`](@ref).
"""
function code_mhlo(ctx, f, args; fn_kwargs=NamedTuple(), kwargs...)
    options = Dict(
        k => v isa QuoteNode ? v.value : v for (k, v) in get_common_compile_options()
    )
    options[:legalize_stablehlo_to_mhlo] = true
    options[:shardy_passes] = :to_mhlo_shardings
    merge!(options, pairs(kwargs))
    return first(compile_mlir(ctx, f, args; fn_kwargs, options...))
end

"""
    @code_mhlo [optimize = ...] [no_nan = <true/false>] f(args...)

Similar to `@code_hlo`, but runs additional passes to export the stablehlo module to MHLO.

## Options

$(COMMON_COMPILE_OPTIONS_DOCS)

See also [`@code_xla`](@ref), [`@code_hlo`](@ref).
"""
macro code_mhlo(args...)
    (; f, args, kwargs, options) = parse_call_expr(
        merge(
            get_common_compile_options(),
            Dict{Symbol,Any}(
                :legalize_stablehlo_to_mhlo => true,
                :shardy_passes => :(:to_mhlo_shardings),
                :debug => false,
                :strip => :(:none),
            ),
        ),
        args...,
    )
    debug = get(() -> Expr(:kw, :debug, false), options, something(findfirst(opt -> opt.args[1] === :debug, options), -1)).args[2]
    options = filter(opt -> opt.args[1] !== :debug, options)
    return quote
        $MLIR.IR.@dispose ctx = $Reactant.ReactantContext() begin
            debug = $(esc(debug))
            mod = $code_mhlo(
                ctx,
                $(esc(f)),
                $(esc(args));
                fn_kwargs=(; $(esc.(kwargs)...)),
                $(esc.(options)...),
            )
            try
                $TextualModule(mod; debug)
            finally
                $MLIR.IR.dispose(mod)
            end
        end
    end
end

"""
    code_xla(ctx, f, args; fn_kwargs = NamedTuple(), kwargs...)

Compile the function `f` with arguments `args` and return the compiled HLO module.

See also: [`@code_xla`](@ref).
"""
function code_xla(ctx, f, args; fn_kwargs=NamedTuple(), kwargs...)
    options = Dict(
        k => v isa QuoteNode ? v.value : v for (k, v) in get_common_compile_options()
    )
    options[:before_xla_optimizations] = false
    merge!(options, pairs(kwargs))
    return compile_xla(ctx, f, args; fn_kwargs, options...)[2]
end

"""
    @code_xla [optimize = ...] [no_nan = <true/false>] f(args...)

Similar to [`@code_hlo`](@ref), but runs additional XLA passes and exports MLIR to XLA HLO.
This is the post optimizations XLA HLO module.

## Options

$(COMMON_COMPILE_OPTIONS_DOCS)
  - `before_xla_optimizations`: If `true`, return the `before_optimizations` HLO module.

See also [`@code_mhlo`](@ref), [`@code_hlo`](@ref).
"""
macro code_xla(args...)
    (; f, args, kwargs, options) = parse_call_expr(
        merge(
            get_common_compile_options(),
            Dict{Symbol,Any}(:before_xla_optimizations => false),
        ),
        args...,
    )

    return quote
        $MLIR.IR.@dispose ctx = $Reactant.ReactantContext() begin
            $code_xla(
                ctx,
                $(esc(f)),
                $(esc(args));
                fn_kwargs=(; $(esc.(kwargs)...)),
                $(esc.(options)...),
            )
        end
    end
end

"""
    @compile [optimize = ...] [no_nan = <true/false>] [sync = <true/false>] f(args...)

Compile the function `f` with arguments `args` and return the compiled function.

## Note

Note that `@compile foo(bar(x))` is equivalent to
```julia
y = bar(x)  # first compute the output of `bar(x)`, say `y`
@compile foo(y) # then compile `foo` for `y`
```
That is, like `@jit`, `@compile` only applies to the outermost function call; it does *not* compile the composed function `foo(bar(x))` jointly.
Hence, if you want to compile the composed function `foo(bar(x))` jointly, you need to introduce an intermediate function, i.e.,
```julia
baz(x) = foo(bar(x))
@compile baz(x)
```

## Options

$(SYNC_DOCS)
$(COMMON_COMPILE_OPTIONS_DOCS)
  - `serializable`: If `true`, the compiled function will be serializable. This is needed
    for saving the compiled function to disk and loading it later. Defaults to `false`.

See also [`@jit`](@ref), [`@code_hlo`](@ref), [`@code_mhlo`](@ref), [`@code_xla`](@ref).
"""
macro compile(args...)
    (; f, args, kwargs, options) = parse_call_expr(
        merge(
            get_common_compile_options(),
            Dict{Symbol,Any}(:sync => false, :serializable => false),
        ),
        args...,
    )
    return quote
        $MLIR.IR.@dispose ctx = $Reactant.ReactantContext() begin
            $compile(
                ctx,
                $(esc(f)),
                $(esc(args));
                fn_kwargs=(; $(esc.(kwargs)...)),
                $(esc.(options)...),
            )
        end
    end
end

"""
    @jit [optimize = ...] [no_nan = <true/false>] [sync = <true/false>] f(args...)

Run @compile f(args..) then immediately execute it. Most users should use [`@compile`](@ref)
instead to cache the compiled function and execute it later.

## Note

Note that `@jit foo(bar(x))` is equivalent to
```julia
y = bar(x)  # first compute the output of `bar(x)`, say `y`
@jit foo(y) # then compile `foo` for `y` and execute it
```
That is, like `@compile`, `@jit` only applies to the outermost function call; it does *not* compile the composed function `foo(bar(x))` jointly.
Hence, if you want to compile the composed function `foo(bar(x))` jointly, you need to introduce an intermediate function, i.e.,
```julia
baz(x) = foo(bar(x))
@jit baz(x)
```

## Options

$(SYNC_DOCS)
$(COMMON_COMPILE_OPTIONS_DOCS)

See also [`@compile`](@ref), [`@code_hlo`](@ref), [`@code_mhlo`](@ref), [`@code_xla`](@ref).
"""
macro jit(args...)
    default_options = merge(get_common_compile_options(), Dict{Symbol,Any}(:sync => false))
    (; f, args, kwargs, options) = parse_call_expr(default_options, args...)
    return quote
        $MLIR.IR.@dispose ctx = $Reactant.ReactantContext() begin
            fn = $compile(
                ctx,
                $(esc(f)),
                $(esc(args));
                fn_kwargs=(; $(esc.(kwargs)...)),
                $(esc.(options)...),
            )
            fn($(esc(args))...)
        end
    end
end

function parse_call_expr(options::Dict, args...)
    while length(args) > 1
        option, args = args[1], args[2:end]
        if !Meta.isexpr(option, :(=))
            error("Invalid option $(option)")
        else
            option_name = option.args[1]
            @assert haskey(options, option_name) "Invalid option name '$(option_name)'. Valid options are $(join(keys(options), ", "))"
            options[option_name] = option.args[2]
        end
    end

    call = only(args)

    if Meta.isexpr(call, :call)
        bcast, fname, fname_full = correct_maybe_bcast_call(call.args[1])
        fname = if bcast
            quote
                if isdefined(mod, $(Meta.quot(fname_full)))
                    $(fname_full)
                else
                    Base.Broadcast.BroadcastFunction($(fname))
                end
            end
        else
            :($(fname))
        end
        args_rhs = call.args[2:end]

        # if (;) is used, we need to extract the kwargs
        if length(args_rhs) ≥ 1 && Meta.isexpr(args_rhs[1], :parameters)
            kwargs_rhs = args_rhs[1].args
            args_rhs = args_rhs[2:end]
        else
            kwargs_rhs = ()
        end
        kw_idxs = findall(Base.Fix2(Meta.isexpr, :kw), args_rhs)
        arg_idxs = setdiff(1:length(args_rhs), kw_idxs)

        kwargs_rhs = (kwargs_rhs..., args_rhs[kw_idxs]...)
        args_rhs = Expr(:tuple, args_rhs[arg_idxs]...)
    elseif Meta.isexpr(call, :(.), 2) && Meta.isexpr(call.args[2], :tuple)
        fname = :($(Base.Broadcast.BroadcastFunction)($(call.args[1])))
        args_rhs = only(call.args[2:end])
        kwargs_rhs = ()
    else
        error("Invalid function call: $(call)")
    end

    return (;
        f=fname,
        args=args_rhs,
        kwargs=kwargs_rhs,
        options=Expr.(:kw, keys(options), values(options)),
    )
end

function get_common_compile_options()
    return Dict{Symbol,Any}(
        :optimize => true,
        :no_nan => false,
        :client => nothing,
        :raise => false,
        :raise_first => false,
        :shardy_passes => :(:post_sdy_propagation),
        :assert_nonallocating => false,
        :donated_args => :(:auto),
        :transpose_propagate => :(:up),
        :reshape_propagate => :(:up),
        :optimize_then_pad => true,
        :optimize_communications => true,
        :cudnn_hlo_optimize => false,
        :legalize_chlo_to_stablehlo => false,
        :compile_options => missing,
        :strip_llvm_debuginfo => false,
    )
end

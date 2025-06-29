# TODO: at some point, we should use the TF C++ API to export the SavedModel

function Reactant.Serialization.serialization_supported(::Val{:SavedModel})
    return SAVED_MODEL_EXPORT_SUPPORTED[]
end

function _extract_call_parameters(args::Tuple, input_locations, state_dict)
    call_args = []
    for loc in input_locations
        if loc isa Reactant.Serialization.TFSavedModel.InputArgument
            push!(call_args, args[loc.position])
        else
            push!(call_args, state_dict[loc.name])
        end
    end
    return call_args
end

function _wrap_as_tf_func(spec::Reactant.Serialization.TFSavedModel.ReactantFunctionSpec)
    Touts = pylist([string(sig.dtype) for sig in spec.output_signature])
    Souts = pylist([pylist(sig.shape) for sig in spec.output_signature])
    return pyfunc(
        function (args...)
            return tf2xlaptr[].call_module(
                pytuple(
                    _extract_call_parameters(args, spec.input_locations, spec.state_dict)
                );
                version=5,
                Tout=Touts,  # dtype information
                Sout=Souts,  # Shape information
                function_list=pylist([]),  # No functions to call
                :module => spec.bytecode,
            )
        end,
    )
end

function _make_input_signatures(
    fn_spec::Reactant.Serialization.TFSavedModel.ReactantFunctionSpec
)
    input_pos_to_spec = Dict(
        loc.position => spec for
        (loc, spec) in zip(fn_spec.input_locations, fn_spec.input_signature) if
        loc isa Reactant.Serialization.TFSavedModel.InputArgument
    )

    sigs = []
    for i in 1:length(input_pos_to_spec)
        spec = input_pos_to_spec[i]
        dtype = getproperty(tfptr[], spec.dtype)
        push!(
            sigs,
            tfptr[].TensorSpec(;
                shape=pylist(spec.shape), dtype=dtype, name="args_$(i - 1)"
            ),
        )
    end
    return sigs
end

function Reactant.Serialization.TFSavedModel.__to_tf_saved_model(
    fn_spec::Reactant.Serialization.TFSavedModel.ReactantFunctionSpec, path::String
)
    tfm = tfptr[].Module()

    state_dict = Dict(
        k => tfptr[].Variable(
            npptr[].asarray(permutedims(v, collect(ndims(v):-1:1)));
            # npptr[].asarray(v);
            trainable=false,
            name=k,
        ) for (k, v) in fn_spec.state_dict
    )

    @show fn_spec.input_signature
    @show fn_spec.output_signature

    input_signatures = _make_input_signatures(fn_spec)

    tfm.f = getproperty(tfptr[], :function)(
        _wrap_as_tf_func(fn_spec); input_signature=pylist(input_signatures)
    )
    tfm._variables = pylist(collect(values(state_dict)))

    signatures = Dict(
        tfptr[].saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY =>
            tfm.f.get_concrete_function(pylist(input_signatures)...),
    )
    save_options = tfptr[].saved_model.SaveOptions(; function_aliases=Dict("" => tfm.f))

    tfptr[].saved_model.save(tfm, path; signatures=signatures, options=save_options)

    return nothing
end

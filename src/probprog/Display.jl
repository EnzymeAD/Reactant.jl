# Reference: https://github.com/probcomp/Gen.jl/blob/91d798f2d2f0c175b1be3dc6daf3a10a8acf5da3/src/choice_map.jl#L104

function _format_array(arr::AbstractArray; n_show::Int=3, indent::Int=0)
    nd = ndims(arr)
    if nd == 0
        return string(arr[])
    elseif nd == 1
        len = length(arr)
        if len <= 2 * n_show
            return "[" * join(arr, " ") * "]"
        end
        first_part = join(arr[1:n_show], " ")
        last_part = join(arr[(end - n_show + 1):end], " ")
        return "[$first_part ... $last_part]"
    else
        n_slices = size(arr, 1)
        indent_str = " "^(indent + 1)

        if n_slices <= 2 * n_show
            slice_strs = [
                _format_array(selectdim(arr, 1, i); n_show=n_show, indent=indent + 1) for
                i in 1:n_slices
            ]
            return "[" * join(slice_strs, "\n" * indent_str) * "]"
        else
            first_slices = [
                _format_array(selectdim(arr, 1, i); n_show=n_show, indent=indent + 1) for
                i in 1:n_show
            ]
            last_slices = [
                _format_array(selectdim(arr, 1, i); n_show=n_show, indent=indent + 1) for
                i in (n_slices - n_show + 1):n_slices
            ]
            return "[" *
                   join(first_slices, "\n" * indent_str) *
                   "\n" *
                   indent_str *
                   "..." *
                   "\n" *
                   indent_str *
                   join(last_slices, "\n" * indent_str) *
                   "]"
        end
    end
end

function _format_digest(value; n_show::Int=3)
    if isa(value, Tuple)
        if length(value) == 1
            return _format_digest(value[1]; n_show=n_show)
        else
            formatted = [_format_digest(v; n_show=n_show) for v in value]
            return "(" * join(formatted, ", ") * ")"
        end
    elseif isa(value, AbstractArray)
        return _format_array(value; n_show=n_show, indent=0)
    else
        return string(value)
    end
end

function _show_pretty(io::IO, trace::ProbProgTrace, pre::Int, vert_bars::Tuple)
    VERT = '\u2502'
    PLUS = '\u251C'
    HORZ = '\u2500'
    LAST = '\u2514'

    indent_vert = vcat(Char[' ' for _ in 1:pre], Char[VERT, '\n'])
    indent = vcat(Char[' ' for _ in 1:pre], Char[PLUS, HORZ, HORZ, ' '])
    indent_last = vcat(Char[' ' for _ in 1:pre], Char[LAST, HORZ, HORZ, ' '])

    for i in vert_bars
        indent_vert[i] = VERT
        indent[i] = VERT
        indent_last[i] = VERT
    end

    indent_vert_str = join(indent_vert)
    indent_str = join(indent)
    indent_last_str = join(indent_last)

    sorted_choices = sort(collect(trace.choices); by=x -> x[1])
    n = length(sorted_choices)

    if trace.retval !== nothing
        n += 1
    end

    if trace.weight !== nothing
        n += 1
    end

    cur = 1

    if trace.retval !== nothing
        print(io, indent_vert_str)
        retval_str = _format_digest(trace.retval)
        print(io, (cur == n ? indent_last_str : indent_str) * "retval : $retval_str\n")
        cur += 1
    end

    if trace.weight !== nothing
        print(io, indent_vert_str)
        print(io, (cur == n ? indent_last_str : indent_str) * "weight : $(trace.weight)\n")
        cur += 1
    end

    for (key, value) in sorted_choices
        print(io, indent_vert_str)
        value_str = _format_digest(value)
        if contains(value_str, '\n')
            indent_continuation = " "^(length(indent_str) + length(repr(key)) + 3)
            value_str = replace(value_str, "\n" => "\n" * indent_continuation)
        end
        print(io, (cur == n ? indent_last_str : indent_str) * "$(repr(key)) : $value_str\n")
        cur += 1
    end

    sorted_subtraces = sort(collect(trace.subtraces); by=x -> x[1])
    n += length(sorted_subtraces)

    for (key, subtrace) in sorted_subtraces
        print(io, indent_vert_str)
        print(io, (cur == n ? indent_last_str : indent_str) * "subtrace on $(repr(key))\n")
        _show_pretty(
            io, subtrace, pre + 4, cur == n ? (vert_bars...,) : (vert_bars..., pre + 1)
        )
        cur += 1
    end
end

function Base.show(io::IO, ::MIME"text/plain", trace::ProbProgTrace)
    println(io, "ProbProgTrace:")
    if isempty(trace.choices) && trace.retval === nothing && trace.weight === nothing
        println(io, "  (empty)")
    else
        _show_pretty(io, trace, 0, ())
    end
end

function Base.show(io::IO, trace::ProbProgTrace)
    if get(io, :compact, false)
        choices_count = length(trace.choices)
        has_retval = trace.retval !== nothing
        print(io, "ProbProgTrace($(choices_count) choices")
        if has_retval
            print(io, ", retval=$(trace.retval), weight=$(trace.weight)")
        end
        print(io, ")")
    else
        show(io, MIME"text/plain"(), trace)
    end
end

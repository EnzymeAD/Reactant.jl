# Reference: https://github.com/probcomp/Gen.jl/blob/91d798f2d2f0c175b1be3dc6daf3a10a8acf5da3/src/choice_map.jl#L104
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
        print(io, (cur == n ? indent_last_str : indent_str) * "retval : $(trace.retval)\n")
        cur += 1
    end

    if trace.weight !== nothing
        print(io, indent_vert_str)
        print(io, (cur == n ? indent_last_str : indent_str) * "weight : $(trace.weight)\n")
        cur += 1
    end

    for (key, value) in sorted_choices
        print(io, indent_vert_str)
        print(io, (cur == n ? indent_last_str : indent_str) * "$(repr(key)) : $value\n")
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

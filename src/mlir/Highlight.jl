module Highlight

using ..API: API
using ..IR: IR
using StyledStrings: Face, AnnotatedString, addface!

export highlight

struct MLIRToken
    kind::Symbol
    text::UnitRange{Int}
end

# Adding faces to the global Faces Dict should be prefaced with Reactant
# to avoid naming conflicts.
const REACTANT_THEME = [
    # Metadata / Layout
    :Reactant_default => Face(),
    :Reactant_whitespace => Face(),

    # Identifiers
    :Reactant_SSA => Face(; foreground=:cyan),
    :Reactant_symbol => Face(; foreground=:magenta),
    :Reactant_block => Face(; foreground=:green),

    # Literals & Comments
    :Reactant_comment => Face(; foreground=:gray),
    :Reactant_string => Face(; foreground=:red),
    :Reactant_punct => Face(; foreground=:red),

    # Additional categories from the C lexer
    :Reactant_keyword => Face(; foreground=:yellow),
    :Reactant_bare_identifier => Face(),
    :Reactant_hash_identifier => Face(; foreground=:magenta),
    :Reactant_exclamation_identifier => Face(; foreground=:magenta),
    :Reactant_number => Face(; foreground=:blue),
    :Reactant_inttype => Face(; foreground=:blue),
    :Reactant_error => Face(; foreground=:red, bold=true),
]

function register_reactant_theme()
    return foreach(addface!, REACTANT_THEME)
end

# Map C-side token category integers to Julia face symbols.
# Must match mlirTokenKindToCategory in API.cpp.
const TOKEN_CATEGORY_MAP = Dict{Int32,Symbol}(
    1 => :Reactant_SSA,                    # %foo
    2 => :Reactant_symbol,                 # @foo
    3 => :Reactant_block,                  # ^foo
    4 => :Reactant_string,                 # "..."
    5 => :Reactant_punct,                  # punctuation
    6 => :Reactant_keyword,               # keywords (func, tensor, etc.)
    7 => :Reactant_bare_identifier,       # bare identifiers
    8 => :Reactant_hash_identifier,       # #foo
    9 => :Reactant_exclamation_identifier, # !foo
    10 => :Reactant_number,               # integers, floats
    11 => :Reactant_inttype,              # i32, si8, ui16
    -1 => :Reactant_error,               # error tokens
    0 => :Reactant_default,              # default / unknown
)

"""
    tokenize(str::AbstractString)

Parses an MLIR string using the real MLIR C++ lexer via `ReactantLexMLIR`.
Returns a `Vector{MLIRToken}`.
"""
function tokenize(str::AbstractString)
    input = String(str)
    input_len = Int32(ncodeunits(input))

    # Estimate max tokens (generous upper bound)
    max_tokens = Int32(max(input_len, Int32(16)))

    token_kinds = Vector{Int32}(undef, max_tokens)
    token_offsets = Vector{Int32}(undef, max_tokens)
    token_lengths = Vector{Int32}(undef, max_tokens)

    # We need an MLIRContext for the lexer. Use existing if available,
    # otherwise create a temporary one.
    has_ctx = IR.has_context()
    ctx = has_ctx ? IR.current_context() : IR.Context()

    tokens = MLIRToken[]
    try
        count = API.ReactantLexMLIR(
            ctx, input, input_len, token_kinds, token_offsets, token_lengths, max_tokens
        )

        sizehint!(tokens, count)
        for i in 1:count
            kind_int = token_kinds[i]
            # C offsets are 0-based byte offsets, Julia strings are 1-indexed
            offset = token_offsets[i]
            length = token_lengths[i]
            start_idx = offset + 1  # convert to 1-based
            end_idx = offset + length  # inclusive end in 1-based

            face = get(TOKEN_CATEGORY_MAP, kind_int, :Reactant_default)
            push!(tokens, MLIRToken(face, start_idx:end_idx))
        end
    finally
        if !has_ctx
            IR.dispose(ctx)
        end
    end
    return tokens
end

"""
    style(str::AbstractString, tokens::Vector{MLIRToken})

Takes a string to be styled and a Vector with information for styling then
uses Julia's AnnotatedStrings to make pretty colors when printing.
"""
function style(str::AbstractString, tokens::Vector{MLIRToken})
    @static if VERSION < v"1.11"
        styling_info = Vector{Tuple{UnitRange{Int},Pair{Symbol,Symbol}}}()
        sizehint!(styling_info, length(tokens))
        for token in tokens
            push!(styling_info, (token.text, :face => token.kind))
        end
        styled_text = AnnotatedString(str, styling_info)
        return styled_text
    else
        styling_info = Vector{Tuple{UnitRange{Int},Symbol,Symbol}}()
        sizehint!(styling_info, length(tokens))
        for token in tokens
            push!(styling_info, (token.text, :face, token.kind))
        end
        styled_text = AnnotatedString(str, styling_info)
        return styled_text
    end
end

"""
    highlight(str::AbstractString)

Returns a `StyledString` (that prints colorfully in the REPL) by lexing the MLIR
string using the real MLIR C++ lexer for accurate tokenization.
"""
function highlight(str::AbstractString)
    tokens = tokenize(str)
    styled_text = style(str, tokens)
    return styled_text
end

end # module Highlight

using StyledStrings: Face, AnnotatedString, addface!
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
]

function register_reactant_theme()
    foreach(addface!, REACTANT_THEME)
end
"""
rule_prefixed_identifier(str::AbstractString, i, prefix::Char, kind::Symbol)

Matches a prefix character and then any number of alphanumeric chars including '_'
Useful for e.g. SSA and Symbols.
"""
@inline function rule_prefixed_identifier(
    str::AbstractString, i::Int, prefix::Char, kind::Symbol
)
    if str[i] != prefix
        return nothing
    end
    start = i
    i = nextind(str, i)
    while i <= lastindex(str) && (isletter(str[i]) || isdigit(str[i]) || str[i] == '_')
        i = nextind(str, i)
    end
    return MLIRToken(kind, range(start, prevind(str, i))), i
end

"""
rule_single_chars(str::AbstractString, i, chars::Set, kind::Symbol)

Matches a Set of singular chars.
"""
@inline function rule_single_chars(str::AbstractString, i::Int, chars::Set, kind::Symbol)
    if !(str[i] in chars)
        return nothing
    end
    start = i
    i = nextind(str, i)
    while i <= lastindex(str) && str[i] in chars
        i = nextind(str, i)
    end
    return MLIRToken(kind, range(start, prevind(str, i))), i
end

"""
rule_comment(str, i, kind::Symbol)

Matches the comment lines.
"""
@inline function rule_comment(str::AbstractString, i::Int)
    if (str[i] != '/' || nextind(str, i) > lastindex(str) || str[nextind(str, i)] != '/')
        return nothing
    end
    start = i
    i = nextind(str, i)
    while i <= lastindex(str) && str[i] != '\n'
        i = nextind(str, i)
    end
    return MLIRToken(:Reactant_comment, range(start, prevind(str, i))), i
end

"""
rule_whitespace(str, i, kind::Symbol)

Matches whitespace. 
"""
@inline function rule_whitespace(str::AbstractString, i::Int)
    if !(isspace(str[i]))
        return nothing
    end
    start = i
    i = nextind(str, i)
    while i <= lastindex(str) && isspace(str[i])
        i = nextind(str, i)
    end

    return MLIRToken(:Reactant_whitespace, range(start, prevind(str, i))), i
end
@inline rule_ssa(str, i) = rule_prefixed_identifier(str, i, '%', :Reactant_SSA)

@inline rule_symbol(str, i) = rule_prefixed_identifier(str, i, '@', :Reactant_symbol)

@inline rule_block(str, i) = rule_prefixed_identifier(str, i, '^', :Reactant_block)

const PUNCTUATION_SET = Set(":-?*=><|")
@inline rule_punctuation(str, i) =
    rule_single_chars(str, i, PUNCTUATION_SET, :Reactant_punct)

# Unrolling the tuple of functions
# There might be a smarter way to do this 
function apply_rules(str, i, rules::Tuple{})
    return nothing
end

function apply_rules(str, i, rules::Tuple)
    result = first(rules)(str, i)
    if result !== nothing
        return result
    end
    return apply_rules(str, i, Base.tail(rules))
end
"""
MLIR_parse(str::AbstractString)
Takes a string of MLIR does very simple parsing to be later used for simple highliting.
"""
function parse_MLIR(str::AbstractString, token_rules::Tuple)
    tokens = MLIRToken[]
    sizehint!(tokens, length(str))
    i = firstindex(str)
    last_idx = lastindex(str)

    while i <= last_idx
        # This call is now type-stable because 'rules' is a fixed Tuple
        res = apply_rules(str, i, token_rules)

        if res !== nothing
            token, new_i = res
            push!(tokens, token)
            i = new_i
        else
            # fallback single-char token
            push!(tokens, MLIRToken(:Reactant_default, i:i))
            i = nextind(str, i)
        end
    end
    return tokens
end
"""
MLIR_highlight(str::AbstractString, tokens::Vector{MLIRToken})

Takes a string to be styled and a Vector with information for styling then 
usess Julias AnnotatedStrings to make pretty colors when printing
"""
function style_MLIR(str::AbstractString, tokens::Vector{MLIRToken})
    styling_info = Vector{Tuple{UnitRange{Int},Symbol,Symbol}}()
    sizehint!(styling_info, length(tokens))
    for token in tokens
        push!(styling_info, (token.text, :face, token.kind))
    end
    styled_text = AnnotatedString(str, styling_info)
    return styled_text
end

const DEFAULT_RULES = (
    rule_whitespace, rule_symbol, rule_ssa, rule_block, rule_comment, rule_punctuation
)# Order matters
"""
highlight_MLIR(str::AbstractString, rules::Tuple=DEFAULT_RULES)

Returns a StyledString (that prints colorfully in the REPL) by parsing a string according to a tuple of rules. 
Rules should return an MLIRToken or nothing if there was no match.
A few simple rules are provided as the default using Reactant's default faces.
"""
function highlight_MLIR(str::AbstractString, rules::Tuple=DEFAULT_RULES)
    tokens = parse_MLIR(str, rules)
    styled_text = style_MLIR(str, tokens)
    return styled_text
end

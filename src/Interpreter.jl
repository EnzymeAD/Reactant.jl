# Taken from https://github.com/JuliaLang/julia/pull/52964/files#diff-936d33e524bcd097015043bd6410824119be5c210d43185c4d19634eb4912708
# Other references:
# - https://github.com/JuliaLang/julia/blob/0fd1f04dc7d4b905b0172b7130e9b1beab9bc4c9/test/compiler/AbstractInterpreter.jl#L228-L234
# - https://github.com/JuliaLang/julia/blob/v1.10.4/test/compiler/newinterp.jl#L9

const CC = Core.Compiler

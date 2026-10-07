module ReactantTablesExt

using Reactant: TracedRNumber
using Tables: Tables

# `Tables.Col` comparisons build lazy scan predicates and are defined against `Number`,
# which is ambiguous with Reactant's catch-all `TracedRNumber` comparisons. A `Col` is a
# `ScanExpr`, not something that can be promoted to a traced number, so the mixed cases
# should always run Tables' method.
@static if isdefined(Tables, :Col)
    for jlop in (:(<), :(<=))
        @eval begin
            function Base.$(jlop)(lhs::TracedRNumber, rhs::Tables.Col)
                return invoke(Base.$(jlop), Tuple{Number,Tables.Col}, lhs, rhs)
            end
            function Base.$(jlop)(lhs::Tables.Col, rhs::TracedRNumber)
                return invoke(Base.$(jlop), Tuple{Tables.Col,Number}, lhs, rhs)
            end
        end
    end
end

end

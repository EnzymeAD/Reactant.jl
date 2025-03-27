using BenchmarkTools
using OffsetArrays

const dim = 1000

x = Array{Float64}(undef, 2*dim);
y = OffsetArray{Float64}(undef, -dim + 1 : dim);
x2d = Array{Float64}(undef, 2*dim, 2*dim);
y2d = OffsetArray{Float64}(undef, -dim + 1 : dim, -dim + 1 : dim);

s = OffsetVector(1:dim, 0);
sur = 1:dim;
sior = OffsetArrays.IdOffsetRange(parent(s));

fill1d(x) = for i in axes(x,1); x[i] = i; end
fill2d(x) = for j in axes(x,2); for i in axes(x,1); x[i,j] = i + j; end; end
update(x) = for i in axes(x,1); x[i] = x[i] + i; end
update2d(x) = for j in axes(x,2); for i in axes(x,1); x[i,j] = x[i,j] + i + j; end; end
update_eachindex(x) = for i in eachindex(x); x[i] = x[i] + i; end

unsafe_fill(x) = @inbounds(for i in axes(x,1); x[i] = i; end)
unsafe_fill2d(x) = @inbounds(for j in axes(x,2); for i in axes(x,1); x[i,j] = i + j; end; end)
unsafe_update(x) = @inbounds(for i in axes(x,1); x[i] = x[i] + i; end)
unsafe_update2d(x) = @inbounds(for j in axes(x,2); for i in axes(x,1); x[i,j] = x[i,j] + i + j; end; end)
unsafe_update_eachindex(x) = @inbounds(for i in eachindex(x); x[i] = x[i] + i; end)

vectorlinearindexing(a, ax) = a[ax]
vectorCartesianindexing(a, ax1, ax2) = a[ax1, ax2]
nestedvectorlinearindexing(a, ax1, ax2) = a[ax1[ax2]]

macro showbenchmark(ex)
	print(ex, " : ")
	quote
		println(@benchmark esc($ex))
	end
end

@showbenchmark fill1d(x)
@showbenchmark fill1d(y)
@showbenchmark fill2d(x2d)
@showbenchmark fill2d(y2d)
@showbenchmark update(x)
@showbenchmark update(y)
@showbenchmark update2d(x2d)
@showbenchmark update2d(y2d)
@showbenchmark update_eachindex(x)
@showbenchmark update_eachindex(y)

@showbenchmark unsafe_fill(x)
@showbenchmark unsafe_fill(y)
@showbenchmark unsafe_fill2d(x2d)
@showbenchmark unsafe_fill2d(y2d)
@showbenchmark unsafe_update(x)
@showbenchmark unsafe_update(y)
@showbenchmark unsafe_update2d(x2d)
@showbenchmark unsafe_update2d(y2d)
@showbenchmark unsafe_update_eachindex(x)
@showbenchmark unsafe_update_eachindex(y)

# Benchmarks of vector indexing using OffsetRanges as axes
@showbenchmark vectorlinearindexing(x, s)
@showbenchmark vectorlinearindexing(x, sur)
@showbenchmark vectorlinearindexing(x, sior)
@showbenchmark vectorlinearindexing(y, s)
@showbenchmark vectorlinearindexing(y, sur)
@showbenchmark vectorlinearindexing(y, sior)

@showbenchmark vectorlinearindexing(sur, s)
@showbenchmark vectorlinearindexing(sur, sur)
@showbenchmark vectorlinearindexing(sur, sior)

@showbenchmark vectorCartesianindexing(x2d, s, s)
@showbenchmark vectorCartesianindexing(x2d, sur, sur)
@showbenchmark vectorCartesianindexing(x2d, sior, sior)

@showbenchmark nestedvectorlinearindexing(x, s, s)
@showbenchmark nestedvectorlinearindexing(x, sur, sur)
@showbenchmark nestedvectorlinearindexing(x, s, sior)
@showbenchmark nestedvectorlinearindexing(x, sur, sior)
@showbenchmark nestedvectorlinearindexing(x, sior, sior)
@showbenchmark vectorlinearindexing(x, sior[sior])
@showbenchmark nestedvectorlinearindexing(x, sur, sior)
@showbenchmark vectorlinearindexing(x, sur[sior])
@showbenchmark nestedvectorlinearindexing(x, sior, sur)
@showbenchmark vectorlinearindexing(x, sior[sur])
@showbenchmark nestedvectorlinearindexing(x, sur, sur)

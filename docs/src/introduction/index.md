# [Getting Started](@id getting-started)

## Installation

Install [Julia v1.10 or above](https://julialang.org/downloads/). Reactant.jl is available
through the Julia package manager. You can enter it by pressing `]` in the REPL and then
typing `add Reactant`. Alternatively, you can also do

```julia
import Pkg
Pkg.add("Reactant")
```

## Quick Start

Reactant provides two new array types at its core, a ConcreteRArray and a TracedRArray. A
ConcreteRArray is an underlying buffer to whatever device data you wish to store and can be
created by converting from a regular Julia Array.

```@example quickstart
using Reactant

julia_data = ones(2, 10)
reactant_data = Reactant.ConcreteRArray(julia_data)
```

You can also create a ConcreteRArray-version of an arbitrary data type by tracing through
the structure, like below.

```@example quickstart
struct Pair{A,B}
   x::A
   y::B
end

pair = Pair(ones(3), ones(10))

reactant_pair = Reactant.to_rarray(pair)
```

To compile programs using ConcreteRArray's, one uses the compile function, like as follows:

```@example quickstart
input1 = Reactant.ConcreteRArray(ones(10))
input2 = Reactant.ConcreteRArray(ones(10))

function sinsum_add(x, y)
   return sum(sin.(x) .+ y)
end

f = @compile sinsum_add(input1,input2)

# one can now run the program
f(input1, input2)
```


## Tips

### Empty Cache

When you encounter OOM (Out of Memory) errors, you can try to clear the cache by using Julia's builtin `GC.gc()` between memory-intensive operations.

Note：This only will free memory which is not currently live. If the result of compiled function was stored in a vector, it would still be live and `GC.gc()` would not free it.

```julia
using Reactant
n = 500_000_000
input1 = Reactant.ConcreteRArray(ones(n))
input2 = Reactant.ConcreteRArray(ones(n))

function sin_add(x, y)
   return sin.(x) .+ y
end

f = @compile sin_add(input1,input2)

for i = 1:10
   GC.gc()
   @info "gc... $i"
   f(input1, input2) # May cause OOM here for a 24GB GPU if GC is not used
end
```

If you **don't** use `GC.gc()` here, this may cause an OOM:

<details>
  <summary>View Outputs</summary>

```bash
[ Info: gc... 1
[ Info: gc... 2
[ Info: gc... 3
2025-01-05 09:48:28.754214: W external/xla/xla/tsl/framework/bfc_allocator.cc:501] Allocator (GPU_0_bfc) ran out of memory trying to allocate 3.72GiB (rounded to 4000000000)requested by op 
2025-01-05 09:48:28.754302: I external/xla/xla/tsl/framework/bfc_allocator.cc:1058] BFCAllocator dump for GPU_0_bfc
2025-01-05 09:48:28.754326: I external/xla/xla/tsl/framework/bfc_allocator.cc:1065] Bin (256):  Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2025-01-05 09:48:28.754348: I external/xla/xla/tsl/framework/bfc_allocator.cc:1065] Bin (512):  Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2025-01-05 09:48:28.754366: I external/xla/xla/tsl/framework/bfc_allocator.cc:1065] Bin (1024):         Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2025-01-05 09:48:28.754389: I external/xla/xla/tsl/framework/bfc_allocator.cc:1065] Bin (2048):         Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2025-01-05 09:48:28.754414: I external/xla/xla/tsl/framework/bfc_allocator.cc:1065] Bin (4096):         Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2025-01-05 09:48:28.754432: I external/xla/xla/tsl/framework/bfc_allocator.cc:1065] Bin (8192):         Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2025-01-05 09:48:28.754449: I external/xla/xla/tsl/framework/bfc_allocator.cc:1065] Bin (16384):        Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2025-01-05 09:48:28.754467: I external/xla/xla/tsl/framework/bfc_allocator.cc:1065] Bin (32768):        Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2025-01-05 09:48:28.754484: I external/xla/xla/tsl/framework/bfc_allocator.cc:1065] Bin (65536):        Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2025-01-05 09:48:28.754513: I external/xla/xla/tsl/framework/bfc_allocator.cc:1065] Bin (131072):       Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2025-01-05 09:48:28.754530: I external/xla/xla/tsl/framework/bfc_allocator.cc:1065] Bin (262144):       Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2025-01-05 09:48:28.754550: I external/xla/xla/tsl/framework/bfc_allocator.cc:1065] Bin (524288):       Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2025-01-05 09:48:28.754568: I external/xla/xla/tsl/framework/bfc_allocator.cc:1065] Bin (1048576):      Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2025-01-05 09:48:28.754590: I external/xla/xla/tsl/framework/bfc_allocator.cc:1065] Bin (2097152):      Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2025-01-05 09:48:28.754607: I external/xla/xla/tsl/framework/bfc_allocator.cc:1065] Bin (4194304):      Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2025-01-05 09:48:28.754625: I external/xla/xla/tsl/framework/bfc_allocator.cc:1065] Bin (8388608):      Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2025-01-05 09:48:28.754642: I external/xla/xla/tsl/framework/bfc_allocator.cc:1065] Bin (16777216):     Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2025-01-05 09:48:28.754670: I external/xla/xla/tsl/framework/bfc_allocator.cc:1065] Bin (33554432):     Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2025-01-05 09:48:28.754688: I external/xla/xla/tsl/framework/bfc_allocator.cc:1065] Bin (67108864):     Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2025-01-05 09:48:28.754705: I external/xla/xla/tsl/framework/bfc_allocator.cc:1065] Bin (134217728):    Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2025-01-05 09:48:28.754770: I external/xla/xla/tsl/framework/bfc_allocator.cc:1065] Bin (268435456):    Total Chunks: 5, Chunks in use: 4. 17.69GiB allocated for chunks. 14.90GiB in use in bin. 14.90GiB client-requested in use in bin.
2025-01-05 09:48:28.754796: I external/xla/xla/tsl/framework/bfc_allocator.cc:1081] Bin for 3.72GiB was 256.00MiB, Chunk State: 
2025-01-05 09:48:28.754830: I external/xla/xla/tsl/framework/bfc_allocator.cc:1087]   Size: 2.79GiB | Requested Size: 0B | in_use: 0 | bin_num: 20, prev:   Size: 3.72GiB | Requested Size: 3.72GiB | in_use: 1 | bin_num: -1
2025-01-05 09:48:28.754850: I external/xla/xla/tsl/framework/bfc_allocator.cc:1094] Next region of size 18995773440
2025-01-05 09:48:28.754874: I external/xla/xla/tsl/framework/bfc_allocator.cc:1114] InUse at 797202000000 of size 4000000000 next 1
2025-01-05 09:48:28.754893: I external/xla/xla/tsl/framework/bfc_allocator.cc:1114] InUse at 7972f06b2800 of size 4000000000 next 2
2025-01-05 09:48:28.754910: I external/xla/xla/tsl/framework/bfc_allocator.cc:1114] InUse at 7973ded65000 of size 4000000000 next 3
2025-01-05 09:48:28.754928: I external/xla/xla/tsl/framework/bfc_allocator.cc:1114] InUse at 7974cd417800 of size 4000000000 next 4
2025-01-05 09:48:28.754945: I external/xla/xla/tsl/framework/bfc_allocator.cc:1114] Free  at 7975bbaca000 of size 2995773440 next 18446744073709551615
2025-01-05 09:48:28.754965: I external/xla/xla/tsl/framework/bfc_allocator.cc:1119]      Summary of in-use Chunks by size: 
2025-01-05 09:48:28.754986: I external/xla/xla/tsl/framework/bfc_allocator.cc:1122] 4 Chunks of size 4000000000 totalling 14.90GiB
2025-01-05 09:48:28.755006: I external/xla/xla/tsl/framework/bfc_allocator.cc:1126] Sum Total of in-use chunks: 14.90GiB
2025-01-05 09:48:28.755023: I external/xla/xla/tsl/framework/bfc_allocator.cc:1128] Total bytes in pool: 18995773440 memory_limit_: 18995773440 available bytes: 0 curr_region_allocation_bytes_: 37991546880
2025-01-05 09:48:28.755051: I external/xla/xla/tsl/framework/bfc_allocator.cc:1133] Stats: 
Limit:                     18995773440
InUse:                     16000000000
MaxInUse:                  16000000000
NumAllocs:                           4
MaxAllocSize:               4000000000
Reserved:                            0
PeakReserved:                        0
LargestFreeBlock:                    0

2025-01-05 09:48:28.755086: W external/xla/xla/tsl/framework/bfc_allocator.cc:512] *************************************************************************************_______________
E0105 09:48:28.755177  110350 pjrt_stream_executor_client.cc:3088] Execution of replica 0 failed: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 4000000000 bytes.
ERROR: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 4000000000 bytes.

Stacktrace:
 [1] reactant_err(msg::Cstring)
   @ Reactant.XLA ~/.julia/packages/Reactant/7m11i/src/XLA.jl:104
 [2] macro expansion
   @ ~/.julia/packages/Reactant/7m11i/src/XLA.jl:357 [inlined]
 [3] ExecutableCall
   @ ~/.julia/packages/Reactant/7m11i/src/XLA.jl:334 [inlined]
 [4] macro expansion
   @ ~/.julia/packages/Reactant/7m11i/src/Compiler.jl:798 [inlined]
 [5] (::Reactant.Compiler.Thunk{…})(::ConcreteRArray{…}, ::ConcreteRArray{…})
   @ Reactant.Compiler ~/.julia/packages/Reactant/7m11i/src/Compiler.jl:909
 [6] top-level scope
   @ ./REPL[7]:4
Some type information was truncated. Use `show(err)` to see complete types.
```

</details>

After using Julia's built-in `GC.gc()`:

<details>
  <summary>View Details</summary>

```bash
[ Info: gc... 1
[ Info: gc... 2
[ Info: gc... 3
[ Info: gc... 4
[ Info: gc... 5
[ Info: gc... 6
[ Info: gc... 7
[ Info: gc... 8
[ Info: gc... 9
[ Info: gc... 10
```
</details>





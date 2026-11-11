## weak key dictionary by object identity
mutable struct WeakKeyIdDictFinalizer{T}
    const d::T
end
(f::WeakKeyIdDictFinalizer)(_) = f.d.dirty = true

"""
    WeakKeyIdDict{K,V}()

A `WeakKeyDict` whose keys are compared with `===` (like in `IdDict`) instead of `isequal`, so any `==` or `hash`
defined on the keys is ignored. Keys must be mutable, since only mutable objects can be weakly referenced.

Like in `WeakKeyDict`, the entry of a key is not deleted as soon as the key is garbage collected: a finalizer marks the
dictionary as dirty, and the next `setindex!` or `length` deletes the dead entries. Until then, the entry is skipped
but its value is kept alive.

Like `IdDict`, it is an open-addressing hash table with linear probing, hashed by `objectid`.
"""
mutable struct WeakKeyIdDict{K,V} <: AbstractDict{K,V}
    # slot `i` holds the weak reference to its key at `2i - 1` and its value at `2i`. A key that is `#undef` marks a
    # never used slot, which ends a probe sequence; a key whose weak reference is empty marks a deleted or dead entry.
    ht::Memory{Any}
    # number of entries, including the dead ones not cleaned up yet
    count::Int
    # number of deleted entries
    ndel::Int
    const lock::ReentrantLock
    dirty::Bool
    finalizer::WeakKeyIdDictFinalizer

    function WeakKeyIdDict{K,V}() where {K,V}
        d = new{K,V}(Memory{Any}(undef, 32), 0, 0, ReentrantLock(), false)
        d.finalizer = WeakKeyIdDictFinalizer(d)
        return d
    end
end
function WeakKeyIdDict{K,V}(kv) where {K,V}
    d = WeakKeyIdDict{K,V}()
    for (k, v) in kv
        d[k] = v
    end
    return d
end
WeakKeyIdDict(ps::Pair{K,V}...) where {K,V} = WeakKeyIdDict{K,V}(ps)
WeakKeyIdDict() = WeakKeyIdDict{Any,Any}()

Base.copy(d::WeakKeyIdDict{K,V}) where {K,V} = WeakKeyIdDict{K,V}(d)
Base.empty(::WeakKeyIdDict, ::Type{K}, ::Type{V}) where {K,V} = WeakKeyIdDict{K,V}()
Base.IteratorSize(::Type{<:WeakKeyIdDict}) = Base.SizeUnknown()

Base.islocked(d::WeakKeyIdDict) = islocked(d.lock)
Base.lock(d::WeakKeyIdDict) = lock(d.lock)
Base.unlock(d::WeakKeyIdDict) = unlock(d.lock)
Base.lock(f, d::WeakKeyIdDict) = lock(f, d.lock)
Base.trylock(f, d::WeakKeyIdDict) = trylock(f, d.lock)

_nslots(ht::Memory{Any}) = length(ht) >> 1

# first slot of the probe sequence of `key`; the number of slots is a power of 2
_first_slot(ht::Memory{Any}, key) = Int(objectid(key) & (_nslots(ht) - 1)) + 1
_next_slot(ht::Memory{Any}, i::Int) = (i & (_nslots(ht) - 1)) + 1

# return the slot of `key`, or `0` if it has none
function _slot_locked(d::WeakKeyIdDict, key)
    ismutable(key) || return 0
    ht = d.ht
    i = _first_slot(ht, key)
    for _ in 1:_nslots(ht)
        isassigned(ht, 2i - 1) || return 0
        (ht[2i - 1]::WeakRef).value === key && return i
        i = _next_slot(ht, i)
    end
    return 0
end

# move the live entries to a table of `nslots` slots, dropping the deleted and dead ones
function _rehash_locked!(d::WeakKeyIdDict, nslots::Int)
    old = d.ht
    ht = Memory{Any}(undef, 2nslots)
    count = 0
    for j in 1:2:length(old)
        isassigned(old, j) || continue
        ref = old[j]::WeakRef
        k = ref.value
        GC.safepoint() # ensure `k` is now gc-rooted
        isnothing(k) && continue
        i = _first_slot(ht, k)
        while isassigned(ht, 2i - 1)
            i = _next_slot(ht, i)
        end
        ht[2i - 1] = ref
        ht[2i] = old[j + 1]
        count += 1
    end
    d.ht = ht
    d.count = count
    d.ndel = 0
    return d
end

function _cleanup_locked(d::WeakKeyIdDict)
    d.dirty || return d
    d.dirty = false
    _rehash_locked!(d, _nslots(d.ht))
    return d
end

function Base.setindex!(d::WeakKeyIdDict{K,V}, v, key) where {K,V}
    key isa K ||
        throw(ArgumentError("$(Base.limitrepr(key)) is not a valid key for type $K"))
    ismutable(key) || throw(ArgumentError("$(Base.limitrepr(key)) is not mutable"))
    v = convert(V, v)
    @lock d begin
        _cleanup_locked(d)
        i = _slot_locked(d, key)
        if i != 0
            d.ht[2i] = v
            return d
        end

        # keep the load factor (counting the deleted entries) under 3/4
        nslots = _nslots(d.ht)
        if 4 * (d.count + d.ndel + 1) > 3nslots
            _rehash_locked!(d, 4 * (d.count + 1) > 3nslots ? 2nslots : nslots)
        end

        ht = d.ht
        i = _first_slot(ht, key)
        while isassigned(ht, 2i - 1)
            i = _next_slot(ht, i)
        end
        finalizer(d.finalizer, key)
        ht[2i - 1] = WeakRef(key)
        ht[2i] = v
        d.count += 1
    end
    return d
end

function Base.get!(default::Base.Callable, d::WeakKeyIdDict{K,V}, key) where {K,V}
    @lock d begin
        i = _slot_locked(d, key)
        i == 0 || return d.ht[2i]::V
        v = convert(V, default())
        d[key] = v
        return v
    end
end
Base.get!(d::WeakKeyIdDict, key, default) = get!(() -> default, d, key)

function Base.get(default::Base.Callable, d::WeakKeyIdDict{K,V}, key) where {K,V}
    @lock d begin
        i = _slot_locked(d, key)
        return i == 0 ? default() : d.ht[2i]::V
    end
end
Base.get(d::WeakKeyIdDict, key, default) = get(() -> default, d, key)

function Base.getindex(d::WeakKeyIdDict{K,V}, key) where {K,V}
    @lock d begin
        i = _slot_locked(d, key)
        i == 0 && throw(KeyError(key))
        return d.ht[2i]::V
    end
end

Base.haskey(d::WeakKeyIdDict, key) = @lock d _slot_locked(d, key) != 0

# keys are compared by identity, so the stored key is `key` itself
function Base.getkey(d::WeakKeyIdDict{K}, key, default) where {K}
    return haskey(d, key) ? key::K : default
end

function Base.pop!(d::WeakKeyIdDict{K,V}, key, default) where {K,V}
    @lock d begin
        i = _slot_locked(d, key)
        i == 0 && return default
        v = d.ht[2i]::V
        # the empty weak reference marks the slot as deleted without ending the probe sequences that go through it
        d.ht[2i - 1] = WeakRef()
        d.ht[2i] = nothing
        d.count -= 1
        d.ndel += 1
        return v
    end
end
function Base.pop!(d::WeakKeyIdDict, key)
    @lock d begin
        haskey(d, key) || throw(KeyError(key))
        return pop!(d, key, nothing)
    end
end

function Base.delete!(d::WeakKeyIdDict, key)
    pop!(d, key, nothing)
    return d
end

function Base.empty!(d::WeakKeyIdDict)
    @lock d begin
        d.ht = Memory{Any}(undef, 32)
        d.count = 0
        d.ndel = 0
    end
    return d
end

function Base.sizehint!(d::WeakKeyIdDict, n::Integer)
    @lock d begin
        nslots = max(16, nextpow(2, cld(4n, 3)))
        nslots > _nslots(d.ht) && _rehash_locked!(d, nslots)
    end
    return d
end

function Base.length(d::WeakKeyIdDict)
    @lock d begin
        _cleanup_locked(d)
        return d.count
    end
end
Base.isempty(d::WeakKeyIdDict) = length(d) == 0

function Base.iterate(d::WeakKeyIdDict{K,V}, i::Int=0) where {K,V}
    @lock d begin
        ht = d.ht
        while i < _nslots(ht)
            i += 1
            isassigned(ht, 2i - 1) || continue
            k = (ht[2i - 1]::WeakRef).value
            GC.safepoint() # ensure `k` is now gc-rooted
            isnothing(k) && continue # deleted, or dead and scheduled for deletion
            return (Pair{K,V}(k::K, ht[2i]::V), i)
        end
        return nothing
    end
end

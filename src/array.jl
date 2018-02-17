using OpenCL, Transpiler

import Transpiler: cli
using Transpiler: GlobalPointer
import GPUArrays: GPUArray, unsafe_reinterpret, LocalMemory, gpu_sub2ind
using Sugar: to_tuple

import Base: pointer, similar, size, copy!, convert
using Base: RefValue

mutable struct CLArray{T, N} <: GPUArray{T, N}
    size::NTuple{N, Cuint}
    ptr::OwnedPtr{T}
end


# arguments are swapped to not override default constructor
function (::Type{CLArray{T, N}})(ptr::OwnedPtr{T}, size::NTuple{N, Integer}) where {T, N}
    arr = CLArray{T, N}(size, ptr)
    finalizer(arr, unsafe_free!)
    arr
end
size(x::CLArray) = Int.(x.size)
pointer(x::CLArray) = x.ptr
context(p::CLArray) = context(pointer(p))

# Avoid conflict with OpenCL.cl
module Shorthands
    using ..CLArrays: CLArray
    import Adapt: adapt, adapt_

    adapt_(::Type{<:CLArray}, xs::AbstractArray) = isbits(xs) ? xs : convert(CLArray, xs)

    cl(x) = adapt(CLArray{Float32}, x)

    export cl
end

function (::Type{CLArray{T, N}})(size::NTuple{N, Integer}, ctx::cl.Context = global_context()) where {T, N}
    # element type has different padding from cl type in julia
    # for fixedsize arrays we use vload/vstore, so we can use it packed
    clT = T #!Transpiler.is_fixedsize_array(T) ? cl.packed_convert(T) : T
    elems = prod(size)
    elems = elems == 0 ? 1 : elems # OpenCL can't allocate 0 sized buffers
    ptr = Mem.alloc(clT, elems, ctx)
    CLArray{clT, N}(ptr, size)
end

raw_print(msg::AbstractString...) =
    ccall(:write, Cssize_t, (Cint, Cstring, Csize_t), 1, join(msg), length(join(msg)))

similar(::Type{<: CLArray}, ::Type{T}, size::Base.Dims{N}) where {T, N} = CLArray{T, N}(size)

function unsafe_free!(a::CLArray)
    ptr = pointer(a)
    ctxid = context(ptr).id
    err = Mem.free(ptr)
    if err == cl.CL_SUCCESS
        Mem.current_allocated_mem[] -= sizeof(eltype(a)) * length(a)
    elseif err == cl.CL_INVALID_CONTEXT
        # ignore (log it!)
    else
        cl.@check err
    end
end

function unsafe_reinterpret(::Type{T}, A::CLArray{ET}, size::NTuple{N, Integer}) where {T, ET, N}
    ptr = pointer(A)
    Mem.retain(ptr) # we now have 2 finalizers for ptr, so it needs to be retained/increase refcount
    ptrt = OwnedPtr{T}(ptr)
    CLArray{T, N}(size, ptrt)
end

function copy!{T}(
        dest::Array{T}, d_offset::Integer,
        source::CLArray{T}, s_offset::Integer, amount::Integer
    )
    amount == 0 && return dest
    s_offset = (s_offset - 1) * sizeof(T)
    q = global_queue(source)
    # TODO unpack elements if they were converted
    unsafe_copy!(q, Ref(dest, d_offset), s_offset, pointer(source), amount * sizeof(T))
    dest
end

function copy!{T}(
        dest::CLArray{T}, d_offset::Integer,
        source::Array{T}, s_offset::Integer, amount::Integer
    )
    amount == 0 && return dest
    q = global_queue(dest)
    d_offset = (d_offset - 1) * sizeof(T)
    buff = pointer(dest)
    clT = eltype(buff)
    # element type has different padding from cl type in julia
    # for fixedsize arrays we use vload/vstore, so we can use it packed
    if sizeof(clT) != sizeof(T) && !Transpiler.is_fixedsize_array(T)
        # TODO only convert the range in the offset, or maybe convert elements and directly upload?
        # depends a bit on the overhead of cl_writebuffer
        source = map(cl.packed_convert, source)
    end
    unsafe_copy!(q, buff, d_offset, Ref(source, s_offset), amount * sizeof(clT))
    dest
end


function copy!{T}(
        dest::CLArray{T}, d_offset::Integer,
        src::CLArray{T}, s_offset::Integer, amount::Integer
    )
    amount == 0 && return dest
    q = global_queue(src)
    d_offset = (d_offset - 1) * sizeof(T)
    s_offset = (s_offset - 1) * sizeof(T)

    n_evts  = UInt(0)
    evt_ids = C_NULL
    ret_evt = Ref{cl.CL_event}()
    cl.@check cl.api.clEnqueueCopyBuffer(
        q.id, pointer(src), pointer(dest),
        s_offset, d_offset, amount * sizeof(T),
        n_evts, evt_ids, ret_evt
    )
    dest
end

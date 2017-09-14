using OpenCL, Transpiler

import Transpiler: cli
using Transpiler: GlobalPointer
import GPUArrays: GPUArray, unsafe_reinterpret, LocalMemory, gpu_sub2ind
using Sugar: to_tuple

import Base: pointer, similar, size, copy!, convert
using Base: RefValue

# pointer MUST be a type parameter, to make it easier to replace it with a non pointer type for host upload
mutable struct CLArray{T, N} <: GPUArray{T, N}
    ptr::OwnedPtr{T}
    size::NTuple{N, Cuint}
end


struct HostPtr{T}
    ptr::Int32
    (::Type{HostPtr{T}})() where T = new{T}(Int32(0))
end
Base.eltype(::Type{HostPtr{T}}) where T = T

struct DeviceArray{T, N, Ptr} <: AbstractArray{T, N}
    ptr::Ptr
    size::NTuple{N, Cuint}
end
const PreDeviceArray{T, N} = DeviceArray{T, N, HostPtr{T}}
const OnDeviceArray{T, N} = DeviceArray{T, N, GlobalPointer{T}}

kernel_convert(A::CLArray{T, N}) where {T, N} = PreDeviceArray{T, N}(HostPtr{T}(), A.size)
predevice_type(::Type{OnDeviceArray{T, N}}) where {T, N} = PreDeviceArray{T, N}
device_type(::CLArray{T, N}) where {T, N} = OnDeviceArray{T, N}
reconstruct(x::PreDeviceArray{T, N}, ptr::GlobalPointer{T}) where {T, N} = OnDeviceArray{T, N}(ptr, x.size)

kernel_convert(x::RefValue{T}) where T <: CLArray = RefValue(kernel_convert(x[]))
predevice_type(::Type{RefValue{T}}) where T <: OnDeviceArray = RefValue{predevice_type(T)}
device_type(x::RefValue{T}) where T <: CLArray = RefValue{device_type(x[])}
reconstruct(x::RefValue{T}, ptr::GlobalPointer) where T <: PreDeviceArray = RefValue(reconstruct(x[], ptr))

kernel_convert(x::Tuple) = kernel_convert.(x)
predevice_type(::Type{T}) where T <: Tuple = Tuple{predevice_type.((T.parameters...))...}
device_type(x::T) where T <: Tuple = Tuple{device_type.(x)...}

@generated function reconstruct(x::Tuple, ptrs::GlobalPointer...)
    ptrlist = to_tuple(ptrs)
    tup = Expr(:tuple)
    ptr_idx = 0
    for (xi, T) in enumerate(to_tuple(x))
        hasptr, fields = contains_pointer(T)
        if hasptr
            # consume the n pointers that T contains
            ptr_args = ntuple(i-> :(ptrs[$(i + ptr_idx)]), length(fields))
            ptr_idx += 1
            push!(tup.args, :(reconstruct(x[$xi], $(ptr_args...))))
        else
            push!(tup.args, :(x[$xi]))
        end
    end
    return tup
end


Base.size(x::OnDeviceArray) = x.size


function Base.getindex(x::Arr{T, N}, i::Vararg{Integer, N}) where {T, N}
    return x.x[i...]
end

function Base.getindex(x::OnDeviceArray{T, N}, i::Vararg{Integer, N}) where {T, N}
    ilin = gpu_sub2ind(size(x), Cuint.(i))
    return x.ptr[ilin]
end
function Base.setindex!(x::OnDeviceArray{T, N}, val, i::Vararg{Integer, N}) where {T, N}
    ilin = gpu_sub2ind(size(x), Cuint.(i))
    x.ptr[ilin] = T(val)
    return
end
function Base.setindex!(x::OnDeviceArray{T, N}, val, ilin::Integer) where {T, N}
    x.ptr[ilin] = T(val)
    return
end

function Base.getindex(x::OnDeviceArray, ilin::Integer)
    return x.ptr[ilin]
end
# arguments are swapped to not override default constructor
function (::Type{CLArray{T, N}})(size::NTuple{N, Integer}, ptr::OwnedPtr{T}) where {T, N}
    arr = CLArray{T, N}(ptr, size)
    finalizer(arr, unsafe_free!)
    arr
end
size(x::CLArray) = Int.(x.size)
pointer(x::CLArray) = x.ptr
context(p::CLArray) = context(pointer(p))

function (::Type{CLArray{T, N}})(size::NTuple{N, Integer}, ctx = global_context()) where {T, N}
    # element type has different padding from cl type in julia
    # for fixedsize arrays we use vload/vstore, so we can use it packed
    clT = !Transpiler.is_fixedsize_array(T) ? cl.packed_convert(T) : T
    elems = prod(size)
    elems = elems == 0 ? 1 : elems # OpenCL can't allocate 0 sized buffers
    ptr = Mem.alloc(clT, elems, ctx)
    arr = CLArray{clT, N}(size, ptr)
    arr
end


similar(::Type{<: CLArray}, ::Type{T}, size::Base.Dims{N}) where {T, N} = CLArray{T, N}(size)

function unsafe_free!(a::CLArray)
    ptr = pointer(a)
    if !cl.is_ctx_id_alive(context(ptr).id)
        #TODO logging that we don't free since context is not alive
    else
        Mem.free(ptr)
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

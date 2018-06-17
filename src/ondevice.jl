import Base: setindex!, getindex, size, IndexStyle, sum, eltype
using Base: IndexLinear

using Transpiler.cli: LocalPointer
using Transpiler.cli: get_local_id, get_global_id, barrier,  CLK_LOCAL_MEM_FENCE
using Transpiler.cli: get_local_size, get_global_size, get_group_id, get_num_groups
import GPUArrays: synchronize, synchronize_threads, device, global_size, linear_index

import GPUArrays: LocalMemory
using GPUArrays: AbstractDeviceArray


"""
Array type on the device
"""
struct DeviceArray{T, N, Ptr} <: AbstractDeviceArray{T, N}
    size::NTuple{N, Cuint}
    ptr::Ptr
end
# shaninagans for uploading CLArrays to OpenCL as a DeviceArray
# (spoiler alert: they can't contain pointers while uploading, but can on the device)
"""
Dummy pointer type for inlining into structs that get uploaded to the GPU
"""
struct HostPtr{T}
    ptr::Int32
    (::Type{HostPtr{T}})() where T = new{T}(Int32(0))
end
eltype(::Type{HostPtr{T}}) where T = T
const PreDeviceArray{T, N} = DeviceArray{T, N, HostPtr{T}} # Pointer free variant for kernel upload
const GlobalArray{T, N} = DeviceArray{T, N, GlobalPointer{T}}
const LocalArray{T, N} = DeviceArray{T, N, LocalPointer{T}}

const OnDeviceArray{T, N} = Union{GlobalArray{T, N}, LocalArray{T, N}} # Variant on the device containing the correct pointer

size(x::DeviceArray) = x.size
size(x::DeviceArray, i::Integer) = x.size[i]

getindex(x::OnDeviceArray, ilin::Integer) =  x.ptr[ilin]
function getindex(x::OnDeviceArray{T, N}, i::Vararg{Integer, N}) where {T, N}
    ilin = gpu_sub2ind(size(x), Cuint.(i))
    return x.ptr[ilin]
end
function setindex!(x::OnDeviceArray{T, N}, val, ilin::Integer) where {T, N}
    x.ptr[ilin] = T(val)
    return
end

function setindex!(x::OnDeviceArray{T, N}, val, i::Vararg{Integer, N}) where {T, N}
    ilin = gpu_sub2ind(size(x), Cuint.(i))
    x.ptr[ilin] = T(val)
    return
end


predevice_type(::Type{T}) where T = T


kernel_convert(A::CLArray{T, N}) where {T, N} = PreDeviceArray{T, N}(A.size, HostPtr{T}())
predevice_type(::Type{GlobalArray{T, N}}) where {T, N} = PreDeviceArray{T, N}
device_type(::CLArray{T, N}) where {T, N} = GlobalArray{T, N}
reconstruct(x::PreDeviceArray{T, N}, ptr::GlobalPointer{T}) where {T, N} = GlobalArray{T, N}(x.size, ptr)

# some converts to inline CLArrays into tuples and refs
kernel_convert(x::RefValue{T}) where T <: CLArray = RefValue(kernel_convert(x[]))
predevice_type(::Type{RefValue{T}}) where T <: GlobalArray = RefValue{predevice_type(T)}
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


#synchronize
function synchronize(x::CLArray)
    cl.finish(global_queue(x)) # TODO figure out the diverse ways of synchronization
end


immutable KernelState
    empty::Int32
    KernelState() = new(Int32(0))
end

for (i, sym) in enumerate((:x, :y, :z))
    for (f, fcl, isidx) in (
            (:blockidx, get_group_id, true),
            (:blockdim, get_local_size, false),
            (:threadidx, get_local_id, true),
            (:griddim, get_num_groups, false)
        )

        fname = Symbol(string(f, '_', sym))
        if isidx
            @eval GPUArrays.$fname(::KernelState)::Cuint = $fcl($(i-1)) + Cuint(1)
        else
            @eval GPUArrays.$fname(::KernelState)::Cuint = $fcl($(i-1))
        end
    end
end

global_size(state::KernelState) = get_global_size(0)
linear_index(state::KernelState) = get_global_id(0) + Cuint(1)

synchronize_threads(::KernelState) = cli.barrier(CLK_LOCAL_MEM_FENCE)

LocalMemory(state::KernelState, ::Type{T}, ::Val{N}, ::Val{C}) where {T, N, C} = Transpiler.cli.LocalPointer{T}()

function (::Type{AbstractDeviceArray})(ptr::PtrT, shape::Vararg{Integer, N}) where PtrT <: Transpiler.cli.LocalPointer{T} where {T, N}
    DeviceArray{T, N, PtrT}(shape, ptr)
end
function (::Type{AbstractDeviceArray})(ptr::PtrT, shape::NTuple{N, Integer}) where PtrT <: Transpiler.cli.LocalPointer{T} where {T, N}
    DeviceArray{T, N, PtrT}(shape, ptr)
end

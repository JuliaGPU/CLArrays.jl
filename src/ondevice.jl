import Base: setindex!, getindex, size, IndexStyle, next, done, start, sum, eltype
using Base: IndexLinear
"""
Array type on the device
"""
struct DeviceArray{T, N, Ptr} <: AbstractArray{T, N}
    ptr::Ptr
    size::NTuple{N, Cuint}
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
const OnDeviceArray{T, N} = DeviceArray{T, N, GlobalPointer{T}} # Variant on the device containing the correct pointer

size(x::OnDeviceArray) = x.size
IndexStyle(::OnDeviceArray) = IndexLinear()
start(x::OnDeviceArray) = Cuint(1)
next(x::OnDeviceArray, state::Cuint) = x[state], state + Cuint(1)
done(x::OnDeviceArray, state::Cuint) = state > length(x)

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



kernel_convert(A::CLArray{T, N}) where {T, N} = PreDeviceArray{T, N}(HostPtr{T}(), A.size)
predevice_type(::Type{OnDeviceArray{T, N}}) where {T, N} = PreDeviceArray{T, N}
device_type(::CLArray{T, N}) where {T, N} = OnDeviceArray{T, N}
reconstruct(x::PreDeviceArray{T, N}, ptr::GlobalPointer{T}) where {T, N} = OnDeviceArray{T, N}(ptr, x.size)

# some converts to inline CLArrays into tuples and refs
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



function sum(A::CLArrays.DeviceArray{T}) where T
    acc = zero(T)
    for elem in A
        acc += elem
    end
    acc
end

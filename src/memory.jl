export Mem

module Mem

import ..CLArrays: context
using OpenCL
import Base: pointer, eltype, cconvert, convert, unsafe_copy!
using GPUArrays: supports_double, device, global_memory
using GPUArrays

immutable OwnedPtr{T}
    ptr::cl.CL_mem
    mapped::Bool
    context::cl.Context
end
pointer(p::OwnedPtr) = p.ptr
eltype{T}(::Type{OwnedPtr{T}}) = T
cconvert(::Type{cl.CL_mem}, p::OwnedPtr) = pointer(p)
context(p::OwnedPtr) = p.context


function convert{T}(::Type{OwnedPtr{T}}, p::OwnedPtr)
    OwnedPtr{T}(pointer(p), p.mapped, context(p))
end

function retain(p::OwnedPtr)
    err = cl.api.clRetainMemObject(p)
    return err
end




const current_allocated_mem = Ref(0)
function free(p::OwnedPtr)
    cl.api.clReleaseMemObject(p)
end

function pressure_gc!(device, bytes)
    # If we used up 80% of our device. lets free see if we can free stuff with a gc swipe.
    # try three times, first only with a an unforced swipe
    mem = current_allocated_mem[] + bytes
    if (global_memory(device) * 0.8) < mem
        println("calling unforced gc")
        gc(false) # non force
    end
    if (global_memory(device) * 0.8) < mem
        println("force swipe")
        gc() # non forced
    end
    current_allocated_mem[] += bytes
end

function alloc(T, elems::Integer, ctx::cl.Context, flags = cl.CL_MEM_READ_WRITE)
    dev = GPUArrays.device(ctx)
    if T == Float64 && !supports_double(dev)
        error("Float64 is not supported by your device: $dev. Make sure to convert all types for the GPU to Float32")
    end
    nbytes = cl.cl_uint(elems * sizeof(T))
    pressure_gc!(dev, nbytes)
    err_code = Ref{cl.CL_int}()
    mem_id = cl.api.clCreateBuffer(
        ctx.id, flags, nbytes,
        C_NULL,
        err_code
    )
    if err_code[] != cl.CL_SUCCESS
        throw(cl.CLError(err_code[]))
    end
    OwnedPtr{T}(mem_id, false, ctx)
end

function cl.set_arg!(k::cl.Kernel, idx::Integer, arg::Mem.OwnedPtr)
    arg_boxed = Base.RefValue(pointer(arg))
    cl.@check cl.api.clSetKernelArg(k.id, cl.cl_uint(idx-1), sizeof(cl.CL_mem), arg_boxed)
    return k
end

function unsafe_copy!(
        queue, hostref::Ref, dev_offset::Integer, ptr::OwnedPtr, nbytes::Integer
    )
    n_evts  = UInt(0)
    evt_ids = C_NULL
    ret_evt = Ref{cl.CL_event}()
    cl.@check cl.api.clEnqueueReadBuffer(
        queue.id, ptr, cl.cl_bool(true),
        dev_offset, nbytes, hostref,
        n_evts, evt_ids, ret_evt
    )
    return
end

function unsafe_copy!(
        queue, ptr::OwnedPtr, dev_offset::Integer, hostref::Ref, nbytes::Integer
    )
    n_evts  = UInt(0)
    evt_ids = C_NULL
    ret_evt = Ref{cl.CL_event}()
    cl.@check cl.api.clEnqueueWriteBuffer(
        queue.id, ptr, cl.cl_bool(true),
        dev_offset, nbytes, hostref,
        n_evts, evt_ids, ret_evt
    )
    return
end

export OwnedPtr

end

using .Mem

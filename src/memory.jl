export Mem

module Mem

import ..CLArrays: context
using OpenCL
import Base: pointer, eltype, cconvert, convert, unsafe_copy!

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
    cl.@check cl.api.clRetainMemObject(p)
    return
end

function free(p::OwnedPtr)
    cl.@check_release cl.api.clReleaseMemObject(p)
    return
end

function alloc(T, elems::Integer, ctx::cl.Context, flags = cl.CL_MEM_READ_WRITE)
    err_code = Ref{cl.CL_int}()
    mem_id = cl.api.clCreateBuffer(
        ctx.id, flags, cl.cl_uint(elems * sizeof(T)),
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

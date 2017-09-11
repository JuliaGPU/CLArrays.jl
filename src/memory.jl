immutable OwnedPtr{T}
    ptr::cl.CL_mem
    mapped::Bool
    context::cl.Context
end


function alloc(ctx::cl.Context, nbytes::Integer, flags = cl.CL_MEM_READ_WRITE)
    err_code = Ref{CL_int}()
    mem_id = cl.api.clCreateBuffer(
        ctx.id, flags, cl.cl_uint(nbytes),
        C_NULL,
        err_code
    )
    if err_code[] != cl.CL_SUCCESS
        throw(cl.CLError(err_code[]))
    end
    OwnedPtr{Void}(mem_id, false, ctx)
end

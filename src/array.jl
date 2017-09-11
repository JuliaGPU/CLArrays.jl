immutable CLArray{T, N} <: GPUArray{T, N}
    buffer::cl.Buffer{T}
    size::NTuple{N, Int}
end

buffer(x::CLArray) = x.buffer

function cl_readbuffer(q, buf, dev_offset, hostref, nbytes)
    n_evts  = UInt(0)
    evt_ids = C_NULL
    ret_evt = Ref{cl.CL_event}()
    cl.@check cl.api.clEnqueueReadBuffer(
        q.id, buf.id, cl.cl_bool(true),
        dev_offset, nbytes, hostref,
        n_evts, evt_ids, ret_evt
    )
end
function cl_writebuffer(q, buf, dev_offset, hostref, nbytes)
    n_evts  = UInt(0)
    evt_ids = C_NULL
    ret_evt = Ref{cl.CL_event}()
    cl.@check cl.api.clEnqueueWriteBuffer(
        q.id, buf.id, cl.cl_bool(true),
        dev_offset, nbytes, hostref,
        n_evts, evt_ids, ret_evt
    )
end

function Base.copy!{T}(
        dest::Array{T}, d_offset::Integer,
        source::CLArray{T}, s_offset::Integer, amount::Integer
    )
    amount == 0 && return dest
    s_offset = (s_offset - 1) * sizeof(T)
    q = global_queue(source)
    cl.finish(q)
    cl_readbuffer(q, buffer(source), unsigned(s_offset), Ref(dest, d_offset), amount * sizeof(T))
    dest
end

function Base.copy!{T}(
        dest::CLArray{T}, d_offset::Integer,
        source::Array{T}, s_offset::Integer, amount::Integer
    )
    amount == 0 && return dest
    q = global_queue(source)
    cl.finish(q)
    d_offset = (d_offset - 1) * sizeof(T)
    buff = buffer(dest)
    clT = eltype(buff)
    # element type has different padding from cl type in julia
    # for fixedsize arrays we use vload/vstore, so we can use it packed
    if sizeof(clT) != sizeof(T) && !Transpiler.is_fixedsize_array(T)
        # TODO only convert the range in the offset, or maybe convert elements and directly upload?
        # depends a bit on the overhead of cl_writebuffer
        source = map(cl.packed_convert, source)
    end
    cl_writebuffer(q, buff, unsigned(d_offset), Ref(source, s_offset), amount * sizeof(clT))
    dest
end


function Base.copy!{T}(
        dest::CLArray{T}, d_offset::Integer,
        src::CLArray{T}, s_offset::Integer, amount::Integer
    )
    amount == 0 && return dest
    q = global_queue(source)
    cl.finish(q)
    d_offset = (d_offset - 1) * sizeof(T)
    s_offset = (s_offset - 1) * sizeof(T)
    cl.enqueue_copy_buffer(
        q, buffer(src), buffer(dest),
        Csize_t(amount * sizeof(T)), Csize_t(s_offset), Csize_t(d_offset),
        nothing
    )
    dest
end

function (AT::Type{CLArray{T, N}})(
        size::NTuple{N, Int};
        ctx = global_context(),
        flag = :rw, kw_args...
    ) where {T, N}
    # element type has different padding from cl type in julia
    # for fixedsize arrays we use vload/vstore, so we can use it packed
    clT = !Transpiler.is_fixedsize_array(T) ? cl.packed_convert(T) : T
    buffsize = prod(size)
    buff = buffsize == 0 ? cl.Buffer(clT, ctx, flag, 1) : cl.Buffer(clT, ctx, flag, buffsize)
    CLArray{T, N}(buff, size)
end

function unsafe_reinterpret(::Type{T}, A::CLArray{ET}, size::Tuple) where {T, ET}
    buff = buffer(A)
    # TODO preserve!?
    newbuff = cl.Buffer{T}(buff.id, true, prod(size))
    ctx = global_context(A)
    CLArray{T, N}(buff, size)
end

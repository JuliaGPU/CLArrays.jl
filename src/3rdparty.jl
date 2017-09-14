# This hack will go away once CLBLAS doesn't
# rely on the clunky cl.CLArray type in OpenCL anymore. Sadly, we need cl.Buffer for that, which  overwrites the default
# constructor registering a finalizer, which we don't want here, so we need to
# get the real constructor.
@generated function clbuffer(ptr::OwnedPtr{T}, len::Integer) where T
    Expr(:new, cl.Buffer{T}, true, :(pointer(ptr)), :(len), :(ptr.mapped), Ptr{T}(C_NULL))
end

function clbuffer(A::CLArray)
    ptr = pointer(A)
    clbuffer(ptr, length(A))
end

import CLFFT

# figure out a gc safe way to store plans.
# weak refs won't work, since the caching should keep them alive.
# But at the end, we need to free all of these, otherwise CLFFT will crash
# at closing time.
# An atexit hook here, which will empty the dictionary seems to introduce racing
# conditions.
#const plan_dict = Dict()
import Base: *, plan_ifft!, plan_fft!, plan_fft, plan_ifft, size, plan_bfft, plan_bfft!

immutable CLFFTPlan{Direction, Inplace, T, N} <: Base.FFTW.FFTWPlan{T, Direction, Inplace}
    plan::CLFFT.Plan{T}
    function (::Type{CLFFTPlan{Direction, Inplace}})(A::CLArray{T, N}) where {T, N, Direction, Inplace}
        ctx = global_context(A)
        p = CLFFT.Plan(T, ctx, size(A))
        CLFFT.set_layout!(p, :interleaved, :interleaved)
        if Inplace
            CLFFT.set_result!(p, :inplace)
        else
            CLFFT.set_result!(p, :outofplace)
        end
        CLFFT.set_scaling_factor!(p, Direction, 1f0)
        CLFFT.bake!(p, global_queue(A))
        new{Direction, Inplace, T, N}(p)
    end
end
size(x::CLFFTPlan) = (CLFFT.lengths(x.plan)...,)

# ignore flags, but have them to make it base compatible.
# TODO can we actually implement the flags?
function plan_fft(A::CLArray; flags = nothing, timelimit = Inf)
    CLFFTPlan{:forward, false}(A)
end
function plan_fft!(A::CLArray; flags = nothing, timelimit = Inf)
    CLFFTPlan{:forward, true}(A)
end
function plan_bfft(A::CLArray, region; flags = nothing, timelimit = Inf)
    CLFFTPlan{:backward, false}(A)
end
function plan_bfft!(A::CLArray, region; flags = nothing, timelimit = Inf)
    CLFFTPlan{:backward, true}(A)
end

const _queue_ref = Vector{cl.CmdQueue}(1)
function *(plan::CLFFTPlan{Direction, true, T, N}, A::CLArray{T, N}) where {T, N, Direction}
    _queue_ref[] = global_queue(A)
    CLFFT.enqueue_transform(plan.plan, Direction, _queue_ref, clbuffer(A), nothing)
    A
end
function *(plan::CLFFTPlan{Direction, false, T, N}, A::CLArray{T, N}) where {T, N, Direction}
    _queue_ref[] = global_queue(A)
    y = typeof(A)(size(plan))
    CLFFT.enqueue_transform(plan.plan, Direction, _queue_ref, clbuffer(A), clbuffer(y))
    y
end

# Enable BLAS support via GPUArrays
import CLBLAS
import GPUArrays: blas_module, blasbuffer
blas_module(::CLArray) = CLBLAS
blasbuffer(A::CLArray) = cl.CLArray(clbuffer(A), global_queue(A), size(A))

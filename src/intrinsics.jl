using Transpiler.cli: get_local_id, get_global_id, barrier,  CLK_LOCAL_MEM_FENCE
using Transpiler.cli: get_local_size, get_global_size, get_group_id
import GPUArrays: synchronize, synchronize_threads, device
#synchronize
function synchronize(x::CLArray)
    cl.finish(context(x).queue) # TODO figure out the diverse ways of synchronization
end


immutable KernelState
    empty::Int32
    KernelState() = new(Int32(0))
end

for (f, fcl, isidx) in (
        (:blockidx, get_group_id, true),
        (:blockdim, get_local_size, false),
        (:threadidx, get_local_id, true)
    )
    for (i, sym) in enumerate((:x, :y, :z))
        fname = Symbol(string(f, '_', sym))
        if isidx
            @eval GPUArrays.$fname(::KernelState)::Cuint = $fcl($(i-1)) + Cuint(1)
        else
            @eval GPUArrays.$fname(::KernelState)::Cuint = $fcl($(i-1))
        end
    end
end

synchronize_threads(::KernelState) = cli.barrier(CLK_LOCAL_MEM_FENCE)

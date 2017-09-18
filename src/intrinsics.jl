using Transpiler.cli: get_local_id, get_global_id, barrier,  CLK_LOCAL_MEM_FENCE
using Transpiler.cli: get_local_size, get_global_size, get_group_id, get_num_groups
import GPUArrays: synchronize, synchronize_threads, device, global_size, linear_index
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

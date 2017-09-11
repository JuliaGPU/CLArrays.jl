using Transpiler.cli: get_local_id, get_global_id, barrier,  CLK_LOCAL_MEM_FENCE
using Transpiler.cli: get_local_size, get_global_size, get_group_id

#synchronize
function synchronize(x::CLArray)
    cl.finish(context(x).queue) # TODO figure out the diverse ways of synchronization
end

for (f, fcl) in (:blockidx => get_group_id, :blockdim => get_local_size, :threadidx => get_local_id)
    for (i, sym) in enumerate((:x, :y, :z))
        fname = Symbol(string(f, '_', sym))
        @eval $fname(A)::Cuint = $fcl($(i-1)) + Cuint(1)
    end
end

synchronize_threads(A::cli.CLArray) = cli.barrier(CLK_LOCAL_MEM_FENCE)

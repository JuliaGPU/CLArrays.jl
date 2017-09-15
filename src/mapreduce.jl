import GPUArrays: acc_mapreduce
using Transpiler.cli: get_local_id, get_global_id, barrier, CLK_LOCAL_MEM_FENCE
using Transpiler.cli: get_local_size, get_global_size, get_group_id
using GPUArrays: blockdim_x, blockidx_x, threadidx_x, synchronize, synchronize_threads, device

for i = 0:10
    args = ntuple(x-> Symbol("arg_", x), i)
    fargs = ntuple(x-> :(broadcast_index($(args[x]), length, global_index)), i)
    @eval begin
        function reduce_kernel(state, f, op, v0, A, tmp_local, length, result, $(args...))
            ui1 = Cuint(1)
            global_index = get_global_id(0) + ui1
            local_v0 = v0
            # Loop sequentially over chunks of input vector
            while (global_index <= length)
                element = f(A[global_index], $(fargs...))
                local_v0 = op(local_v0, element)
                global_index += get_global_size(0)
            end

            # Perform parallel reduction
            local_index = threadidx_x(state)
            tmp_local[local_index + ui1] = local_v0
            barrier(CLK_LOCAL_MEM_FENCE)
            offset = blockdim_x(state) ÷ ui1
            while offset > 0
                if (local_index < offset)
                    other = tmp_local[local_index + offset + ui1]
                    mine = tmp_local[local_index + ui1]
                    tmp_local[local_index + ui1] = op(mine, other)
                end
                barrier(CLK_LOCAL_MEM_FENCE)
                offset = offset ÷ Cuint(2)
            end
            if local_index == Cuint(0)
                result[blockidx_x(state) + ui1] = tmp_local[1]
            end
            return
        end
    end
end

function acc_mapreduce{T, OT, N}(
        f, op, v0::OT, A::CLArray{T, N}, rest::Tuple
    )
    dev = device(A)
    block_size = 16
    group_size = ceil(Int, length(A) / block_size)
    out = similar(A, OT, (group_size,))
    fill!(out, v0)
    lmem = LocalMemory{OT}(block_size)
    args = (f, op, v0, A, lmem, Cuint(length(A)), out, rest...)
    gpu_call(reduce_kernel, A, args, (group_size * block_size,), (block_size,))
    println(Array(out))
    reduce(op, Array(out))
end

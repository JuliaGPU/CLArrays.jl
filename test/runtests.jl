using CLArrays
using Base.Test
import GPUArrays: gpu_call, linear_index

Ac = rand(Float32, 10, 10)
A = CLArray(Ac);
B = CLArray(Ac);

Array(A .+ B) ≈ (Ac .+ Ac)

function kernel(state, A)
    idx = @(state, A)
    A[idx] = 22f0
    return
end



@testset "Custom kernel from string function" begin
    copy_source = """
    __kernel void copy(
            __global float *dest,
            __global float *source
        ){
        int gid = get_global_id(0);
        dest[gid] = source[gid];
    }
    """
    source = GPUArray(rand(Float32, 1023, 11))
    dest = GPUArray(zeros(Float32, size(source)))
    f = (copy_source, :copy)
    gpu_call(f, dest, (dest, source))
    @test Array(dest) == Array(source)
end

using OpenCL
device, ctx, queue = cl.create_compute_context()

src = """
struct test{
    __global float * ptr;
};
typedef struct test Test;
__kernel void copy(__global float *dest){
    Test lol = {dest};
}
"""

p = cl.Program(ctx, source=src) |> cl.build!

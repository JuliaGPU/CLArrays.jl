using CLArrays
using Base.Test

# write your own tests here
@test 1 == 2



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

using CLArrays, CLArrays.Shorthands
using GPUArrays.TestSuite, Base.Test

for dev in CLArrays.devices()
    # we only test gpu devices for now - cpu drivers use different
    # inconsistent struct alignment, which has low priority to be fixed right now
    is_gpu(dev) || continue
    CLArrays.init(dev)
    @testset "Device: $dev" begin

        TestSuite.run_tests(CLArray)

        @testset "muladd & abs" begin
            a = rand(Float32, 32) - 0.5f0
            A = CLArray(a)
            x = abs.(A)
            @test Array(x) == abs.(a)
            y = muladd.(A, 2f0, x)
            @test Array(y) == muladd(a, 2f0, abs.(a))
            u0 = CLArray(ones(Float32, 32, 32)); u1 = zeros(u0);
            @test muladd.(2, Array(u0), Array(u1)) == Array(muladd.(2, u0, u1))
            ###########
            # issue #20
            against_base(a-> abs.(a), CLArray{Float32}, (10, 10))

            #### bools in kernel:
        end
        @testset "bools" begin
            A, B = rand(Bool, 10), rand(Bool, 10)
            Ag, Bg = CLArray(A), CLArray(B)
            res = A .& B
            resg = Ag .& Bg
            @test res == Array(resg)
            # this version needs to have a fix in GPUArrays, since it uses T.(array)
            # in copy to convert to array type, but that actually convert Array{Bool} to BitArray
            # against_base((a, b)-> a .& b, CLArray{Bool}, (10,), (10,))
        end

        @testset "Shorthand Test" begin
            GPUArrays.allowslow(true)
            @test collect(cl([1,2])) == [1,2]
            @test collect(cl([1 2;3 4])) == [1 2;3 4]
            @test cl([1,2,3]) == CLArray([1,2,3])
        end

        @testset "Compilation from String" begin
            copy_source = """
                __kernel void copy(
                    __global float *dest,
                    __global float *source
                ){
                int gid = get_global_id(0);
                dest[gid] = source[gid];
                }
            """
            source = rand(CLArray{Float32}, 1023, 11)
            dest = zeros(CLArray{Float32}, size(source))
            gpu_call(:copy => copy_source, dest, (dest, source))
            Array(dest) == Array(source)
        end
    end
end

# Indexing with
# Issue
#   ([CartesianIndex(2,2), CartesianIndex(2,1)],) # Array{CartesianIndex} # FAIL

# #The above is equal to:
# Typ = CLArray
# GPUArrays.allowslow(false)
# TestSuite.run_gpuinterface(Typ)
# TestSuite.run_base(Typ)
# TestSuite.run_blas(Typ)
# TestSuite.run_broadcasting(CLArray)
# TestSuite.run_construction(Typ)
# TestSuite.run_fft(Typ)
# TestSuite.run_linalg(Typ)
# TestSuite.run_mapreduce(Typ)
# TestSuite.run_indexing(Typ)
#
# function test_sizes6(state, out)
#     x1 = (1, 2, 3)
#     out[1] = sizeof(x1)
#
#     x2 = (1f0, 2f0, 3f0)
#     out[2] = sizeof(x2)
#
#     x3 = ((1f0, 2f0), 1.0)
#     out[3] = sizeof(x3)
#
#     x4 = ((1f0, 2f0, 3f0), 1.0)
#     out[4] = sizeof(x4)
#
#     x5 = ((1f0, 2f0, 3f0, 4f0), 1.0)
#     out[5] = sizeof(x5)
#
#     x6 = (1.0, (1f0, 2f0))
#     out[6] = sizeof(x6)
#
#     x7 = (1.0, (1f0, 2f0, 3f0))
#     out[7] = sizeof(x7)
#
#     x8 = (1.0, (1f0, 2f0, 3f0, 4f0))
#     out[8] = sizeof(x8)
#
#     x9 = ((1f0, 2f0), UInt32(1), 1.0)
#     out[9] = sizeof(x9)
#
#     return
# end
#
# function test_sizes6(state, out)
#     x1 = 1
#     out[1] = sizeof(x1)
#
#     x2 = (1, 2)
#     out[2] = sizeof(x2)
#
#     x3 = (1, 2, 3)
#     out[3] = sizeof(x3)
#
#     x4 = 1f0
#     out[4] = sizeof(x4)
#
#     x5 = (1f0, 2f0)
#     out[5] = sizeof(x5)
#
#     x6 = (1f0, 2f0, 3f0)
#     out[6] = sizeof(x6)
#
#     x7 = (1f0, 2f0, 3f0, 4f0, 5f0)
#     out[7] = sizeof(x7)
#
#     x8 =  (1f0, 2f0, 3f0, 4f0, 5f0, 6f0)
#     out[8] = sizeof(x8)
#
#     x9 = (1, 2, 3, 4, 5, 6)
#     out[9] = sizeof(x9)
#
#     x10 = UInt8(1)
#     out[10] = sizeof(x10)
#
#     x11 = (x10, x10)
#     out[11] = sizeof(x11)
#
#     x12 = (x10, x10, x10)
#     out[12] = sizeof(x12)
#
#
#     x13 = CLArrays.DeviceArray{Complex{Float64},3,CLArrays.HostPtr{Complex{Float64}}}(
#         CLArrays.HostPtr{Complex{Float64}}(),
#         (Cuint(2), Cuint(2), Cuint(2))
#     )
#     out[13] = sizeof(x13)
#     #
#     # x14 = (x13, x13)
#     # out[14] = sizeof(x14)
#     #
#     # x15 = (x13, x13, x13)
#     # out[15] = sizeof(x15)
#     return
# end

using CLArrays
using GPUArrays.TestSuite, Base.Test
for dev in CLArrays.devices()
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
    end
end

# The above is equal to:
# Typ = CuArray
# GPUArrays.allowslow(false)
# TestSuite.run_gpuinterface(Typ)
# TestSuite.run_base(Typ)
# TestSuite.run_blas(Typ)
# TestSuite.run_broadcasting(Typ)
# TestSuite.run_construction(Typ)
# TestSuite.run_fft(Typ)
# TestSuite.run_linalg(Typ)
# TestSuite.run_mapreduce(Typ)
# TestSuite.run_indexing(Typ)

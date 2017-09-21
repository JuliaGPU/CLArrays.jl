if (
        get(ENV, "TRAVIS", "") == "true" ||
        get(ENV, "APPVEYOR", "") == "true" ||
        get(ENV, "CI", "") == "true"
    )
    Pkg.clone("Sugar")
    Pkg.clone("Transpiler")
    Pkg.clone("GPUArrays")

    Pkg.checkout("OpenCL", "sd/pointerfree")
    Pkg.checkout("Sugar", "sd/struct_buffer")
    Pkg.checkout("Transpiler", "sd/struct_buffer")
    Pkg.checkout("GPUArrays", "sd/abstractgpu")

end

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

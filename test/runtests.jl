if (
        get(ENV, "TRAVIS", "") == "true" ||
        get(ENV, "APPVEYOR", "") == "true" ||
        get(ENV, "CI", "") == "true"
    )
    Pkg.clone("Transpiler")
    Pkg.clone("Sugar")
    Pkg.clone("GPUArrays")
    Pkg.clone("OpenCL")

    Pkg.checkout("Transpiler", "sd/struct_buffer")
    Pkg.checkout("Sugar", "sd/struct_buffer")
    Pkg.checkout("GPUArrays", "sd/abstractgpu")
    Pkg.checkout("OpenCL", "sd/pointerfree")
end

using CLArrays
using GPUArrays.TestSuite

TestSuite.run_tests(CLArray)

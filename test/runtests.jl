if (
        get(ENV, "TRAVIS", "") == "true" ||
        get(ENV, "APPVEYOR", "") == "true" ||
        get(ENV, "CI", "") == "true"
    )

    Pkg.checkout("Transpiler", "sd/struct_buffer")
    Pkg.checkout("Sugar", "sd/struct_buffer")
    Pkg.checkout("GPUArrays", "sd/abstractgpu")
end

using CLArrays
using GPUArrays.TestSuite, Base.Test

TestSuite.run_tests(CLArray)

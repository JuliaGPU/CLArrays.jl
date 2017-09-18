using CLArrays
using GPUArrays.TestSuite, Base.Test
using GPUArrays: global_size
using CUDAnative, CUDAdrv
TestSuite.run_tests(CLArray)

using CLArrays

x = CLArray(rand(Float32, 10))

GPUArrays.gpu_call(x, (x,)) do state, l
    l[1] = 1f0 ^ 1.0
    return
end

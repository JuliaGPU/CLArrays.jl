# CLArrays

Implementation of the [abstract GPU Array Interface](https://github.com/JuliaGPU/GPUArrays.jl)

CLArray uses [Transpiler.jl](https://github.com/SimonDanisch/Transpiler.jl) to compile Julia functions for the GPU using OpenCL.

It implements the full abstract gpu interface in GPUArrays.
To learn how to use it, please refer to the GPUArray documentations:

[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://JuliaGPU.github.io/GPUArrays.jl/latest)

The only noteworthy interface that isn't mentioned in GPUArrays is how to select and initialize devices:

```Julia
using CLArrays

for dev in CLArrays.devices()
    CLArrays.init(dev)
    x = zeros(CLArray{Float32}, 5, 5) # create a CLArray on device `dev`
end
# you can also filter with is_gpu, is_cpu
gpu_devices = CLArrays.devices(is_gpu)
```

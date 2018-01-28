# CLArrays


[![](http://ci.maleadt.net/shields/build.php?builder=CLArrays-julia06-x86-64bit&name=julia%200.6)](http://ci.maleadt.net/shields/url.php?builder=CLArrays-julia06-x86-64bit)

Implementation of the [abstract GPU Array Interface](https://github.com/JuliaGPU/GPUArrays.jl)

CLArray uses [Transpiler.jl](https://github.com/SimonDanisch/Transpiler.jl) to compile Julia functions for the GPU using OpenCL.

It implements the full abstract gpu interface from GPUArrays.
To learn how to use it, please refer to the GPUArray documentations:

[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://JuliaGPU.github.io/GPUArrays.jl/latest)

The only noteworthy functionality that isn't part of the GPUArray interface is how to select and initialize devices:

```Julia
using CLArrays

for dev in CLArrays.devices()
    CLArrays.init(dev)
    x = zeros(CLArray{Float32}, 5, 5) # create a CLArray on device `dev`
end
# you can also filter with is_gpu, is_cpu
gpu_devices = CLArrays.devices(is_gpu)
```

Note that CLArrays.jl does not handle installing OpenCL drivers for your machine. You will need to make sure you have the appropriate drivers installed for your hardware.

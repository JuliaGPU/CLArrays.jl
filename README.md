# CLArrays



**Build status**: [![][gitlab-img]][gitlab-url]

[gitlab-img]: https://gitlab.com/JuliaGPU/CLArrays.jl/badges/master/pipeline.svg
[gitlab-url]: https://gitlab.com/JuliaGPU/CLArrays.jl/pipelines


Implementation of the [abstract GPU Array Interface](https://github.com/JuliaGPU/GPUArrays.jl)

CLArray uses [Transpiler.jl](https://github.com/SimonDanisch/Transpiler.jl) to compile Julia functions for the GPU using OpenCL.

It implements the full abstract gpu interface from GPUArrays, and most interactions will be through those functions.
To learn how to use it, please refer to the GPUArray documentation:

[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://JuliaGPU.github.io/GPUArrays.jl/latest)

CLArrays includes several other OpenCL-specific functions:

* `CLArrays.devices()` returns a list of the OpenCL compute devices (CPU and GPU) available on the system.
* `CLArrays.init(dev::OpenCL.cl.Device)` will set the given device to be the active device. If you do not initialize a device explicitly, a default device will be chosen automatically, prioritizing GPU devices over CPU devices.
* `is_gpu(dev::OpenCL.cl.Device)` returns `true` if the given device is a GPU.
* `is_cpu(dev::OpenCL.cl.Device)` returns `true` if the given device is a CPU.
* `gpu_call(kernel::Function, A::GPUArray, args::Tuple, configuration = length(A))` calls the given function on the GPU. See the function documentation for more details.

### Example

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

#### Install OpenCL drivers for intel graphics in Linux

```
cd $HOME
git clone https://github.com/intel/beignet
cd $HOME/beignet
sudo apt-get install cmake pkg-config python ocl-icd-dev libegl1-mesa-dev ocl-icd-opencl-dev libdrm-dev libxfixes-dev libxext-dev llvm-3.6-dev clang-3.6 libclang-3.6-dev libtinfo-dev libedit-dev zlib1g-devD
mkdir build; cd build; cmake ..
make
make utest; . utests/setenv.sh; utests/utest_run
sudo make install
```

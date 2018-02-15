__precompile__(true)
module CLArrays

using GPUArrays
using GPUArrays: LocalMemory
using OpenCL
using Transpiler
import Transpiler: cli

function context end

include("memory.jl")
include("array.jl")
include("ondevice.jl")
include("device.jl")
include("context.jl")
include("compilation.jl")
include("3rdparty.jl")

export CLArray, gpu_call, is_gpu, is_cpu

end # module

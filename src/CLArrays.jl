module CLArrays


using GPUArrays
using OpenCL
using OpenCL: cl
using Transpiler
import Transpiler: cli

include("memory.jl")
include("array.jl")
include("gpu_device.jl")
include("context.jl")
include("intrinsics.jl")
include("compilation.jl")
include("mapreduce.jl")
include("opencl.jl")
include("3rdparty.jl")

end # module

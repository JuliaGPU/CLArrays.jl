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
include("device.jl")
include("context.jl")
include("intrinsics.jl")
include("compilation.jl")
include("mapreduce.jl")
include("3rdparty.jl")

export CLArray

end # module

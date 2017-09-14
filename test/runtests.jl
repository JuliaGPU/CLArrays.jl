using CLArrays
using Base.Test
import GPUArrays: gpu_call, linear_index

Ac = rand(Float32, 10, 10)
A = CLArray(Ac);
B = CLArray(Ac);
Array(A .+ B) ≈ (Ac .+ Ac)


x = Base.typename(CLArrays.KernelState)
using OpenCL

cl.packed_convert(CLArrays.cl_convert(A));
CLArrays.cl_convert(Base.RefValue(A))
T = typeof(Base.RefValue(A))
ctr, hoists = CLArrays.reconstruct((), T, :x, (), CLArrays.HostPtr)
println(ctr)
hoists
isa(Array, Type)

using CLArrays

function test2(a, b)
    a[23] = b
    return
end
CLArrays.empty_compile_cache!()
x = CLArray(zeros(Float32, 32))
args = (x, 33f0);
f = CLArrays.CLFunction(test2, args)
f(args, (1,), (1,));
Array(x)[23]


immutable MyArray2 <: AbstractArray{Float32, 2}
    x::Float32
end
Base.copy!(x::MyArray2, y::MyArray2) = error("Why?!")
x = Base.RefValue{MyArray2}(MyArray2());



m = Transpiler.CLMethod((GPUArrays.apply_broadcast,
    Tuple{UInt32, CLArrays.KernelState, typeof(+),
    Tuple{UInt32,UInt32}, UInt32,
    Tuple{GPUArrays.BroadcastDescriptorN{Array,2}, GPUArrays.BroadcastDescriptorN{Array,2}},
    CLArrays.CLArray{Float32,2,Transpiler.CLIntrinsics.GlobalPointer{Float32}},
    CLArrays.CLArray{Float32,2,Transpiler.CLIntrinsics.GlobalPointer{Float32}}
}))

ast = Sugar.sugared(m.signature..., code_typed)
Sugar.expr_type.(m, ast.args[5].args[2].args[2:end])

Core.Inference.return_type(m.signature...)
Sugar.expr_type.(m, ast.args[5].args[2].args[2:end])

Sugar.returntype(m)
f = eval(ast.args[5].args[2].args[1])
m2 = Transpiler.CLMethod((f, Tuple{Sugar.expr_type.(m, ast.args[5].args[2].args[2:end])...}))
getast!(m2)
ast = Sugar.sugared(m2.signature..., code_typed)
CLArrays.empty_compile_cache!()

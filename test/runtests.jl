using CLArrays
using Base.Test
import GPUArrays: gpu_call, linear_index

Ac = rand(Float32, 10, 10)
A = CLArray(Ac);
B = CLArray(Ac);

Array(A .+ B) ≈ (Ac .+ Ac)

using CLArrays

function test2(a, b)
    a[Cuint(1), Cuint(1)] = b
    return
end

x = CLArray(rand(Float32, 32, 32))
f = CLArrays.CLFunction(test2, (CLArray{Float32, 2}, Float32))


contains, field_list = CLArrays.contains_tracked_type(CLArray{Float32, 2})

typeof(x)

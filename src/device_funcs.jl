function Base.sum(x::CLArrays.DeviceArray{T}) where T
    acc = zero(T)
    for i = Cuint(1):length(x)
        acc += x[i]
    end
    acc
end

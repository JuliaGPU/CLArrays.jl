const lib = joinpath(@__DIR__, "2016-03-28", "GPUVerify.py")
ENV["PATH"] = ENV["PATH"] * ":$lib"

function verify_kernel(source, lsize_gsize)
    lsize, gsize = lsize_gsize
    mktempdir() do f
        path = joinpath(f, "kernel.cl")
        io = open(path, "w")
        print(io, source)
        close(io)
        run(`$lib --timeout=0 --local_size=$(join(lsize, ",")) --num_groups=$(join(gsize, ",")) $(path)`)
    end
end

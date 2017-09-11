type CLContext <: Context
    device::cl.Device
    context::cl.Context
    queue::cl.CmdQueue
    function CLContext(device::cl.Device)
        ctx = cl.Context(device)
        queue = cl.CmdQueue(ctx)
        new(device, ctx, queue)
    end
end

is_opencl(ctx::CLContext) = true

function Base.show(io::IO, ctx::CLContext)
    println(io, "OpenCL context with:")
    println(io, "CL version: ", cl.info(ctx.device, :version))
    device_summary(io, ctx.device)
end


global all_contexts, current_context, current_device
let contexts = Dict{cl.Device, CLContext}(), active_device = cl.Device[]
    all_contexts() = values(contexts)
    function current_device()
        if isempty(active_device)
            push!(active_device, CUDAnative.default_device[])
        end
        active_device[]
    end
    function current_context()
        dev = current_device()
        get!(contexts, dev) do
            new_context(dev)
        end
    end
    function GPUArrays.init(dev::cl.Device)
        GPUArrays.setbackend!(CLBackend)
        if isempty(active_device)
            push!(active_device, dev)
        else
            active_device[] = dev
        end
        ctx = get!(()-> new_context(dev), contexts, dev)
        ctx
    end

    function GPUArrays.destroy!(context::CLContext)
        # don't destroy primary device context
        dev = context.device
        if haskey(contexts, dev) && contexts[dev] == context
            error("Trying to destroy primary device context which is prohibited. Please use reset!(context)")
        end
        finalize(context.ctx)
        return
    end
end

function reset!(context::CLContext)
    device = context.device
    finalize(context.context)
    context.context = cl.Context(device)
    context.queue = cl.CmdQueue(context.context)
    return
end

new_context(dev::cl.Device) = CLContext(dev)

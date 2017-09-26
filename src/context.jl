import Base: show
import GPUArrays: device


struct CLContext
    context::cl.Context
    device::cl.Device
    queue::cl.CmdQueue
end

function show(io::IO, ctx::CLContext)
    println(io, "OpenCL context with:")
    println(io, "CL version: ", cl.info(ctx.device, :version))
    GPUArrays.device_summary(io, ctx.device)
end

function CLContext(device::cl.Device)
    context = cl.Context(device)
    queue = cl.CmdQueue(context)
    CLContext(context, device, queue)
end

new_context(dev::cl.Device) = CLContext(dev)
function default_device()
    devs = sort(devices(), by = x-> !is_gpu(x))
    if isempty(devs)
        error("No OpenCL devices found")
    end
    first(devs)
end

global all_contexts, global_context, current_device, getcontext!

global_context(A::CLArray) = context(A)

global_queue(A::CLArray) = global_queue(context(A))

function global_queue(cl_ctx::cl.Context = global_context())
    getcontext!(cl_ctx).queue
end

function device(x::CLArray)
    getcontext!(context(x)).device
end
function device(x::cl.Context)
    getcontext!(x).device
end


global init
let contexts = Dict{cl.Device, CLContext}(), active_device = cl.Device[], clcontext2context = Dict{cl.Context, CLContext}()
    function getcontext!(ctx::cl.Context)
        if haskey(clcontext2context, ctx)
            return clcontext2context[ctx]
        else
            device = first(cl.devices(ctx))
            getcontext!(device)
        end
    end
    function getcontext!(device::cl.Device)
        get!(contexts, device) do
            ctx = new_context(device)
            clcontext2context[ctx.context] = ctx
            ctx
        end
    end
    all_contexts() = values(contexts)
    function current_device()
        device = default_device()
        if isempty(active_device)
            push!(active_device, device)
        end
        active_device[]
    end
    function global_context()
        device = current_device()
        getcontext!(device).context
    end
    function init(device::cl.Device)
        if isempty(active_device)
            push!(active_device, device)
        else
            active_device[] = device
        end
        ctx = get!(()-> new_context(device), contexts, device)
        ctx
    end

    function destroy!(context::CLContext)
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

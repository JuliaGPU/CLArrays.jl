using OpenCL: cl
using Transpiler: CLMethod, EmptyStruct
using Sugar
import GPUArrays: _gpu_call, linear_index
using Transpiler: CLMethod
using Sugar: method_nargs, getslots!, isintrinsic, getcodeinfo!, sugared
using Sugar: returntype, type_ast, rewrite_ast, newslot!, to_tuple
using Sugar: isfunction

using Base: tail

function _gpu_call(f, A::CLArray, args::Tuple, blocks_threads::Tuple{T, T}) where T <: NTuple{N, Integer} where N
    ctx = context(A)
    _args = (KernelState(), args...) # CLArrays "state"
    clfunc = CLFunction(f, _args, ctx)
    blocks, threads = blocks_threads
    global_size = blocks .* threads
    clfunc(_args, global_size, threads)
end


device_type{T}(::T) = T
device_type{T}(::Type{T}) = Type{T}
device_type{T}(x::LocalMemory{T}) = cli.LocalPointer{T}
device_type(arg::Mem.OwnedPtr{T}) where T = cli.GlobalPointer{T}

kernel_convert{T}(x::LocalMemory{T}) = cl.LocalMem(T, x.size)
kernel_convert{T}(x::Mem.OwnedPtr{T}) = x

string_kernel_convert(x::CLArray) = pointer(x)
string_kernel_convert(x) = kernel_convert(x)


function kernel_convert(x::T) where T
    contains, fields = contains_pointer(T)
    if contains
        error("Contains pointers. Please define the correct kernel_convert/reconstruct methods")
    else
        if (isbits(x) && sizeof(x) == 0 && nfields(x) == 0) || T <: Type
            return EmptyStruct()
        end
        isbits(T) || error("Only isbits types allowed for gpu kernel")
        return x
    end
end

function match_field(field_list, field)
    for elem in field_list
        if first(elem) == field
            return [Base.tail(elem)]
        end
    end
    ()
end

makeslot(m::Sugar.LazyMethod, T, name) = newslot!(m, T, name)
makeslot(m::Tuple{}, T, name) = name

function getfield_expr(m, T, name, fname)
    # julia doesn't like symbols in getfield in its ast for rount trip
    expr = Expr(:call, getfield, name, QuoteNode(fname))
    expr.typ = T
    expr
end
function getfield_expr(m::LazyMethod, T, name, fname)
    # when we have a lazy method, we're generating code for the GPU
    expr = Expr(:call, getfield, name, fname)
    expr.typ = T
    expr
end


isa_pointer_type(T) = isa(T, Type) && (T <: OwnedPtr || T <: HostPtr || T <: cli.GlobalPointer)


function contains_pointer(
        ::Type{T},
        parent_field = (),
        pointer_fields = []
    ) where T
    @assert isleaftype(T) "Arguments must be fully concrete (leaftypes)"
    # pointers must be in parameters
    hasptr = false
    for fname in fieldnames(T)
        FT = fieldtype(T, fname)
        if isa_pointer_type(FT)
            push!(pointer_fields, (parent_field..., fname))
            hasptr = true
        else
            _hasptr, pointer_fields = contains_pointer(FT, (parent_field..., fname), pointer_fields)
            hasptr |= _hasptr
        end
    end
    hasptr, pointer_fields
end

get_fields_type(T, fields::Tuple{X}) where X = fieldtype(T, first(fields))
get_fields_type(T, fields::NTuple{N, Any}) where N = get_fields_type(fieldtype(T, first(fields)), Base.tail(fields))

get_fields(x, fields::NTuple{1}) = getfield(x, first(fields))
get_fields(x, fields::NTuple{N, Any}) where N = get_fields(getfield(x, first(fields)), Base.tail(fields))



function _getast(x::CLMethod)
    if isfunction(x)
        if isintrinsic(x)
            Expr(:block)
        else
            getcodeinfo!(x) # make sure codeinfo is present
            expr = sugared(x.signature..., code_typed)
            expr.typ = returntype(x)
            expr
        end
    else
        type_ast(x.signature)
    end
end


function funcheader_string(method, args)
    sprint() do io
        cio = Transpiler.CIO(io, method)
        Transpiler.show_returntype(cio, method)
        print(cio, ' ')
        Base.show_unquoted(cio, method)
        print(cio, '(')
        for (i, elem) in enumerate(args)
            if i != 1
                print(cio, ", ")
            end
            name, T = elem.args
            Transpiler.show_type(cio, T)
            print(cio, ' ')
            Transpiler.show_name(cio, name)
        end
        print(cio, ')')
    end
end

"""
Assembles a kernel considering that you can't pass structs containing pointers.
Will generate a kernel from function f:
```
function f(arg1, contains_ptr, arg3)
    ...
end
```
in the form of
```
// ## indicating gensymed/anonymous name
__kernel void f(arg1, ##contains_ptr, ##ptr, arg3){
    contains_ptr = reconstruct(##contains_ptr, ##ptr);
    ...
}
```
The function returns the body of the function string and a list of expressions
to set up the kernel call for clEnqueueNDRangeKernel.
"""
function assemble_kernel(m::CLMethod)
    kernel_args = []
    kernel_ptrs = []
    body = Expr(:block)
    nargs = method_nargs(m)
    # declare rest of slots
    for (i, (T, name)) in enumerate(getslots!(m)[nargs+1:end])
        slot = TypedSlot(i + nargs, T)
        push!(m.decls, slot)
        push!(m, T)
        tmp = :($name::$T)
        tmp.typ = T
        push!(body.args, tmp)
    end
    st = getslots!(m)[2:nargs] # don't include self
    arg_idx = 1
    ptr_extract = []
    for (i, (T, name)) in enumerate(st)
        argslot = TypedSlot(i + 1, T)
        contains, field_list = contains_pointer(T)
        if contains # type contains pointer to global memory
            # replace variable in arguments to free up slot
            fake_arg = gensym(name)
            T̂ = predevice_type(T) # get the pointer free equivalent type
            fake_arg_slot = newslot!(m, T̂, fake_arg)
            push!(m, T̂) # include in dependencies
            push!(kernel_args, :($fake_arg_slot::$T̂)) # put pointer free variant in kernel arguments
            push!(m.decls, fake_arg_slot)
            # we extract all pointers, pass them seperately and then assemble them in the function body
            ptr_fields = []
            ptr_slots = []
            for (ptridx, fields) in enumerate(field_list)
                ptr_arg = gensym(string("ptr_", ptridx, "_", name, arg_idx))
                PT = get_fields_type(T, fields)
                PT = cli.GlobalPointer{eltype(PT)}
                ptr_slot = newslot!(m, PT, ptr_arg)
                push!(ptr_slots, ptr_slot)
                push!(kernel_ptrs, :($ptr_slot::$PT)) # add pointer to kernel arguments
                push!(ptr_fields, (fields..., ptr_slot)) # add name of ptr to fields
                push!(ptr_extract, (i, fields...))
            end
            constr = Expr(:call, reconstruct, fake_arg_slot, ptr_slots...)
            constr.typ = T
            push!(body.args, :($argslot = $constr))
        else
            push!(m.decls, argslot)
            push!(kernel_args, :($argslot::$T))
        end
    end

    append!(kernel_args, kernel_ptrs)
    real_body = _getast(m)
    body.typ = real_body.typ # use real type
    append!(body.args, real_body.args)
    body = rewrite_ast(m, body)
    io = Transpiler.CLIO(IOBuffer(), m)
    println(io, "// Inbuilds")
    println(io, "typedef char JLBool;")
    println(io, "// dependencies")
    visited = Set()
    for dep in Sugar.dependencies!(m)
        Transpiler.print_dependencies(io, dep, visited) # this is recursive but would print method itself
    end
    println(io, "// ########################")
    println(io, "// Main inner function")
    println(io, "// ", m.signature)
    print(io, "__kernel ") # mark as kernel function
    print(io, funcheader_string(m, kernel_args))
    Base.show_unquoted(io, body, 0, 0)
    fname = string(Sugar.functionname(io, m))
    String(take!(io.io)), fname, ptr_extract
end
Transpiler.functionname(x, ::typeof(getfield)) = :getfield


const compiled_functions = Dict()

function empty_compile_cache!()
    empty!(compiled_functions)
    return
end

immutable CLFunction{F, Args, Ptrs}
    kernel::cl.Kernel
end

function CLFunction(f::F, args::T, ctx = global_context()) where {T, F}
    device = getcontext!(ctx).device
    if !supports_double(device) && any(x-> isa(x, Float64), args)
        error("Float64 is not supported by your device: $device. Make sure to convert all types for the GPU to Float32")
    end
    version = cl_version(device)
    cltypes = device_type.(args)
    get!(compiled_functions, (ctx.id, f, cltypes)) do # TODO make this faster
        method = CLMethod((f, cltypes))
        source, fname, ptr_extract = assemble_kernel(method)
        options = "-cl-denorms-are-zero -cl-mad-enable -cl-unsafe-math-optimizations"
        if version > v"1.2"
            options *= " -cl-std=CL1.2"
        end
        p = try
            cl.build!(
                cl.Program(ctx, source = source),
                options = options
            )
        catch e
            println(source)
            rethrow(e)
        end
        kernel = cl.Kernel(p, fname)
        CLFunction{F, T, Tuple{ptr_extract...}}(kernel)
    end
end

"""
Compiles a kernel from f = :kernel_name => source, with arguments args.
"""
function CLFunction(f::Pair{Symbol, String}, args::T, ctx = global_context(), options = "-cl-denorms-are-zero -cl-mad-enable -cl-unsafe-math-optimizations") where T
    device = getcontext!(ctx).device
    if !supports_double(device) && any(x-> isa(x, Float64), args)
        error("Float64 is not supported by your device: $device. Make sure to convert all types for the GPU to Float32")
    end
    version = cl_version(device)
    fname, source = f
    get!(compiled_functions, (ctx.id, fname, T)) do # TODO make this faster
        if version > v"1.2"
            options *= " -cl-std=CL1.2"
        end
        p = try
            cl.build!(
                cl.Program(ctx, source = source),
                options = options
            )
        catch e
            println(source)
            rethrow(e)
        end
        kernel = cl.Kernel(p, string(fname))
        # drop kernelstate from args
        CLFunction{String, T, Tuple{}}(kernel)
    end
end

function (clf::CLFunction{F, Args, Ptrs})(
        args::Args, blocks::NTuple{N, Integer}, threads, queue = global_queue()
    ) where {F, Args, Ptrs, N}
    @assert N in (1, 2, 3) "invalid block size"
    gsize = Csize_t[b for b in blocks]
    lsize = isa(threads, Tuple) ? Csize_t[threads[i] for i=1:length(blocks)] : threads
    kernel = clf.kernel
    idx = 1
    for elem in args
        tmp = kernel_convert(elem)
        cl.set_arg!(kernel, idx, tmp)
        idx += 1
    end
    for fields in to_tuple(Ptrs)
        arg_idx, fields = first(fields), tail(fields)
        cl.set_arg!(kernel, idx, kernel_convert(get_fields(args[arg_idx], fields)))
        idx += 1
    end
    ret_event = Ref{cl.CL_event}()
    cl.@check cl.api.clEnqueueNDRangeKernel(
        queue.id, kernel.id,
        length(gsize), C_NULL, gsize, lsize,
        0, C_NULL, ret_event
    )
    return cl.Event(ret_event[], retain = false)
end

# Kernels compiled from a string have a different conversion semantic.
function (clf::CLFunction{String, Args, Tuple{}})(
        args::Args, blocks::NTuple{N, Integer}, threads, queue = global_queue()
    ) where {Args, N}
    @assert N in (1, 2, 3) "invalid block size"
    gsize = Csize_t[b for b in blocks]
    lsize = isa(threads, Tuple) ? Csize_t[threads[i] for i=1:length(blocks)] : threads
    kernel = clf.kernel
    idx = 1
    for elem in Iterators.drop(args, 1) # drop kernel state
        tmp = string_kernel_convert(elem)
        cl.set_arg!(kernel, idx, tmp)
        idx += 1
    end
    ret_event = Ref{cl.CL_event}()
    cl.@check cl.api.clEnqueueNDRangeKernel(
        queue.id, kernel.id,
        length(gsize), C_NULL, gsize, lsize,
        0, C_NULL, ret_event
    )
    return cl.Event(ret_event[], retain = false)
end

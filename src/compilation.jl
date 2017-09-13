using OpenCL: cl
using Transpiler: CLMethod, EmptyStruct
using Sugar
import GPUArrays: gpu_call, linear_index
using Transpiler: CLMethod
using Sugar: method_nargs, getslots!, isintrinsic, getcodeinfo!, sugared, returntype, type_ast, rewrite_ast

function CLFunction{T, N}(A::CLArray{T, N}, f, args...)
    CLFunction(f, args, context(A))
end
function (clfunc::CLFunction{T}){T, T2, N}(A::CLArray{T2, N}, args...)
    # TODO use better heuristic
    clfunc(args, length(A))
end

function gpu_call(f, A::CLArray, args::Tuple, blocks = nothing, thread = nothing)
    ctx = context(A)
    _args = if !isa(f, Tuple{String, Symbol})
        (KernelState(), args...) # CLArrays "state"
    else
        args
    end
    clfunc = CLFunction(f, _args, global_queue(ctx))
    if blocks == nothing
        blocks, thread = thread_blocks_heuristic(length(A))
    end
    clfunc(_args, blocks, thread)
end

#_to_cl_types{T}(::Type{T}) = T
_to_cl_types{T}(::T) = T
_to_cl_types{T}(::Type{T}) = Type{T}
_to_cl_types(::Int64) = Int32
_to_cl_types(::Float64) = Float32
_to_cl_types{T, N}(arg::CLArray{T, N}) = Transpiler.DeviceArray{T, N}
_to_cl_types{T}(x::LocalMemory{T}) = cli.LocalPointer{T}
_to_cl_types(x::Ref{<: CLArray}) = _to_cl_types(x[])
_to_cl_types(arg::Mem.OwnedPtr{T}) where T = cli.GlobalPointer{T}

function to_cl_types(args::Union{Vector, Tuple})
    map(_to_cl_types, args)
end

cl_convert(x::Ref{<: CLArray}) = cl_convert(x[])
cl_convert(x::CLArray{T, N}) where {T, N} = Transpiler.DeviceArray{T, N}(pointer(x), Cuint.(size(x)))
cl_convert{T}(x::LocalMemory{T}) = cl.LocalMem(T, x.size)
cl_convert{T}(x::Mem.OwnedPtr{T}) = x

function cl_convert{T}(x::T)
    # empty objects are empty and are only usable for dispatch
    isbits(x) && sizeof(x) == 0 && nfields(x) == 0 && return EmptyStruct()
    # same applies for types
    isa(x, Type) && return EmptyStruct()
    convert(_to_cl_types(x), x)
end



function match_field(field_list, field)
    for elem in field_list
        if first(elem) == field
            return Base.tail(elem)
        end
    end
    ()
end

function reconstruct(T, name, ptr_list, hoisted = [])
    if T <: OwnedPtr#GlobalPointer
        @assert length(ptr_list) == 1
        return first(ptr_list), hoisted
    elseif nfields(T) == 0
        return name, hoisted
    else
        constr = Expr(:new, T)
        for fname in fieldnames(T)
            hoistname = gensym(string(fname))
            gfield = Expr(:call, getfield, name, QuoteNode(fname))

            current_ptr_list = match_field(ptr_list, fname)
            if isempty(current_ptr_list)
                push!(hoisted, :($hoistname = $gfield))
                push!(constr.args, hoistname)
            else
                FT = fieldtype(T, fname)
                _constr, list = reconstruct(FT, hoistname, current_ptr_list, hoisted)
                push!(constr.args, _constr)
            end
        end
        return constr, hoisted
    end
end

function contains_tracked_type(
        ::Type{T},
        parent_field = (),
        trackedfields = []
    ) where T
    @assert isleaftype(T) "Arguments must be fully concrete (leaftypes)"
    any(x-> x <: OwnedPtr, Sugar.to_tuple(T.parameters))
    for fname in fieldnames(T)
        FT = fieldtype(T, fname)
        if FT <: OwnedPtr
            has_tracked_field = true
            push!(trackedfields, (parent_field..., fname))
        else
            _has_tracked_field, trackedfields = contains_tracked_type(FT, (fname,), trackedfields)
            has_tracked_field |= _has_tracked_field
        end
    end
    has_tracked_field, trackedfields
end

function pointerfree(::Type{T}, dependencies = [], curr_fields = (:x,)) where T
    fields = Expr(:block)
    newname = gensym(Base.typename(T).name)
    type_expr = Expr(:type, false, newname, fields)
    constructor = Expr(:new, newname)
    has_pointer = false
    for fname in fieldnames(T)
        FT = fieldtype(T, fname)
        if FT <: OwnedPtr
            has_pointer = true
            push!(fields.args, :($fname::UInt32))
            push!(constructor.args, UInt32(0))
        else
            newfields = (curr_fields..., fname)
            tname, hasptr, deps = pointerfree(FT, dependencies, newfields)
            has_pointer |= hasptr
            getf_expr = foldl((l, r)-> :(getfield($l, $(QuoteNode(r)))), newfields)
            if hasptr
                push!(constructor.args, :(cl_convert($getf_expr)))
            else
                push!(constructor.args, getf_expr)
            end
            push!(fields.args, :($fname::$tname))
        end
    end
    T̂ = if has_pointer
        Mod = CLArrays
        eval(Mod, type_expr)
        argname = first(curr_fields)
        eval(Mod, :(
            $Mod.cl_convert($argname::$T) = $constructor
        ))
        getfield(Mod, newname)
    else
        T
    end
    T̂, has_pointer, dependencies
end

get_fields_type(T, fields::NTuple{1}) = fieldtype(T, first(fields))
get_fields_type(T, fields::NTuple{N}) where N = get_fields_type(fieldtype(T, first(fields)), Base.tail(fields))

get_fields(x, fields::NTuple{1}) = getfield(x, first(fields))
get_fields(x, fields::NTuple{N}) where N = get_fields(getfield(x, first(fields)), Base.tail(fields))


using Sugar: method_nargs, getslots!, isintrinsic, getcodeinfo!, sugared, returntype, type_ast, rewrite_ast, isfunction

function _getast(x::CLMethod)
    ast = if isfunction(x)
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
    rewrite_ast(x, ast)
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
    kernel_setup = []
    body = Expr(:block)
    nargs = method_nargs(m)
    st = getslots!(m)[2:end] # don't include self
    arg_idx = 1
    for (i, (T, name)) in enumerate(st)
        argslot = TypedSlot(i + 1, T)
        push!(kernel_setup, :(cl.set_arg!(kernel, $arg_idx, cl_convert(args[$i]))))
        arg_idx += 1
        contains, field_list = contains_tracked_type(T)
        if contains # type contains pointer to global memory
            # replace variable in arguments to free up slot
            fake_arg = gensym(name)
            T̂, has, deps = pointerfree(T) # get the pointer free equivalent type
            for elem in (T̂, deps...)
                push!(m, elem) # include in dependencies
            end
            push!(kernel_args, :($fake_arg::$T̂)) # put pointer free variant in kernel arguments
            push!(body.args, :($argslot::$T)) # define slot that was the argument in body
            # we extract all pointers, pass them seperately and then assemble them in the function body
            ptr_fields = []
            for (ptridx, fields) in enumerate(field_list)
                ptr_arg = gensym(string("ptr_", ptridx, "_", name, arg_idx))
                PT = get_fields_type(T, fields)
                push!(kernel_args, :($ptr_arg::$PT)) # add pointer to kernel arguments
                push!(ptr_fields, (fields..., ptr_arg)) # add name of ptr to fields
                push!(kernel_setup, :(
                    cl.set_arg!(kernel, $arg_idx, cl_convert(get_fields(args[$i], $(fields))))
                ))
                arg_idx += 1
            end
            # get reconstruction constructor
            constr, hoisted_fields = reconstruct(T, fake_arg, ptr_fields)
            append!(body.args, hoisted_fields)
            push!(body.args, :($argslot = $constr))
        else
            push!(kernel_args, :($argslot::$T))
        end
    end
    real_body = _getast(m)
    body.typ = real_body.typ # use real type
    append!(body.args, real_body.args)

    io = Transpiler.CLIO(IOBuffer(), m)
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
    String(take!(io.io)), fname, kernel_setup
end
Transpiler.functionname(x, ::typeof(getfield)) = :getfield


const compiled_functions = Dict()

function empty_compile_cache!()
    empty!(compiled_functions)
    return
end

function CLFunction(f::Function, args::T, ctx = global_context()) where T
    device = getcontext!(ctx).device
    version = cl_version(device)
    cltypes = to_cl_types(args)
    get!(compiled_functions, (ctx.id, f, cltypes)) do # TODO make this faster
        method = CLMethod((f, args))
        source, fname, kernel_setup = assemble_kernel(method)
        options = "-cl-denorms-are-zero -cl-mad-enable -cl-unsafe-math-optimizations"
        if version > v"1.2"
            options *= " -cl-std=CL1.2"
        end
        p = cl.build!(
            cl.Program(ctx, source = source),
            options = options
        )
        kernel = cl.Kernel(p, fname)
        kernel_call_function(kernel, kernel_setup)
    end
end


function kernel_call_function(kernel, kernel_setup)
    @eval begin
        function (args, blocks::NTuple{N, Integer}, threads::NTuple{N, Integer}, queue = global_queue()) where N
            @assert N in (1, 2, 3) "invalid block size"
            lsize = Csize_t[b for b in blocks]
            gsize = Csize_t[threads[i] for i=1:length(blocks)]
            kernel = $kernel
            $(kernel_setup...)
            ret_event = Ref{cl.CL_event}()
            cl.@check cl.api.clEnqueueNDRangeKernel(
                queue.id, kernel.id,
                length(gsize), C_NULL, gsize, lsize,
                0, C_NULL, ret_event
            )
            return cl.Event(ret_event[], retain = false)
        end
    end
end

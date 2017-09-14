using OpenCL: cl
using Transpiler: CLMethod, EmptyStruct
using Sugar
import GPUArrays: gpu_call, linear_index
using Transpiler: CLMethod
using Sugar: method_nargs, getslots!, isintrinsic, getcodeinfo!, sugared
using Sugar: returntype, type_ast, rewrite_ast, newslot!, to_tuple
using Base: tail

struct HostPtr{T}
    ptr::Int32
    (::Type{HostPtr{T}})() where T = new{T}(Int32(0))
end

Base.eltype(::Type{HostPtr{T}}) where T = T

const UploadArray{T, N} = CLArray{T, N, HostPtr{T}}

Base.copy!{T}(dest::CLArray{T}, src::UploadArray{T}) = src
Base.showarray(io::IO, ::UploadArray{T, N}, repr::Bool) where {T, N} = print(io, "DeviceArray{$T, $N}(...)")

function gpu_call(f, A::CLArray, args::Tuple, blocks = nothing, thread = C_NULL)
    ctx = context(A)
    _args = if !isa(f, Tuple{String, Symbol})
        (KernelState(), args...) # CLArrays "state"
    else
        args
    end
    clfunc = CLFunction(f, _args, ctx)
    if blocks == nothing
        blocks, thread = thread_blocks_heuristic(length(A))
        blocks = (blocks,)
        thread = (thread,)
    end
    if isa(blocks, Integer)
        blocks = (blocks,)
    end
    clfunc(_args, blocks, thread)
end

#_to_cl_types{T}(::Type{T}) = T
_to_cl_types{T}(::T) = replace_ptr_parameter(T)
_to_cl_types{T}(::Type{T}) = Type{replace_ptr_parameter(T)}
_to_cl_types(::Int64) = Int32
_to_cl_types(::Float64) = Float32
_to_cl_types{T}(x::LocalMemory{T}) = cli.LocalPointer{T}
_to_cl_types(arg::Mem.OwnedPtr{T}) where T = cli.GlobalPointer{T}

to_cl_types(args::Union{Vector, Tuple}) = _to_cl_types.(args)

cl_convert(x::Int) = Int32(x)
cl_convert(x::Float64) = Float32(x)
cl_convert{T}(x::LocalMemory{T}) = cl.LocalMem(T, x.size)
cl_convert{T}(x::Mem.OwnedPtr{T}) = x


function cl_convert(x)
    contains, fields = contains_pointer(T)
    if contains
        error("Contains pointers. Please define the correct cl_convert/reconstruct methods")
    else
        if (isbits(x) && sizeof(x) == 0 && nfields(x) == 0) || x <: Type
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

function reconstruct(m, T, name, ptr_list, ptrtype = cli.GlobalPointer, hoisted = [])
    if T <: HostPtr || T <: cli.GlobalPointer # when on host for kernel
        length(ptr_list) == 1 && return (first(ptr_list[1]), hoisted)
        throw(AssertionError("internal reconstruct error, $ptr_list has wrong length"))
    elseif T <: OwnedPtr
        @assert isempty(ptr_list) # we pass an empty ptr_list if we convert for kernel
        return :(HostPtr{$(eltype(T))}()), hoisted
    elseif nfields(T) == 0
        return name, hoisted
    else
        constr = Expr(:new, replace_ptr_parameter(T, ptrtype))
        for fname in fieldnames(T)
            hoistname = gensym(string(fname))
            FT = fieldtype(T, fname)
            gfield = getfield_expr(m, FT, name, fname)
            hoistslot = makeslot(m, FT, hoistname)
            hasptr, list = contains_pointer(FT)
            if hasptr || isa_pointer_type(FT)
                current_ptr_list = match_field(ptr_list, fname)
                _constr, list = reconstruct(m, FT, hoistname, current_ptr_list, ptrtype, hoisted)
                push!(constr.args, _constr)
                if isa(_constr, Expr) && _constr.head == :new # we need the hoist when its a real constructor
                    idx_expr = getfield_expr(m, FT, name, fname)
                    push!(hoisted, :($hoistslot = $idx_expr))
                end
            else
                push!(constr.args, gfield)
            end
        end
        return constr, hoisted
    end
end

isa_pointer_type(T) = isa(T, Type) && (T <: OwnedPtr || T <: HostPtr || T <: cli.GlobalPointer)

function parameter_contain_ptr(T)
    isa(T, DataType) || return false
    params = Sugar.to_tuple(T.parameters)
    isempty(params) && return false
    any(params) do param
        isa_pointer_type(param) && return true
        parameter_contain_ptr(param)
    end
end

function contains_pointer(
        ::Type{T},
        parent_field = (),
        pointer_fields = []
    ) where T
    @assert isleaftype(T) "Arguments must be fully concrete (leaftypes)"
    # pointers must be in parameters
    parameter_contain_ptr(T) || return false, pointer_fields
    for fname in fieldnames(T)
        FT = fieldtype(T, fname)
        if isa_pointer_type(FT)
            push!(pointer_fields, (parent_field..., fname))
        else
            _hasptrfield, pointer_fields = contains_pointer(FT, (fname,), pointer_fields)
        end
    end
    true, pointer_fields
end

@generated function replace_ptr_parameter(::Type{T}, ptrtyp::Type{PTRT} = cli.GlobalPointer) where {T,PTRT}
    isleaftype(T) || return T
    params = Sugar.to_tuple(T.parameters)
    isempty(params) && return :($T)
    parameters = map(params) do param
        isa(param, Type) || return param
        isa_pointer_type(param) && return PTRT{eltype(param)}
        replace_ptr_parameter(param, PTRT)
    end
    tname = Base.typename(T)
    tmod = getfield(tname, :module)
    name = getfield(tname, :name)
    :($tmod.$name{$(parameters...)})
end



get_fields_type(T, fields::Tuple{X}) where X = fieldtype(T, first(fields))
get_fields_type(T, fields::Tuple{Vararg{Any, N}}) where N = get_fields_type(fieldtype(T, first(fields)), Base.tail(fields))

get_fields(x, fields::NTuple{1}) = getfield(x, first(fields))
get_fields(x, fields::NTuple{N, Any}) where N = get_fields(getfield(x, first(fields)), Base.tail(fields))


using Sugar: method_nargs, getslots!, isintrinsic, getcodeinfo!, sugared, returntype, type_ast, rewrite_ast, isfunction

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
    st = getslots!(m)[2:nargs] # don't include self
    arg_idx = 1
    ptr_extract = []
    for (i, (T, name)) in enumerate(st)
        argslot = TypedSlot(i + 1, T)
        contains, field_list = contains_pointer(T)
        if contains # type contains pointer to global memory
            # replace variable in arguments to free up slot
            fake_arg = gensym(name)
            T̂ = replace_ptr_parameter(T, HostPtr) # get the pointer free equivalent type
            fake_arg_slot = newslot!(m, T̂, fake_arg)
            push!(m, T̂) # include in dependencies
            push!(kernel_args, :($fake_arg_slot::$T̂)) # put pointer free variant in kernel arguments
            push!(m.decls, fake_arg_slot)
            # we extract all pointers, pass them seperately and then assemble them in the function body
            ptr_fields = []
            for (ptridx, fields) in enumerate(field_list)
                ptr_arg = gensym(string("ptr_", ptridx, "_", name, arg_idx))
                PT = get_fields_type(T, fields)
                PT = cli.GlobalPointer{eltype(PT)}
                ptr_slot = newslot!(m, PT, ptr_arg)
                push!(kernel_ptrs, :($ptr_slot::$PT)) # add pointer to kernel arguments
                push!(ptr_fields, (fields..., ptr_slot)) # add name of ptr to fields
                push!(ptr_extract, (i, fields...))
            end
            # get reconstruction constructor
            constr, hoisted_fields = reconstruct(m, T̂, fake_arg_slot, ptr_fields)
            append!(body.args, hoisted_fields)
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
    version = cl_version(device)
    cltypes = to_cl_types(args)
    get!(compiled_functions, (ctx.id, f, cltypes)) do # TODO make this faster
        method = CLMethod((f, cltypes))
        source, fname, ptr_extract = assemble_kernel(method)
        options = "-cl-denorms-are-zero -cl-mad-enable -cl-unsafe-math-optimizations"
        if version > v"1.2"
            options *= " -cl-std=CL1.2"
        end
        p = cl.build!(
            cl.Program(ctx, source = source),
            options = options
        )
        kernel = cl.Kernel(p, fname)
        CLFunction{F, T, Tuple{ptr_extract...}}(kernel)
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
        tmp = cl_convert(elem)
        cl.set_arg!(kernel, idx, tmp)
        idx += 1
    end
    for fields in to_tuple(Ptrs)
        arg_idx, fields = first(fields), tail(fields)
        cl.set_arg!(kernel, idx, cl_convert(get_fields(args[arg_idx], fields)))
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


#
# function kernel_call_function(kernel, kernel_setup)
#     @eval begin
#         function (args, blocks::NTuple{N, Integer}, threads, queue = global_queue()) where N
#             @assert N in (1, 2, 3) "invalid block size"
#             gsize = Csize_t[b for b in blocks]
#             lsize = isa(threads, Tuple) ? Csize_t[threads[i] for i=1:length(blocks)] : threads
#             kernel = $kernel
#             $(kernel_setup...)
#             ret_event = Ref{cl.CL_event}()
#             cl.@check cl.api.clEnqueueNDRangeKernel(
#                 queue.id, kernel.id,
#                 length(gsize), C_NULL, gsize, lsize,
#                 0, C_NULL, ret_event
#             )
#             return cl.Event(ret_event[], retain = false)
#         end
#     end
# end

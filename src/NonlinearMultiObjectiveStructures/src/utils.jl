

maybe_push!!(tup, ::Nothing)=tup
maybe_push!!(tup, val)=push!!(tup, val)

function vcat_and_merge(::NTuple{0}, r)
    return r
end
function vcat_and_merge(l::Tuple, r)
    _l = stack_tuple_of_tuples(l)
    return (_l, r...)
end

function selectfirstdim(arr, i)
    return selectdim(arr, 1, i)
end
function selectfirstdim(arrs::Tuple, i)
    return selectfirstdim.(arrs, i)
end

obtainfirstdim(arr::AbstractVector)=arr
obtainfirstdim(arr, i)=selectfirstdim(arr, i)
obtainfirstdim(arr::AbstractVector, i) = arr[i]
function obtainfirstdim(arrs::Tuple, i)
    return obtainfirstdim.(arrs, i)
end

function sync!(trgt::AbstractArray, src::Union{Number, AbstractArray})
    return copyto!(trgt, src)
end

function sync!(trgt_tup::Tuple{Vararg{<:Any, N}}, res::Tuple{Vararg{<:Any, N}}) where N
    sync!.(trgt_tup, res)
    return trgt_tup
end
function sync!(trgt_tup::Tuple{<:AbstractArray}, src::Union{Number, AbstractArray})
    sync!(only(trgt_tup), src)
    return trgt_tup
end

copy_array(arr)=copy(arr)
function copy_array(arr::StackView)
    deepcopy(arr)
end

_ensure_tuple(t::Tuple)=t
_ensure_tuple(x)=(x,)
_ensure_tuple(::Nothing)=nothing

function is_implemented(
    func, op, inspect_methods=Val(true)
)
    return is_implemented(implemented(func, op), func, op, inspect_methods)
end
is_implemented(impl, func, op, inspect_methods)=impl

@generated function is_implemented(
    impl::MaybeImplemented, func, op::OP{op_symb}, ::Val{true}
) where {op_symb}
    return quote
        if !isempty(methodswith(func_Type, $(op_symb); supertypes=false))
            ## TODO signature checking (?) 
            return IsImplemented()
        end
        return NotImplemented()
    end
end

#=
num_inplace_args(op) = num_inplace_args(OperatingTrait(op))
num_inplace_args(::OutOfPlace) = 0
num_inplace_args(::InPlace{N}) where N = N

for op in OP_SYMBS
    trgt_tup = [gensym() for _=1:num_inplace_args(op)]
    i = OutputTrait(op) isa AllOutputs ? Symbol[] : [:i,]
    @eval begin
        function $(op)($(trgt_tup...), func::AbstractNonlinearFunction, x, $(i...), params; context=DefaultContext())
            ttup = ($(trgt_tup...),)
            return compute!(context, ttup, func, Val($(Meta.quot(op))), x, params, $(i...))
        end
        function $(op)($(trgt_tup...), func::AbstractNonlinearFunction, x, $(i...); context=DefaultContext())
            ttup = ($(trgt_tup...),)
            return compute!(context, ttup, func, Val($(Meta.quot(op))), x, nothing, $(i...))
        end
    end
end
=#
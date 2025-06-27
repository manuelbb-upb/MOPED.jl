# Defaults are implemented for `OP{symb}`, where `symb` is a function name.
struct OP{symb} <: AbstractOperator{symb} end
OP(symb::Symbol) = OP{symb}()
OP(op::OP)=op   # no-op constructor

# These are all available operator symbols for `OP`:
const OP_SYMBS = (
    :primals, :primals!, :primal, :primal!,    
    :gradients, :gradients!, :gradient, :gradient!,
    :hessians, :hessians!, :hessian, :hessian!,
    :primals_and_gradients, :primals_and_gradients!,
    :primal_and_gradient, :primal_and_gradient!,
    :primals_and_gradients_and_hessians, :primals_and_gradients_and_hessians!,
    :primal_and_gradient_and_hessian, :primal_and_gradient_and_hessian!,
    :jac_vec_prod, :grad_vec_prod
)

abstract type AbstractOPTrait end

abstract type DiffOrderTrait <: AbstractOPTrait end
struct ZerothOrder <: DiffOrderTrait end
struct FirstOrder <: DiffOrderTrait end
struct SecondOrder <: DiffOrderTrait end

function diff_order(::Type{<:AbstractOperator{op_symb}}) where op_symb
    error("`diff_order` not defined for `$(op_symb)`.")
end
diff_order(op::AbstractOperator)=diff_order(typeof(op))

function diff_order(::Type{op_Type}) where {op_Type<:Union{
    OP{:primals}, OP{:primals!}, OP{:primal}, OP{:primal!}
}}
    return ZerothOrder()
end

function diff_order(::Type{op_Type}) where {op_Type<:Union{
    OP{:gradients}, OP{:gradients!}, OP{:gradient}, OP{:gradient!},
    OP{:primals_and_gradients}, OP{:primals_and_gradients!}, 
    OP{:primal_and_gradient}, OP{:primal_and_gradient!},
    OP{:jac_vec_prod}, OP{:grad_vec_prod}
}}
    return FirstOrder()
end

function diff_order(::Type{op_Type}) where {op_Type<:Union{
    OP{:hessians}, OP{:hessians!}, OP{:hessian}, OP{:hessian!},
    OP{:primals_and_gradients_and_hessians}, OP{:primals_and_gradients_and_hessians!}, 
    OP{:primal_and_gradient_and_hessian}, OP{:primal_and_gradient_and_hessian!},
}}
    return SecondOrder()
end

abstract type CoValenceTrait <: AbstractOpTrait end
struct AllOutputs <: CoValenceTrait end
struct SingleOutput <: CoValenceTrait end

function covalence(::Type{<:AbstractOperator{op_symb}}) where {op_symb}
    error("`covalence` not defined for `$(op_symb).`")
end
covalence(op::AbstractOperator)=covalence(typeof(op))

function covalence(::Type{op_Type}) where {op_Type<:Union{
    OP{:primals}, OP{:primals!}, #OP{:primal}, OP{:primal!}
    OP{:gradients}, OP{:gradients!}, #OP{:gradient}, OP{:gradient!},
    OP{:primals_and_gradients}, OP{:primals_and_gradients!}, 
    #OP{:primal_and_gradient}, OP{:primal_and_gradient!},
    OP{:jac_vec_prod}, #OP{:grad_vec_prod}
    OP{:hessians}, OP{:hessians!}, #OP{:hessian}, OP{:hessian!},
    OP{:primals_and_gradients_and_hessians}, OP{:primals_and_gradients_and_hessians!}, 
    #OP{:primal_and_gradient_and_hessian}, OP{:primal_and_gradient_and_hessian!},
}}
    return AllOutputs()
end
function covalence(::Type{op_Type}) where {op_Type<:Union{
    #OP{:primals}, OP{:primals!}, 
    OP{:primal}, OP{:primal!},
    #OP{:gradients}, OP{:gradients!}, 
    OP{:gradient}, OP{:gradient!},
    #OP{:primals_and_gradients}, OP{:primals_and_gradients!}, 
    OP{:primal_and_gradient}, OP{:primal_and_gradient!},
    #OP{:jac_vec_prod}, 
    OP{:grad_vec_prod},
    #OP{:hessians}, OP{:hessians!}, 
    OP{:hessian}, OP{:hessian!},
    #OP{:primals_and_gradients_and_hessians}, OP{:primals_and_gradients_and_hessians!}, 
    OP{:primal_and_gradient_and_hessian}, OP{:primal_and_gradient_and_hessian!},
}}
    return SingleOutput()
end
# ## Internal Calls

# Bridging is performed with `compute!`.
# Internally it calls `operate!`.
# The `operate!` function can be specialized.
function operate!(
    trgt_tup, func::AbstractNonlinearFunction, op::AbstractOperator, op_args
)
    error("operate! not defined")
end

# For basic operators, it falls back to mnemonic function calls.
@generated function operate!(
    trgt_tup,
    func::AbstractNonlinearFunction,
    op::OP{op_symb},
    op_args
) where {op_symb}
    return quote
        ttype = trgts_type(Val(true), func, op)
        @assert trgt_tup isa ttype
        _op_args = check_op_args(func, op, op_args)
        res = $(op_symb)(trgt_tup..., func, _op_args...)
        rtype = res_type(Val(false), func, op)
        if !(res isa rtype)
            return res
        end
        return _ensure_tuple(res)
   end
end

function check_op_args(func, op, op_args)
    _op_args = maybe_remove_params(func, op_args)
    atype = args_type(Val(true), func, op)
    @smart_assert _op_args isa atype
    return _op_args
end

function maybe_remove_params(func::func_Type, op_args) where func_Type
    maybe_remove_params(HasParametersTrait(func_Type), op_args)
end
maybe_remove_params(::HasParameters, op_args)=op_args
maybe_remove_params(::NoParameters, op_args)=Base.front(op_args)

inplace_arrays(func, op, op_args) = ()

function inplace_arrays(func, ::OP{:gradient!}, (x, i, params))
    return (array_gradient(func, x, i, params),)
end

function inplace_arrays(func, ::OP{:hessian!}, (x, i, params))
    return (array_hessian(func, x, i, params),)
end

function inplace_arrays(func, ::OP{:primals!}, (x, params))
    return (array_primals(func, x, params),)
end

function inplace_arrays(func, ::OP{:gradients!}, (x, params))
    return (array_gradients(func, x, params),)
end

function inplace_arrays(func, ::OP{:hessians!}, (x, params))
    return (array_hessians(func, x, params),)
end

function inplace_arrays(
    func, 
    op::Union{OP{:primals_and_gradients!}, OP{:primal_and_gradient!}}, 
    op_args
)
    y = inplace_arrays(func, _zeroth_order_op(op), op_args)
    Dy = inplace_arrays(func, _first_order_op(op), op_args)
    return (y..., Dy...)
end
function _zeroth_order_op(::Union{OP{:primal_and_gradient!}, OP{:primal_and_gradient_and_hessian!}})
    return OP{:primal!}()
end
function _zeroth_order_op(::Union{OP{:primals_and_gradients!}, OP{:primals_and_gradients_and_hessians!}})
    return OP{:primals!}()
end
function _first_order_op(::Union{OP{:primal_and_gradient!}, OP{:primal_and_gradient_and_hessian!}})
    return OP{:gradient!}()
end
function _first_order_op(::Union{OP{:primals_and_gradients!}, OP{:primals_and_gradients_and_hessians!}})
    return OP{:gradients}()
end

function inplace_arrays(
    func,
    op::Union{OP{:primals_and_gradients_and_hessians!}, OP{:primal_and_gradient_and_hessian!}}, 
    op_args
)
    y_Dy = inplace_arrays(func, _zeroth_and_first_order_op(op), op_args)
    Hy = inplace_arrays(func, _second_order_op(op), op_args)
    return (y_Dy..., Hy...)
end
function _zeroth_and_first_order_op(::OP{:primal_and_gradient_and_hessian!})
    return OP{:primal_and_gradient!}()
end
function _zeroth_and_first_order_op(::OP{:primals_and_gradients_and_hessians!})
    return OP{:primals_and_gradients!}()
end
function _second_order_op(::OP{:primal_and_gradient_and_hessian!})
    return OP{:hessian!}()
end
function _second_order_op(::OP{:primals_and_gradients_and_hessians!})
    return OP{:hessians!}()
end
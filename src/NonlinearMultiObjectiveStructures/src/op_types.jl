for op_types_func in (
    :res_type,
    :trgts_type,
    :args_type
)
    @eval begin
        function $(op_types_func)(
            operate!_call::Val{false}, 
            func::AbstractNonlinearFunction, 
            op::AbstractOperator
        )
            return $(op_types_func)(func, op)
        end

        function $(op_types_func)(
            operate!_call::Val{true}, 
            func::AbstractNonlinearFunction, 
            op::AbstractOperator
        )
            return _ensure_tuple_type($(op_types_func)(func, op))
        end
    end
end

function res_type(func::AbstractNonlinearFunction, op::AbstractOperator)
    return Union{Real, AbstractArray{<:Real}, Nothing}
end

function trgts_type(func::AbstractNonlinearFunction, op::AbstractOperator)
    return Tuple{}
end

function args_type(func::AbstractNonlinearFunction, op::AbstractOperator)
    return Tuple{Vararg}
end

_ensure_tuple_type(T::Type{<:Tuple})=T
_ensure_tuple_type(T::Type)=Tuple{<:T}
_ensure_tuple_type(::Type{<:Nothing})=Nothing

res_type(func::AbstractNonlinearFunction, op::OP)=_res_type(op)
trgts_type(func::AbstractNonlinearFunction, op::OP)=_trgts_type(op)
args_type(func::AbstractNonlinearFunction, op::OP)=_args_type(op)

_res_type(::OP{:primal})=Real
_trgts_type(::OP{:primal})=Tuple{}
_args_type(::OP{:primal})=Tuple{<:RVector, <:Integer}

_res_type(::OP{:primals})=RVector
_trgts_type(::OP{:primals})=Tuple{}
_args_type(::OP{:primals})=RVector

_res_type(::OP{:primal!})=Real
_trgts_type(::OP{:primal!})=Tuple{}
_args_type(::OP{:primal!})=Tuple{<:RVector, <:Integer}

_res_type(::OP{:primals!})=Nothing
_trgts_type(::OP{:primals!})=RVector
_args_type(::OP{:primals!})=Tuple{<:RVector}

_res_type(::OP{:gradient})=RVector
_trgts_type(::OP{:gradient})=Tuple{}
_args_type(::OP{:gradient})=Tuple{<:RVector, <:Integer}

_res_type(::OP{:gradients})=RMatrix
_trgts_type(::OP{:gradients})=Tuple{}
_args_type(::OP{:gradients})=RVector

_res_type(::OP{:gradient!})=Nothing
_trgts_type(::OP{:gradient!})=RVector
_args_type(::OP{:gradient!})=Tuple{<:RVector, <:Integer}

_res_type(::OP{:gradients!})=Nothing
_trgts_type(::OP{:gradients!})=RMatrix
_args_type(::OP{:gradients!})=RVector

_res_type(::OP{:hessian})=RMatrix
_trgts_type(::OP{:hessian})=Tuple{}
_args_type(::OP{:hessian})=Tuple{<:RVector, <:Integer}

_res_type(::OP{:hessians})=RTensor
_trgts_type(::OP{:hessians})=Tuple{}
_args_type(::OP{:hessians})=RVector

_res_type(::OP{:hessian!})=Nothing
_trgts_type(::OP{:hessian!})=RMatrix
_args_type(::OP{:hessian!})=Tuple{<:RVector, <:Integer}

_res_type(::OP{:hessians!})=Nothing
_trgts_type(::OP{:hessians!})=RTensor
_args_type(::OP{:hessians!})=RVector

_res_type(::OP{:primal_and_gradient})=Tuple{<:Real, <:RVector}
_trgts_type(::OP{:primal_and_gradient})=Tuple{}
_args_type(::OP{:primal_and_gradient})=Tuple{<:RVector, <:Integer}

_res_type(::OP{:primals_and_gradients})=Tuple{<:RVector, <:RMatrix}
_trgts_type(::OP{:primals_and_gradients})=Tuple{}
_args_type(::OP{:primals_and_gradients})=RVector

_res_type(::OP{:primal_and_gradient!})=Real
_trgts_type(::OP{:primal_and_gradient!})=RMatrix
_args_type(::OP{:primal_and_gradient!})=Tuple{<:RVector, <:Integer}

_res_type(::OP{:primals_and_gradients!})=Nothing
_trgts_type(::OP{:primals_and_gradients!})=Tuple{<:RVector, <:RMatrix}
_args_type(::OP{:primals_and_gradients!})=RVector

_res_type(::OP{:primal_and_gradient_and_hessian})=Tuple{<:Real, <:RVector, <:RMatrix}
_trgts_type(::OP{:primal_and_gradient_and_hessian})=Tuple{}
_args_type(::OP{:primal_and_gradient_and_hessian})=Tuple{<:RVector, <:Integer}

_res_type(::OP{:primals_and_gradients_and_hessians})=Tuple{<:RVector, <:RMatrix, <:RTensor}
_trgts_type(::OP{:primals_and_gradients_and_hessians})=Tuple{}
_args_type(::OP{:primals_and_gradients_and_hessians})=RVector

_res_type(::OP{:primal_and_gradient_and_hessian!})=Real
_trgts_type(::OP{:primal_and_gradient_and_hessian!})=Tuple{<:RVector, <:RMatrix}
_args_type(::OP{:primal_and_gradient_and_hessian!})=Tuple{<:RVector, <:Integer}

_res_type(::OP{:primals_and_gradients_and_hessians!})=Nothing
_trgts_type(::OP{:primals_and_gradients_and_hessians!})=Tuple{<:RMatrix, <:RTensor}
_args_type(::OP{:primals_and_gradients_and_hessians!})=RVector

_res_type(::OP{:jac_vec_prod})=Tuple{Vararg{<:RVector}}
_trgts_type(::OP{:jac_vec_prod})=Tuple{}
_args_type(::OP{:jac_vec_prod})=Tuple{RVector, Vararg{<:RVector}}

_res_type(::OP{:grad_vec_prod})=Tuple{Vararg{<:Real}}
_trgts_type(::OP{:grad_vec_prod})=Tuple{}
_args_type(::OP{:grad_vec_prod})=Tuple{<:RVector, <:Integer, Vararg{<:RVector}}
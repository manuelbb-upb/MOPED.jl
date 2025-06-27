struct WrappedOP{
    op_symb, 
    op_Type <: AbstractOperator{op_symb},
    prep_Type
} <: AbstractOperator{op_symb}
    op :: op_Type 
    prep :: prep_Type
end

function _wrapped_type(
    ::Type{<:WrappedOP{op_symb, op_Type, prep_Type}}
) where {op_symb, op_Type, prep_Type}
    return op_Type
end

function WrappedOP(op::OP{op_symb}, prep::Prep) where op_symb
    return WrappedOP{op_symb}(op, prep)
end

function operate!(
    trgt_tup, func::AbstractNonlinearFunction, wop::WrappedOP, op_args
)
    @unpack prep, op = wop
    return prepped_compute!(trgt_tup, prep, func, op, op_args)
end

implemented(::AbstractNonlinearFunction, wop::WrappedOP)=IsImplemented()

res_type(func::AbstractNonlinearFunction, wop::WrappedOP)=res_type(func, wop.op)
trgts_type(func::AbstractNonlinearFunction, wop::WrappedOP)=trgts_type(func, wop.op)
args_type(func::AbstractNonlinearFunction, wop::WrappedOP)=args_type(func, wop.op)

inplace_arrays(func, wop::WrappedOP, op_args)=inplace_arrays(func, wop.op, op_args)

diff_order(wT::Type{<:WrappedOP})=diff_order(_wrapped_type(wT))
covalence(wT::Type{<:WrappedOP})=covalence(_wrapped_type(wT))
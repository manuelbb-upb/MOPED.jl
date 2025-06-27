function prepare(
    fctx::FirstOrderContext, trgt_tup, func, op, op_args
)
    prep = prepare(
        fctx.ctx, trgt_tup, func, op, op_args
    )
    if ismissing(prep) && diff_order(op) isa FirstOrder
        if !(op isa OP{:jac_vec_prod} || op isa OP{:grad_vec_prod})
            prep = prepare_jvp(fctx, trgt_tup, func, op, op_args)
        end
    end
    return prep
end

function prepare_jvp(fctx, trgt_tup, func, op, op_args)
    prep = _prep_pushforward(fctx, trgt_tup, func, op, op_args)
    return _prepare_jvp(prep, fctx, trgt_tup, func, op, op_args)
end

function _prep_pushforward(fctx, trgt_tup, func, op, op_args)
    op_pf = _pushforward_op(op)
    return prep(fctx.ctx, trgt_tup, func, op_pf, op_args)
end

_pushforward_op(op)=_pushforward_op(covalence(op))
_pushforward_op(::SingleOutput)=OP(:grad_vec_prod)
_pushforward_op(::AllOutputs)=OP(:jac_vec_prod)

function _prepare_jvp(jvp_prep, fctx, trgt_tup, func, op, op_args)
    return jvp_prep
end

function _prepare_jvp(jvp_prep::Prep, fctx, trgt_tup, func, op, op_args)
    
end

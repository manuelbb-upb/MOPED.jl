abstract type AbstractFBackResult end
import BangBang: push!!, popfirst!!

struct UnsuccessfulFBack <: AbstractFBackResult end
struct EndOfChain <: AbstractFBackResult end

## Initialize fallback chain:
function generic_fbackchain(trgt, callee, mop, attr)
    ctxs = all_ctx(trgt, callee)
    return generic_fbackchain(trgt, callee, ctxs, mop, attr)
end

function all_ctx(trgt, callee)
    _ctxs = all_fbacks(trgt, callee)
    ctxs = filter(
        ctx -> check_compat(trgt, callee, ctx) isa Val{true}, 
        _ctxs
    )
    return push!!(ctxs, missing)
end
all_fbacks(trgt, callee)=()

function generic_fbackchain(trgt, callee, ctxs, mop, attr)
    fin = _generic_fbackchain(trgt, callee, ctxs, mop, attr)
    return final_res(trgt, callee, fin)
end

final_res(trgt, callee, res)=res

function final_res(
    trgt::T, callee::T, res::UnsuccessfulFBack) where T
    error("`$(error_str(trgt))` not implemented.")
end
error_str(trgt)=string(trgt)

@generated function _generic_fbackchain(
    trgt, callee, ctxs::Tuple, mop, attr
)
    return _generic_fbackchain_expr(trgt, callee, ctxs, mop, attr)
end

function _generic_fbackchain_expr(trgt, callee, ctxs, mop, attr)
   return quote
        res = UnsuccessfulFBack()
        $((
            :(if res isa UnsuccessfulFBack
                res = _generic_fback(trgt, callee, $(ctx), mop, attr)
            end) for ctx = ctxs.parameters
        )...)
        if res isa EndOfChain
            res = UnsuccessfulFBack()
        end
        return res
    end
end

#=
function _generic_fbackchain(trgt, callee, ctxs::Tuple, mop, attr)
    _ctxs, ctx = popfirst!!(ctxs)
    return __generic_fbackchain(ctx, trgt, callee, _ctxs, mop, attr)
end
function __generic_fbackchain(ctx, trgt, callee, ctxs, mop, attr)
    res = _generic_fback(trgt, callee, ctx, mop, attr)
    return __generic_fbackchain(res, ctx, trgt, callee, ctxs, mop, attr)
end

function __generic_fbackchain(res::FIN_TYPE, ctx, trgt, callee, ctxs, mop, attr)
    if res isa EndOfChain
        return UnsuccessfulFBack()
    end
    return res
end
function __generic_fbackchain(res, ctx, trgt, callee, ctxs, mop, attr)
    return _generic_fbackchain(trgt, callee, ctxs, mop, attr)
end
=#
check_compat(trgt, callee, ctx)=Val(true)
check_compat(trgt, callee::c_Type, ctx::c_Type) where{c_Type}=Val(false)
check_compat(trgt, callee::c_Type, ctx::Type{c_Type}) where{c_Type}=Val(false)

## End of chain:
function _generic_fback(trgt, callee, ctx::Type{Missing}, mop, attr)
    return EndOfChain()
end

## Generic return value for any fallback implementation:
function _generic_fback(trgt, callee, ctx, mop, attr)
    ctx_compat = check_compat(trgt, callee, ctx)
    @assert ctx_compat isa Val{true}
    res = generic_fback(trgt, callee, ctx, mop, attr)
    return res
end
function generic_fback(trgt, callee, ctx, mop, attr)
    @info "fallback"
    return UnsuccessfulFBack()
end
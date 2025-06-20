## Initialize fallback chain:
function generic_fbackchain(trgt, callee, mop, attr)
    ctx = next_ctx(trgt, callee)
    return generic_fbackchain(trgt, callee, ctx, mop, attr)
end

function generic_fbackchain(trgt, callee, ctx, mop, attr)
    fin = _generic_fbackchain(trgt, callee, ctx, mop, attr)
    return final_res(trgt, callee, fin)
end

function _generic_fbackchain(trgt, callee, ctx, mop, attr)
    res = _generic_fback(trgt, callee, ctx, mop, attr)
    res_success = check_result(trgt, res)::Union{Val{true}, Val{false}}
    return _generic_fbackchain(res_success, res, trgt, callee, ctx, mop, attr)
end

final_res(trgt, callee, res)=res

struct EndOfChain end
function final_res(trgt, callee, res::EndOfChain)
    error("`$(error_str(trgt))` not implemented.")
end
error_str(trgt)=string(trgt)

check_result(trgt, res)=Val(false)
check_result(trgt, ::EndOfChain)=Val(true)

function _generic_fbackchain(::Val{true}, res, trgt, callee, ctx, mop, attr)
    return res
end
function _generic_fbackchain(::Val{false}, res, trgt, callee, ctx, mop, attr)
    return next_fback(trgt, callee, ctx, mop, attr)
end

function next_fback(trgt, callee, ctx, mop, attr)
    _ctx = next_ctx(trgt, callee, ctx)
    return _generic_fbackchain(trgt, callee, _ctx, mop, attr)
end

## Get next `ctx`, e.g. some dispatch signal for fallback to try.
## Making this depend on callee should allow for re-ordering of fback chain;
## It can also be used to disable/skip fbacks -- it can be easiear however to simply
## have `generic_fback` return nothing for certain `callees`.
## Or force `nothing` via `check_compat(...)=Val(false)`.
next_ctx(trgt, callee)=missing
next_ctx(trgt, callee, ctx)=missing

check_compat(trgt, callee, ctx)=Val(true)
check_compat(trgt, callee::c_Type, ctx::c_Type) where{c_Type}=Val(false)

## End of chain:
function _generic_fback(trgt, callee, ctx::Missing, mop, attr)
    return EndOfChain()
end

## Generic return value for any fallback implementation:
function _generic_fback(trgt, callee, ctx, mop, attr)
    ctx_compat = check_compat(trgt, callee, ctx) :: Union{Val{true}, Val{false}}
    return _generic_fback(
        ctx_compat, trgt, callee, ctx, mop, attr)
end
function _generic_fback(::Val{true}, trgt, callee, ctx, mop, attr)
    return generic_fback(trgt, callee, ctx, mop, attr)
end
function _generic_fback(::Val{false}, trgt, callee, ctx, mop, attr)
    nothing
end
generic_fback(trgt, callee, ctx, mop, attr)=nothing
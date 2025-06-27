struct Prep{
    pair_Type <: Tuple{<:OP, <:OP},
    ctx_Type,
    cache_Type
}
    op_fback :: pair_Type
    ctx :: ctx_Type
    cache :: cache_Type
end

function Prep(;
    op::OP, fback::OP, ctx, cache
)
    return Prep((op, fback), ctx, cache)
end

which_fback(prep::Prep) = prep.op_fback[2]

function finalize_arrays(prep, arrs)
    return maybe_copy(is_onetime(prep.ctx), arrs)
end
function maybe_copy(onetime::Val{true}, res)
    return res
end
function maybe_copy(onetime::Val{false}, res)
    return copy_array(res)
end
function maybe_copy(onetime::Val{false}, res::Tuple)
    return copy_array.(res)
end
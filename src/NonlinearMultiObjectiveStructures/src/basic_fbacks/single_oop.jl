# Compute single-output out-of-place operators indirectly, using some other operator
# as fallback.

# ## First Case
function __prepare_fback(
    (op, fback) :: pair_Type,
    ctx, trgt_tup, func, (x, i, params)
) where {pair_Type <: Union{
    Tuple{<:AbstractOperator{:primal}, <:AbstractOperator{:primals}},
    Tuple{<:AbstractOperator{:gradient}, <:AbstractOperator{:gradients}},
    Tuple{<:AbstractOperator{:hessian}, <:AbstractOperator{:hessians}},
    Tuple{<:AbstractOperator{:primal_and_gradient}, <:AbstractOperator{:primals_and_gradients}},
    Tuple{<:AbstractOperator{:primal_and_gradient_and_hessian}, <:AbstractOperator{:primals_and_gradients_and_hessians}},
    Tuple{<:AbstractOperator{:grad_vec_prod}, <:AbstractOperator{:jac_vec_prod}}
}} 
    cache = nothing
    return Prep(; op, fback, cache, ctx)
end

function prepped_compute!(
    trgt_tup, prep::Prep{pair_Type}, func, op, (x, i, args...)
) where {pair_Type <: Union{
    Tuple{<:AbstractOperator{:primal}, <:AbstractOperator{:primals}},
    Tuple{<:AbstractOperator{:gradient}, <:AbstractOperator{:gradients}},
    Tuple{<:AbstractOperator{:hessian}, <:AbstractOperator{:hessians}},
    Tuple{<:AbstractOperator{:primal_and_gradient}, <:AbstractOperator{:primals_and_gradients}},
    Tuple{<:AbstractOperator{:primal_and_gradient_and_hessian}, <:AbstractOperator{:primals_and_gradients_and_hessians}},
}}
    fback = which_fback(prep)
    rtype = res_type(Val(true), func, fback)
    @assert rtype <: Tuple
    res = operate!((), func, fback, (x, args...))
    if !(res isa rtype)
        return StopObject(res, (;op, prep))
    end
    return obtainfirstdim(res, i)
end

# ## Second Case
# `Dyi = gradient(func, i, ...)` ⇔ `gradients!(Dy, func, ...)`
# `yi, Dyi = primal_and_gradient(func, i, ...)` ⇔ `primals_and_gradients!(y, Dy, func, ...)`
function __prepare_fback(
    (op, fback) :: pair_Type,
    ctx, trgt_tup, func, (x, i, params)
) where {pair_Type <: Union{
    Tuple{<:AbstractOperator{:primal}, <:AbstractOperator{:primals!}},
    Tuple{<:AbstractOperator{:gradient}, <:AbstractOperator{:gradients!}},
    Tuple{<:AbstractOperator{:hessian}, <:AbstractOperator{:hessians!}},
    Tuple{<:AbstractOperator{:primal_and_gradient}, <:AbstractOperator{:primals_and_gradients!}},
    Tuple{<:AbstractOperator{:primal_and_gradient_and_hessian}, <:AbstractOperator{:primals_and_gradients_and_hessians!}},
}} 
    cache = inplace_arrays(func, fback, (x, params))
    return Prep(; op, fback, cache, ctx)
end

function prepped_compute!(
    trgt_tup, prep::Prep{pair_Type}, func, op, (x, i, params)
) where {pair_Type <: Union{
    Tuple{<:AbstractOperator{:primal}, <:AbstractOperator{:primals!}},
    Tuple{<:AbstractOperator{:gradient}, <:AbstractOperator{:gradients!}},
    Tuple{<:AbstractOperator{:hessian}, <:AbstractOperator{:hessians!}},
    Tuple{<:AbstractOperator{:primal_and_gradient}, <:AbstractOperator{:primals_and_gradients!}},
    Tuple{<:AbstractOperator{:primal_and_gradient_and_hessian}, <:AbstractOperator{:primals_and_gradients_and_hessians!}},
}}
    fback = which_fback(prep)
    rtype = res_type(Val(true), func, fback)
    @assert rtype <: Nothing
    trgt = prep.cache
    res = operate!(trgt, func, fback, (x, params))
    if !(res isa rtype)
        return StopObject(res, (;op, prep))
    end
    ret = obtainfirstdim(trgt, i)
    return finalize_arrays(prep, ret)
end

# ## Third Case
# `yi, Dyi = primal_and_gradient(func, i, ...)` ⇔ `yi = primal_and_gradient!(Dyi, func, ...)`
function __prepare_fback(
    (op, fback)::pair_Type,
    ctx, trgt_tup, func, (x, i, params)
) where {pair_Type <: Union{
    Tuple{<:AbstractOperator{:primal}, <:AbstractOperator{:primal!}},
    Tuple{<:AbstractOperator{:gradient}, <:AbstractOperator{:gradient!}},
    Tuple{<:AbstractOperator{:hessian}, <:AbstractOperator{:hessian!}},
    Tuple{<:AbstractOperator{:primal_and_gradient}, <:AbstractOperator{:primal_and_gradient!}},
    Tuple{<:AbstractOperator{:primal_and_gradient_and_hessian}, <:AbstractOperator{:primal_and_gradient_and_hessian!}},
}}
    cache = inplace_arrays(func, fback, (x, i, params))
    return Prep(; op, fback, cache, ctx)
end

function prepped_compute!(
    trgt_tup, prep::Prep{pair_Type}, func, op, (x, i, params)
) where {pair_Type <: Union{
    Tuple{<:AbstractOperator{:primal}, <:AbstractOperator{:primal!}},
    Tuple{<:AbstractOperator{:gradient}, <:AbstractOperator{:gradient!}},
    Tuple{<:AbstractOperator{:hessian}, <:AbstractOperator{:hessian!}},
    Tuple{<:AbstractOperator{:primal_and_gradient}, <:AbstractOperator{:primal_and_gradient!}},
    Tuple{<:AbstractOperator{:primal_and_gradient_and_hessian}, <:AbstractOperator{:primal_and_gradient_and_hessian!}},
}}    
    fback = which_fback(prep)
    rtype = res_type(Val(true), func, fback)
    @assert rtype <: Nothing || rtype <: Tuple{<:Real}
    trgt = prep.cache
    l = ()
    res = operate!(trgt, func, fback, (x, i, params))
    if !(res isa rtype)
        return StopObject(res, (;op, prep))
    end
    l = maybe_push!!(l, res)
    r = finalize_arrays(prep, trgt)
    return vcat_and_merge(l, r)
end
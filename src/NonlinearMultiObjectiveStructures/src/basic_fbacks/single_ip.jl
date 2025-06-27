# ## First Case
# `gradient!(Dyi, func, i, ...)` ⇔ `gradients!(Dy, func, ...)`
# `yi = primal_and_gradient!(Dyi, func, i, ...)` ⇔ `primals_and_gradients!(y, Dy, func, ...)`
function __prepare_fback(
    (op, fback)::pair_Type,
    ctx, trgt_tup, func, (x, i, params)
) where {pair_Type <: Union{
    Tuple{<:AbstractOperator{:primal!}, <:AbstractOperator{:primals!}},
    Tuple{<:AbstractOperator{:gradient!}, <:AbstractOperator{:gradients!}},
    Tuple{<:AbstractOperator{:hessian!}, <:AbstractOperator{:hessians!}},
    Tuple{<:AbstractOperator{:primal_and_gradient!}, <:AbstractOperator{:primals_and_gradients!}},
    Tuple{<:AbstractOperator{:primal_and_gradient_and_hessian!}, <:AbstractOperator{:primals_and_gradients_and_hessians!}},
}}
    cache = inplace_arrays(func, fback, (x, params))
    return Prep(; op, fback, cache, ctx)
end

function prepped_compute!(
    trgt_tup, prep::Prep{pair_Type}, func, op, (x, i, params)
) where {pair_Type <: Union{
    Tuple{<:AbstractOperator{:primal!}, <:AbstractOperator{:primals!}},
    Tuple{<:AbstractOperator{:gradient!}, <:AbstractOperator{:gradients!}},
    Tuple{<:AbstractOperator{:hessian!}, <:AbstractOperator{:hessians!}},
    Tuple{<:AbstractOperator{:primal_and_gradient!}, <:AbstractOperator{:primals_and_gradients!}},
    Tuple{<:AbstractOperator{:primal_and_gradient_and_hessian!}, <:AbstractOperator{:primals_and_gradients_and_hessians!}},
}}
    fback = which_fback(prep)
    rtype = res_type(Val(true), func, fback)
    @assert rtype <: Nothing
    trgt = prep.cache
    res = operate!(trgt, func, fback, (x, params))
    if !(res isa rtype)
        return StopObject(res, (;op, prep))
    end
    @assert isnothing(res)
    return retval_and_sync!(trgt_tup, trgt, i)
end

# If ∃ same number of objects (arrays) in `trgt_tup` and `fback_tup`, 
# then just sync into targets:
function retval_and_sync!(
    trgt_tup::Tuple{Vararg{<:Any, N}}, fback_tup::Tuple{Vararg{<:Any, N}}
) where {N}
    sync!(trgt_tup, fback_tup)
    return nothing
end
 
# If ∃ same number of objects (arrays) in `trgt_tup` and `fback_tup`, 
# and if there is a dimension index `i`, first select first dim at index `i` in all 
# arrays in `fback_tup` and then sync into targets:
function retval_and_sync!(
    trgt_tup::Tuple{Vararg{<:Any, N}}, fback_tup::Tuple{Vararg{<:Any, N}}, i
) where {N}
    return retval_and_sync!(trgt_tup, selectfirstdim(fback_tup, i))
end

# If there is one entry more in `fback_tup`, return first entry as a result and sync 
# remaining objects:
function retval_and_sync!(
    trgt_tup::Tuple, fback_tup::Tuple, i...
)
    @assert length(trgt_tup) == length(fback_tup) - 1
    ftup = Base.tail(fback_tup)
    retval_and_sync!(trgt_tup, ftup, i...)
    y = first(fback_tup)
    return (obtainfirstdim(y, i...),)
end

# ## Second Case
# `gradient!(Dyi, func, i, ...)` ⇔ `Dy = gradients(func, ...)`
# `yi = primal_and_gradient!(Dyi, func, i, ...)` ⇔ `y, Dy = primals_and_gradients(func, ...)`
function __prepare_fback(
    (op, fback)::pair_Type,
    ctx, trgt_tup, func, (x, i, params)
) where {pair_Type <: Union{
    Tuple{<:AbstractOperator{:primal!}, <:AbstractOperator{:primals}},
    Tuple{<:AbstractOperator{:gradient!}, <:AbstractOperator{:gradients}},
    Tuple{<:AbstractOperator{:hessian!}, <:AbstractOperator{:hessians}},
    Tuple{<:AbstractOperator{:primal_and_gradient!}, <:AbstractOperator{:primals_and_gradients}},
    Tuple{<:AbstractOperator{:primal_and_gradient_and_hessian!}, <:AbstractOperator{:primals_and_gradients_and_hessians}},
}}
    cache = nothing
    return Prep(; op, fback, cache, ctx)
end

function prepped_compute!(
    trgt_tup, prep::Prep{pair_Type}, func, op, (x, i, params)
) where {pair_Type <: Union{
    Tuple{<:AbstractOperator{:primal!}, <:AbstractOperator{:primals}},
    Tuple{<:AbstractOperator{:gradient!}, <:AbstractOperator{:gradients}},
    Tuple{<:AbstractOperator{:hessian!}, <:AbstractOperator{:hessians}},
    Tuple{<:AbstractOperator{:primal_and_gradient!}, <:AbstractOperator{:primals_and_gradients}},
    Tuple{<:AbstractOperator{:primal_and_gradient_and_hessian!}, <:AbstractOperator{:primals_and_gradients_and_hessians}},
}}
    fback = which_fback(prep)
    rtype = res_type(Val(true), func, fback)
    @assert rtype <: Tuple
    res = operate!((), func, fback, (x, params))
    if !(res isa rtype)
        return StopObject(res, (;op, prep))
    end
    return retval_and_sync!(trgt_tup, res, i)
end

# ## Third Case
# `gradient!(Dyi, func, i, ...)` ⇔ `Dyi = gradient(func, i, ...)`
# `yi = primal_and_gradient!(Dyi, func, i, ...)` ⇔ `yi, Dyi = primal_and_gradient(func, i, ...)`
function __prepare_fback(
    (op, fback)::pair_Type,
    ctx, trgt_tup, func, (x, i, params)
) where {pair_Type <: Union{
    Tuple{<:AbstractOperator{:primal!}, <:AbstractOperator{:primal}},
    Tuple{<:AbstractOperator{:gradient!}, <:AbstractOperator{:gradient}},
    Tuple{<:AbstractOperator{:hessian!}, <:AbstractOperator{:hessian}},
    Tuple{<:AbstractOperator{:primal_and_gradient!}, <:AbstractOperator{:primal_and_gradient}},
    Tuple{<:AbstractOperator{:primal_and_gradient_and_hessian!}, <:AbstractOperator{:primal_and_gradient_and_hessian}},
}}
    cache = nothing
    return Prep(; op, fback, cache, ctx)
end

function prepped_compute!(
    trgt_tup, prep::Prep{pair_Type}, func, op, (x, i, params)
) where {pair_Type <: Union{
    Tuple{<:AbstractOperator{:primal!}, <:AbstractOperator{:primal}},
    Tuple{<:AbstractOperator{:gradient!}, <:AbstractOperator{:gradient}},
    Tuple{<:AbstractOperator{:hessian!}, <:AbstractOperator{:hessian}},
    Tuple{<:AbstractOperator{:primal_and_gradient!}, <:AbstractOperator{:primal_and_gradient}},
    Tuple{<:AbstractOperator{:primal_and_gradient_and_hessian!}, <:AbstractOperator{:primal_and_gradient_and_hessian}},
}}
    fback = which_fback(prep)
    rtype = res_type(Val(true), func, fback)
    @assert rtype <: Tuple
    res = operate!((), func, fback, (x, i, params))
    if !(res isa rtype)
        return StopObject(res, (;op, prep))
    end
    return retval_and_sync!(trgt_tup, res)
end
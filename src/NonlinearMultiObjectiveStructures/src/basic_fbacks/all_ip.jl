# Compute vector-valued out-of-place operators indirectly, using some other operator
# as fallback.
# We don't need a cache for any of the bridges here, because we manipulate arrays
# in `trgt_tup`:
function __prepare_fback(
    (op, fback)::pair_Type,
    ctx, trgt_tup, func, op_args 
) where {pair_Type<:Union{
    Tuple{<:AbstractOperator{:primals!}, <:AbstractOperator},
    Tuple{<:AbstractOperator{:gradients!}, <:AbstractOperator},
    Tuple{<:AbstractOperator{:hessians!}, <:AbstractOperator},
    Tuple{<:AbstractOperator{:primals_and_gradients!}, <:AbstractOperator},
    Tuple{<:AbstractOperator{:primals_and_gradients_and_hessians!}, <:AbstractOperator},
}}
    cache = nothing
    return Prep(; op, fback, cache, ctx)
end

# E.g., compute `primals!(y, func, x, params)` 
# using `y = primals(func, x, params)`.
# Applicable for all pairs of operators where in-place targets correspond to 
# out-of-place return values.
function prepped_compute!(
    trgt_tup, prep::Prep{pair_Type}, func, op, (x, params)
) where {pair_Type<:Union{
    Tuple{<:AbstractOperator{:primals!}, <:AbstractOperator{:primals}},
    Tuple{<:AbstractOperator{:gradients!}, <:AbstractOperator{:gradients}},
    Tuple{<:AbstractOperator{:hessians!}, <:AbstractOperator{:hessians}},
    Tuple{<:AbstractOperator{:primals_and_gradients!}, <:AbstractOperator{:primals_and_gradients}},
    Tuple{<:AbstractOperator{:primals_and_gradients_and_hessians!}, <:AbstractOperator{:primals_and_gradients_and_hessians}},
}}
    fback = which_fback(prep)
    rtype = res_type(Val(true), func, fback)
    res = operate!((), func, fback, (x, params))
    if !(res isa rtype)
        return StopObject(res, (;op, prep))
    end
    sync!(trgt_tup, res)
    return nothing
end

# E.g., compute `primals!(y, func, x, params)` 
# using `yi = primal(func, x, i, params)` for all output indices.
# But also `primals_and_gradients!(y, Dy, ...)` and `yi, Dyi = primal_and_gradient(...)`.
# Applicable for all pairs of operators where number of in-place targets corresponds to 
# number of out-of-place return values.
function prepped_compute!(
    trgt_tup, prep::Prep{pair_Type}, func, op, (x, params)
) where {pair_Type<:Union{
    Tuple{<:AbstractOperator{:primals!}, <:AbstractOperator{:primal}},
    Tuple{<:AbstractOperator{:gradients!}, <:AbstractOperator{:gradient}},
    Tuple{<:AbstractOperator{:hessians!}, <:AbstractOperator{:hessian}},
    Tuple{<:AbstractOperator{:primals_and_gradients!}, <:AbstractOperator{:primal_and_gradient}},
    Tuple{<:AbstractOperator{:primals_and_gradients_and_hessians!}, <:AbstractOperator{:primal_and_gradient_and_hessian}},
}}
    fback = which_fback(prep)
    rtype = res_type(Val(true), func, fback)
    @assert rtype <: Tuple
    for i = 1 : dim_out(func)
        res = operate!((), func, fback, (x, i, params))
        if !(res isa rtype)
            return StopObject(res, (;op, prep))
        end
        for (trgt_arr, res_val) = zip(trgt_tup, res)
            copyto!(selectfirstdim(trgt_arr, i), res_val)
        end
    end
    return nothing
end

# Now, we would like to compute a vector-valued in-place operator by using a 
# single-output in-place operator.
# E.g., `gradients!(Dy, func, x, params)` with 
# `Dyi = gradient(func, x, i, params)` for all output indices.
# We have to be careful with operator pairs, where the number of in-place targets of `op`
# does not match the number of in-place targets of `fback`, e.g.
# `primals_and_gradients!(y, Dy, ...)` and `yi = primal_and_gradient!(Dyi, ...)`.
#
# Easy case, matching number of arguments:
function prepped_compute!(
    trgt_tup, prep::Prep{pair_Type}, func, op, (x, params)
) where {pair_Type<:Union{
    Tuple{<:AbstractOperator{:gradients!}, <:AbstractOperator{:gradient!}},
    Tuple{<:AbstractOperator{:hessians!}, <:AbstractOperator{:hessian!}},
}}
    fback = which_fback(prep)
    rtype = res_type(Val(true), func, fback)
    @assert rtype <: Nothing
    for i = 1 : dim_out(func)
        trgts = selectfirstdim(trgt_tup, i)
        res = operate!(trgts, func, fback, (x, i, params))
        if !(res isa rtype)
            return StopObject(res, (;op, prep))
        end
        ## no sync needed, `trgts` has views which are modified by `operate!`
    end
    return nothing
end
# Advanced case, in-place `fback` returns a number for position `i` in `first(trgt_tup)`:
function prepped_compute!(
    trgt_tup, prep::Prep{pair_Type}, func, op, (x, params)
) where {pair_Type<:Union{
    Tuple{<:AbstractOperator{:primals!}, <:AbstractOperator{:primal!}},
    Tuple{<:AbstractOperator{:primals_and_gradients!}, <:AbstractOperator{:primal_and_gradient!}},
    Tuple{<:AbstractOperator{:primals_and_gradients_and_hessians!}, <:AbstractOperator{:primal_and_gradient_and_hessian!}},
}}
    fback = which_fback(prep)
    rtype = res_type(Val(true), func, fback)
    @assert rtype <: Tuple{<:Real}
    y = first(trgt_tup)
    _trgt_tup = Base.tail(trgt_tup)
    for i = 1 : dim_out(func)
        trgts = selectfirstdim(_trgt_tup, i)
        res = operate!(trgts, func, fback, (x, i, params))
        if !(res isa rtype)
            return StopObject(res, (;op, prep))
        end
        ## sync needed for primal values only, other arrays are mutated by `operate!`
        y[i] = only(res)
    end
    return nothing
end
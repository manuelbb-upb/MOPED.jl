# Compute vector-valued out-of-place operators indirectly, using some other operator
# as fallback.

# ## First Case
# `y = primals(func, ...)` ⇔ `primals!(y, func, ...)`
# `Dy = gradients(func, ...)` ⇔ `gradients!(Dy, func, ...)`
# `y, Dy = primals_and_gradients(func, ...)` ⇔ `primals_and_gradients!(y, Dy, func, ...)`
# We need to allocate the fallback in-place targets:
function __prepare_fback(
    (op, fback)::pair_Type,
    ctx, trgt_tup, func, (x, params)
) where {pair_Type<:Union{
    Tuple{<:AbstractOperator{:primals}, <:AbstractOperator{:primals!}},
    Tuple{<:AbstractOperator{:gradients}, <:AbstractOperator{:gradients!}},
    Tuple{<:AbstractOperator{:hessians}, <:AbstractOperator{:hessians!}},
    Tuple{<:AbstractOperator{:primals_and_gradients}, <:AbstractOperator{:primals_and_gradients!}},
    Tuple{<:AbstractOperator{:primals_and_gradients_and_hessians}, <:AbstractOperator{:primals_and_gradients_and_hessians!}},
}}
    cache = inplace_arrays(func, fback, (x, params))
    @assert cache isa Tuple
    return Prep(; op, fback, cache, ctx)
end
# The `Prep` object now has the fallback targets in `cache`:
function prepped_compute!(
    trgt_tup, prep::Prep{pair_Type}, func, op, (x, params)
) where {pair_Type<:Union{
    Tuple{<:AbstractOperator{:primals}, <:AbstractOperator{:primals!}},
    Tuple{<:AbstractOperator{:gradients}, <:AbstractOperator{:gradients!}},
    Tuple{<:AbstractOperator{:hessians}, <:AbstractOperator{:hessians!}},
    Tuple{<:AbstractOperator{:primals_and_gradients}, <:AbstractOperator{:primals_and_gradients!}},
    Tuple{<:AbstractOperator{:primals_and_gradients_and_hessians}, <:AbstractOperator{:primals_and_gradients_and_hessians!}},
}}  
    fback = which_fback(prep)
    rtype = res_type(Val(true), func, fback)
    @assert rtype <: Nothing
    res = operate!(prep.cache, func, fback, (x, params))
    if !(res isa rtype)
        return StopObject(res, (;op, prep))
    end
    return finalize_arrays(prep, prep.cache)
end

# ## Second Case
# `y = primals(func, ...)` ⇔ `yi = primal(func, i, ...)`
# `Dy = gradients(func, ...)` ⇔ `Dyi = gradient(func, i, ...)`
# `y, Dy = primals_and_gradients(func, ...)` ⇔ `yi, Dyi = primal_and_gradient(func, i, ...)`
# We *could* use a cache and mutate; could lead to issues with autodiff.
# Instead, in `prepped_compute!`, we use a `StackView` on collected result tuples.
function __prepare_fback(
    (op, fback)::pair_Type,
    ctx, trgt_tup, func, (x, params)
) where {pair_Type<:Union{
    Tuple{<:AbstractOperator{:primals}, <:AbstractOperator{:primal}},
    Tuple{<:AbstractOperator{:gradients}, <:AbstractOperator{:gradient}},
    Tuple{<:AbstractOperator{:hessians}, <:AbstractOperator{:hessian}},
    Tuple{<:AbstractOperator{:primals_and_gradients}, <:AbstractOperator{:primal_and_gradient}},
    Tuple{<:AbstractOperator{:primals_and_gradients_and_hessians}, <:AbstractOperator{:primal_and_gradient_and_hessian}},
}}
    cache = nothing
    return Prep(; op, fback, cache, ctx)
end

function prepped_compute!(
    trgt_tup, prep::Prep{pair_Type}, func, op, (x, args...)
) where {pair_Type<:Union{
    Tuple{<:AbstractOperator{:primals}, <:AbstractOperator{:primal}},
    Tuple{<:AbstractOperator{:gradients}, <:AbstractOperator{:gradient}},
    Tuple{<:AbstractOperator{:hessians}, <:AbstractOperator{:hessian}},
    Tuple{<:AbstractOperator{:primals_and_gradients}, <:AbstractOperator{:primal_and_gradient}},
    Tuple{<:AbstractOperator{:primals_and_gradients_and_hessians}, <:AbstractOperator{:primal_and_gradient_and_hessian}},
    Tuple{<:AbstractOperator{:jac_vec_prod}, <:AbstractOperator{:grad_vec_prod}}
}} 
    fback = which_fback(prep)
    rtype = res_type(Val(true), func, fback)
    @assert rtype <: Tuple
    tups = ()
    res = operate!((), func, fback, (x, 1, args...))
    if !(res isa rtype)
        return StopObject(res, (;op, prep))
    end
    tups = push!!(tups, res)
    for i=2:dim_out(func)
        res = operate!((), func, fback, (x, i, args...))
        if !(res isa rtype)
            return StopObject(res, (;op, prep))
        end
        tups = push!!(tups, res)
    end
    return stack_tuple_of_tuples(tups)
end

function stack_tuple_of_tuples(tups)
    n_arrs = length(first(tups))
    @assert all(==(n_arrs), length.(tups))
    ret = ()
    for j=1:n_arrs
        slcs = (res -> res[j]).(tups)
        ret = push!!(ret, vcat_tuple(slcs))
    end
    return ret
end
function vcat_tuple(tup::Tuple{Vararg{<:Number}})
    return reduce(vcat, tup)
end
function vcat_tuple(slcs)
    return StackView(slcs, Val(1))
end

# ## Second Case
# ### Matching Number of Arguments
# `Dy = gradients(func, ...)` ⇔ `gradient!(Di, func, i, ...)`
# `Hy = hessians(func, ...)` ⇔ `hessian!(Di, func, i, ...)`
# ### More Out-Of-Place arguments
# `y = primals(func, ...)` ⇔ `yi = primal!(func, i, ...)`
# `y, Dy = primals_and_gradients(func, ...)` ⇔ `yi = primal_and_gradient!(Dyi, func, i, ...)`

function __prepare_fback(
    (op, fback)::pair_Type,
    ctx, trgt_tup, func, (x, params)
) where {pair_Type<:Union{
    Tuple{<:AbstractOperator{:primals}, <:AbstractOperator{:primal!}},
    Tuple{<:AbstractOperator{:gradients}, <:AbstractOperator{:gradient!}},
    Tuple{<:AbstractOperator{:hessians}, <:AbstractOperator{:hessian!}},
    Tuple{<:AbstractOperator{:primals_and_gradients}, <:AbstractOperator{:primal_and_gradient!}},
    Tuple{<:AbstractOperator{:primals_and_gradients_and_hessians}, <:AbstractOperator{:primal_and_gradient_and_hessian!}},
}}
    trgt_tups = tuple(
        (inplace_arrays(func, fback, (x, i, params)) for i=1:dim_out(func))...
    )
    stacked = stack_tuple_of_tuples(trgt_tups) # TODO for onetime use, this is very inefficient
    cache = (; trgt_tups, stacked)
    return Prep(; op, fback, cache, ctx)
end

function prepped_compute!(
    trgt_tup, prep::Prep{pair_Type}, func, op, (x, params)
) where {pair_Type<:Union{
    Tuple{<:AbstractOperator{:primals}, <:AbstractOperator{:primal!}},
    Tuple{<:AbstractOperator{:gradients}, <:AbstractOperator{:gradient!}},
    Tuple{<:AbstractOperator{:hessians}, <:AbstractOperator{:hessian!}},
    Tuple{<:AbstractOperator{:primals_and_gradients}, <:AbstractOperator{:primal_and_gradient!}},
    Tuple{<:AbstractOperator{:primals_and_gradients_and_hessians}, <:AbstractOperator{:primal_and_gradient_and_hessian!}},
}}
    fback = which_fback(prep)
    rtype = res_type(Val(true), func, fback)
    @assert rtype <: Nothing || rtype <: Tuple{<:Real}
    cache = prep.cache
    @unpack trgt_tups, stacked = cache
    lhs = ()
    for (i, trgt) = enumerate(trgt_tups)
        res = operate!(trgt, func, fback, (x, i, params))
        if !(res isa rtype)
            return StopObject(res, (;op, prep))
        end
        lhs = maybe_push!!(lhs, res)
    end
    r = finalize_arrays(prep, stacked)
    return (vcat_and_merge(lhs, r))
end
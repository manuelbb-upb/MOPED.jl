abstract type AbstractContext end

abstract type AbstractMethodCheckStrategy end
struct RuntimeMethodCheck <: AbstractMethodCheckStrategy end
struct IndicatorOnlyCheck <: AbstractMethodCheckStrategy end
struct PreferIndicatorCheck <: AbstractMethodCheckStrategy end

runtime_check(::AbstractMethodCheckStrategy)=Val(false)
runtime_check(::RuntimeMethodCheck)=Val(true)

Base.@kwdef @concrete struct DefaultContext <: AbstractContext
    is_onetime :: Union{Val{true}, Val{false}} = Val(true)
    method_check_strategy <: AbstractMethodCheckStrategy = PreferIndicatorCheck()
end
is_onetime(ctx::DefaultContext) = ctx.is_onetime
method_check_strategy(ctx::DefaultContext) = ctx.method_check_strategy
runtime_check(ctx::DefaultContext)=runtime_check(method_check_strategy(ctx))

@concrete struct FirstOrderContext <: AbstractContext
    ctx
end

function compute(
    func::AbstractNonlinearFunction,
    op::AbstractOperator, 
    op_args
)
    return compute!((), func, op, op_args)
end

function compute!(
    trgt_tup,
    func::AbstractNonlinearFunction,
    op::AbstractOperator,
    op_args
)
    return compute!(
        DefaultContext(), trgt_tup, func, op, op_args
    )
end

function compute!(
    ctx, trgt_tup, 
    func::AbstractNonlinearFunction, 
    op::AbstractOperator,
    op_args
)
    prep = prepare(ctx, trgt_tup, func, op, op_args)
    return prepped_compute!(trgt_tup, prep, func, op, op_args)
end

# Function to prepare a call to `prepped_compute!`.
# Arguments are passed to `_prepare` and -- using multiple dispatch --
# we check if the corresponding method is implemented or if we 
# can use fallbacks:
function prepare(
    ctx::DefaultContext, trgt_tup, func, op, op_args
)
    prep = _prepare(
        is_implemented(func, op, runtime_check(ctx)), ctx, trgt_tup, op, func, op_args
    )
    if ismissing(prep)
        if method_check_strategy(ctx) isa PreferIndicatorCheck
            @reset ctx.method_check_strategy = RuntimeMethodCheck()
            return prepare(ctx, trgt_tup, func, op, op_args)
        end
    end
    return prep
end

# If `op` is implemented, no preparation should be needed, return nothing:
function _prepare(
    op_impl::IsImplemented, ctx, trgt_tup, op, func, op_args
)
    return nothing
end

# Otherwise, start fallback chain to find suitable implemented method and allocate
# a preparation cache.
function _prepare(
    op_impl::NotImplemented, ctx, trgt_tup, op, func, op_args
)
    return _prepare_fback(
        (op, _next_fback(ctx, op)), ctx, trgt_tup, func, op_args
    )
end

_next_fback(ctx, op)=missing
_next_fback(ctx, op, failed_fback)=missing

function _prepare_fback(
    (op, fback), ctx, trgt_tup, func, op_args
)
    return _prepare_fback(
        is_implemented(func, fback, runtime_check(ctx)), (op, fback), ctx, trgt_tup, func, op_args)
end

function _prepare_fback(
    fback_impl::UndefImplemented, (op, fback), ctx, trgt_tup, func, op_args
)
    return missing
end
function _prepare_fback_undef_error(
    (op, fback), ctx, trgt_tup, func, op_args
)
    error("No suitable fallback defined/implemented for `$(op)`.")
end

function _prepare_fback(
    fback_impl::UndefImplemented, (op, fback), ctx::DefaultContext, trgt_tup, func, op_args
)
    if method_check_strategy(ctx) isa PreferIndicatorCheck
        @reset ctx.method_check_strategy = RuntimeMethodCheck()
        return prepare(ctx, trgt_tup, func, op, op_args)
    end
    return _prepare_fback_undef_error((op, fback), ctx, trgt_tup, func, op_args)
end

function _prepare_fback(
    fback_impl::NotImplemented, (op, fback), ctx, trgt_tup, func, op_args
)
    return _prepare_fback(
        (op, _next_fback(ctx, op, fback)), ctx, trgt_tup, func, op_args)
end

function _prepare_fback(
    fback_impl::IsImplemented, (op, fback), ctx, trgt_tup, func, op_args
)
    return __prepare_fback(
        (op, fback), ctx, trgt_tup, func, op_args
    )
end

function __prepare_fback(
    (op, fback), ctx, trgt_tup, func, op_args
)
    error("`$(fback)` seems like a suitable fallback for `$(op)`, but `__prepare_fback` is not specialized.")
end

# If `isnothing(prep)`, the default is to just call the operator with all arguments:
function prepped_compute!(
    trgt_tup, prep::Nothing, func, op, op_args
)
    @debug "No fallback needed for `$(op)`."
    operate!(trgt_tup, func, op, op_args)
end

# This should be specialized:
function prepped_compute!(
    trgt_tup, prep::Missing, func, op, op_args
)
    return prepped_compute!_error(
        trgt_tup, prep, func, op, op_args
    )
end

function prepped_compute!_error(
    trgt_tup, prep, func, op, op_args
)
    error("Call to `prepped_compute!` not implemented, prep=$(prep).")
end
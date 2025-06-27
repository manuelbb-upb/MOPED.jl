# # Interface 
# A nonlinear function should be defined with a type that subtypes `AbstractNonlinearFunction`:
abstract type AbstractNonlinearFunction end

# This package is aimed at vector-valued functions with vector input.
# Define the following methods to access dimension information:
function dim_in(::AbstractNonlinearFunction) end
function dim_out(::AbstractNonlinearFunction) end

# To implement the `AbstractNonlinearFunction` interface, it is necessary
# to specialize methods w.r.t. the type of `func<:AbstractNonlinearFunction`
# and w.r.t. the type of (basic) operators that are implemented.
abstract type AbstractOperator{op_symb} end

# ## (Recommended) Implementation Indicator
# When bridging, we try to check if a certain operator has been implemented.
# This is not totally reliable, and there is a runtime cost...
# You can manually specialize `implemented` to return `IsImplemented()` for 
# supported operators.

function implemented(::AbstractNonlinearFunction, ::AbstractOperator)
    return MaybeImplemented()
end

# For anything that is not a valid operator, we return `UndefImplemented()`.
# This should not be changed! 
function implemented(::AbstractNonlinearFunction, op)
    return UndefImplemented()
end

abstract type HasParametersTrait end
struct HasParameters <: HasParametersTrait end
struct NoParameters <: HasParametersTrait end
HasParametersTrait(::Type{<:AbstractNonlinearFunction})=HasParameters()

# ## Return Types
# We expect the out-of-place operators to return arrays, and in-place operators 
# to return `nothing` (most of the time).
# Other return value are allowed, but disrupt bridging.
# This is desired and can be exploited for early stopping and to propagate stop codes.
# To see what return type is expected, query `res_type(func, op)`.

# ## Interface

# ### Zeroth Order
# Define at least one of the following in-place or out-of-place methods by specializing
# on the `func` argument:
# #### Out-of-place
function primals(func, x, params...) end
function primal(func, x, i, params...) end
# #### In-place
function primals!(y, func, x, params...) end
function primal!(func, x, i, params...) end # TODO deprecate

# #### Pre-Allocation
# If you want to use some special vector type for the in-place-methods, override 
# `array_primals`:
function array_primals(func, x, params...)
    sz = (dim_out(func),)
    return similar(x, sz)
end

# ### First Order
# Define at least one of the following in-place or out-of-place methods by specializing
# on the `func` argument:
# ### Out-of-place
function gradients(func, x, params...) end
function gradient(func, x, i, params...) end

# #### In-place
function gradients!(Dy, func, x, params...) end
function gradient!(Dyi, func, x, i, params...) end

# #### Pre-Allocation
# If you want to use some special matrix type for the in-place-methods, override 
# `array_gradients:
function array_gradients(func, x, params...)
    sz = (dim_out(func), dim_in(func))
    return similar(x, sz)
end
# Likewise, the `array_gradient` method can be overriden
function array_gradient(func, x, i, params...)
    sz = (dim_in(func),)
    return similar(x, sz)
end

# ### Second Order
# Define at least one of the following in-place or out-of-place methods by specializing
# on the `func` argument:
# #### Out-of-place
function hessians(func, x, params...) end
function hessian(func, x, i, params...) end

# #### In-place
function hessians!(Hy, func, x, params...) end
function hessian!(Hyi, func, x, i, params...) end

# #### Pre-Allocation
# If you want to use some special tensor type for the in-place-methods, override 
# `array_hessians`:
function array_hessians(func, x, params...)
    sz = (dim_out(func), dim_in(func), dim_in(func))
    return similar(x, sz)
end
# Likewise, the `array_hessian` method can be overriden for the matrix case:
function array_hessian(func, x, i, params...)
    sz = (dim_in(func), dim_in(func))
    return similar(x, sz)
end

# ### Zeroth and First Order
# #### Out-of-place
function primals_and_gradients(func, x, params...) end
function primal_and_gradient(func, x, i, params...) end
# #### In-place
function primals_and_gradients!(y, Dy, func, x, params...) end
function primal_and_gradient!(Dyi, func, x, i, params...) end

# ## Zeroth and First and Second Order
# #### Out-of-place
function primals_and_gradients_and_hessians(func, x, params...) end
function primal_and_gradient_and_hessian(func, x, i, params...) end
# #### In-place
function primals_and_gradients_and_hessians!(y, Dy, Hy, func, x, params...) end
function primal_and_gradient_and_hessian!(Dyi, Hyi, func, x, i, params...) end

function jac_vec_prod(func, x, vecs, params...) end
function grad_vec_prod(func, x, i, vecs, params...) end
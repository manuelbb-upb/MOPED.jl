# Draft of Problem Interface

Problems must subtype `AbstractProblem`.

````julia
abstract type AbstractProblem end
````

The problem type should indicate what precision is required:

````julia
float_type(::Type{<:AbstractProblem})=Float64
````

We allow to also give objects instead of types:

````julia
float_type(mop::mop_Type) where {mop_Type<:AbstractProblem}=float_type(mop_Type)
````

Like with `float_type`, other properties are queried from the problem **type**, to
enable type-based bridging.
To this end, we have several attribute types:

````julia
abstract type AbstractAttribute end

struct Variables <: AbstractAttribute end
````

A supertype for something that can be evaluated:

````julia
abstract type AbstractFunctionQualifier <: AbstractAttribute end

struct Objectives <: AbstractFunctionQualifier end
struct LinEqConstraints <: AbstractFunctionQualifier end
struct LinIneqConstraints <: AbstractFunctionQualifier end
struct NonlinEqConstraints <: AbstractFunctionQualifier end
struct NonlinIneqConstraints <: AbstractFunctionQualifier end
````

## Dimension Information
**Mandatory**: Specialize `dimval` for `Variables` and other attributes as needed.
For `AbstractFunctionQualifier`s, this should return `Val{i}`, where `i` is the
integer dimension of output vectors.

````julia
function dimval(
    mop_Type::Type{<:AbstractProblem}, attr_Type::Type{<:AbstractAttribute})::Val
    error("`dimval` not applicable for `$(mop_Type)` and `$(attr_Type)`.")
end
dimval(::Type{<:AbstractProblem}, ::Type{<:AbstractFunctionQualifier})=Val(0)
````

We have some helpers that also take objects instead of types:

````julia
function dimval(_mop, _attr)
    dimval(_typeof(_mop), _typeof(_attr))
end
_typeof(T::Type)=error("Cannot extract concrete type from `$T`.")
_typeof(T::DataType)=T
_typeof(obj)=typeof(obj)
````

The integer dimension is returned by `dim`:

````julia
dim(mop, attr)=_extract_val(dimval(mop, attr))
_extract_val(::Val{i}) where {i} = i
````

## Variable Bounds

An attribute to indicate how variables are constrained:

````julia
abstract type AbstractVarBoundsAttr <: AbstractAttribute end
struct NoBounds <: AbstractVarBoundsAttr end
struct LowerBounds <: AbstractVarBoundsAttr end
struct UpperBounds <: AbstractVarBoundsAttr end
struct BoxBounds <: AbstractVarBoundsAttr end
````

**Suggested**: `var_bounds` defaults to `NoBounds()`; adapt as needed:

````julia
var_bounds(::Type{<:AbstractProblem}) = NoBounds()
var_bounds(mop::mop_Type) where mop_Type<:AbstractProblem=var_bounds(mop_Type)
````

If `var_bounds` returns `LowerBounds()` or `BoxBounds()`, then `lower_var_bounds`
should be implemented.

````julia
function lower_var_bounds(mop::AbstractProblem)
    return lower_var_bounds(var_bounds(mop), mop)
end
function lower_var_bounds(::Union{LowerBounds, BoxBounds}, mop)
    error("`lower_var_bounds` not implemented.")
end
````

Otherwise, we fall back to -∞:

````julia
function lower_var_bounds(var_bounds_attr, mop)
    return _bounds_vec(dimval(mop, Variables), float_type(mop), mop, -Inf)
end
````

Likewise, implement `upper_var_bounds` if needed:

````julia
function upper_var_bounds(mop::AbstractProblem)
    return upper_var_bounds(var_bounds(mop), mop)
end
function upper_var_bounds(::Union{UpperBounds, BoxBounds}, mop)
    error("`upper_var_bounds` not implemented.")
end
function upper_var_bounds(var_bounds_attr, mop)
    return _bounds_vec(dimval(mop, Variables), float_type(mop), mop, Inf)
end
@generated function _bounds_vec(::Val{nvars}, ::Type{F}, mop, val) where {nvars, F}
    return quote
        Fval = convert($F, val)
        fill(Fval, $nvars)
    end
end

const LinConstraints = Union{LinEqConstraints, LinIneqConstraints}
````

## Linear Constraints
If there are linear constraints, as indicated by `dimval`,
you are advised to implement `constraint_matrices`.
(There are bridges for 0-dim constraints and to fall back to `calc`).

````julia
function constraint_matrices(mop::AbstractProblem, attr::LinConstraints)
    # return (A, b) for constraints A ≤ b
    # return (E, c) for constraints E = c
    error("`constraint_matrices` not defined.")
end
````

## Evaluation
If there are evaluators (as indicated by `dimval`), implement `calc`.\
For linear inequality constraints ``A x ≤ b``, this should return the residual ``A - b``.\
For linear equality constraints ``E x = c``, this should return the residual ``E - c``.\
Nonlinear constraints take the form ``g(x) ≤ 0`` or ``h(x) = 0``.\
There are some bridges for 0-dim evaluators and to translate between matrices and vectors.

````julia
function calc(
    mop::AbstractProblem, attr::AbstractFunctionQualifier, x::AbstractVecOrMat
)
    error("`calc` not implemented.")
end
````

## Differentiation
Maybe implement the Jacobian:

````julia
function diff(
    mop::AbstractProblem, attr::AbstractFunctionQualifier, x::AbstractVector
)
    error("`diff` not implemented.")
end
````

## Internals
Are there user defined methods?

````julia
function is_implemented(
    @nospecialize(func::func_Type),
    @nospecialize(args_Type::Type{<:Tuple{mop_Type, Vararg}})
) where {
    mop_Type <: AbstractProblem,
    func_Type <: Function
}
    m = static_which(func, args_Type, Val(false))
    if isnothing(m)
        return NotImplemented()
    end
    if m.sig.parameters[2] >: AbstractProblem
        return NotImplemented()
    end
    return IsImplemented()
end
````

Is there an autodiff backend?

````julia
function _backend_Type(::Type{<:AbstractProblem})
    return Nothing
end
````

If `_backend_Type` not `Nothing`, implement `_backend` accordingly:

````julia
function _backend(::AbstractProblem)
    return nothing
end
````

Show some info:

````julia
function Base.show(io::IO, mop::AbstractProblem)
    mop_Type = typeof(mop)
    show_problem(io, mop)
    if !get(io, :compact, false)
        print(io, "\n\t| ")
        print(io, "Variables=$(dim(mop, Variables)), ")
        print(io, "VarBounds=$(var_bounds(mop_Type)), ")
        print(io, "Objectives=$(dim(mop, Objectives)), ")
        print(io, "\n\t| ")
        print(io, "LinEqConstraints=$(dim(mop, LinEqConstraints)), ")
        print(io, "LinIneqConstraints=$(dim(mop, LinIneqConstraints)), ")
        print(io, "NonlinEqConstraints=$(dim(mop, NonlinEqConstraints)), ")
        print(io, "NonlinIneqConstraints=$(dim(mop, NonlinIneqConstraints)) ")
    end
end

function show_problem(io::IO, prop::AbstractProblem)
    Base.show_default(io, prop)
end
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*


abstract type AbstractProblem end

float_type(::Type{<:AbstractProblem})=Float64

abstract type AbstractAttribute end

struct Variables <: AbstractAttribute end

abstract type AbstractFunctionQualifier <: AbstractAttribute end

struct Objectives <: AbstractFunctionQualifier end
struct LinEqConstraints <: AbstractFunctionQualifier end
struct LinIneqConstraints <: AbstractFunctionQualifier end
struct NonlinEqConstraints <: AbstractFunctionQualifier end
struct NonlinIneqConstraints <: AbstractFunctionQualifier end

# internal!!
const LinConstraints = Union{LinEqConstraints, LinIneqConstraints}
const NonlinConstraints = Union{NonlinEqConstraints, NonlinIneqConstraints}
const NonlinFunctions = Union{Objectives, NonlinConstraints}

# ## Dimension Information
function dimval(
    mop_Type::Type{<:AbstractProblem}, attr_Type::Type{<:AbstractAttribute})
    error("`dimval` not applicable for `$(mop_Type)` and `$(attr_Type)`.")
end
dimval(::Type{<:AbstractProblem}, ::Type{<:AbstractFunctionQualifier})=Val(0)

function dimval(_mop, _attr)
    dimval(_typeof(_mop), _typeof(_attr))
end
_typeof(T::Type)=error("Cannot extract concrete type from `$T`.")
_typeof(T::DataType)=T
_typeof(obj)=typeof(obj)

dim(mop, attr)=_extract_val(dimval(mop, attr))
_extract_val(::Val{i}) where {i} = i

float_type(mop::mop_Type) where {mop_Type<:AbstractProblem}=float_type(mop_Type)

# ## Variable Bounds
abstract type AbstractVarBoundsAttr <: AbstractAttribute end
struct NoBounds <: AbstractVarBoundsAttr end
struct LowerBounds <: AbstractVarBoundsAttr end
struct UpperBounds <: AbstractVarBoundsAttr end
struct BoxBounds <: AbstractVarBoundsAttr end

var_bounds(::Type{<:AbstractProblem}) = NoBounds()
var_bounds(mop::mop_Type) where mop_Type<:AbstractProblem=var_bounds(mop_Type)

function lower_var_bounds(mop::AbstractProblem)
    return lower_var_bounds(var_bounds(mop), mop)
end
function lower_var_bounds(::Union{LowerBounds, BoxBounds}, mop)
    error("`lower_var_bounds` not implemented.")
end
function lower_var_bounds(var_bounds_attr, mop)
    return _bounds_vec(dimval(mop, Variables), float_type(mop), mop, -Inf)
end
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

# ## Linear Constraints
function constraint_matrices(mop::AbstractProblem, attr::LinConstraints)   
    error("`constraint_matrices` not defined.")
end

## Evaluation
function calc(
    mop::AbstractProblem, attr::AbstractFunctionQualifier, x::AbstractVecOrMat
)
    error("`calc` not implemented.")
end

function is_implemented(
    @nospecialize(func::func_Type),
    @nospecialize(args_Type::Type{<:Tuple{mop_Type, Vararg}})
) where {
    mop_Type <: AbstractProblem,
    func_Type <: Union{
        typeof(constraint_matrices),
        typeof(calc),
    }
}
    m = static_which(func, args_Type)
    if m.sig.parameters[2] >: AbstractProblem
        return NotImplemented()
    end
    return IsImplemented()
end
import LinearAlgebra as LA
import ConcreteStructs: @concrete
import UnPack: @unpack

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
function dimval(mop_Type::Type{<:AbstractProblem}, attr::AbstractAttribute)
    error("`dimval` not applicable for `$(mop_Type)` and `$(typeof(attr))`.")
end
function dimval(mop_Type::Type{<:AbstractProblem}, attr::Variables)
    error("`dimval` not defined for `$(mop_Type)` and `$(typeof(attr))`.")
end
dimval(::Type{<:AbstractProblem}, ::AbstractFunctionQualifier)=Val(0)
dimval(::mop_Type, attr) where mop_Type<:AbstractProblem=dimval(mop_Type, attr)

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
    return _bounds_vec(mop, -Inf)
end
function upper_var_bounds(mop::AbstractProblem)
    return upper_var_bounds(var_bounds(mop), mop)
end
function upper_var_bounds(::Union{UpperBounds, BoxBounds}, mop)
    error("`upper_var_bounds` not implemented.")
end
function upper_var_bounds(var_bounds_attr, mop)
    return _bounds_vec(mop, Inf)
end
function _bounds_vec(mop, val)
    nvars = dim(mop, Variables())
    F = float_type(mop)
    Fval = convert(F, val)
    return fill(Fval, nvars)
end
  
# ## Linear Constraints
include("generic_fallbacks.jl")
struct CMatsTrgt end
error_str(::CMatsTrgt)="constraint_matrices"
@concrete struct CalcTrgt
    attr
    x
end
error_str(::CalcTrgt)="calc"
# Overwrite this:
function _constraint_matrices(mop::AbstractProblem, attr::LinConstraints)   
    return UnsuccessfulFBack() 
end

# This becomes available:
function constraint_matrices(mop::AbstractProblem, attr::LinConstraints)
    callee = CMatsTrgt()
    return constraint_matrices(mop, attr, callee)
end

function constraint_matrices(mop, attr, callee)
    trgt = CMatsTrgt()
    return generic_fbackchain(trgt, callee, mop, attr)
end

function all_fbacks(trgt::CMatsTrgt, callee)
    return (
        Val{:_constraint_matrices}(),
        Val{:_cmats_zero_dim}(),
        Val{:_cmats_from_calc}()
    )
end

# Implementation of "fallback" to `constraint_matrices`;
# In case `callee==ctx`, this will be skipped.
# Only relevant for bridges depending on constraint matrices:
function generic_fback(
    trgt::CMatsTrgt, callee, ctx::Type{Val{:constraint_matrices}},
    mop, attr
)
    mats = _constraint_matrices(mop, attr)
    if mats isa Tuple{<:AbstractMatrix, <:AbstractVector}
        return mats
    end
    return mats
end

## Implementation of fallback for zero-dim constraints:
function generic_fback(
    trgt::CMatsTrgt, callee, ctx::Type{Val{:_cmats_zero_dim}},
    mop, attr 
)
    @info "_cmats_zero_dim"
    return _cmats_zero_dim(mop, attr)
end
function _cmats_zero_dim(mop, attr)
    _cmats_zero_dim(dimval(mop, attr), mop, attr)
end
_cmats_zero_dim(attr_dimval, mop, attr)=nothing
function _cmats_zero_dim(::Val{0}, mop, attr)
    ## NOTE this is one of some functions I have tried to make `@generated`
    #       I didn't know that generated functions do not take method specializations into
    #       account. E.g., `dim` call will error, even if user defines specialization.
    nvars = dim(mop, Variables())
    F = float_type(mop)
    A = Matrix{F}(undef, 0, nvars)
    b = Vector{F}(undef, 0)
    return (A, b)
end

function generic_fback(
    trgt::CMatsTrgt, callee, ctx::Type{Val{:_cmats_from_calc}},
    mop, attr 
)
    return _cmats_from_calc(mop, attr, ctx)
end

function _cmats_from_calc(mop, attr, ctx)
    F = float_type(mop)
    nvars = dim(mop, Variables())
    z = zeros(F, nvars)
    ## sub-call to `calc` with special callee
    b = calc(mop, attr, z, ctx)
    return __cmats_from_calc(b, mop, attr, ctx)
end
__cmats_from_calc(b, mop, attr, ctx)=b
function __cmats_from_calc(b::AbstractVector{F}, mop, attr, ctx) where F
    b .*= -1
    nvars = dim(mop, Variables())
    X = Matrix{F}(LA.I(nvars))
    Y = calc(mop, attr, X, ctx)
    # A * X - b = Y ⇔ A = (Y + b) X⁻¹ 
    A = Y .+ b
    return (A, b)
end
## Prevent infinite recursion / StackOverflow by disabling this bridge in sub-sub-call
## from within `calc`:

function check_compat(
    trgt::CMatsTrgt, callee::Type{Val{:_calc_from_cmats}}, ctx::Type{Val{:_cmats_from_calc}}
)
    return Val(false)
end

## Evaluation

## Overwrite this:
function _calc(mop::AbstractProblem, attr::AbstractFunctionQualifier, x::AbstractVecOrMat)
    return UnsuccessfulFBack()
end
## This becomes available:
function calc(
    mop::AbstractProblem, attr::AbstractFunctionQualifier, x::AbstractVecOrMat
)
    callee = CalcTrgt(attr, x)
    return calc(mop, attr, x, callee)
end

function calc(mop, attr, x, callee)
    trgt = CalcTrgt(attr, x) 
    return generic_fbackchain(trgt, callee, mop, attr)
end

function all_fbacks(
    trgt::CalcTrgt, callee
) 
    return (
        Val{:_calc}(),
        Val{:_calc_zero_dim}(),
        Val{:_calc_vec2mat}(),
        Val{:_calc_mat2vecs}(),
        Val{:_calc_from_cmats}()
    )
end

function generic_fback(
    trgt::CalcTrgt, callee, ctx::Type{Val{:_calc}},
    mop, attr
)
    y = _calc(mop, attr, trgt.x)
    if y isa AbstractArray
        return y 
    end
    return y 
end

function generic_fback(
    trgt::CalcTrgt, callee, ctx::Type{Val{:_calc_zero_dim}},
    mop, attr
)
    return _calc_zero_dim(dimval(mop, attr), mop, attr, trgt.x)
end
_calc_zero_dim(attr_dimval, mop, attr, x)=UnsuccessfulFBack()
function _calc_zero_dim(::Val{0}, mop, attr, x::AbstractVector)
    return Vector{float_type(mop)}(undef, 0)
end
function _calc_zero_dim(::Val{0}, mop, attr, X::AbstractMatrix)
    return Matrix{float_type(mop)}(undef, 0, size(X, 2))
end

function generic_fback(
    trgt::CalcTrgt, callee, ctx::Type{Val{:_calc_vec2mat}},
    mop, attr
)
    return _calc_vec2mat(mop, attr, trgt.x)
end
_calc_vec2mat(mop, attr, x)=UnsuccessfulFBack()
function _calc_vec2mat(mop, attr, x::AbstractVector)
    y = _calc(mop, attr, reshape(x, :, 1))
    if y isa AbstractMatrix
        return vec(y)
    end
    return y
end

function generic_fback(
    trgt::CalcTrgt, callee, ctx::Type{Val{:_calc_mat2vecs}},
    mop, attr
)
    return _calc_mat2vecs(mop, attr, trgt.x)
end
_calc_mat2vecs(mop, attr, x)=UnsuccessfulFBack()
function _calc_mat2vecs(mop, attr, X::AbstractMatrix)
    x1 = @view(X[:, 1])
    y1 = _calc(mop, attr, x1)
    return __calc_mat2vecs(y1, mop, attr, X)
end
__calc_mat2vecs(y1, mop, attr, X)=y1
function __calc_mat2vecs(y1::AbstractVector{F}, mop, attr, X) where F
    Y = hcat(
        y1, 
        reduce(
            hcat,
            _calc(mop, attr, x) for x = Iterators.drop(eachcol(X), 1)
        )
    ) :: AbstractMatrix{F}
    return Y
end

function generic_fback(
    trgt::CalcTrgt, callee, ctx::Type{Val{:_calc_from_cmats}}, mop, attr)
    return UnsuccessfulFBack()
end
function generic_fback(
    trgt::CalcTrgt, callee, ctx::Val{:_calc_from_cmats}, mop, attr::LinConstraints
)
    mats = constraint_matrices(mop, attr, ctx)
    return __calc_from_cmats(mats, trgt.x)
end
__calc_from_cmats(mats, x)=mats
function __calc_from_cmats((A,b)::Tuple{<:AbstractMatrix,<:AbstractVector},x)
    y = A * x .- b
    return y
end

function check_compat(
    trgt::CalcTrgt, callee::Type{Val{:_cmats_from_calc}}, ctx::Type{Val{:_calc_from_cmats}}
)
    return Val(false)
end
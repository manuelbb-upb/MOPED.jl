abstract type AbstractCMatsBridge <: AbstractBridge end
abstract type AbstractCMatsPreparation <: AbstractPreparation end

struct CMatsZeroDimBridge <: AbstractCMatsBridge end

@concrete struct CMatsZeroDimPrep{ft, n_vars} <: AbstractCMatsPreparation end

function is_implemented(
    bridge::CMatsZeroDimBridge, 
    ::typeof(constraint_matrices),
    args_Type::Type{<:Tuple{mop_Type, attr_Type}}
) where {mop_Type<:AbstractProblem, attr_Type <: LinConstraints}
    if dimval(mop_Type, attr_Type) isa Val{0}
        return IsImplemented()
    end
    return NotImplemented()
end

function prep(
    bridge::CMatsZeroDimBridge, 
    bw::BridgedWrapper, 
    ::typeof(constraint_matrices),
    args_Type::Type{<:Tuple{mop_Type, attr_Type}}
) where {mop_Type<:AbstractProblem, attr_Type <: LinConstraints}
    @info "Prepping CMatsZeroDimBridge"
    return CMatsZeroDimPrep{
        float_type(mop_Type),
        dim(mop_Type, Variables)
    }()
end

@generated function compute!(
    ::CMatsZeroDimPrep{ft, n_vars}, 
    @nospecialize(mop::AbstractProblem), @nospecialize(attr::LinConstraints)
) where {ft, n_vars}
    A = Matrix{ft}(undef, 0, n_vars)
    b = Vector{ft}(undef, 0)
    return :($A, $b)
end

struct CMatsCalcBridge <: AbstractCMatsBridge end
@concrete struct CMatsCalcPrep <: AbstractCMatsPreparation
    prep_vec
    prep_mat
    # TODO cache
end

function bridging_cost(
    bridge::CMatsCalcBridge, 
    ::typeof(constraint_matrices),
    args_Type::Type{<:Tuple{mop_Type, attr_Type}}
) where {mop_Type<:AbstractProblem, attr_Type <: LinConstraints}
    2.0
end

function is_implemented(
    bridge::CMatsCalcBridge, 
    ::typeof(constraint_matrices),
    args_Type::Type{<:Tuple{mop_Type, attr_Type}}
) where {mop_Type<:AbstractProblem, attr_Type <: LinConstraints}
    return IsImplemented()
end

function required_funcs_with_argtypes(
    bridge::CMatsCalcBridge, 
    ::typeof(constraint_matrices),
    args_Type::Type{<:Tuple{mop_Type, attr_Type}}
) where {mop_Type<:AbstractProblem, attr_Type <: LinConstraints}
    return (
        (calc, Tuple{mop_Type, attr_Type, AbstractVector}),
        (calc, Tuple{mop_Type, attr_Type, AbstractMatrix})
    )
end

function prep(
    bridge::CMatsCalcBridge, 
    bw::BridgedWrapper, 
    ::typeof(constraint_matrices),
    args_Type::Type{<:Tuple{mop_Type, attr_Type}}
) where {mop_Type<:AbstractProblem, attr_Type <: LinConstraints}
    prep_vec = prep(bw, calc, Tuple{mop_Type, attr_Type, AbstractVector})
    prep_mat = prep(bw, calc, Tuple{mop_Type, attr_Type, AbstractMatrix})
    return CMatsCalcPrep(
        #float_type(mop_Type), dimval(mop_Type, Variables()), 
        prep_vec, prep_mat
    )
end

function compute!(
    p::CMatsCalcPrep, mop::AbstractProblem, attr::LinConstraints
)
    F = float_type(mop)
    n_vars = dim(mop, n_vars)
    z = zeros(F, n_vars)
    b = compute!(p.prep_vec, (mop, attr, z))
    I = Matrix{F}(LA.I(n_vars))
    A = compute!(p.prep_mat, (mop, attr, I))
    A .+= b
    return (A, b)
end

abstract type AbstractCalcBridge <: AbstractBridge end
abstract type AbstractCalcPrep <: AbstractPreparation end

struct CalcZeroDimBridge <: AbstractCalcBridge end
@concrete struct CalcZeroDimPrep <: AbstractCalcBridge end

function is_implemented(
    bridge::CalcZeroDimBridge,
    ::typeof(calc),
     args_Type::Type{<:Tuple{mop_Type, attr_Type, x_Type}}
) where {
    mop_Type<:AbstractProblem,
    attr_Type<:AbstractFunctionQualifier,
    x_Type<:AbstractVecOrMat
}
    if dimval(mop_Type, attr_Type) <: Val{0}
        return IsImplemented()
    end
    return NotImplemented()
end

function bridging_cost(
    bridge::CalcZeroDimBridge,
    ::typeof(calc),
    args_Type::Type{<:Tuple{mop_Type, attr_Type, x_Type}}
) where {
    mop_Type<:AbstractProblem,
    attr_Type<:AbstractFunctionQualifier,
    x_Type<:AbstractVecOrMat
}
    return 0.5
end

function prep(
    bridge::CalcZeroDimBridge,
    bw::BridgedWrapper,
    ::typeof(calc),
    args_Type::Type{<:Tuple{mop_Type, attr_Type, x_Type}}
) where {
    mop_Type<:AbstractProblem,
    attr_Type<:AbstractFunctionQualifier,
    x_Type<:AbstractVecOrMat
}
    return CalcZeroDimPrep(float_type(mop_Type), dimval(mop_Type, attr_Type))
end

function compute!(
    p::CalcZeroDimPrep,
    mop::AbstractProblem, attr::AbstractFunctionQualifier, x::AbstractVecOrMat
)
    return _zerodim_array(x)
end
_zerodim_array(x::AbstractVector)=similar(x, 0)
_zerodim_array(X::AbstractArray)=similar(X, (0, size(X)[2:end]...))

struct CalcInputVecAsMatBridge <: AbstractCalcBridge end
@concrete struct CalcInputVecAsMatPrep <: AbstractCalcPrep 
    prep_mat
end

function is_implemented(
    bridge::CalcInputVecAsMatBridge,
    ::typeof(calc),
    args_Type::Type{<:Tuple{mop_Type, attr_Type, x_Type}}
) where {
    mop_Type<:AbstractProblem,
    attr_Type<:AbstractFunctionQualifier,
    x_Type<:AbstractVector
}
    return Implemented()
end

function required_funcs_with_argtypes(
    bridge::CalcInputVecAsMatBridge,
    ::typeof(calc),
     args_Type::Type{<:Tuple{mop_Type, attr_Type, x_Type}}
) where {
    mop_Type<:AbstractProblem,
    attr_Type<:AbstractFunctionQualifier,
    x_Type<:AbstractVector
}
    return (
        (calc, Tuple{mop_Type, attr_Type, AbstractMatrix{eltype(x_Type)}}),
    )
end

function prep(
    bridge::CalcInputVecAsMatBridge,
    bw::BridgedWrapper, 
    ::typeof(calc),
    args_Type::Type{<:Tuple{mop_Type, attr_Type, x_Type}}
) where {
    mop_Type<:AbstractProblem,
    attr_Type<:AbstractFunctionQualifier,
    x_Type<:AbstractVector
}
    return CalcInputVecAsMatPrep(
        prep(bw, calc, Tuple{mop_Type, attr_Type, AbstractMatrix{eltype(x_Type)}})
    )
end

function compute!(
    p::CalcInputVecAsMatPrep, 
    mop::AbstractProblem, attr::AbstractFunctionQualifier, x::AbstractVector
)
    X = reshape(x, :, 1)
    return compute!(p.prep_mat, mop, attr, X)
end

struct CalcInputMatAsVecsBridge <: AbstractCalcBridge end
@concrete struct CalcInputMatAsVecsPrep <: AbstractCalcPrep 
    prep_vecs
end

function is_implemented(
    bridge::CalcInputMatAsVecsBridge,
    ::typeof(calc),
    args_Type::Type{<:Tuple{mop_Type, attr_Type, x_Type}}
) where {
    mop_Type<:AbstractProblem,
    attr_Type<:AbstractFunctionQualifier,
    x_Type<:AbstractMatrix
}
    return Implemented()
end

function required_funcs_with_argtypes(
    bridge::CalcInputMatAsVecsBridge,
    ::typeof(calc),
    args_Type::Type{<:Tuple{mop_Type, attr_Type, x_Type}}
) where {
    mop_Type<:AbstractProblem,
    attr_Type<:AbstractFunctionQualifier,
    x_Type<:AbstractMatrix
}
    return (
        (calc, Tuple{mop_Type, attr_Type, AbstractVector{eltype(x_Type)}}),
    )
end

function prep(
    bridge::CalcInputMatAsVecsBridge,
    bw::BridgedWrapper, 
    ::typeof(calc),
    args_Type::Type{<:Tuple{mop_Type, attr_Type, x_Type}}
) where {
    mop_Type<:AbstractProblem,
    attr_Type<:AbstractFunctionQualifier,
    x_Type<:AbstractMatrix
}
    return CalcInputMatAsVecsPrep(
        prep(bw, calc, Tuple{mop_Type, attr_Type, AbstractVector{eltype(x_Type)}})
    )
end

function compute!(
    p::CalcInputMatAsVecsPrep,
    mop::AbstractProblem, attr::AbstractFunctionQualifier, X::AbstractMatrix
)
    @unpack prep_vecs = p
    return reduce(
        hcat,
        compute!(prep_vecs, mop, attr, x) for x = eachcol(X)
    )
end

struct CalcCMatsBridge <: AbstractCalcBridge end
@concrete struct CalcCMatsPrep <: AbstractCalcPrep 
    prep_cmats
end

function is_implemented(
    bridge::CalcCMatsBridge,
    ::typeof(calc),
    args_Type::Type{<:Tuple{mop_Type, attr_Type, x_Type}}
) where {
    mop_Type<:AbstractProblem,
    attr_Type<:LinConstraints,
    x_Type<:AbstractVecOrMat
}
    return Implemented()
end

function required_funcs_with_argtypes(
    bridge::CalcCMatsBridge,
    ::typeof(calc),
    args_Type::Type{<:Tuple{mop_Type, attr_Type, x_Type}}
) where {
    mop_Type<:AbstractProblem,
    attr_Type<:LinConstraints,
    x_Type<:AbstractVecOrMat
}
    return (
        (constraint_matrices, Tuple{mop_Type, attr_Type}),
    )
end

function prep(
    bridge::CalcCMatsBridge,
    bw::BridgedWrapper, 
    ::typeof(calc),
    args_Type::Type{<:Tuple{mop_Type, attr_Type, x_Type}}
) where {
    mop_Type<:AbstractProblem,
    attr_Type<:LinConstraints,
    x_Type<:AbstractVecOrMat
}
    return CalcCMatsBridge(
        prep(bw, constraint_matrices, Tuple{mop_Type, attr_Type})
    )
end

function compute!(
    p::CalcCMatsPrep,
    mop::AbstractProblem, attr::LinConstraints, x::AbstractVecOrMat
)
    A, b = compute!(p.prep_cmats, mop, attr)
    return A * x .- b
end

all_bridges(::Type{<:AbstractProblem})=(
    CMatsZeroDimBridge(),
    CMatsCalcBridge(),
    CalcZeroDimBridge(),
    CalcInputVecAsMatBridge(),
    CalcInputMatAsVecsBridge(),
    CalcCMatsBridge()
)

import OrderedCollections: freeze

@concrete struct BridgedProblem1 <: AbstractProblem
    mop
    preps
end

BridgedProblem = BridgedProblem1

function dimval(BP::Type{<:BridgedProblem}, attr_Type::Type{<:AbstractAttribute})
    return dimval(_inner_problem_type(BP), attr_Type)
end

_inner_problem_type(::Type{<:BridgedProblem{mop_Type}}) where mop_Type=mop_Type

function make_bridged(mop::AbstractProblem)
    preps = _prep_preps(mop)
    return BridgedProblem(mop, preps)    
end

# TODO turn into generated once done debugging
function _prep_preps(mop::AbstractProblem)
    #mop_Type = mop  # this is a generated function
    @show mop_Type = typeof(mop)
    bw = BridgedWrapper(mop_Type)
    _preps = Dict()
    for (func, args_Type) in (
        (constraint_matrices, Tuple{mop_Type, LinEqConstraints}),
        (constraint_matrices, Tuple{mop_Type, LinIneqConstraints}),
        (calc, Tuple{mop_Type, Objectives, AbstractVector}),
        (calc, Tuple{mop_Type, LinEqConstraints, AbstractVector}),
        (calc, Tuple{mop_Type, LinIneqConstraints, AbstractVector}),
        (calc, Tuple{mop_Type, NonlinEqConstraints, AbstractVector}),
        (calc, Tuple{mop_Type, NonlinIneqConstraints, AbstractVector}),
        (calc, Tuple{mop_Type, Objectives, AbstractMatrix}),
        (calc, Tuple{mop_Type, LinEqConstraints, AbstractMatrix}),
        (calc, Tuple{mop_Type, LinIneqConstraints, AbstractMatrix}),
        (calc, Tuple{mop_Type, NonlinEqConstraints, AbstractMatrix}),
        (calc, Tuple{mop_Type, NonlinIneqConstraints, AbstractMatrix}),
    )
        @show p = prep(bw, func, args_Type)
        _insert!(_preps, func, args_Type, p)
    end
    preps = freeze(_preps)
    return preps
end

function constraint_matrices(bmop::BridgedProblem, attr::LinConstraints)
    @unpack mop, preps = bmop
    mop_Type = typeof(mop)
    p = _get(preps, constraint_matrices, Tuple{mop_Type, typeof(attr)}, nothing)
    isnothing(p) && error("No prep object found for `$(constraint_matrices)`")
    return compute!(p, mop, attr)
end
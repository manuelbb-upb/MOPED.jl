abstract type AbstractCMatsBridge <: AbstractBridge end
abstract type AbstractCMatsPreparation <: AbstractPreparation end

struct CMatsZeroDimBridge <: AbstractCMatsBridge end

@concrete struct CMatsZeroDimPrep{ft, n_vars} <: AbstractCMatsPreparation end

@nospecialize
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
    mop::mop_Type, attr::attr_Type
) where {mop_Type<:AbstractProblem, attr_Type <: LinConstraints}
    return CMatsZeroDimPrep{
        float_type(mop_Type),
        dim(mop_Type, Variables)
    }()
end
@specialize

@generated function compute!(
    ::CMatsZeroDimPrep{ft, n_vars}, 
    @nospecialize(mop::AbstractProblem), 
    @nospecialize(attr::LinConstraints)
) where {ft, n_vars}
    A = Matrix{ft}(undef, 0, n_vars)
    b = Vector{ft}(undef, 0)
    return :($A, $b)
end

struct CMatsCalcBridge <: AbstractCMatsBridge end
struct CMatsCalcPrep{F<:AbstractFloat} <: AbstractCMatsPreparation
    A :: Matrix{F}
    b :: Vector{F}
end

@nospecialize
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
@specialize

function prep(
    bridge::CMatsCalcBridge, 
    bw::BridgedWrapper, 
    ::typeof(constraint_matrices),
    mop::mop_Type, attr::attr_Type
) where {mop_Type<:AbstractProblem, attr_Type <: LinConstraints}
    F = float_type(mop_Type)
    n_vars = dim(mop_Type, Variables)
    z = zeros(F, n_vars)
    b = bridged_compute!(bw, calc, mop, attr, z)
    I = Matrix{F}(LA.I(n_vars))
    A = bridged_compute!(bw, calc, mop, attr, I)
    A .+= b
    return CMatsCalcPrep{F}(A, b)
end

function compute!(
    p::CMatsCalcPrep, mop::AbstractProblem, attr::LinConstraints
)
    @unpack A, b = p
    return (A, b)
end

abstract type AbstractCalcBridge <: AbstractBridge end
abstract type AbstractCalcPrep <: AbstractPreparation end

struct CalcZeroDimBridge <: AbstractCalcBridge end
@concrete struct CalcZeroDimPrep{F} <: AbstractCalcPrep end

@nospecialize
function is_implemented(
    bridge::CalcZeroDimBridge,
    ::typeof(calc),
    args_Type::Type{<:Tuple{mop_Type, attr_Type, x_Type}}
) where {
    mop_Type<:AbstractProblem,
    attr_Type<:AbstractFunctionQualifier,
    x_Type<:AbstractVecOrMat
}
    if dimval(mop_Type, attr_Type) isa Val{0}
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
    mop::mop_Type, attr::attr_Type, x::x_Type
) where {
    mop_Type<:AbstractProblem,
    attr_Type<:AbstractFunctionQualifier,
    x_Type<:AbstractVecOrMat
}
    return CalcZeroDimPrep{float_type(mop_Type)}()
end
@specialize

function compute!(
    p::CalcZeroDimPrep{F},
    mop::AbstractProblem, attr::AbstractFunctionQualifier, x::AbstractVecOrMat
) where F
    return _zerodim_array(F, x)
end
_zerodim_array(F::DataType, x::AbstractVector)=Vector{F}()
_zerodim_array(F::DataType, X::AbstractArray)=Matrix{F}(undef, (0, size(X)[2:end]...))

struct CalcInputVecAsMatBridge <: AbstractCalcBridge end
@concrete struct CalcInputVecAsMatPrep <: AbstractCalcPrep 
    prep_mat
end

@nospecialize
function is_implemented(
    bridge::CalcInputVecAsMatBridge,
    ::typeof(calc),
    args_Type::Type{<:Tuple{mop_Type, attr_Type, x_Type}}
) where {
    mop_Type<:AbstractProblem,
    attr_Type<:AbstractFunctionQualifier,
    x_Type<:AbstractVector
}
    return IsImplemented()
end
@specialize

function required_funcs_with_argtypes(
    bridge::CalcInputVecAsMatBridge,
    ::typeof(calc),
    args_Type::Type{Tuple{mop_Type, attr_Type, x_Type}}
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
    mop::mop_Type, attr::attr_Type, x::x_Type
) where {
    mop_Type<:AbstractProblem,
    attr_Type<:AbstractFunctionQualifier,
    x_Type<:AbstractVector
}
    X = reshape(x, :, 1)
    return CalcInputVecAsMatPrep(
        prep(bw, calc, mop, attr, X) 
    )
end

function compute!(
    p::CalcInputVecAsMatPrep, 
    mop::AbstractProblem, attr::AbstractFunctionQualifier, x::AbstractVector
)
    ## `x` might be different here than in `prep`
    X = reshape(x, :, 1)
    return compute!(p.prep_mat, mop, attr, X)
end

struct CalcInputMatAsVecsBridge <: AbstractCalcBridge end
@concrete struct CalcInputMatAsVecsPrep <: AbstractCalcPrep 
    prep_vecs
end

@nospecialize
function is_implemented(
    bridge::CalcInputMatAsVecsBridge,
    ::typeof(calc),
    args_Type::Type{<:Tuple{mop_Type, attr_Type, x_Type}}
) where {
    mop_Type<:AbstractProblem,
    attr_Type<:AbstractFunctionQualifier,
    x_Type<:AbstractMatrix
}
    return IsImplemented()
end
@specialize

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
        #(calc, Tuple{mop_Type, attr_Type, AbstractVector}),
    )
end

function prep(
    bridge::CalcInputMatAsVecsBridge,
    bw::BridgedWrapper, 
    ::typeof(calc),
    mop::mop_Type, attr::attr_Type, X::x_Type
) where {
    mop_Type<:AbstractProblem,
    attr_Type<:AbstractFunctionQualifier,
    x_Type<:AbstractMatrix
}
    x = first(eachcol(X))
    return CalcInputMatAsVecsPrep(
        prep(bw, calc, mop, attr, x) 
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
    A <: AbstractMatrix
    b <: AbstractVector
end

@nospecialize
function is_implemented(
    bridge::CalcCMatsBridge,
    ::typeof(calc),
    args_Type::Type{<:Tuple{mop_Type, attr_Type, x_Type}}
) where {
    mop_Type<:AbstractProblem,
    attr_Type<:LinConstraints,
    x_Type<:AbstractVecOrMat
}
    return IsImplemented()
end
@specialize

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
    mop::mop_Type, attr::attr_Type, X::x_Type
) where {
    mop_Type<:AbstractProblem,
    attr_Type<:LinConstraints,
    x_Type<:AbstractVecOrMat
}
    A, b = bridged_compute!(bw, constraint_matrices, mop, attr)
    return CalcCMatsPrep(A, b)
end

function compute!(
    p::CalcCMatsPrep,
    @nospecialize(mop::AbstractProblem), 
    @nospecialize(attr::LinConstraints), 
    x::AbstractVecOrMat
)
    @unpack A, b = p
    return A * x .- b
end
abstract type AbstractDiffBridge <: AbstractBridge end
abstract type AbstractDiffPrep <: AbstractPreparation end

struct DiffCalcBridge <: AbstractDiffBridge end
@concrete terse struct DiffCalcPrep <: AbstractDiffPrep
    cc <: Function
    bcknd
    di_prep
end

@nospecialize
function is_implemented(
    ::DiffCalcBridge,
    ::typeof(diff),
    args_Type::Type{<:Tuple{mop_Type, attr_Type, x_Type}}
) where {mop_Type<:AbstractProblem, attr_Type<:AbstractFunctionQualifier, x_Type<:AbstractVector}
    if _backend_Type(mop_Type) <: AbstractADType
        return IsImplemented()
    end
    return NotImplemented()
end

function bridging_cost(
    ::DiffCalcBridge,
    ::typeof(diff),
    args_Type::Type{<:Tuple{mop_Type, attr_Type, x_Type}}
) where {mop_Type<:AbstractProblem, attr_Type<:AbstractFunctionQualifier, x_Type<:AbstractVector}
    2.0     # TODO this should be bigger than 1.0 (I think), but needs tuning probably
end
@specialize

function required_funcs_with_argtypes(
    ::DiffCalcBridge,
    ::typeof(diff),
    args_Type::Type{<:Tuple{mop_Type, attr_Type, x_Type}}
) where {mop_Type<:AbstractProblem, attr_Type<:AbstractFunctionQualifier, x_Type<:AbstractVector}
    @show Tuple{mop_Type, attr_Type, x_Type}
    return (
        (calc, Tuple{mop_Type, attr_Type, x_Type}),
    )
end

@concrete terse struct CalcClosure <: Function
    mop
    attr
    prep_calc
end

function (cc::CalcClosure)(x::AbstractVector)
    @unpack mop, attr, prep_calc = cc
    return compute!(cc.prep_calc, mop, attr, x)
end

function prep(
    ::DiffCalcBridge,
    bw::BridgedWrapper,
    ::typeof(diff),
    mop::mop_Type, attr::attr_Type, x::x_Type
) where {mop_Type<:AbstractProblem, attr_Type<:AbstractFunctionQualifier, x_Type<:AbstractVector}
    prp_clc = prep(bw, calc, mop, attr, x)
    cc = CalcClosure(mop, attr, prp_clc)
    bcknd = _backend(mop)
    di_prep = DI.prepare_jacobian(cc, bcknd, x)
    return DiffCalcPrep(cc, bcknd, di_prep)
end

function compute!(
    p::DiffCalcPrep,
    @nospecialize(mop::mop_Type), 
    @nospecialize(attr::attr_Type), 
    x::x_Type
) where {mop_Type<:AbstractProblem, attr_Type<:AbstractFunctionQualifier, x_Type<:AbstractVector}
    @unpack cc, bcknd, di_prep = p
    return DI.jacobian(cc, di_prep, bcknd, x)
end
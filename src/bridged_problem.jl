
import OrderedCollections as OC
import OrderedCollections: freeze, LittleDict

all_bridges(::Type{<:AbstractProblem})=(
    CMatsZeroDimBridge(),
    CMatsCalcBridge(),
    CalcZeroDimBridge(),
    CalcInputVecAsMatBridge(),
    CalcInputMatAsVecsBridge(),
    CalcCMatsBridge(),
    DiffCalcBridge()
)
all_bridges(mop::mop_Type) where mop_Type <: AbstractProblem = all_bridges(mop_Type)

abstract type AbstractWrappedProblem{mop_Type} <: AbstractProblem end
_inner_problem_type(::Type{<:AbstractWrappedProblem{mop_Type}}) where mop_Type = mop_Type

function show_problem(io::IO, mop::mop_Type) where mop_Type<:AbstractWrappedProblem
    _io = IOContext(io, :compact => true)
    print(io, "$(_typename_symb(mop_Type))($(repr(mop.mop; context=_io))")
end

_typename_symb(T::Type)=Base.typename(T).name::Symbol

@concrete terse struct BridgedProblem{mop_Type} <: AbstractWrappedProblem{mop_Type}
    mop :: mop_Type
    bw
    preps
    did_warn
end

@concrete terse struct ProblemWithBackend{mop_Type, backend_Type} <: AbstractWrappedProblem{mop_Type}
    mop :: mop_Type
    backend :: backend_Type
end

function show_problem(io::IO, mop::ProblemWithBackend)
    _io = IOContext(io, :compact => true)
    print(io, "ProblemWithBackend($(repr(mop.mop; context=_io)), $(mop.backend))")
end

float_type(BP::Type{<:AbstractWrappedProblem}) = float_type(_inner_problem_type(BP))
all_bridges(BP::Type{<:AbstractWrappedProblem}) = all_bridges(_inner_problem_type(BP))

function dimval(BP::Type{<:AbstractWrappedProblem}, attr_Type::Type{<:AbstractAttribute})
    return dimval(_inner_problem_type(BP), attr_Type)
end
function dimval(BP::Type{<:AbstractWrappedProblem}, attr_Type::Type{<:AbstractFunctionQualifier})
    return dimval(_inner_problem_type(BP), attr_Type)
end
function var_bounds(BP::Type{<:AbstractWrappedProblem})
    var_bounds(_inner_problem_type(BP))
end
lower_var_bounds(mop::AbstractWrappedProblem)=lower_var_bounds(mop.mop)
upper_var_bounds(mop::AbstractWrappedProblem)=upper_var_bounds(mop.mop)

function _backend_Type(BT::Type{<:AbstractWrappedProblem})
    return _backend_Type(_inner_problem_type(BT))
end
function _backend_Type(
    ::Type{<:ProblemWithBackend{mop_Type, backend_Type}}
) where {mop_Type, backend_Type}
    return backend_Type
end
_backend(_mop::ProblemWithBackend)=_mop.backend

function constraint_matrices(wmop::AbstractWrappedProblem, attr::LinConstraints)
    return constraint_matrices(wmop.mop, attr)
end
function calc(
    wmop::AbstractWrappedProblem, attr::AbstractFunctionQualifier, x::AbstractVecOrMat
)
    return calc(wmop.mop, attr, x)
end
function diff(
    wmop::AbstractWrappedProblem, attr::AbstractFunctionQualifier, x::AbstractVector
)
    return diff(wmop.mop, attr, x)
end

function is_implemented(
    @nospecialize(func::func_Type),
    @nospecialize(args_Type::Type{<:Tuple{mop_Type, Vararg}})
) where {
    mop_Type <: ProblemWithBackend,
    func_Type <: Function
}
    _mop_Type = _inner_problem_type(mop_Type)
    _args_Type = _change_mop_Type(args_Type, _mop_Type)
    return is_implemented(func, _args_Type)
end

@generated function _change_mop_Type(
    _T::Type{T}, ::Type{mop_Type}
) where {T<:Tuple, mop_Type<:AbstractProblem}
    ts = T.parameters[2:end]
    return Tuple{mop_Type, ts...}
end

function init_bridged(
    mop::AbstractProblem; backend::Union{Nothing, AbstractADType}=nothing
)
    did_warn = Dict{Tuple{Type, Type}, Bool}()
    _mop = ProblemWithBackend(mop, backend)
    return BridgedProblem(_mop, BridgedWrapper(_mop), Dict(), did_warn)
end

function fully_prepped(
    mop::AbstractProblem; backend::Union{Nothing, AbstractADType}=nothing
)
    _bmop = init_bridged(mop; backend)
    return prep_all(_bmop)
end

function prep!(
    bmop::BridgedProblem,
    @nospecialize(func::func_Type),
    @nospecialize(args...)
) where func_Type <: Function
    @unpack mop, preps = bmop
    args = pushfirst!!(args, mop)
    args_Type = typeof(args)
    p = _get(preps, func, args_Type, nothing)
    !isnothing(p) && return bmop
    @unpack bw = bmop
    p = prep(bw, func, args...)
    _insert!(preps, func, args_Type, p)
    return bmop
end

function prep_all(bmop::BridgedProblem)
    prep!(bmop, constraint_matrices, LinEqConstraints())
    prep!(bmop, constraint_matrices, LinIneqConstraints())

    F = float_type(bmop)
    n_vars = dim(bmop, Variables)
    x = rand(F, n_vars)
    #prep!(bmop, calc, Objectives(), x)
    prep!(bmop, calc, LinEqConstraints(), x)
    #prep!(bmop, calc, LinIneqConstraints(), x)
    #prep!(bmop, calc, NonlinEqConstraints(), x)
    #prep!(bmop, calc, NonlinIneqConstraints(), x)

    #prep!(bmop, diff, Objectives(), x)
    prep!(bmop, diff, LinEqConstraints(), x)
    #prep!(bmop, diff, LinIneqConstraints(), x)
    #prep!(bmop, diff, NonlinEqConstraints(), x)
    #prep!(bmop, diff, NonlinIneqConstraints(), x)

    #=
    x = rand(F, n_vars, 1)
    prep!(bmop, calc, Objectives(), x)
    prep!(bmop, calc, LinEqConstraints(), x)
    prep!(bmop, calc, LinIneqConstraints(), x)
    prep!(bmop, calc, NonlinEqConstraints(), x)
    prep!(bmop, calc, NonlinIneqConstraints(), x)
    =#
    # TODO `diff`
    return lock_preps(bmop)
end

function lock_preps(bmop::BridgedProblem)
    little_preps = freeze(bmop.preps)
    _bmop = @set bmop.preps = little_preps
    return _bmop 
end

function constraint_matrices(wmop::BridgedProblem, attr::LinConstraints)
    return _bridged_invoke(
        constraint_matrices, wmop, wmop.mop, attr)
end
function calc(
    wmop::BridgedProblem, attr::AbstractFunctionQualifier, x::AbstractVecOrMat
)
    return _bridged_invoke(
        calc, wmop, wmop.mop, attr, x)
end
function diff(
    wmop::BridgedProblem, attr::AbstractFunctionQualifier, x::AbstractVector
)
    return _bridged_invoke(
        diff, wmop, wmop.mop, attr, x)
end

@nospecialize
function _bridged_invoke(
    func::Function, wmop::BridgedProblem, args...
)
    _bridged_invoke(is_implemented(func, typeof(args)), func, wmop, args...)
end
function _bridged_invoke(
    ::IsImplemented,
    func::Function, wmop::BridgedProblem, args...
)
    args_Type = typeof(args)
    return invoke(func, args_Type, args...)
end

function _bridged_invoke(
    ::NotImplemented,
    func::Function, 
    wmop::BridgedProblem, 
    args...
)
    @unpack preps, bw, did_warn = wmop
    __bridged_invoke(func, preps, bw, did_warn, args...)
end
@specialize

function __bridged_invoke(@nospecialize(func::Function), preps, bw, did_warn, args...)
    args_Type = typeof(args)
    p = _get(preps, func, args_Type, nothing)
    if !isnothing(p)
        return compute!(p, args...)
    end
    if !_get(did_warn, func, args_Type, false)
        @warn "Calling `$func` without proper cache."
        _insert!(did_warn, func, args_Type, true)
    end
    return bridged_compute!(bw, func, args...)
end
import ADTypes: 
    AbstractADType, mode, inplace_support, InPlaceSupported, InPlaceNotSupported,
    ForwardMode

import DifferentiationInterface as DI
#import DifferentiationInterface:

struct PushforwardOPBackend{
    op_Type
} <: AbstractADType
    op :: op_Type
end

struct PushforwardOPPrep{
    SIG,
    op_Type
} <: DI.PushforwardPrep
    _sig::Val{SIG}
    op::op_Type
end

function DI.prepare_pushforward_nokwarg(
    strict::Val, f::F, backend::PushforwardOPBackend, 
    x, tx::NTuple, contexts::Vararg{DI.Context,C};
) where {F,C}
    _sig = DI.signature(f, backend, x, tx, contexts...; strict)
    return PushforwardOPPrep(_sig, backend.op) 
end

function DI.pushforward(
    f::F,
    prep::PushforwardOPPrep{SIG,op_Type},
    backend::PushforwardOPBackend{op_Type},
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {F,SIG,op_Type,C}
    DI.check_prep(f, prep, backend, x, tx, contexts...)
    ydual = compute_ydual_onearg(f, prep, x, tx, contexts...)
    ty = mypartials(T, Val(B), ydual)
    return ty
end
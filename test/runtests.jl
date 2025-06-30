using MOPEDS
using Test

@testset "MOPEDS.jl" begin
    # Write your tests here.
#=
abstract type AbstractProb end

function afunc(::AbstractProb)
    nothing
end
function is_implemented(
    ::typeof(afunc), ::Type{<:Tuple{<:AbstractProb}})
    return NotImplemented()
end

function bfunc(::AbstractProb)
    nothing
end
function is_implemented(
    ::typeof(bfunc), ::Type{<:Tuple{<:AbstractProb}})
    return NotImplemented()
end

struct ExProb <: AbstractProb end

function bfunc(::ExProb)
    Inf
end
function is_implemented(::typeof(bfunc), ::Type{<:Tuple{<:ExProb}})
    IsImplemented()
end

struct afunc_bfunc_Bridge <: AbstractBridge end

function is_implemented(
    ::afunc_bfunc_Bridge, ::typeof(afunc), ::Type{<:Tuple{<:AbstractProb}})
    return IsImplemented()
end
function required_funcs_with_argtypes(
    ::afunc_bfunc_Bridge, ::typeof(afunc), args_Type::Type{<:Tuple{<:AbstractProb}})
    return ((bfunc, args_Type),)
end

struct ExProb2 <: AbstractProb end

all_bridges(::AbstractProb) = (afunc_bfunc_Bridge(),)

function prep(
    bridge::afunc_bfunc_Bridge, bw::BridgedWrapper, ::typeof(afunc), args_Type::Type{<:Tuple{<:AbstractProb}}
)
    return WrappedPrep(prep(bw, bfunc, args_Type))
end
=#
end

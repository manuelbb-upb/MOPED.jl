using Pkg
Pkg.activate(@__DIR__)

import NonlinearMultiObjectiveStructures as NMOS
import NonlinearMultiObjectiveStructures: AbstractNonlinearFunction, HasParametersTrait, 
    NoParameters, dim_in, dim_out

function parabola(::Val{1}, x)
    return sum( (x .- 1).^2 )
end
function dx_parabola(::Val{1}, x)
    return 2 .* (x .- 1)
end

function parabola(::Val{2}, x)
    return sum( (x .+ 1).^2 )
end
function dx_parabola(::Val{2}, x)
    return 2 .* (x .+ 1)
end

function parabolas(x)
    return [parabola(Val(1), x), parabola(Val(2), x)]
end
function dx_parabolas(x)
    return stack((dx_parabola(Val(1), x), dx_parabola(Val(2), x)); dims=1)
end

function parabola!(vi, x)
    return parabola(vi, x)
end
function dx_parabola!(gi, vi, x)
    gi .= dx_parabola(vi, x)
    return nothing
end

function parabolas!(y, x)
    y .= parabolas(x)
    return nothing
end
function dx_parabolas!(Dy, x)
    Dy .= dx_parabolas(x)
    return nothing
end

abstract type AbstractParabolaFunction <: AbstractNonlinearFunction end
NMOS.HasParametersTrait(::Type{<:AbstractParabolaFunction}) = NMOS.NoParameters()
dim_in(::AbstractParabolaFunction) = 3
dim_out(::AbstractParabolaFunction) = 2
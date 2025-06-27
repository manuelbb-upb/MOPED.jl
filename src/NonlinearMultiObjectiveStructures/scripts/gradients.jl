using Pkg
Pkg.activate(@__DIR__)
using Test
import NonlinearMultiObjectiveStructures as NMOS
import NonlinearMultiObjectiveStructures: AbstractNonlinearFunction, implemented, IsImplemented
import NonlinearMultiObjectiveStructures: dim_in, dim_out 
import NonlinearMultiObjectiveStructures: gradients, gradients!, gradient, gradient! 
import NonlinearMultiObjectiveStructures: OP, compute!, @concrete

include("parabola.jl")

#%%
@kwdef @concrete struct ParabolaTestFuncFirstOrder <: AbstractParabolaFunction
    strict_impl_trait = Val(true)
    all_oop = Val(true)
    all_ip = Val(false)
    single_oop = Val(false)
    single_ip = Val(false)
end

function implemented(f::ParabolaTestFuncFirstOrder, op::NMOS.AbstractBasicOperator)
    if f.strict_impl_trait isa Val{true}
        return NMOS.NotImplemented()
    end
    return NMOS.MaybeImplemented()
end

function implemented(
    f::ParabolaTestFuncFirstOrder{impl,all_oop,all_ip,single_oop,single_ip}, 
    ::OP{:gradients}
) where {
    impl<:Union{Val{true}, Val{false}},
    all_oop<:Val{true},
    all_ip, 
    single_oop,
    single_ip
}
    return IsImplemented()
end
function gradients(
    f::ParabolaTestFuncFirstOrder{impl,all_oop,all_ip,single_oop,single_ip}, 
    x
) where {
    impl<:Union{Val{true}, Val{false}},
    all_oop<:Val{true},
    all_ip, 
    single_oop,
    single_ip
}
    return dx_parabolas(x)
end
 
function implemented(
    f::ParabolaTestFuncFirstOrder{impl,all_oop,all_ip,single_oop,single_ip}, 
    ::OP{:gradients!}
) where {
    impl<:Union{Val{true}, Val{false}},
    all_oop,
    all_ip<:Val{true}, 
    single_oop,
    single_ip
}
    return IsImplemented()
end
function gradients!(
    y,
    f::ParabolaTestFuncFirstOrder{impl,all_oop,all_ip,single_oop,single_ip}, 
    x
) where {
    impl<:Union{Val{true}, Val{false}},
    all_oop,
    all_ip<:Val{true}, 
    single_oop,
    single_ip
}
    return dx_parabolas!(y, x)
end
function implemented(
    f::ParabolaTestFuncFirstOrder{impl,all_oop,all_ip,single_oop,single_ip}, 
    ::OP{:gradient}
) where {
    impl<:Union{Val{true}, Val{false}},
    all_oop,
    all_ip, 
    single_oop<:Val{true},
    single_ip
}
    return IsImplemented()
end
function gradient(
    f::ParabolaTestFuncFirstOrder{impl,all_oop,all_ip,single_oop,single_ip}, 
    vi::Val{i},
    x,
) where {
    i,
    impl<:Union{Val{true}, Val{false}},
    all_oop,
    all_ip, 
    single_oop<:Val{true},
    single_ip
}
    return dx_parabola(vi, x)
end

function implemented(
    f::ParabolaTestFuncFirstOrder{impl,all_oop,all_ip,single_oop,single_ip}, 
    ::OP{:gradient!}
) where {
    impl<:Union{Val{true}, Val{false}},
    all_oop,
    all_ip, 
    single_oop,
    single_ip<:Val{true}
}
    return IsImplemented()
end

function gradient!(
    Dyi,
    f::ParabolaTestFuncFirstOrder{impl,all_oop,all_ip,single_oop,single_ip}, 
    vi::Val{i},
    x
) where {
    i,
    impl<:Union{Val{true}, Val{false}},
    all_oop,
    all_ip, 
    single_oop,
    single_ip<:Val{true}
}
    return dx_parabola!(Dyi, vi, x)
end
#%%
for (i, propvals) in enumerate(Iterators.product(
    Iterators.repeated((Val(false), Val(true)), 5)...
))
    i <= 2 && continue
    @show i, propvals
    f = ParabolaTestFuncFirstOrder(propvals...)

    x = rand(dim_in(f))
    Dy = dx_parabolas(x)

    _Dy = compute!((), f, OP(:gradients), (x, missing))
    @test Dy ≈ _Dy
    __Dy = similar(_Dy)
    res = compute!((__Dy,), f, OP(:gradients!), (x, missing))
    @test isnothing(res)
    @test _Dy ≈ __Dy
    
    for i = 1:dim_out(f)
        Dyi = compute!((), f, OP(:gradient), (Val(i), x, missing))
        @test Dy[i, :] ≈ Dyi
        _Dyi = similar(Dyi)
        res = compute!((_Dyi,), f, OP(:gradient!), (Val(i), x, missing))
        @test isnothing(res)
        @test Dyi ≈ _Dyi
    end
    
end
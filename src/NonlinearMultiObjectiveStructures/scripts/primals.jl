include("parabola.jl")
using Test
import NonlinearMultiObjectiveStructures: implemented, IsImplemented
import NonlinearMultiObjectiveStructures: primals, primals!, primal, primal! 
import NonlinearMultiObjectiveStructures: OP, compute!, @concrete

#%%
@kwdef @concrete struct ParabolaTestFuncZerothOrder <: AbstractParabolaFunction
    strict_impl_trait = Val(true)
    all_oop = Val(true)
    all_ip = Val(false)
    single_oop = Val(false)
    single_ip = Val(false)
end

function implemented(f::ParabolaTestFuncZerothOrder, op::NMOS.AbstractBasicOperator)
    if f.strict_impl_trait isa Val{true}
        return NMOS.NotImplemented()
    end
    return NMOS.MaybeImplemented()
end

function implemented(
    f::ParabolaTestFuncZerothOrder{impl,all_oop,all_ip,single_oop,single_ip}, 
    ::OP{:primals}
) where {
    impl<:Union{Val{true}, Val{false}},
    all_oop<:Val{true},
    all_ip, 
    single_oop,
    single_ip
}
    return IsImplemented()
end
function primals(
    f::ParabolaTestFuncZerothOrder{impl,all_oop,all_ip,single_oop,single_ip}, 
    x
) where {
    impl<:Union{Val{true}, Val{false}},
    all_oop<:Val{true},
    all_ip, 
    single_oop,
    single_ip
}
    return parabolas(x)
end
 
function implemented(
    f::ParabolaTestFuncZerothOrder{impl,all_oop,all_ip,single_oop,single_ip}, 
    ::OP{:primals!}
) where {
    impl<:Union{Val{true}, Val{false}},
    all_oop,
    all_ip<:Val{true}, 
    single_oop,
    single_ip
}
    return IsImplemented()
end
function primals!(
    y,
    f::ParabolaTestFuncZerothOrder{impl,all_oop,all_ip,single_oop,single_ip}, 
    x
) where {
    impl<:Union{Val{true}, Val{false}},
    all_oop,
    all_ip<:Val{true}, 
    single_oop,
    single_ip
}
    return parabolas!(y, x)
end
function implemented(
    f::ParabolaTestFuncZerothOrder{impl,all_oop,all_ip,single_oop,single_ip}, 
    ::OP{:primal}
) where {
    impl<:Union{Val{true}, Val{false}},
    all_oop,
    all_ip, 
    single_oop<:Val{true},
    single_ip
}
    return IsImplemented()
end
function primal(
    f::ParabolaTestFuncZerothOrder{impl,all_oop,all_ip,single_oop,single_ip}, 
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
    return parabola(vi, x)
end

function implemented(
    f::ParabolaTestFuncZerothOrder{impl,all_oop,all_ip,single_oop,single_ip}, 
    ::OP{:primal!}
) where {
    impl<:Union{Val{true}, Val{false}},
    all_oop,
    all_ip, 
    single_oop,
    single_ip<:Val{true}
}
    return IsImplemented()
end

function primal!(
    f::ParabolaTestFuncZerothOrder{impl,all_oop,all_ip,single_oop,single_ip}, 
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
    return parabola!(vi, x)
end
#%%
for (i, propvals) in enumerate(Iterators.product(
    Iterators.repeated((Val(false), Val(true)), 5)...
))
    i <= 2 && continue
    @show i, propvals
    f = ParabolaTestFuncZerothOrder(propvals...)

    x = rand(dim_in(f))
    y = parabolas(x)

    _y = compute!((), f, OP(:primals), (x, missing))
    @test y ≈ _y
    __y = similar(_y)
    compute!((__y,), f, OP(:primals!), (x, missing))
    @test _y ≈ __y
    
    for i = 1:dim_out(f)
        yi = compute!((), f, OP(:primal), (Val(i), x, missing))
        @test y[i] ≈ yi
        _yi = compute!((), f, OP(:primal!), (Val(i), x, missing))
        @test yi ≈ _yi
    end
    
end
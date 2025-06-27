include("parabola.jl")
using Test
using Profile
import NonlinearMultiObjectiveStructures: implemented, IsImplemented,
    primals, AbstractOperator, OP, NotImplemented, compute!, 
    HasParametersTrait, NoParameters, primals!
#%%
struct Func1 <: AbstractParabolaFunction end
struct Func2 <: AbstractParabolaFunction end
#%%
Func! = Func2

implemented(::Func!, ::AbstractOperator)=NotImplemented()

implemented(::Func!, ::OP{:primals!})=IsImplemented()
primals!(y,::Func!, x)=parabolas!(y, x)
#%%
f! = Func!()
x = rand(dim_in(f!))
y = compute!((), f!, OP(:primals), (x, missing)) |> only
compute!((), f!, OP(:primal), (x, 1, missing))
compute!((), f!, OP(:primal!), (x, 1, missing))
_y = similar(y)
compute!((_y,), f!, OP(:primals!), (x, missing))
#%%
Func = Func1

implemented(::Func, ::AbstractOperator)=NotImplemented()

implemented(::Func, ::OP{:primals})=IsImplemented()
primals(::Func, x)=parabolas(x)

#%%
f = Func()
x = rand(dim_in(f))
y = compute!((), f, OP(:primals), (x, missing)) |> only
compute!((), f, OP(:primal), (x, 1, missing))
compute!((), f, OP(:primal!), (x, 1, missing))
_y = similar(y)
compute!((_y,), f, OP(:primals!), (x, missing))
@test _y ≈ y

Profile.Allocs.clear()
@profview_allocs for _=1:100
    compute!((), f, OP(:primals), (x, missing))
end
@time y = parabolas(x)
#%%
using DifferentiationInterface
import ForwardDiff

backend = AutoForwardDiff()

jacobian(parabolas, backend, x)
pushforward(parabolas, backend, x, ([1, 0, 0],))
pushforward(x -> parabola(Val(1), x), backend, x, ([1, 0, 0],))
pushforward!(parabolas, (zeros(2),), backend, x, ([1, 0, 0],))
__y = rand(dim_out(f))
pushforward!(parabolas!, __y, (zeros(2),), backend, x, ([1, 0, 0],))
___y = similar(__y)
parabolas!(___y, x)
__y, ___y

_f = x -> compute!((), f!, OP(:primals), (x, missing)) |> only
@time jacobian(
    _f,
    backend,
    x
)
@time dx_parabolas(x)
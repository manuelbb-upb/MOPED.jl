module NonlinearMultiObjectiveStructures

import ConcreteStructs: @concrete
import UnPack: @unpack
import InteractiveUtils: methodswith
import StackViews: StackView
import BangBang: push!!
import Accessors: @reset
import SmartAsserts: @smart_assert

# Methods should accept arguments with the following types:
const RZDim = AbstractArray{<:Real, 0}
const RVector = AbstractArray{<:Real, 1}
const RMatrix = AbstractArray{<:Real, 2}
const RTensor = AbstractArray{<:Real, 3}

abstract type ImplementedTrait end
struct IsImplemented <: ImplementedTrait end
struct MaybeImplemented <: ImplementedTrait end
struct NotImplemented <: ImplementedTrait end
struct UndefImplemented <: ImplementedTrait end

@concrete struct StopObject
    val
    meta
end
StopObject(val)=StopObject(val, (;))

include("abstract_nonlinear_function.jl")
include("basic_ops.jl")
include("op_types.jl")
include("fallback_chains.jl")
include("prep.jl")

include("basic_fbacks/all_oop.jl")
include("basic_fbacks/all_ip.jl")
include("basic_fbacks/single_oop.jl")
include("basic_fbacks/single_ip.jl")

include("basic_fbacks/primal__.jl")
include("basic_fbacks/gradient__.jl")

include("utils.jl")
end # module NonlinearMultiObjectiveStructures
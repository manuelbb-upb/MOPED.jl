module MOPED

import LinearAlgebra as LA

import ConcreteStructs: @concrete
import UnPack: @unpack
import BangBang: pushfirst!!
import Accessors: @set

import DifferentiationInterface as DI
import DifferentiationInterface: AbstractADType

include("static_which.jl")

include("problem_interface.jl")

include("graph.jl")
include("bridging_interface.jl")

include("bridges.jl")
include("diff_bridges.jl")

include("bridged_problem.jl")
end

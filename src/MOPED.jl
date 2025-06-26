module MOPED

import LinearAlgebra as LA

import ConcreteStructs: @concrete
import UnPack: @unpack


include("static_which.jl")

include("fallbacks.jl")

include("interface.jl")
include("bridges.jl")
end

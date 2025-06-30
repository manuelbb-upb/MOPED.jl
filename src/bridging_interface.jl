# ## Bridges
# Every bridge has to subtype `AbstractBridge`:
abstract type AbstractBridge end

# For the shortest-path computation(s) in the bridging hyper-graph, the following
# methods should be implemented.
@nospecialize
"""
    is_implemented(
        bridge::AbstractBridge, func::Function, args_Type::Type{<:Tuple})

Trait function to indicate if a method exists."""
function is_implemented(
    bridge::AbstractBridge, func::func_Type, args_Type::Type{<:Tuple}
) where func_Type <: Function
    return NotImplemented()
end

"""
    required_funcs_with_argtypes(
        bridge::AbstractBridge, func::Function, args_Type::Type{<:Tuple}
    )

Return an indexable iterable of function-args_Type tuples required by `bridge`.
"""
function required_funcs_with_argtypes(
    bridge::AbstractBridge, func::func_Type, args_Type::Type{<:Tuple}
) where func_Type <: Function
    return ()
end
@specialize

# Optionally, define a bridging cost:
@nospecialize
"""
    bridging_cost(
        bridge::AbstractBridge, func::Function, args_Type::Type{<:Tuple}
    )

(Positive) cost of bridging with `bridge`."""
function bridging_cost(
    bridge::AbstractBridge, func::func_Type, args_Type::Type{<:Tuple}
) where func_Type <: Function
    return 1.0
end
@specialize

# !!! note
#     We use `args_Type` instead of `args...` here to enable investigating the graph
#     without allocations.
#     It is certainly more prohibitive because we must be able to infer enough information
#     from types alone. But for now it appears sufficient.

# ## BridgedWrapper

# To enable bridges, implement `all_bridges` to return objects with types that subtype
# `AbstractBridge`:
all_bridges(obj) = ()

# This then enables us to build a minimal wrapper storing all the bridges,
# a type-stable graph, and a dict for quick retrieval of nodes for specific
# function calls (based on the call signature).
# `BridgedWrapper` is loosely inspired by the `LazyBridgeOptimizer` in `MathOptInterface`.
@kwdef struct BridgedWrapper{bridges_Type}
    bridges :: bridges_Type
    graph :: Graph = Graph()
    node_dict :: Dict{Tuple{Type, Type}, Node} = Dict{Tuple{Type, Type}, Node}()
end
"""
    BridgedWrapper(obj)

A wrapper around `obj` with bridges `all_bridges(obj)` enabled.
A `BridgedWrapper` can be used to efficiently query shortest paths."""
BridgedWrapper(obj) = BridgedWrapper(; bridges = all_bridges(obj))

# More compact printing:
function Base.show(io::IO, bw::BridgedWrapper)
    println(io, "BridgedWrapper(:bridges($(length(bw.bridges))), :graph, :node_dict)")
end

# Helper for resetting a wrapper.
# (We don't really support dynamically adding bridges as of yet.)
function reset!(bw::BridgedWrapper)
    empty!(bw.graph)
    empty!(bw.node_dict)
    return bw
end

# ### Adding Nodes to Wrapped Graph
# A node is only added for unimplemented function call signatures.
function node(
    bw::BridgedWrapper, @nospecialize(func::func_Type), @nospecialize(args_Type::Type)
) where func_Type <: Function
    return node(is_implemented(func, args_Type), bw, func, args_Type)
end

# `IsImplemented`, return trivial node:
function node(
    bw_impls_func::IsImplemented, 
    bw::BridgedWrapper, @nospecialize(func::func_Type), @nospecialize(args_Type)
) where func_Type <: Function
    return Node(0)
end

# `NotImplemented`, actually add a node to the graph:
function node(
    bw_impls_func::NotImplemented, 
    bw::BridgedWrapper, @nospecialize(func::func_Type), @nospecialize(args_Type)
) where func_Type <: Function
    @unpack bridges, graph, node_dict = bw
    ## is node already in graph?
    nd = _get(node_dict, func, args_Type, nothing)
    if !isnothing(nd)
        return nd
    end
    ## create new node
    nd = add_node!(graph)
    ## also store in dict
    _insert!(node_dict, func, args_Type, nd)

    ## check if it is supported by any bridges:
    for (i, bridge) = enumerate(bridges)
        _maybe_add_bridge!(bw, nd, func, args_Type, i, bridge)
    end
    return nd
end

# Like with nodes, bridges/edges are only added if they support a certain function call.
function _maybe_add_bridge!(
    bw, nd, @nospecialize(func::func_Type), @nospecialize(args_Type), i, bridge
) where func_Type
    is_implemented(bridge, func, args_Type)
    return _maybe_add_bridge!(
        is_implemented(bridge, func, args_Type), bw, nd, func, args_Type, i, bridge)
end
# `NotImplemented`, do nothing:
function _maybe_add_bridge!(
    bridge_impls_func::NotImplemented, 
    bw, nd, @nospecialize(func::func_Type), @nospecialize(args_Type), i, bridge
) where func_Type
    return bw
end
# `IsImplemented`, add hyper-edge to wrapped graph:
function _maybe_add_bridge!(
    bridge_impls_func::IsImplemented, 
    bw, nd, @nospecialize(func::func_Type), @nospecialize(args_Type), i, bridge
) where func_Type
    e = _edge(bw, func, args_Type, i, bridge)
    add_edge!(bw.graph, nd, e)
    return bw
end
# Helper to also add successor nodes; this is where recursion becomes obvious:
function _edge(
    bw, @nospecialize(func::func_Type), @nospecialize(args_Type), i, bridge
) where func_Type
    new_nodes = Node[
        node(bw, _func, _args_Type) 
            for (_func, _args_Type) = required_funcs_with_argtypes(bridge, func, args_Type)
    ]
    bc = bridging_cost(bridge, func, args_Type)
    edge = Edge(i, new_nodes, bc)
    return edge
end
 
# ## Preparation and Computation

# When everything is set up, there is a two-stage execution procedure to
# actually perform a queried function call.
# First, some `AbstractPreparation` is instantiated.
# This object is then used to execute the function calls -- with varying argument values
# (but not varying types or sizes).
# This is very similar to how `DifferentiationInterface` works.
# In fact, the fact that `prep` takes `args...` instead of `args_Type` is to enable
# bridges using `DifferentiationInterface`.
# (Otherwise, `prep` would also resemble `concrete_bridge_type` in `MathOptInterface`.)
abstract type AbstractPreparation end

# For a bridge to work, `prep` has to be specialized and return some `AbstractPreparation`
# object.
# The argument `bw` is needed to prepare successor calls/bridges:
@nospecialize
function prep(
    bridge::AbstractBridge, bw::BridgedWrapper, func::func_Type, args...
) where func_Type <: Function
    error("`prep` not implemented for `$(bridge)` and `$(func)`.")
end
@specialize

# Then, `compute!` has to be specialized to return the value for `args...`:
@nospecialize
function compute!(prep::AbstractPreparation, args...)
    error("`compute!` not implemented for `$(prep)`.")
end
@specialize

# We can derive a compact helper:
function bridged_compute!(
    bw::BridgedWrapper, @nospecialize(func::func_Type), args...
) where func_Type <: Function
    p = prep(bw, func, args...)
    return compute!(p, args...)
end

# ### Default Preparation Objects

# When there is no path in the hyper-graph, an `UnsuccessfulPrep` is returned:
@kwdef struct UnsuccessfulPrep <: AbstractPreparation
    msg :: Union{Nothing, String} = nothing
end

function UnsuccessfulPrep(@nospecialize(func::Function))
    msg = "Call to `$(func)` not possible."
    UnsuccessfulPrep(; msg)
end

# It just errors:
function compute!(p::UnsuccessfulPrep, @nospecialize(args...))
    _compute_unsucc_prep(p.msg)
end
_compute_unsucc_prep(msg::String)=throw(ErrorException(msg))
_compute_unsucc_prep(::Nothing)=error("`compute!` not applicable for `UnsuccessfulPrep`.")

# If a function call is supported, then we just wrap the function:
struct FuncPrep{func_Type} <: AbstractPreparation
    func :: func_Type
end

# This enables us to specialize `compute!` based on dispatch:
function compute!(@nospecialize(prep::FuncPrep), args...)
    @unpack func = prep
    return _compute_func_prep(func, args)
end
# In the end, we just `invoke` the function with arguments `args`:
function _compute_func_prep(@nospecialize(func::func_Type), args::args_Type) where {func_Type, args_Type}
    return invoke(func, args_Type, args...)
end

# ## Graph-Backed Preparation

# For a function call, a node is added (lazily), the shortest path is calculated (lazily),
# and a corresponding `AbstractPreparation` object is returned: 
function prep(
    bw::BridgedWrapper, @nospecialize(func::func_Type), args...
) where {func_Type<:Function}
    args_Type = typeof(args)
    n = node(bw, func, args_Type)
    if iszero(n.index)
        return FuncPrep(func)
    end
    _bellman_ford!(bw.graph)
    if isinf(bw.graph.dist[n.index])
        return UnsuccessfulPrep(func)
    end
    return prep_node(bw, n, func, args...)
end

# `prep_node` is just a helper around `prep` for the best bridge (currently):
function prep_node(
    bw::BridgedWrapper, n::Node, @nospecialize(func::func_Type), args...
) where {func_Type<:Function}
    bi = bw.graph.best[n.index]
    b = bw.bridges[bi]
    p = prep(b, bw, func, args...)
    return _prep_node(p, bw, n, func, args...)
end
# If preparation was successful, return `p`:
@nospecialize
function _prep_node(
    p::AbstractPreparation, bw::BridgedWrapper, n::Node, func::func_Type, args...
) where {func_Type<:Function}
    return p
end
@specialize

# Otherwise, invalidate bridge and try again:
function _prep_node(
    @nospecialize(p), bw::BridgedWrapper, n::Node, @nospecialize(func::func_Type), args...
) where {func_Type<:Function}
    ni = n.index
    bi = bw.graph.best[n.index]
    @unpack graph = bw
    es = graph.edges[ni]
    @debug "Invalidating bridge $bi for node with index $ni."
    for (i, e) in enumerate(es)
        if e.bridge_index == bi
            _e = @set e.cost = Inf
            deleteat!(es, i)
            insert!(es, i, _e)
            ## TODO modifying an array during iteration is dangerous
            ##      alternative: make `Edge` mutable?
        end
    end
    graph.dist[ni] = Inf
    graph.last_correct_ref[] = ni - 1
    return prep(bw, func, args...)
end

# # Hypergraph Types
#
# ## Introduction
# This file describes a graph structure inspired by `MathOptInterface`.
# The structure is used to find the shortest path in the hypergraph of function call bridges
# in a type-stable manner.
# To be precise, we want to enable inter-dependent fallbacks for certain function calls with
# known argument types.

# ## Method Implementation Trait
# Whether or not a suitable method exists is indicated with a trait:
abstract type AbstractImplementedTrait end
struct IsImplemented <: AbstractImplementedTrait end
struct NotImplemented <: AbstractImplementedTrait end
# The trait is queried with `is_implemented`:
@nospecialize
is_implemented(func, args_Type)=NotImplemented()
@specialize

# This trait function can be specialized. It falls back to checking if an applicable
# method exists (statically, thanks to `Tricks.jl`)
@nospecialize
function is_implemented(
    func::func_Type,
    args_Type::Type{<:Tuple}
) where func_Type <: Function
    if _applicable(func, args_Type)
        return IsImplemented()
    else
        return NotImplemented()
    end
end
@specialize


# ## The Hypergraph of Bridges

# The requested function calls can be thought of as vertices in a directed hypergraph.
# More precisely, vertices in an F-graph.
# Fallback bridges are then hyper-edges, starting in a function call vertex and 
# pointing to other vertices that enable the fallback calculations.

# !!! note
#     In certain situations, a bridge can itself perform all the necessary operations
#     based on information inferred from the argument types.
#     Then the hyper-edge points to "virtual" vertices, which are just assumed to be implemented.

# The hypergraph idea allows us to compute shortest bridging paths.
# But to do so quickly, we don't store the actual func-arg types or bridge types
# in the graph structure.
# Instead, we take inspiration from 
# [MathOptInterface](https://github.com/jump-dev/MathOptInterface.jl/blob/master/src/Bridges/graph.jl).

# A simple node type just wrapping an index:
"Node(index::Int)"
struct Node
    "Integer index. Either `0` or an index into arrays within a `Graph` object."
    index :: Int
end

# The hyper-edge type also stores source information, but also indicates the 
# successor nodes and a cost:
"Edge(bridge_index::Int, added_nodes::Vector{Node}, cost::Float64)"
struct Edge
    "Integer index of bridge inducing this edge in some sorted iterable of bridges."
    bridge_index :: Int
    "Successor nodes this hyper-edge is pointing to."
    added_nodes :: Vector{Node}
    "Briding cost for shortest path calculation"
    cost :: Float64
end

# The graph has only a few fields needed for shortest path computation.
# There is no list of `nodes`, but a dict mapping function calls to respective nodes
# is stored in a `BridgedWrapper`.
# For shortest path algorithm to work, the node information stored in the edges is sufficient.
@kwdef struct Graph
    edges :: Vector{Vector{Edge}} = []
    dist :: Vector{Float64} = []
    last_correct_ref :: Base.RefValue{Int} = Ref(0)
    best :: Vector{Int} = []
end

# Helper to reset a graph:
function Base.empty!(graph::Graph)
    @unpack edges, dist, best, last_correct_ref = graph
    empty!(edges)
    empty!(dist)
    empty!(best)
    last_correct_ref[] = 0
    return graph
end

# Helper to add a new node and return it:
function add_node!(graph)
    @unpack edges, dist, best = graph
    push!(edges, Edge[])
    push!(best, 0)
    push!(dist, Inf)
    return Node(length(dist))
end

# Helper for pushing an `edge` starting at `node`: 
function add_edge!(graph, node, edge)
    push!(graph.edges[node.index], edge)
    return edge
end

# ### Bellman Ford -- Shortest Path

# The best bridge is available in `graph.best` after shortest path search:
function bridge_index(graph::Graph, n::Node)
    _bellman_ford!(graph)
    return graph.best[n.index]
end

# The implementation is adapted from `MathOptInterface`:
function _bellman_ford!(graph::Graph)
    @unpack best, last_correct_ref = graph
    lc = last_correct_ref[]
    # Has a distance changed in the last iteration?
    changed = true
    while changed
        changed = false
        for i in (lc+1):length(best)
            dist, best_index = _updated_dist(
                graph,
                graph.dist[i],
                graph.edges[i],
            )
            if !iszero(best_index)
                graph.dist[i] = dist
                graph.best[i] = best_index
                changed = true
            end
        end
    end
    last_correct_ref[] = length(best)
    return graph
end

# Helper to update distance estimate for a node with current distance `current`
# and leaving hyper-edges `edges`:
function _updated_dist(
    graph::Graph,
    current::Float64,
    edges::Vector{Edge},
)
    bridge_index = 0
    for edge in edges
        dist = _dist(graph, edge)
        if isinf(dist)
            continue
        end
        dist += edge.cost
        if dist < current
            current = dist
            bridge_index = edge.bridge_index
        end
    end
    return current, bridge_index
end

# Return cumulative distance for all successor nodes:
function _dist(graph::Graph, edge::Edge)
    return _dist(graph, edge.added_nodes)
end

function _dist(graph::Graph, nodes::Vector{<:Node})
    dist = 0
    for node in nodes
        d = _dist(graph, node)
        if isinf(d)
            return d
        end
        dist += d
    end
    return dist
end

# Distance for a single node:
function _dist(graph::Graph, node::Node)
    ni = node.index
    ## zero-index: implemented function call ⇒ no cost
    return iszero(ni) ? 0 : graph.dist[ni]
end

## Helpers to allow insertion into `Dict{Tuple{Type,Type}, Something}` 
## without giving the type of the function `func` as argument, but rather function itself.
## We use the helper and this kind of dict, because operations on a dict 
## `Dict{Tuple{Function, Type}, Something}` are rather slow. 
## The helper has `@nospecialize` to avoid costly re-compilations for different functions:
function _insert!(dict, @nospecialize(func::func_Type), @nospecialize(args_Type), val) where func_Type
    dict[(func_Type, args_Type)] = val
    return val
end

## Similar helper to obtain values from dict:
function _get(dict, @nospecialize(func::func_Type), @nospecialize(args_Type), default) where func_Type
    return get(dict, (func_Type, args_Type), default)
end


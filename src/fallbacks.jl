abstract type AbstractImplementedTrait end
struct IsImplemented <: AbstractImplementedTrait end
struct NotImplemented <: AbstractImplementedTrait end
is_implemented()=NotImplemented()
#=
function unspecialized_method(func::typeof(some_func), args_type::Type{<:Tuple{Any}})
    return static_which(func, args_type)
end
=#
function is_implemented(
    @nospecialize(func::func_Type),
    args_Type::Type{<:Tuple}
) where func_Type <: Function
    if _applicable(func, args_Type)
        return IsImplemented()
    else
        return NotImplemented()
    end
end

all_bridges(obj) = ()

abstract type AbstractBridge end

function is_implemented(
    bridge::AbstractBridge, @nospecialize(func::func_Type), args_Type::Type{<:Tuple}
) where func_Type <: Function
    return NotImplemented()
end
function required_funcs_with_argtypes(
    bridge::AbstractBridge, @nospecialize(func::func_Type), args_Type::Type{<:Tuple}
) where func_Type <: Function
    return ()
end
function bridging_cost(
    bridge::AbstractBridge, @nospecialize(func::func_Type), args_Type::Type{<:Tuple}
) where func_Type <: Function
    return 1.0
end

struct Node
    index :: Int
end

struct Edge
    bridge_index :: Int
    added_nodes :: Vector{Node}
    cost :: Float64
end

@kwdef struct Graph
    edges :: Vector{Vector{Edge}} = []
    dist :: Vector{Float64} = []
    last_correct_ref :: Base.RefValue{Int} = Ref(0)
    best :: Vector{Int} = []
end

function Base.empty!(graph::Graph)
    @unpack edges, dist, best, last_correct_ref = graph
    empty!(edges)
    empty!(dist)
    empty!(best)
    last_correct_ref[] = 0
    return graph
end

function add_node!(graph)
    @unpack edges, dist, best = graph
    push!(edges, Edge[])
    push!(best, 0)
    push!(dist, Inf)
    return Node(length(dist))
end

function _insert!(dict, @nospecialize(func::func_Type), @nospecialize(args_Type), val) where func_Type
    dict[(func_Type, args_Type)] = val
    return val
end

function _get(dict, @nospecialize(func::func_Type), @nospecialize(args_Type), default) where func_Type
    return get(dict, (func_Type, args_Type), default)
end

function add_edge!(graph, node, edge)
    push!(graph.edges[node.index], edge)
    return edge
end

@kwdef struct BridgedWrapper{bridges_Type}
    bridges :: bridges_Type
    graph :: Graph = Graph()
    node_dict :: Dict{Tuple{Type, Type}, Node} = Dict{Tuple{Type, Type}, Node}()
end
BridgedWrapper(obj) = BridgedWrapper(; bridges = all_bridges(obj))

function reset!(bw::BridgedWrapper)
    empty!(bw.graph)
    empty!(bw.node_dict)
    return bw
end

abstract type AbstractPreparation end
compute!(prep::AbstractPreparation, @nospecialize(args...))=error("compute not implemented for `$(prep)`.")

struct UnsuccessfulPrep <: AbstractPreparation end

struct FuncPrep{func_Type} <: AbstractPreparation
    func :: func_Type
end

function FuncPrep(func::func_Type, args_Type::Type) where func_Type<:Function
    return FuncPrep(func)
end

function compute!(prep::FuncPrep, @nospecialize(args...))
    @unpack func = prep
    _compute_func_prep(func, args)
end

function _compute_func_prep(@nospecialize(func::func_Type), args::args_Type) where {func_Type, args_Type}
    return invoke(func, args_Type, args...)
end

@concrete struct WrappedPrep
    prep
end

function compute!(@nospecialize(wprep::WrappedPrep), @nospecialize(args...))
    return compute!(wprep.prep, args...)
end

function bridged_compute(
    bw::BridgedWrapper, @nospecialize(func::func_Type), args...
) where func_Type <: Function
    p = prep(bw, func, typeof(args))
    return compute!(p, args...)
end

function prep(
    bw::BridgedWrapper, @nospecialize(func::func_Type), @nospecialize(args_Type)
) where {func_Type<:Function}
    n = node(bw, func, args_Type)
    if iszero(n.index)
        return FuncPrep(func, args_Type)
    end
    _bellman_ford!(bw.graph)
    if isinf(bw.graph.dist[n.index])
        return UnsuccessfulPrep()
    end
    bi = bw.graph.best[n.index]
    #error("`func` allegedly supported by bridges, but `prep` not implemented.")
    b = bw.bridges[bi]
    return prep(b, bw, func, args_Type)
end

function prep(
    bridge::AbstractBridge, bw::BridgedWrapper, @nospecialize(func::func_Type), @nospecialize(args_Type)
) where func_Type <: Function
    error("`prep` not implemented for `$(bridge)` and `$(func)`.")
end

function node(
    bw::BridgedWrapper, @nospecialize(func::func_Type), @nospecialize(args_Type)
) where func_Type <: Function
    return node(is_implemented(func, args_Type), bw, func, args_Type)
end
function node(
    bw_impls_func::IsImplemented, 
    bw::BridgedWrapper, @nospecialize(func::func_Type), @nospecialize(args_Type)
) where func_Type <: Function
    @info "$(func) is implemented."
    return Node(0)
end

function node(
    bw_impls_func::NotImplemented, 
    bw::BridgedWrapper, @nospecialize(func::func_Type), @nospecialize(args_Type)
) where func_Type <: Function
    @unpack bridges, graph, node_dict = bw
    nd = _get(node_dict, func, args_Type, nothing)
    if !isnothing(nd)
        return nd
    end
    nd = add_node!(graph)
    _insert!(node_dict, func, args_Type, nd)
    for (i, bridge) = enumerate(bridges)
        _maybe_add_bridge!(bw, nd, func, args_Type, i, bridge)
    end
    return nd
end

function _maybe_add_bridge!(
    bw, nd, @nospecialize(func::func_Type), @nospecialize(args_Type), i, bridge
) where func_Type
    @show bridge
    @show is_implemented(bridge, func, args_Type)
    return _maybe_add_bridge!(
        is_implemented(bridge, func, args_Type), bw, nd, func, args_Type, i, bridge)
end
function _maybe_add_bridge!(
    bridge_impls_func::NotImplemented, 
    bw, nd, @nospecialize(func::func_Type), @nospecialize(args_Type), i, bridge
) where func_Type
    return bw
end
function _maybe_add_bridge!(
    bridge_impls_func::IsImplemented, 
    bw, nd, @nospecialize(func::func_Type), @nospecialize(args_Type), i, bridge
) where func_Type
    e = _edge(bw, func, args_Type, i, bridge)
    add_edge!(bw.graph, nd, e)
    return bw
end

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
 
function bridge_index(graph::Graph, n::Node)
    #=if iszero(n.index)
        return 0
    end=#
    _bellman_ford!(graph)
    return graph.best[n.index]
end

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

function _dist(graph::Graph, node::Node)
    ni = node.index
    return iszero(ni) ? 0 : graph.dist[ni]
end
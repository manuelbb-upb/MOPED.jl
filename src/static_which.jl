import Tricks: _combine_signature_type, create_codeinfo_with_returnvalue
"""
    static_which(f, [type_tuple::Type{<:Tuple}, throw_error::Union{Val{true}, Val{false}}])

Similar to `Base.which` this returns the method of `f` that would be used by `invoke` for 
the given argument type tuple.
If `throw_error==Val(true)`, then an error is thrown if no suitable method is found.
If `throw_error==Val(false)` and there is no method, then `nothing` is returned.
Default is `throw_error=Val(false)`.
"""
static_which(@nospecialize(f)) = static_which(f, Tuple{Vararg{Any}})
@static if VERSION >= v"1.10.0-DEV.609"
    function __static_which(world, source, T, self, f, _T, _throw_error::Type{throw_error}) where throw_error
        tt = _combine_signature_type(f, T)
        match, _ = Core.Compiler._findsup(tt, nothing, world)
       
        if isnothing(match)
            if throw_error <: Val{true}
                #=
                This is what happens in `Base`:
                ```
                me = MethodError(f, T, world)
                ee = ErrorException(sprint(io -> begin
                    println(io, "Calling invoke(f, t, args...) would throw:");
                    Base.showerror(io, me);
                end))
                throw(ee)
                ```
                We cannot easily replicate this behavior:
                1) Instead of `f::Function`, in the generated code we have `f::Type{<:Function}`.
                To define `me` we could do something like `f.instance`, but only for 
                singleton function types; function-like structs wouldn't work.
                2) `Base.showerror(io, me)` performs code reflection, disallowed in generated 
                functions. Moreover, the method in `base/errorshow.jl` is non-trivial to 
                replicate with our static alternatives.

                Thus, we only throw with some short message:
                =#
                throw("Calling invoke(f, t, args...) would throw an error.")
            else
                matched_method = match
            end
        else
            matched_method = match.method
        end
        
        # Now we add the edges so if a method is defined this recompiles
        ci = create_codeinfo_with_returnvalue([Symbol("#self#"), :f, :_T, :throw_error], [:T], (:T,), :($matched_method))
        return ci
    end
    @eval function static_which(@nospecialize(f) , @nospecialize(_T::Type{T}) , @nospecialize(throw_error::Union{Val{true}, Val{false}}=Val(false))) where {T <: Tuple}
        $(Expr(:meta, :generated, __static_which))
        $(Expr(:meta, :generated_only))
    end
else
    @generated function static_which(@nospecialize(f) , @nospecialize(_T::Type{T}) , @nospecialize(throw_error::Union{Val{true}, Val{false}}=Val(false))) where {T <: Tuple}
        world = typemax(UInt)
        tt = _combine_signature_type(f, T)
        match, _ = Core.Compiler._findsup(tt, nothing, world)
        if isnothing(match)
            if throw_error <: Val{true}
               throw("Calling invoke(f, t, args...) would throw an error.")
            else
                matched_method = match
            end
        else
            matched_method = match.method
        end
        ci = create_codeinfo_with_returnvalue([Symbol("#self#"), :f, :_T, :throw_error], [:T], (:T,), :($matched_method))
        return ci
    end
end

function _applicable(@nospecialize(f), _T::Type{<:Tuple})
    return !isnothing(static_which(f, _T, Val(false)))
end
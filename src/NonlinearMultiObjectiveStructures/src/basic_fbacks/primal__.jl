_next_fback(ctx, ::OP{:primal!}) = OP(:primal)
_next_fback(ctx, ::OP{:primal!}, ::OP{:primal}) = OP(:primals)
_next_fback(ctx, ::OP{:primal!}, ::OP{:primals}) = OP(:primals!)

_next_fback(ctx, ::OP{:primals!}) = OP(:primal!)
_next_fback(ctx, ::OP{:primals!}, ::OP{:primal!}) = OP(:primals)
_next_fback(ctx, ::OP{:primals!}, ::OP{:primals}) = OP(:primal)

_next_fback(ctx, ::OP{:primal}) = OP{:primal!}()
_next_fback(ctx, ::OP{:primal}, ::OP{:primal!}) = OP{:primals}()
_next_fback(ctx, ::OP{:primal}, ::OP{:primals}) = OP{:primals!}()

_next_fback(ctx, ::OP{:primals}) = OP{:primal}()
_next_fback(ctx, ::OP{:primals}, ::OP{:primal}) = OP{:primals!}()
_next_fback(ctx, ::OP{:primals}, ::OP{:primals!}) = OP{:primal!}()

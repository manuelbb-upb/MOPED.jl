_next_fback(ctx, ::OP{:gradient!}) = OP(:gradient)
_next_fback(ctx, ::OP{:gradient!}, ::OP{:gradient}) = OP(:gradients)
_next_fback(ctx, ::OP{:gradient!}, ::OP{:gradients}) = OP(:gradients!)

_next_fback(ctx, ::OP{:gradients!}) = OP(:gradient!)
_next_fback(ctx, ::OP{:gradients!}, ::OP{:gradient!}) = OP(:gradients)
_next_fback(ctx, ::OP{:gradients!}, ::OP{:gradients}) = OP(:gradient)

_next_fback(ctx, ::OP{:gradient}) = OP{:gradient!}()
_next_fback(ctx, ::OP{:gradient}, ::OP{:gradient!}) = OP{:gradients}()
_next_fback(ctx, ::OP{:gradient}, ::OP{:gradients}) = OP{:gradients!}()

_next_fback(ctx, ::OP{:gradients}) = OP{:gradient}()
_next_fback(ctx, ::OP{:gradients}, ::OP{:gradient}) = OP{:gradients!}()
_next_fback(ctx, ::OP{:gradients}, ::OP{:gradients!}) = OP{:gradient!}()
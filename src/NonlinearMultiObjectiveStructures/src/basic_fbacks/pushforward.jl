_next_fback(ctx, ::OP{:jac_vec_prod}) = OP(:grad_vec_prod)

_next_fback(ctx, ::OP{:grad_vec_prod}) = OP(:jac_vec_prod)
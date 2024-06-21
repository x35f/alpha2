import jax.numpy as jnp
import jax
from functools import partial

# unary operators
@jax.jit
def OpRev(val):
    return - val

# binary operators
@jax.jit
def OpAdd(val0, val1):
    return jnp.add(val0, val1)

@jax.jit
def OpSub(val0, val1):
    # definition of subtract here
    return 

# ternary operators
@partial(jax.jit, static_argnums=(2,))
def OpCorrelation(val0, val1, lookback):
    # definition of correlation here
    return 




import jax
import jax.numpy as jnp
from alpha2.expression.structure import Value
def fast_evaluate(alpha: Value): 
    return compute_metric(alpha.value)

@jax.jit
def compute_metric(alpha: jnp.ndarray):
    # definition of evaluation metric here, e.g., ic, sharpe
    return 0



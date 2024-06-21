import numpy as np
import jax.numpy as jnp
from alpha2.expression.structure import Value, DimensionType, Dimension
from alpha2.expression.meta_data import num_registers


register_operands = [i for i in range(num_registers)]

# define you constant operands here
scalar_operands = [
    Value(value=np.float16(0.01), dimension=Dimension(numerator=[], denominator=[])),
]

vector_operands = {
}

matrix_operands = {
    "open":     Value(value=None,     dimension=Dimension(numerator=[DimensionType.price], denominator=[])),
    "close":    Value(value=None,     dimension=Dimension(numerator=[DimensionType.price], denominator=[])),
    "high":     Value(value=None,     dimension=Dimension(numerator=[DimensionType.price], denominator=[])),
    "low":      Value(value=None,     dimension=Dimension(numerator=[DimensionType.price], denominator=[])),
    "vwap":     Value(value=None,     dimension=Dimension(numerator=[DimensionType.price], denominator=[])),
    "volume":   Value(value=None,     dimension=Dimension(numerator=[DimensionType.volume], denominator=[])),
}

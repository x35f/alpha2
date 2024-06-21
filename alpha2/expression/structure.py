from enum import Enum
from typing import NamedTuple, List
import jax.numpy as jnp
from typing import Sequence

class Token(object):
    def __init__(self, s=None):
        self.s = s
        self.name = s

class OperatorToken(object):
    def __init__(self, s=None):
        self.s = s    
      
class UnaryOpToken(OperatorToken):
    def __init__(self, s=None):
        self.s = s

class BinaryOpToken(OperatorToken):
    def __init__(self, s=None):
        self.s = s

class TernaryOpToken(OperatorToken):
    def __init__(self, s=None):
        self.s = s


class ExpressionNode:
    def __init__(self, token, children = [], value=None):
        self.token: Token = token
        self.children: Sequence[ExpressionNode] = children
        self.value = value

class DimensionType(Enum):
    price = 1
    trade = 2
    volume = 3
    condition = 4

class Dimension(NamedTuple):
    numerator: List[DimensionType]
    denominator: List[DimensionType]

class Value(NamedTuple):
    value: jnp.ndarray
    dimension: Dimension


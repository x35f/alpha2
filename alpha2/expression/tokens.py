from alpha2.expression.operators import *
from alpha2.expression.structure import *
from alpha2.expression.operands import *
from copy import copy
def copy_dim(dim):
    # copy the dimension
    return Dimension(numerator=copy(dim.numerator), denominator=copy(dim.denominator))

def add_sub_dim(dim1, dim2):
    # return the new dimension for the addition of two dimensions
    return Dimension(numerator=[], denominator=[])
    

class StartToken(Token):
    def __init__(self):
        self.s = "Start"
        self.name = 'Start'

class FinishToken(Token):
    def __init__(self):
        self.s = "Finish"
        self.name = 'Finish'

# Register
class RegisterToken(Token): # the number of registers corresponds to the max fan-out
    def __init__(self, idx: str):
        self.idx = idx
        self.name = "Reg"+str(idx)
        self.s = "Reg"+str(idx) #just for debugging
        self.value = None # for type check, should not be assigned

# Operators
## Unary Operator
class RevToken(UnaryOpToken):
    def __init__(self):
        self.s = "-{}"
        self.name = 'Reverse'
    
    @staticmethod
    def validity_check(*values):
        # customized dimension check here
        return True
    
    @staticmethod
    def cal(*values):
        val= values[0].value
        res_val = OpRev(val)
        res_dim = copy_dim(values[0].dimension)
        return Value(value=res_val, dimension=res_dim)


## Binary Operator
class AddToken(BinaryOpToken):
    def __init__(self,):
        self.s = "({} + {})"
        self.name = 'Add'

    @staticmethod
    def validity_check(*values):
        # customized validity check here
        return True
        
    @staticmethod
    def cal(*values):
        val0, val1 = values[0], values[1]
        res_val = OpAdd(val0.value, val1.value)
        res_dim = add_sub_dim(val0.dimension, val1.dimension)
        return Value(value=res_val, dimension=res_dim)


class SubToken(BinaryOpToken):
    def __init__(self,):
        self.s = "({} - {})"
        self.name = 'Sub'

    @staticmethod
    def validity_check(*values):
        # customized validity check here
        return True
    @staticmethod
    def cal(*values):
        val0, val1 = values
        res_val = OpSub(val0.value, val1.value)
        res_dim = add_sub_dim(val0.dimension, val1.dimension)
        return Value(value=res_val, dimension=res_dim)

    

# ternary tokens
class CorrelationToken(TernaryOpToken):
    def __init__(self, s=None):
        self.s = "Corr[{},{},{}]"
        self.name = "Corr"

    @staticmethod
    def validity_check(*values):
        # customized validity check here
        return True
        
    @staticmethod
    def cal(*values):
        val0, val1, value2 = values
        res_val = OpCorrelation(val0.value, val1.value, value2.value)
        res_dim =  Dimension(numerator=[], denominator=[])
        return Value(value=res_val, dimension=res_dim)

# Operands
class ConstToken(Token):
    def __init__(self, value, s=None):
        self.value = value
        if s is None:
            self.s = str(s)
        else:
            self.s = s
        self.name = self.s

#  NULL for padding actions to the same length
class NullToken(Token):
    def __init__(self):
        self.s = "Null"
        self.name = 'Null'
        self.value = None

    def cal(self, *values):
        return None

UNARY_OP_TOKENS = [ 
    RevToken(),         
]

BINARY_OP_TOKENS = [
    AddToken(),         SubToken(), 
]
TERNARY_OP_TOKENS = [
    CorrelationToken()
]

start_token = StartToken()
finish_token = FinishToken()
null_token = NullToken()
register_tokens = [RegisterToken(i) for i in register_operands]

operand_tokens = [ConstToken(value=v, s = str(v.value)) for v in scalar_operands] + \
                [ConstToken(value=v, s = k) for k,v in vector_operands.items()] + \
                [ConstToken(value=v, s = k) for k,v in matrix_operands.items()] + \
                 register_tokens + \
                [null_token]

null_token_idx = operand_tokens.index(null_token)
START_FINISH_TOKENS = [start_token, finish_token]
operator_tokens = UNARY_OP_TOKENS + BINARY_OP_TOKENS + TERNARY_OP_TOKENS + START_FINISH_TOKENS 
start_token_idx = len(operator_tokens) - 2
finish_token_idx = len(operator_tokens) - 1
NUM_OPERATORS = len(operator_tokens)
NUM_OPERANDS = len(operand_tokens)
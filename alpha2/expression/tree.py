from alpha2.expression.structure import ExpressionNode,BinaryOpToken, UnaryOpToken, TernaryOpToken
from alpha2.expression.tokens import RegisterToken, FinishToken, StartToken
from alpha2.expression.meta_data import num_registers
    
class ExpressionTree:
    def __init__(self, max_length):
        self.max_length = max_length
        self.root_node = None

        self.reg_nodes = [None for _ in range(num_registers)]
        self.reg_strs = ["" for _ in range(num_registers)]

        self.action_history = []
    
    @property
    def program(self,):
        return self.action_history

    @property
    def length(self):
        return len(self.action_history)
    
    def add_action(self, action):
        # apply action to the expreesion tree
        self.action_history.append(action)
        operator_tk = action[0]
        if isinstance(operator_tk, StartToken):
            return
        elif isinstance(operator_tk, FinishToken):
            self.finished = True
        elif isinstance(operator_tk, UnaryOpToken):
            # apply a unary action
            pass
        elif isinstance(operator_tk, BinaryOpToken): 
            # apply a binary action
            pass
        elif isinstance(operator_tk, TernaryOpToken): 
            # apply a ternary action
            pass
            
    def calculate(self):
        # calculate the value of the nodes
        ret = {
            "val": None,
            "reg0": None,
            "reg1": None
        }
        return ret

    def calculate_node(self, node: ExpressionNode):
        # recursively calculate the node using node.token.cal function
        return 0
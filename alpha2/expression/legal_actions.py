
from alpha2.expression.tokens import FinishToken

def get_legal_action_idxs(tree, computation_data):
    #check the validity of computation actions through token.validity_check
    legal_op_idxs = []
    return legal_op_idxs

def get_legal_action_list(tree, computation_data):
    finish_action_idx = computation_data['finish_action_idx']
    if len(tree.action_history) == 0: 
        # the first action should always be start, and has been executed on game start
        assert 0
    elif isinstance(tree.action_history[-1][0], FinishToken):
        legal_action_list = []
    elif len(tree.action_history) >= tree.max_length:
        legal_action_list = [finish_action_idx]
    else: # cal create a new subtree or operate on the current subtree
        legal_action_list = get_legal_action_idxs(tree,computation_data)
    
    return legal_action_list

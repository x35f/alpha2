import ray
import numpy as np
import pickle
from alpha2.expression.meta_data import num_registers
from alpha2.expression.tokens import RegisterToken, NullToken,null_token_idx, operand_tokens, operator_tokens, UNARY_OP_TOKENS, BINARY_OP_TOKENS, TERNARY_OP_TOKENS, finish_token_idx, start_token_idx, NUM_OPERATORS, NUM_OPERANDS

@ray.remote(num_cpus=1, num_gpus=0.4)
def get_all_UNARY_ACTIONS(start_idx, end_idx, operator_tks, operand_tks,null_idx):
    action_list, token_action_list = [], []
    for i in range(start_idx, end_idx):
        operator_tk = operator_tks[i]
        for j, operand_tk in enumerate(operand_tks):
            # check validity of the action and add to the action_list
            pass
    return action_list, token_action_list

@ray.remote(num_cpus=1, num_gpus=0.05)
def initialize_UNARY_actions( operator_tks, operand_tks, null_idx, num_splits=1):
    num_operator_tokens = len(operator_tks)
    interval = num_operator_tokens / num_splits
    refs = []
    for i in range(num_splits):
        start_index, end_index = i*interval, (i+1)*interval
        start_index = int(np.floor(start_index))
        end_index = int(np.floor(end_index))
        if i ==num_splits - 1:
            end_index = num_operator_tokens
        refs.append(get_all_UNARY_ACTIONS.remote(start_index, end_index, operator_tks, operand_tks, null_idx))
    action_lists = ray.get(refs)
    unary_action_list, unary_token_action_list = [], []
    for action_list, token_action_list in action_lists:
        unary_action_list.extend(action_list)
        unary_token_action_list.extend(token_action_list)
    return unary_action_list, unary_token_action_list

@ray.remote(num_cpus=1, num_gpus=0.4)
def get_all_BINARY_ACTIONS(start_idx, end_idx, operator_tks, operand_tks, action_shift):
    action_list, token_action_list = [], []
    for i in range(start_idx, end_idx):
        operator_tk = operator_tks[i]
        for j, operand_tk1 in enumerate(operand_tks):
            for k, operand_tk2 in enumerate(operand_tks):
                # check validity of the action and add to the action_list
                pass
    return action_list, token_action_list

@ray.remote(num_cpus=1,num_gpus=0.05)
def initialize_BINARY_actions( operator_tks, operand_tks,action_shift,  num_splits=1):
    num_operator_tokens = len(operator_tks)
    interval = num_operator_tokens / num_splits
    refs = []
    for i in range(num_splits):
        start_index, end_index = i*interval, (i+1)*interval
        start_index = int(np.floor(start_index))
        end_index = int(np.floor(end_index))
        if i == num_splits - 1:
            end_index = num_operator_tokens
        refs.append(get_all_BINARY_ACTIONS.remote(start_index, end_index, operator_tks, operand_tks, action_shift))
    action_lists = ray.get(refs)
    
    binary_action_list, binary_token_action_lists = [], []
    for action_list, token_action_list in action_lists:
        binary_action_list.extend(action_list)
        binary_token_action_lists.extend(token_action_list)
    return binary_action_list, binary_token_action_lists

@ray.remote(num_cpus=1, num_gpus=0.3)
def get_all_ternary_actions(start_idx, end_idx, operator_tks, operand_tks, action_shift):
    action_list, token_action_list = [], []
    for i in range(start_idx, end_idx):
        operator_tk = operator_tks[i]
        for j, operand_tk1 in enumerate(operand_tks):
            for k, operand_tk2 in enumerate(operand_tks):
                for l, operand_tk3 in enumerate(operand_tks):
                    # check validity of the action and add to the action_list
                    pass
    return action_list, token_action_list

@ray.remote(num_cpus=1,num_gpus=0.2)
def initialize_ternary_actions( operator_tks, operand_tks, action_shift,  num_splits=1):
    num_operator_tokens = len(operator_tks)
    interval = num_operator_tokens / num_splits
    refs = []
    for i in range(num_splits):
        start_index, end_index = i*interval, (i+1)*interval
        start_index = int(np.floor(start_index))
        end_index = int(np.floor(end_index))
        if i == num_splits - 1:
            end_index = num_operator_tokens
        refs.append(get_all_ternary_actions.remote(start_index, end_index, operator_tks, operand_tks, action_shift))
    action_lists = ray.get(refs)
    
    ternary_action_list, ternary_token_action_lists = [], []
    for action_list, token_action_list in action_lists:
        ternary_action_list.extend(action_list)
        ternary_token_action_lists.extend(token_action_list)
    return ternary_action_list, ternary_token_action_lists


refs = [initialize_UNARY_actions.remote(UNARY_OP_TOKENS, operand_tokens, null_token_idx), initialize_BINARY_actions.remote(BINARY_OP_TOKENS, operand_tokens, action_shift=len(UNARY_OP_TOKENS)),  initialize_ternary_actions.remote(TERNARY_OP_TOKENS, operand_tokens, action_shift=len(UNARY_OP_TOKENS) + len(BINARY_OP_TOKENS))]
[unary_actions, unary_token_actions], [binary_actions, binary_token_actions],  [ternary_actions, ternary_token_actions] = ray.get(refs)


start_action = tuple([start_token_idx, null_token_idx, null_token_idx, null_token_idx])
finish_action = tuple([finish_token_idx, null_token_idx, null_token_idx, null_token_idx])

all_action_list = [start_action, finish_action]  + unary_actions + binary_actions + ternary_actions
start_action_idx = 0
finish_action_idx = 1
NUM_ACTIONS = len(all_action_list)
    
    
register_token_ids = [i for i in range(NUM_OPERANDS) if isinstance(operand_tokens[i], RegisterToken)]

computation_data ={
    "start_action_idx": start_action_idx,
    "finish_action_idx": finish_action_idx,
    "all_action_list": all_action_list,
    "num_registers": num_registers,
    "unary_operator_tokens": UNARY_OP_TOKENS,
    "binary_operator_tokens": BINARY_OP_TOKENS,
    "resgister_token_ids": register_token_ids,
    "operator_tokens": operator_tokens,
    "operand_tokens": operand_tokens,
    "num_operators": NUM_OPERATORS,
    "num_operands": NUM_OPERANDS, 
    "num_actions": len(all_action_list),
}

save_path = "./data/computation_data.pkl"
with open(save_path, 'w+b') as f:
    pickle.dump(computation_data, f)
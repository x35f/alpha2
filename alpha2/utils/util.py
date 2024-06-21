import os
import ast
import random
from jax import random as jrandom
import numpy as np
import importlib
from .logger import Logger

logger: Logger= None
curr_game_id = 0
PRNGkey = None

def set_global_seed(seed):
    global PRNGkey
    np.random.seed(seed)
    random.seed(seed)
    PRNGkey = jrandom.PRNGKey(seed)

def get_PRNGkey():
    global PRNGkey
    return PRNGkey

def set_logger(logger_ent):
    global logger
    logger = logger_ent

def relative_path_to_module_path(relative_path):
    path = relative_path.replace(".py", "").replace(os.path.sep,'.')
    return path
    
def load_config(config_path,update_args=[]):
    default_config_path_elements = config_path.split(os.sep)
    default_config_path_elements[-1] = "default.py"
    default_config_path = os.path.join(*default_config_path_elements)
    default_args_module = importlib.import_module(relative_path_to_module_path(default_config_path), package='my_current_pkg')
    overwrite_args_module = importlib.import_module(relative_path_to_module_path(config_path), package='my_current_pkg')
    default_args_dict = getattr(default_args_module, 'default_args')
    args_dict = getattr(overwrite_args_module, 'overwrite_args')
    assert type(default_args_dict) == dict, "default args file should be default_args=\{...\}"
    assert type(args_dict) == dict, "args file should be default_args=\{...\}"

    #update args is tpule type, convert to dict type
    update_args_dict = {}
    for update_arg in update_args:
        key, val = update_arg.split("=")
        print(val)
        update_args_dict[key] = ast.literal_eval(val)
    
    #update env specific args to default 
    args_dict = merge_dict(default_args_dict, args_dict)
    default_args_dict = update_parameters(default_args_dict, update_args_dict)
    if 'common' in args_dict:
        for sub_key in args_dict:
            if type(args_dict[sub_key]) == dict:
                args_dict[sub_key] = merge_dict(args_dict[sub_key], default_args_dict['common'], "common")
    return args_dict


def merge_dict(source_dict, update_dict, ignored_dict_name=""):
    for key in update_dict:
        if key == ignored_dict_name:
            continue
        if key not in source_dict:
            #print("\033[32m new arg {}: {}\033[0m".format(key, update_dict[key]))
            source_dict[key] = update_dict[key]
        else:
            if type(update_dict[key]) == dict:
                source_dict[key] = merge_dict(source_dict[key], update_dict[key], ignored_dict_name)
            else:
                if source_dict[key] != update_dict[key]:
                    print("updated {} from {} to {}".format(key, source_dict[key], update_dict[key]))
                source_dict[key] = update_dict[key]
    return source_dict


def update_parameters(source_args, update_args):
    print("updating args", update_args)
    #command line overwriting case, decompose the path and overwrite the args
    for key_path in update_args:
        target_value = update_args[key_path]
        print("key:{}\tvalue:{}".format(key_path, target_value))
        source_args = overwrite_argument_from_path(source_args, key_path, target_value)
    return source_args


def overwrite_argument_from_path(source_dict, key_path, target_value):
    key_path = key_path.split("/")
    curr_dict = source_dict
    for key in key_path[:-1]:
        if not key in curr_dict:
            #illegal path
            return source_dict
        curr_dict = curr_dict[key]
    final_key = key_path[-1] 
    curr_dict[final_key] = target_value
    return source_dict


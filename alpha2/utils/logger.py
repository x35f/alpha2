import os
from datetime import datetime
import json
from torch.utils.tensorboard import SummaryWriter
from abc import abstractmethod
import pickle
import numpy as np
from alpha2.expression.evaluate import visualize
from alpha2.expression.operands import train_stock_data
class BaseLogger(object):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def log_str(self):
        pass

    @abstractmethod
    def log_var(self):
        pass

    
class Logger(BaseLogger):
    def __init__(self, computation_data, config, log_path, env_name, seed, info_str="",  warning_level = 3, print_to_terminal = True):
        unique_path = self.make_simple_log_path(info_str, seed)
        log_path = os.path.join(log_path, env_name, unique_path)
        self.log_path = log_path
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.tb_writer = SummaryWriter(log_path)
        self.log_file_path = os.path.join(log_path,"logs.txt")
        self.visualize_path = os.path.join(log_path, 'alphas')
        self.print_to_terminal = print_to_terminal
        self.warning_level = warning_level
        self.log_str("logging to {}".format(self.log_path))
        self.num_games_saved = 0
        self.config = config
        self.computation_data = computation_data
        
    def make_simple_log_path(self, info_str, seed):
        now = datetime.now()
        time_str = now.strftime("%Y-%m-%d-%H-%M")
        pid_str = os.getpid()
        if info_str != "":
            return "{}-{}-{}_{}".format(time_str, seed, pid_str, info_str)
        else:
            return "{}-{}-{}".format(time_str, seed, pid_str)

    @property
    def log_dir(self):
        return self.log_path
        
    def log_str(self, content, level = 4):
        if level < self.warning_level:
            return
        now = datetime.now()
        time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        if self.print_to_terminal:
            print("\033[32m{}\033[0m:\t{}".format(time_str, content))
        with open(self.log_file_path,'a+') as f:
            f.write("{}:\t{}\n".format(time_str, content))

    def log_tb_var(self, name, val, timestamp):
        self.tb_writer.add_scalar(name, val, timestamp)

    def log_str_object(self, name: str, log_dict: dict = None, log_str: str = None):
        if log_dict!=None:
            log_str = json.dumps(log_dict, indent=4)            
        elif log_str!= None:     
            pass
        else:
            assert 0
        if name[-4:] != ".txt":
            name += ".txt"
        target_path = os.path.join(self.log_path, name)
        with open(target_path,'w+') as f:
            f.write(log_str)
        self.log_str("saved {} to {}".format(name, target_path))

    def log_network(self, network_params, timestep):
        with open(os.path.join(self.network_snapshot_path, "{}.pkl".format(timestep)), "w+b") as f:
            pickle.dump(network_params, f)

    def save_buffer(self, buffer, timestep, num_games):
        buffer_save_path = os.path.join(self.buffer_path, "Net{}_Game{}.pkl".format(timestep, num_games))
        with open(buffer_save_path, 'w+b') as f:
            pickle.dump(buffer, f)
        
    def log_game(self, game, network_step):
        
        # log expression
        self.num_games_saved += 1
        alpha_game = game.environment
        expression_tree = alpha_game.alpha
        expression_str = expression_tree.expression_str
        metric = game.environment.evaluate()
        alpha_value = game.environment.alpha.calculate()['val']
        alpha_value = alpha_value.value
        # visulalize the result and save to files
        
 

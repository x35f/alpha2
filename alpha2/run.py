
import click
from alpha2.utils import load_config, set_logger, Logger, set_global_seed
import munch
from alpha2.mcts.buffer import SharedStorage, ReplayBuffer
from alpha2.trainer import train
import jax
import pickle
import ray

@click.command()
@click.argument("config_path", type=str)
@click.option("--data-path", type=str, default='./data/computation_data.pkl')
@click.option("--log_path", type=str, default="./logs", help="path to save logs")
@click.option("--seed", type=int, default=35)
@click.option("--info", type=str, default="", help="notes for this experiment")
def alphadev(config_path, data_path, log_path, seed, info):
  # load data
  ray.init()
  
  with open(data_path, 'rb') as f:
    computation_data = pickle.load(f)

  set_global_seed(seed)
  # logging and device
  #load config
  config_dict = load_config(config_path)
  config = munch.munchify(config_dict)
  logger = Logger(computation_data, config, log_path, "Alpha", seed, info)
  set_logger(logger)
  
  logger.log_str_object("params.txt", log_dict=config_dict)

  #set task related parameters
  config.task_spec.num_actions = computation_data['num_actions']
  config.task_spec.num_operators = computation_data['num_operators']
  config.task_spec.num_operands = computation_data['num_operands']
  config.network.num_inputs = config.task_spec.max_expression_size + 1

  logger.log_str("Available devices:{}".format(jax.devices()))
  config.seed = seed

  #initialize buffer and network storage
  storage = SharedStorage(config.shared_storage)
  replay_buffer = ReplayBuffer(config)

  # start training
  train(config, storage, replay_buffer, computation_data)

if __name__ == "__main__":
  alphadev()







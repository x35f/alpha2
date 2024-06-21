from alpha2.mcts.play import play_game
from alpha2.mcts.update import _update_weights
from alpha2.utils import util
from alpha2.mcts.networks import make_uniform_network, Network
import ray
import jax.profiler
import jax
import optax
from jax import random as jrandom

@ray.remote(num_gpus=0.4) # actor 
class AlphaSearcher:
    def __init__(self, config, computation_data, id):
         
        self.config = config
        self.computation_data = computation_data
        self.id = id
        self.rng = jrandom.PRNGKey(config.seed + id)
        self.network = Network(config.network, config.task_spec, config.seed)
        self.uniform_network = make_uniform_network(config.task_spec.num_actions)
        self.num_alphas_found = 0
        self.gpu_id = int(ray.get_gpu_ids()[0])
        print(f"Searcher {id} initialized on gpu: {self.gpu_id}")


    def run_alpha_search(self, network_params, network_training_step, use_uniform_network=False): # task
        if use_uniform_network:
            game = play_game(
                self.config, self.uniform_network, self.rng, self.computation_data)
        else:
            self.network.set_params(network_params)
            game = play_game(
                self.config, self.network, self.rng, self.computation_data)
        self.num_alphas_found += 1
        metric =game.environment.evaluate()
        info = {"network_step": network_training_step, "searcher_id": self.id, "num_alphas_found": self.num_alphas_found,  'metric': metric}
        return game, info


@ray.remote(num_gpus=0.4,)
class NetworkTrainer:
    def __init__(self, config):
        self.config = config
        self.network = Network(config.network, config.task_spec, config.seed+35)
        self.target_network = Network(config.network, config.task_spec, config.seed + 66)
        self.optimizer = optax.sgd(
            config.trainer.lr.init_value, config.trainer.lr.momentum)
        self.optimizer_state = self.optimizer.init(self.network.get_params())

    def train_network(self, replay_buffer):
        data_batches = replay_buffer.sample_batches(td_steps=self.config.trainer.num_td_steps, batch_size=self.config.trainer.batch_size, num_batches=self.config.trainer.snapshot_interval)
        loss_infos = []
        for i, data_batch in enumerate(data_batches):
            if self.network.num_training_steps % self.config.trainer.update_target_network_interval == 0:
                self.target_network = self.network.copy()
            self.optimizer_state, info = _update_weights(
                self.optimizer, self.optimizer_state, self.network, self.target_network, data_batch)
            loss_infos.append([self.network.num_training_steps, info])
            self.network.num_training_steps += 1
        new_params = self.network.get_params()
        return new_params, loss_infos, self.network.num_training_steps
    
    def get_network(self):
        return self.network.get_params()

def save_search_results(search_results, num_games_finished ):
    for search_result in search_results:
        num_games_finished += 1
        game, info = search_result
        util.logger.log_game(game, info['network_step'])
        util.logger.log_tb_var("time/game_duration",
                               info['game_duration'], num_games_finished)
    return num_games_finished


def train(config, storage, replay_buffer, computation_data):
    computation_data = ray.put(computation_data)
    # initialize env parameters
    num_games_finished = 0
    latest_network, network_train_step = None, -1
    
    # initialized ray actors
    alpha_searchers = [AlphaSearcher.options(name=f"runner_{i}").remote(
        config, computation_data, i) for i in range(config.num_mcts_actors)]
    network_trainer = NetworkTrainer.options(name="trainer").remote(config)

    #submit initial jobs
    searching_list = [searcher.run_alpha_search.remote(
       latest_network, -1, use_uniform_network=True) for searcher in alpha_searchers]
    network_train_ref = []
    curr_train_step = 0

    # start alpha search
    while True:
        #  mcts searcher
        ready_searchers, searching_list = ray.wait(searching_list, timeout=2)
        if len(ready_searchers) > 0:
            searched_results = ray.get(ready_searchers)
            num_games_finished = save_search_results(searched_results, num_games_finished)
            for search_result in searched_results:
                game, info = search_result
                replay_buffer.save_game(game)
                searcher_id = info['searcher_id']
                if latest_network is None:
                    searching_list.append(alpha_searchers[searcher_id].run_alpha_search.remote(
                        latest_network, -1, use_uniform_network=True))
                else:
                    searching_list.append(alpha_searchers[searcher_id].run_alpha_search.remote(
                        latest_network, network_train_step, use_uniform_network=False))

        if replay_buffer.size < config.trainer.start_training_buffer_size: 
            continue

        if len(network_train_ref) == 0 and replay_buffer.size % config.trainer.snapshot_interval <= 10:
            network_train_ref = [network_trainer.train_network.remote(replay_buffer)]

        #network training loop
        ready_network, network_train_ref = ray.wait(
            network_train_ref, num_returns=1, timeout=2)
        ready_network = []
        if len(ready_network) > 0:
            [network_params, loss_infos, curr_train_step] = ray.get(ready_network)[
                0]
            util.logger.log_network(network_params, curr_train_step)
            for loss_info in loss_infos:
                t, losses = loss_info
                for k, v in losses.items():
                    util.logger.log_tb_var(k, v, t)
            storage.save_network(network_params, curr_train_step)
            latest_network, network_train_step = storage.latest_network()   
            util.logger.save_buffer(replay_buffer, curr_train_step, num_games_finished)



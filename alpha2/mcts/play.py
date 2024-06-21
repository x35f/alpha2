import math
import numpy as np
from typing import Sequence
from alpha2.mcts.structure import Node, ActionHistory
from alpha2.mcts.game import AlphaEnv, Game
from alpha2.mcts.networks import Network, NetworkOutput


def play_game(config: dict, network: Network, rng, computation_data) -> Game:

    game: Game = Game(config.task_spec.num_actions,
                      config.mcts.discount_factor, config.task_spec, computation_data)
    while not game.terminal():
        # Initialisation of the root node and addition of exploration noise
        
        root = Node(config.mcts.topk, 0)
        current_observation = game.make_observation(-1)
        network_output = network.inference(current_observation)
        
        _expand_node(
            root, game.legal_actions(), network_output, config.mcts.topk, reward=0
        )
        
        _backpropagate(
            [root],
            network_output.value,
            config.mcts.discount_factor,
        )
        _add_exploration_noise(config, root)
        run_mcts(  
            config,
            root,
            game.action_history(),
            network,
            game.environment,
            config.mcts.topk
        )
        action= _select_action(config, root, network)
        game.apply(action)
        game.store_search_statistics(root)
    return game

def run_mcts(
    config: dict,
    root: Node,
    action_history: ActionHistory,
    network: Network,
    env: AlphaEnv,
    topk: int,
):
    """Runs the Monte Carlo Tree Search algorithm.

    To decide on an action, we run N simulations, always starting at the root of
    the search tree and traversing the tree according to the UCB formula until we
    reach a leaf node.

    Args:
      config: configuration
      root: The root node of the MCTS tree from which we start the algorithm
      action_history: history of the actions taken so far.
      network: instances of the networks that will be used.
      env: an instance of the AlphaEnv.
    """
    #global total_nodes_deleted
    for _ in range(config.mcts.num_simulations):
        history = action_history.clone()
        sim_env = env.clone()
        node = root
        search_path = [node]
        while node.expanded():
            action, node = _select_child(config, node, env.max_expression_size)
            sim_env.step(action)
            history.add_action(action)
            search_path.append(node)
        
        observation, reward = sim_env.step(action)
        network_output = network.inference(observation)
        legal_actions = sim_env.legal_actions()
        _expand_node(
            node, legal_actions, network_output, topk, reward
        )
        _backpropagate(
            search_path,
            network_output.value,
            config.mcts.discount_factor,
        )

def softmax_sample(distribution, temperature: float):
    return 0,0 


def _select_action(
    config,  node: Node, network: Network
):
    visit_counts = [
        (child.visit_count, action) for action, child in node.children.items()
    ]
    t = config.visit_softmax_temperature_fn(
        training_steps=network.training_steps()
    )
    _, action = softmax_sample(visit_counts, t)
    return action

def _select_child(
    config, node: Node
):
  """Selects the child with the highest UCB score."""
  _, action, child = max(
      (_ucb_score(config, node, child), action, child)
      for action, child in node.children.items()
  )
  return action, child


def _ucb_score(
    config ,
    parent: Node,
    child: Node,
) -> float:
  """Computes the UCB score based on its value + exploration based on prior."""
  pb_c = (
      math.log((parent.visit_count + config.pb_c_base + 1) / config.pb_c_base)
      + config.pb_c_init
  )
  pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

  prior_score = pb_c * child.prior
  if child.visit_count > 0:
    value_score = child.reward + config.discount * child.value()
  else:
    value_score = 0
  return prior_score + value_score

def _expand_node(
    node: Node,
    actions: Sequence[int],
    network_output: NetworkOutput,
    topk: np.int8,
    reward: np.float16, 
):
    """Expands the node using value, reward and policy predictions from the NN."""
    node.hidden_state = network_output.hidden_state
    node.reward = reward
    policy = {a: math.exp(network_output.policy_logits[a]) for a in actions}
    policy_sum = sum(policy.values())
    for action, p in policy.items():
        node.children[action] = Node(topk, p / policy_sum)


def _backpropagate(
    search_path: Sequence[Node],
    value: float,
    discount: float,
):
    for node in reversed(search_path):
        node.add_value(value)
        node.visit_count += 1
        value = node.reward + discount * value

def _add_exploration_noise(config, node: Node):
  """Adds dirichlet noise to the prior of the root to encourage exploration."""
  actions = list(node.children.keys())
  noise = np.random.dirichlet([config.root_dirichlet_alpha] * len(actions))
  frac = config.root_exploration_fraction
  for a, n in zip(actions, noise):
    node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac

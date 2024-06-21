from typing import Sequence
from alpha2.mcts.structure import Node, ActionHistory, Target
from copy import deepcopy, copy
from alpha2.expression.port import ExpressionTree, fast_evaluate, get_legal_action_list
import jax.numpy as jnp
import jax
from typing import Dict
import numpy as np

class AlphaEnv(object):
  """The environment AlphaDev is interacting with."""

  def __init__(self, task_spec, computation_data):
    self.task_spec = task_spec
    self.max_expression_size = task_spec.max_expression_size
    self.alpha = ExpressionTree(task_spec.max_expression_size)
    self.program = []
    self.computation_data = computation_data
    self.step(computation_data['start_action_idx']) # always add a start action at the beginning

  def step(self, action):
    self.alpha.apply_action(action)
    reward = self.reward()
    observation = self.observation()
    return observation, reward

  def observation(self):
    program = jnp.ndarray([self.program])
    program_length = jnp.ndarray([[len(self.program)]]) 
    return {
        'program': program,
        'program_length': program_length
    }

  def reward(self) -> float:
    metric = self.evaluate()
    reward = metric - self.prev_metric
    self.prev_metric = metric
    return reward

  def evaluate(self) -> float:
    # evalute the self.alpha using the fast_evaluate function
    return 0

  def legal_actions(self) -> Dict:
    return get_legal_action_list(self.alpha, self.computation_data)
  
      
  def clone(self):
    new_self = copy(self)
    new_self.alpha = deepcopy(self.alpha)
    new_self.program = copy(self.program)
    return new_self

class Game(object):
  """A single episode of interaction with the environment."""

  def __init__(
      self, action_space_size: int, discount: float, task_spec: dict, computation_data: dict
  ):
    self.task_spec = task_spec
    self.computation_data = computation_data
    self.environment = AlphaEnv(task_spec, computation_data)
    self.history = []
    self.rewards = []
    self.heuristic_reward = 0
    self.child_visits = []
    self.root_values = []
    self.action_space_size = action_space_size
    self.finish_action_idx = computation_data['finish_action_idx']
    self.search_infos = []
    self.discount = discount

  def terminal(self) -> bool:
    # Game specific termination rules.
    return self.history[-1] == self.finish_action_idx
  
  def legal_actions(self) -> Sequence[int]:
    # calculation of legal actions.
    return self.environment.legal_actions()

  def apply(self, action: int):
    o, reward = self.environment.step(action)
    self.rewards.append(reward)
    self.history.append(action)
    if self.terminal():
      self.heuristic_reward = 0.005

  def store_search_statistics(self, root: Node, initial_visit_count: Dict):
    sum_visits = sum(root.child_visit_counts)
    #action_space = (index for index in range(self.action_space_size))
    visit_counts = np.zeros((self.action_space_size), dtype=np.float16)
    for i, (a, _) in enumerate(root.children):
      visit_counts[a] = root.child_visit_counts[i]
    self.search_infos.append({ "visits": visit_counts})


  def make_observation(self, state_index: int):
    if state_index == -1:
      return self.environment.observation(), None
    env = AlphaEnv(self.task_spec, self.computation_data)
    observation = env.observation()
    for action in self.history[:state_index]:
      observation, _ = env.step(action)
    return observation

  def make_target(
      self, state_index: int, td_steps: int
  ) -> Target:
    """Creates the value target for training."""
    # The value target is the discounted sum of all rewards until N steps
    # into the future, to which we will add the discounted boostrapped future
    # value.
    td_steps = min(len(self.rewards) - state_index, td_steps)
    bootstrap_index = state_index + td_steps
    value = 0
    
    for i, reward in enumerate(self.rewards[state_index:bootstrap_index]):
      value += reward * self.discount**i  # pytype: disable=unsupported-operands

    if bootstrap_index < len(self.root_values):
      bootstrap_discount = self.discount**td_steps
    else:
      bootstrap_discount = 0
    visit_distribution = jax.nn.softmax(jnp.ndarray(self.child_visits[state_index]))
    return Target(
        value,
        self.heuristic_reward,
        visit_distribution,
        bootstrap_discount,
    )


  def action_history(self) -> ActionHistory:
    return ActionHistory(self.history, self.action_space_size)
  
  def clean_up(self): # cleanup computation tree data to save gpu memory
    self.environment = None
    del self.computation_data
  
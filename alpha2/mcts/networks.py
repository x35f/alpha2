from typing import  Any ,Callable, Optional
import jax
import jax.numpy as jnp
import haiku as hk
import chex
import functools
from alpha2.mcts.structure import  NetworkOutput
from copy import deepcopy
from jax import random as jrandom
class Network(object):
  """Wrapper around Representation and Prediction networks."""

  def __init__(self, hparams: dict, task_spec: dict, seed):
    self.num_operators = task_spec.num_operators
    self.num_operands = task_spec.num_operands
    self.num_actions = task_spec.num_actions
    self.num_inputs = hparams.num_inputs
    self.embedding_dim = hparams.embedding_dim
    self.num_training_steps = 0
    self.PRNGkey = jrandom.PRNGKey(seed)

    self.representation = hk.transform(RepresentationNet(hparams, self.num_inputs, self.num_operators, self.num_operands, embedding_dim=self.embedding_dim))
    
    self.prediction = hk.transform(PredictionNet(
        task_spec=task_spec,
        value_max=hparams.value.max,
        value_num_bins=hparams.value.num_bins,
        embedding_dim=hparams.embedding_dim,
    ))
    rep_key, pred_key = self.PRNGkey.split()
    self.params = {
        'representation': self.representation.init(rep_key),
        'prediction': self.prediction.init(pred_key),
    }

  def inference(self, params: Any, observation: jnp.array) -> NetworkOutput:
    # representation + prediction function
    embedding = self.representation.apply(params['representation'], observation)
    return self.prediction.apply(params['prediction'], embedding)

  def set_params(self, params):
    for k, v in params.items():
      self.params[k]= jax.tree_util.tree_map(lambda x: x, params[k])
    
  def get_params(self):
    # Returns the weights of this network.
    return self.params

  def update_params(self, updates: Any) -> None:
    # Update network weights internally.
    self.params = jax.tree_map(lambda p, u: p + u, self.params, updates)

  @property
  def training_steps(self) -> int:
    return self.num_training_steps
  
  def copy(self):
    return deepcopy(self)


class UniformNetwork(object):
  """Network representation that returns uniform output."""

  # pylint: disable-next=unused-argument
  def inference(self, observation) -> NetworkOutput:
    # representation + prediction function
    return NetworkOutput(0, 0, 0, {})

  def get_params(self):
    # Returns the weights of this network.
    return self.params

  def update_params(self, updates: Any) -> None:
    # Update network weights internally.
    self.params = jax.tree_map(lambda p, u: p + u, self.params, updates)

  def training_steps(self) -> int:
    # How many steps / batches the network has been trained for.
    return 0


######## 2.2 Representation Network ########


class CustomAttentionBlock(hk.MultiHeadAttention): 
  """Attention with multiple query heads and a single shared key and value head.

  Implementation of "Fast Transformer Decoding: One Write-Head is All You Need",
  see https://arxiv.org/abs/1911.02150.
  """

class ResBlockV2(hk.Module):
  """Layer-normed variant of the block from https://arxiv.org/abs/1603.05027."""

def int2bin(integers_array: jnp.array) -> jnp.array:
  """Converts an array of integers to an array of its 32bit representation bits.

  Conversion goes from array of shape (S1, S2, ..., SN) to (S1, S2, ..., SN*32),
  i.e. all binary arrays are concatenated. Also note that the single 32-long
  binary sequences are reversed, i.e. the number 1 will be converted to the
  binary 1000000... . This is irrelevant for ML problems.

  Args:
    integers_array: array of integers to convert.

  Returns:
    array of bits (on or off) in boolean type.
  """
  flat_arr = integers_array.astype(jnp.int32).reshape(-1, 1)
  bin_mask = jnp.tile(2 ** jnp.arange(32), (flat_arr.shape[0], 1))
  return ((flat_arr & bin_mask) != 0).reshape(
      *integers_array.shape[:-1], integers_array.shape[-1] * 32
  )


def bin2int(binary_array: jnp.array) -> jnp.array:
  """Reverses operation of int2bin."""
  u_binary_array = binary_array.reshape(
      *binary_array.shape[:-1], binary_array.shape[-1] // 32, 32
  )
  exp = jnp.tile(2 ** jnp.arange(32), u_binary_array.shape[:-1] + (1,))
  return jnp.sum(exp * u_binary_array, axis=-1)

class RepresentationNet(hk.Module):
  """Representation network."""

  def __init__(
      self,
      hparams: dict,
      num_inputs: int,
      num_operators: int,
      num_operands: int,
      embedding_dim: int,
      name: str = 'representation',
  ):
    super().__init__(name=name)
    self._hparams = hparams
    self.num_inputs = num_inputs
    self.num_operators = num_operators
    self.num_operands = num_operands
    self._embedding_dim = embedding_dim

  def __call__(self, inputs):
    batch_size = inputs['program'].shape[0]

    program_encoding = None
    if self._hparams.representation.use_program:
      program_encoding = self._encode_program(inputs, batch_size)

    if (
        self._hparams.representation.use_locations
        and self._hparams.representation.use_locations_binary
    ):
      raise ValueError(
          'only one of `use_locations` and `use_locations_binary` may be used.'
      )
    locations_encoding = None
    if self._hparams.representation.use_locations:
      locations_encoding = self._make_locations_encoding_onehot(
          inputs, batch_size
      )
    elif self._hparams.representation.use_locations_binary:
      locations_encoding = self._make_locations_encoding_binary(
          inputs, batch_size
      )

    permutation_embedding = None
    if self._hparams.representation.use_permutation_embedding:
      permutation_embedding = self.make_permutation_embedding(batch_size)

    return self.aggregate_locations_program(
        locations_encoding, permutation_embedding, program_encoding, batch_size
    )

  def _encode_program(self, inputs, batch_size):
    program = inputs['program'] # unpadded program
    max_program_size = inputs['program'].shape[1]
    program_length = inputs['program_length'].astype(jnp.int32)
    program_onehot = self.make_program_onehot(
        program, batch_size, max_program_size
    )
    program_encoding = self.apply_program_mlp_embedder(program_onehot)
    program_encoding = self.apply_program_attention_embedder(program_encoding)
    program_encoding = self.pad_program_encoding(
        program_encoding, batch_size, program_length, max_program_size
    )
    return program_encoding

  def aggregate_locations_program(
      self,
      locations_encoding,
      unused_permutation_embedding,
      program_encoding,
      batch_size,
  ):
    locations_embedder = hk.Sequential(
        [
            hk.Linear(self._embedding_dim),
            hk.LayerNorm(axis=-1),
            jax.nn.relu,
            hk.Linear(self._embedding_dim),
        ],
        name='per_locations_embedder',
    )

    # locations_encoding.shape == [B, P, D] so map embedder across locations to
    # share weights
    locations_embedding = hk.vmap(
        locations_embedder, in_axes=1, out_axes=1, split_rng=False
    )(locations_encoding)

    program_encoded_repeat = self.repeat_program_encoding(
        program_encoding, batch_size
    )

    grouped_representation = jnp.concatenate(
        [locations_embedding, program_encoded_repeat], axis=-1
    )

    return self.apply_joint_embedder(grouped_representation, batch_size)
  
  def repeat_program_encoding(self, program_encoding, batch_size):
    return jnp.broadcast_to(
        program_encoding,
        [batch_size, self.num_inputs, program_encoding.shape[-1]],
    )

  def apply_joint_embedder(self, grouped_representation, batch_size):
    all_locations_net = hk.Sequential(
        [
            hk.Linear(self._embedding_dim),
            hk.LayerNorm(axis=-1),
            jax.nn.relu,
            hk.Linear(self._embedding_dim),
        ],
        name='per_element_embedder',
    )
    joint_locations_net = hk.Sequential(
        [
            hk.Linear(self._embedding_dim),
            hk.LayerNorm(axis=-1),
            jax.nn.relu,
            hk.Linear(self._embedding_dim),
        ],
        name='joint_embedder',
    )
    joint_resnet = [
        ResBlockV2(self._embedding_dim, name=f'joint_resblock_{i}')
        for i in range(self._hparams.representation.repr_net_res_blocks)
    ]

    chex.assert_shape(
        grouped_representation, (batch_size, self._task_spec.num_inputs, None)
    )
    permutations_encoded = all_locations_net(grouped_representation)
    # Combine all permutations into a single vector.
    joint_encoding = joint_locations_net(jnp.mean(permutations_encoded, axis=1))
    for net in joint_resnet:
      joint_encoding = net(joint_encoding)
    return joint_encoding
  
  def make_program_onehot(self, program, batch_size, max_program_size):
    operator = program[:, :, 0] #batch_size, program_length, 3
    operand1 = program[:, :, 1]
    operand2 = program[:, :, 2]
    operator_onehot = jax.nn.one_hot(operator, self.num_operators)
    operand1_onehot = jax.nn.one_hot(operand1, self.num_operands)
    operand2_onehot = jax.nn.one_hot(operand2, self.num_operands)
    #register_onehot = jax.nn.one_hot(register, self._task_spec.num_registers)
    program_onehot = jnp.concatenate(
        [operator_onehot, operand1_onehot, operand2_onehot], axis=-1
    )
    chex.assert_shape(program_onehot, (batch_size, max_program_size, None))
    return program_onehot

  def pad_program_encoding(
      self, program_encoding, batch_size, program_length, max_program_size
  ):
    """Pads the program encoding to account for state-action stagger."""
    chex.assert_shape(program_encoding, (batch_size, max_program_size, None))

    empty_program_output = jnp.zeros(
        [batch_size, program_encoding.shape[-1]],
    )
    program_encoding = jnp.concatenate(
        [empty_program_output[:, None, :], program_encoding], axis=1
    )

    program_length_onehot = jax.nn.one_hot(program_length, max_program_size + 1)

    program_encoding = jnp.einsum(
        'bnd,bNn->bNd', program_encoding, program_length_onehot
    )

    return program_encoding


  def apply_program_mlp_embedder(self, program_encoding):
    program_embedder = hk.Sequential(
        [
            hk.Linear(self._embedding_dim),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
            jax.nn.relu,
            hk.Linear(self._embedding_dim),
        ],
        name='per_instruction_alpha_embedder',
    )

    program_encoding = program_embedder(program_encoding)
    return program_encoding

  def apply_program_attention_embedder(self, program_encoding):
    attention_params = self._hparams.attention
    *_, seq_size, feat_size = program_encoding.shape
    make_attention_block = functools.partial(
        CustomAttentionBlock, attention_params,model_size=feat_size# causal_mask=False
    )
    attention_encoders = [
        make_attention_block(name=f'attention_embed_block_{i}')
        for i in range(self._hparams.attention.num_layers)
    ]

    position_encodings = jnp.broadcast_to(
        CustomAttentionBlock.sinusoid_position_encoding(
            seq_size, feat_size, causal=True
        ),
        program_encoding.shape,
    )
    program_encoding += position_encodings
    for i, e in enumerate(attention_encoders):
      program_encoding = e(program_encoding) #todo: encoded_state=None)
    return program_encoding

  def _make_locations_encoding(self, inputs, batch_size):
    """Creates location encoding using onehot representation."""
    registers = inputs['registers']
    locations = registers  # [B, H, P, D]
    locations = jnp.transpose(locations, [0, 2, 1, 3])  # [B, P, H, D]

    # One-hot encode the values in the memory and average everything across
    # permutations.
    locations_onehot = jax.nn.one_hot(
        locations, self._task_spec.num_location_values, dtype=jnp.int32
    )
    locations_onehot = locations_onehot.reshape(
        [batch_size, self._task_spec.num_inputs, -1]
    )

    return locations_onehot
  def _make_locations_encoding_binary(self, inputs, batch_size):
    """Creates location encoding using binary representation."""

    memory_binary = int2bin(inputs['memory']).astype(jnp.float32)
    registers_binary = int2bin(inputs['registers']).astype(jnp.float32)
    # Note the extra I dimension for the length of the binary integer (32)
    locations = jnp.concatenate(
        [memory_binary, registers_binary], axis=-1
    )  # [B, H, P, D*I]
    locations = jnp.transpose(locations, [0, 2, 1, 3])  # [B, P, H, D*I]

    locations = locations.reshape([batch_size, self._task_spec.num_inputs, -1])

    return locations
 

######## 2.3 Prediction Network ########


def make_head_network(
    embedding_dim: int,
    output_size: int,
    resnet_params: dict,
    num_hidden_layers: int = 2,
    name: Optional[str] = None,
) -> Callable[[jnp.ndarray,], jnp.ndarray]:
  return hk.Sequential(
      [ResBlockV2(embedding_dim, num_layers=resnet_params.num_layers, use_projection=resnet_params.use_projection, name=f'head_resblock_{i}') for i in range(num_hidden_layers)]
      + [hk.Linear(output_size)],
      name=name,
  )


class DistributionSupport(object):

  def __init__(self, value_max: float, num_bins: int):
    self.value_max = value_max
    self.num_bins = num_bins

  def mean(self, logits: jnp.ndarray) -> float:
    pass

  def scalar_to_two_hot(self, scalar: float) -> jnp.ndarray:
    pass

class CategoricalHead(hk.Module):
  """A head that represents continuous values by a categorical distribution."""

  def __init__(
      self,
      embedding_dim: int,
      support: DistributionSupport,
      name: str = 'CategoricalHead',
  ):
    super().__init__(name=name)
    self._value_support = support
    self._embedding_dim = embedding_dim
    self._head = make_head_network(
        embedding_dim, output_size=self._value_support.num_bins
    )

  def __call__(self, x: jnp.ndarray):
    # For training returns the logits, for inference the mean.
    logits = self._head(x)
    probs = jax.nn.softmax(logits)
    mean = jax.vmap(self._value_support.mean)(probs)
    return dict(logits=logits, mean=mean)
  

class PredictionNet(hk.Module):
  """MuZero prediction network."""

  def __init__(
      self,
      task_spec,
      value_max: float,
      value_num_bins: int,
      embedding_dim: int,
      name: str = 'prediction',
  ):
    super().__init__(name=name)
    self.task_spec = task_spec
    self.support = DistributionSupport(self.value_max, self.value_num_bins)
    self.embedding_dim = embedding_dim

  def __call__(self, embedding: jnp.ndarray):
    policy_head = make_head_network(
        self.embedding_dim, self.task_spec.num_actions
    )
    value_head = CategoricalHead(self.embedding_dim, self.support)
    latency_value_head = CategoricalHead(self.embedding_dim, self.support)
    correctness_value = value_head(embedding)
    latency_value = latency_value_head(embedding)

    return NetworkOutput(
        value=correctness_value['mean'] + latency_value['mean'],
        metric_logits=correctness_value['logits'],
        heuristic_reward_logits=latency_value['logits'],
        policy=policy_head(embedding),
    )
  

def make_uniform_network(num_actions):
  return UniformNetwork(num_actions)

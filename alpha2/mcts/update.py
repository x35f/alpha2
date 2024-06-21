from typing import Sequence, Any
import optax
import jax.numpy as jnp
import jax
from .networks import Network
from .structure import Sample

def _loss_fn(
    network_params: jnp.array,
    target_network_params: jnp.array,
    network: Network,
    target_network: Network,
    batch: Sequence[Sample]
) -> float:
  """Computes loss."""
  loss = 0
  for observation, bootstrap_obs, target in batch:
    predictions = network.inference(network_params, observation)
    bootstrap_predictions = target_network.inference(
        target_network_params, bootstrap_obs)
    target_correctness, target_latency, target_policy, bootstrap_discount = (
        target
    )
    target_correctness += (
        bootstrap_discount * bootstrap_predictions.metric_logits
    )

    l = optax.softmax_cross_entropy(predictions.policy_logits, target_policy)
    l += scalar_loss(
        predictions.metric_logits, target_correctness, network
    )
    l += scalar_loss(predictions.heuristic_reward_logits, target_latency, network)
    loss += l
  loss /= len(batch)
  return loss

_loss_grad = jax.grad(_loss_fn, argnums=0)

def _update_weights(
    optimizer: optax.GradientTransformation,
    optimizer_state: Any,
    network: Network,
    target_network: Network,
    batch: Sequence[Sample],
) -> Any:
  """Updates the weight of the network."""
  updates = _loss_grad(
      network.get_params(),
      target_network.get_params(),
      network,
      target_network,
      batch)

  optim_updates, new_optim_state = optimizer.update(updates, optimizer_state)
  network.update_params(optim_updates)
  return new_optim_state

def scalar_loss(prediction, target, network) -> float:
  support = network.prediction.support
  return optax.softmax_cross_entropy(
      prediction, support.scalar_to_two_hot(target)
  )

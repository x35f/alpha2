from typing import Sequence
from .structure import Sample


class ReplayBuffer(object):
    """Replay buffer object storing games for training."""

    def __init__(self, config):
        self.max_buffer_size = config.buffer.max_buffer_size
        self.buffer = []

    @property
    def size(self):
        return len(self.buffer)

    def save_game(self, game):
        if len(self.buffer) > self.max_buffer_size:
            self.buffer.pop(0)
        game.clean_up()
        self.buffer.append(game)

    def sample_batches(self, td_steps: int, batch_size: int, num_batches: int):
        batches = []
        for i in range(num_batches):
            batches.append(self.sample_batch(
                td_steps=td_steps, batch_size=batch_size))
        return batches
    def sample_batch(self, td_steps: int) -> Sequence[Sample]:
        games = [self.sample_game() for _ in range(self.batch_size)]
        game_pos = [(g, self.sample_position(g)) for g in games]
        # pylint: disable=g-complex-comprehension
        return [
            Sample(
                observation=g.make_observation(i),
                bootstrap_observation=g.make_observation(i + td_steps),
                target=g.make_target(i, td_steps, g.to_play()),
            )
            for (g, i) in game_pos
        ]
        # pylint: enable=g-complex-comprehension

    def sample_game(self):
        # Sample game from buffer either uniformly or according to some priority.
        return self.buffer[0]

    # pylint: disable-next=unused-argument
    def sample_position(self, game) -> int:
        # Sample position from game either uniformly or according to some priority.
        return -1


# parallel implementation
class SharedStorage(object):
    """Controls which network is used at inference."""

    def __init__(self, config: dict):
        self._networks = []
        self.max_num_networks = config.max_num_networks

    def latest_network(self):
        net, step = self._networks[-1]
        return net, step

    def save_network(self, network_params, step: int, ):
        self._networks.append([network_params, step])
        self.clean_network()

    def clean_network(self):
        num_current_networks = len(self._networks)
        if num_current_networks > self.max_num_networks:
            for i in range(self.max_num_networks - num_current_networks):
                del self._networks[0]

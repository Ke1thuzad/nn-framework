import random
import numpy as np
from ..layers import Linear
from ..model import Sequential, Model
from ..optimizers import GradientClipping


class Genome:
    def __init__(self, input_dim, output_dim, search_space, layers_cfg=None, lr=None, opt_class=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.search_space = search_space
        self.layers_cfg = layers_cfg or self._generate_random_layers()
        self.lr = lr or random.choice(self.search_space.lr_rates)
        self.opt_class = opt_class or random.choice(self.search_space.optimizers)

        self.fitness = -float('inf')
        self.best_score = 0.0
        self.best_penalized_score = -float('inf')
        self.best_weights = None

    def _generate_random_layers(self):
        num_layers = random.randint(self.search_space.min_layers, self.search_space.max_layers)
        return [
            (random.choice(self.search_space.hidden_sizes), random.choice(self.search_space.activations))
            for _ in range(num_layers)
        ]

    def build_model(self, loss_fn, metrics, output_activation=None):
        layers = []
        prev_dim = self.input_dim
        for size, act_class in self.layers_cfg:
            layers.append(Linear(prev_dim, size))
            layers.append(act_class())
            prev_dim = size

        layers.append(Linear(prev_dim, self.output_dim))
        if output_activation:
            layers.append(output_activation())

        net = Sequential(*layers)
        base_opt = self.opt_class(net.parameters(), lr=self.lr)
        opt = GradientClipping(base_opt, clip_value=1.0)
        return Model(net, loss_fn, opt, metrics=metrics)

    def save_weights(self, model):
        self.best_weights = [np.copy(p.data) for p in model.network.parameters()]

    def load_weights(self, model):
        if self.best_weights is not None:
            for p, w in zip(model.network.parameters(), self.best_weights):
                p.data = np.copy(w)

    @property
    def params_count(self):
        count = 0
        prev = self.input_dim
        for size, _ in self.layers_cfg:
            count += (prev * size) + size
            prev = size
        count += (prev * self.output_dim) + self.output_dim
        return count

    def summary(self):
        arch = " -> ".join([f"{s}({a.__name__})" for s, a in self.layers_cfg])
        return f"[{self.opt_class.__name__} | LR: {self.lr}] Arch: In -> {arch} -> Out"
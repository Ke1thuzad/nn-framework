import time
import random
import copy
import math
from .genome import Genome


class Island:
    def __init__(self, name, objective, input_dim, output_dim, search_space, config):
        self.name = name
        self.objective = objective
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.search_space = search_space
        self.config = config
        self.population = [Genome(input_dim, output_dim, search_space) for _ in range(config.pop_size)]

    def evaluate(self, train_loader, val_loader, loss_fn, metrics, output_activation):
        target_metric_key = metrics[0].__class__.__name__.lower()

        for genome in self.population:
            model = genome.build_model(loss_fn, metrics, output_activation)
            genome.load_weights(model)

            start_time = time.time()
            try:
                model.fit(train_loader, epochs=self.config.eval_epochs)

                train_res = model.evaluate(train_loader)
                val_res = model.evaluate(val_loader)

                train_score = train_res.get(target_metric_key, 0.0)
                val_score = val_res.get(target_metric_key, 0.0)

                if math.isnan(val_score) or math.isinf(val_score):
                    val_score, train_score = 0.0, 0.0
            except Exception:
                val_score, train_score = 0.0, 0.0

            duration = time.time() - start_time
            overfit_gap = max(0.0, train_score - val_score)
            penalized_score = val_score - (overfit_gap * self.config.overfit_penalty)

            if penalized_score > genome.best_penalized_score:
                genome.best_penalized_score = penalized_score
                genome.best_score = val_score
                genome.save_weights(model)

            base_fitness = self.objective.calculate(genome.best_score, genome, duration)
            genome.fitness = base_fitness - (overfit_gap * self.config.overfit_penalty)

    def evolve(self):
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        survivors_count = max(1, int(self.config.pop_size * self.config.survival_rate))
        survivors = self.population[:survivors_count]

        new_gen = [copy.deepcopy(survivors[0])]

        while len(new_gen) < self.config.pop_size:
            p1, p2 = random.sample(survivors, 2) if len(survivors) > 1 else (survivors[0], survivors[0])
            child = self._crossover(p1, p2)
            self._mutate(child)
            new_gen.append(child)

        self.population = new_gen

    def _crossover(self, p1, p2):
        split = random.randint(0, min(len(p1.layers_cfg), len(p2.layers_cfg)))
        new_cfg = p1.layers_cfg[:split] + p2.layers_cfg[split:]
        lr = random.choice([p1.lr, p2.lr])
        opt = random.choice([p1.opt_class, p2.opt_class])

        child = Genome(self.input_dim, self.output_dim, self.search_space, new_cfg, lr, opt)
        child.best_weights = None
        return child

    def _mutate(self, genome):
        struct_mutated = False

        if random.random() < self.config.mut_rate_lr:
            genome.lr = random.choice(self.search_space.lr_rates)

        if random.random() < self.config.mut_rate_opt:
            genome.opt_class = random.choice(self.search_space.optimizers)

        if random.random() < self.config.mut_rate_layer and genome.layers_cfg:
            idx = random.randint(0, len(genome.layers_cfg) - 1)
            new_size = random.choice(self.search_space.hidden_sizes)
            new_act = random.choice(self.search_space.activations)
            genome.layers_cfg[idx] = (new_size, new_act)
            struct_mutated = True

        if random.random() < self.config.mut_rate_add_layer and len(genome.layers_cfg) < self.search_space.max_layers:
            new_layer = (random.choice(self.search_space.hidden_sizes), random.choice(self.search_space.activations))
            genome.layers_cfg.append(new_layer)
            struct_mutated = True

        if random.random() < self.config.mut_rate_rm_layer and len(genome.layers_cfg) > self.search_space.min_layers:
            idx = random.randint(0, len(genome.layers_cfg) - 1)
            genome.layers_cfg.pop(idx)
            struct_mutated = True

        if struct_mutated:
            genome.best_weights = None
            genome.best_score = 0.0
            genome.best_penalized_score = -float('inf')
import copy
from .island import Island
from .objectives import AccuracyObjective, EfficiencyObjective, SpeedObjective


class EvolutionaryEngine:
    def __init__(self, input_dim, output_dim, loss_fn, metrics, search_space, config, output_activation=None):
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.output_activation = output_activation
        self.config = config

        self.islands = [
            Island("Sniper", AccuracyObjective(), input_dim, output_dim, search_space, config),
            Island("Mobile", EfficiencyObjective(config.efficiency_penalty), input_dim, output_dim, search_space,
                   config),
            Island("Flash", SpeedObjective(config.time_penalty, config.layer_penalty), input_dim, output_dim,
                   search_space, config)
        ]

    def run(self, train_loader, val_loader):
        target_metric = self.metrics[0].__class__.__name__.lower()

        for gen in range(self.config.generations):
            print(f"\n--- Generation {gen + 1}/{self.config.generations} ---")
            for island in self.islands:
                island.evaluate(train_loader, val_loader, self.loss_fn, self.metrics, self.output_activation)
                island.evolve()
                best = island.population[0]
                print(f"[{island.name}] Val {target_metric.capitalize()}: {best.best_score:.4f} | {best.summary()}")

            if (gen + 1) % self.config.migration_interval == 0:
                self._migrate()

            if (gen + 1) % self.config.cataclysm_interval == 0:
                self._cataclysm()

        return self._get_best()

    def _migrate(self):
        for i in range(len(self.islands)):
            src = self.islands[i]
            tgt = self.islands[(i + 1) % len(self.islands)]
            tgt.population[-1] = copy.deepcopy(src.population[0])

    def _cataclysm(self):
        for island in self.islands:
            survivors_count = max(1, int(self.config.pop_size * self.config.cataclysm_survival_rate))
            island.population[survivors_count:] = [
                type(island.population[0])(island.input_dim, island.output_dim, island.search_space)
                for _ in range(self.config.pop_size - survivors_count)
            ]

    def _get_best(self):
        all_best = [isl.population[0] for isl in self.islands]
        all_best.sort(key=lambda x: x.best_score, reverse=True)
        return all_best[0]
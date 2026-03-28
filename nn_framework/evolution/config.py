class SearchSpace:
    def __init__(self, activations, hidden_sizes, lr_rates, optimizers, min_layers, max_layers):
        self.activations = activations
        self.hidden_sizes = hidden_sizes
        self.lr_rates = lr_rates
        self.optimizers = optimizers
        self.min_layers = min_layers
        self.max_layers = max_layers

class EvolutionConfig:
    def __init__(self,
                 pop_size=10,
                 generations=15,
                 eval_epochs=2,
                 survival_rate=0.5,
                 mut_rate_lr=0.3,
                 mut_rate_opt=0.2,
                 mut_rate_layer=0.4,
                 mut_rate_add_layer=0.2,
                 mut_rate_rm_layer=0.2,
                 migration_interval=3,
                 cataclysm_interval=7,
                 cataclysm_survival_rate=0.2,
                 efficiency_penalty=0.0001,
                 time_penalty=0.05,
                 layer_penalty=0.02,
                 overfit_penalty=0.5):
        self.pop_size = pop_size
        self.generations = generations
        self.eval_epochs = eval_epochs
        self.survival_rate = survival_rate
        self.mut_rate_lr = mut_rate_lr
        self.mut_rate_opt = mut_rate_opt
        self.mut_rate_layer = mut_rate_layer
        self.mut_rate_add_layer = mut_rate_add_layer
        self.mut_rate_rm_layer = mut_rate_rm_layer
        self.migration_interval = migration_interval
        self.cataclysm_interval = cataclysm_interval
        self.cataclysm_survival_rate = cataclysm_survival_rate
        self.efficiency_penalty = efficiency_penalty
        self.time_penalty = time_penalty
        self.layer_penalty = layer_penalty
        self.overfit_penalty = overfit_penalty
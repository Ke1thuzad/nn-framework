class Objective:
    def calculate(self, accuracy, genome, training_time):
        raise NotImplementedError

class AccuracyObjective(Objective):
    def calculate(self, accuracy, genome, training_time):
        return accuracy

class EfficiencyObjective(Objective):
    def __init__(self, penalty_weight):
        self.penalty_weight = penalty_weight

    def calculate(self, accuracy, genome, training_time):
        return accuracy - (genome.params_count * self.penalty_weight)

class SpeedObjective(Objective):
    def __init__(self, time_penalty_weight, layer_penalty_weight):
        self.time_penalty_weight = time_penalty_weight
        self.layer_penalty_weight = layer_penalty_weight

    def calculate(self, accuracy, genome, training_time):
        return accuracy - (training_time * self.time_penalty_weight) - (len(genome.layers_cfg) * self.layer_penalty_weight)
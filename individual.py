import copy

class Individual:

    def __init__(self):
        self.chromosome = []
        self.fitness = None
        self.rank = None
        self.crowding_distance = 0
        self.dominated_individuals = []
        self.domination_count = 0
        self.status = 'parent'
        self.memory = []
        self.swap = None

    def set_parent(self):
        self.status = 'parent'
    
    def set_offspring(self):
        self.status = 'offspring'

    def save_chromosome(self):
        self.memory = copy.deepcopy(self.chromosome)

    def get_memory(self):
        return self.memory
    
    def save_swap(self, swap):
        self.swap = swap

    def get_swap(self):
        return (self.swap[0], self.swap[1])

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.features == other.features
        return False
    
    def dominates(self, other_individual):
        better_or_equal_in_all = True
        better_in_at_least_one = False
        for self_fit, other_fit in zip(self.fitness, other_individual.fitness):
            if self_fit > other_fit:
                better_or_equal_in_all = False
            elif self_fit < other_fit:
                better_in_at_least_one = True
        return better_or_equal_in_all and better_in_at_least_one


    
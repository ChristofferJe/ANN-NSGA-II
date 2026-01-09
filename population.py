class Population:

    def __init__(self):
        self.individuals =  []
        self.fronts = []
    
    def add_individual(self, individual):
        self.individuals.append(individual)

    def extend_individuals(self, individuals):
        self.individuals.extend(individuals)
    
    def __iter__(self):
        return self.individuals.__iter__()
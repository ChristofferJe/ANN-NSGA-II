from individual import Individual
import random

class Problem:

    def __init__(self, objectives, cost_matrix):
        self.objectives = objectives
        self.cost_matrix = cost_matrix
        self.n = len(cost_matrix)

    def generate_individual(self):
        individual = Individual()
        individual.chromosome = random.sample(range(self.n), self.n)
        return individual
    
    def evaluate_fitness(self, individual):
        individual.fitness = [f(individual.chromosome, self.cost_matrix) for f in self.objectives]



    

from population import Population
from swapDataset import SwapDataset
from ANNswap import ANNSwap
from ANNswap import train_ann_epoch
import torch

class Evolution:
    
    def __init__(self, problem, GA, hybrid=True, batch_size = 32, epochs = 1, learning_rate=0.001):
        self.problem = problem
        self.GA = GA
        self.population = GA.create_initial_solution()
        self.hybrid = hybrid 
        self.swapDataset = SwapDataset(self.problem.n)
        self.ANNSwap = ANNSwap(self.problem.n)
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.ANNSwap.parameters(), lr=self.learning_rate)
        
    def evolve(self, n_generations):
        for generation in range(n_generations):
            if generation % 10 == 0:
                print(f"Generation {generation}")
            self.evolve_generation(generation, n_generations)
        self.GA.fast_non_dominated_sort(self.population)

    def evolve_generation(self, generation, n_generations):
        # Define new population
        new_population = Population()

        # Generate offspring
        self.GA.fast_non_dominated_sort(self.population)
        offspring = self.GA.generate_offspring(self.population, generation, n_generations, self.ANNSwap, self.hybrid)
        self.population.extend_individuals(offspring)

        # Sort combined population and select next generation population
        self.GA.fast_non_dominated_sort(self.population)
        for front in self.population.fronts:
            self.GA.calculate_crowding_distance(front)
        while len(new_population.individuals) + len(self.population.fronts[0]) <= self.GA.population_size:
            new_population.extend_individuals(self.population.fronts.pop(0))
        self.population.fronts[0].sort(key = lambda ind: ind.crowding_distance, reverse = True)
        new_population.extend_individuals(self.population.fronts[0][:self.GA.population_size - len(new_population.individuals)])
        self.population = new_population

        # Find offspring in new generation and add to swap dataset
        if self.hybrid:
            for individual in self.population.individuals:
                if individual.status == 'offspring':
                    self.swapDataset.add(individual)
                individual.set_parent()

        # Train ANN swap model if dataset is large enough
            if len(self.swapDataset) >= self.batch_size:
                for epoch in range(self.epochs):
                    train_ann_epoch(self.ANNSwap, self.optimizer, self.swapDataset, batch_size = self.batch_size)


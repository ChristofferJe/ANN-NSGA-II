from population import Population
import random
from ANNSwap import predict_swap_indices


class NSGAII:

    def __init__(self, problem, population_size = 50, tournament_size = 2, ANN_warmup = 0.2, ANN_rampup = 0.6):
        self.problem = problem
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.ANN_warmup = ANN_warmup
        self.ANN_rampup = ANN_rampup

    def create_initial_solution(self):
        population = Population()
        for _ in range(self.population_size):
            individual = self.problem.generate_individual()
            self.problem.evaluate_fitness(individual)
            population.add_individual(individual)
        return population
    
    def fast_non_dominated_sort(self, population):
        population.fronts = [[]]
        for individual in population:
            individual.dominated_individuals = []
            for other_individual in population:
                if individual.dominates(other_individual):
                    individual.dominated_individuals.append(other_individual)
                elif other_individual.dominates(individual):
                    individual.domination_count += 1
            if individual.domination_count == 0:
                individual.rank = 0
                population.fronts[0].append(individual)
        
        i=0
        while(len(population.fronts[i]) > 0): 
            next_front = []
            for individual in population.fronts[i]:
                for dominated_individual in individual.dominated_individuals:
                    dominated_individual.domination_count -= 1
                    if dominated_individual.domination_count == 0:
                        next_front.append(dominated_individual)
                        dominated_individual.rank = i + 1
            i += 1
            population.fronts.append(next_front)
            
    def calculate_crowding_distance(self, front):
        num_individuals = len(front)
        if num_individuals == 0:
            return
        for m in range(len(self.problem.objectives)):
            sorted_front = sorted(front, key = lambda individual: individual.fitness[m])
            scale = sorted_front[-1].fitness[m] - sorted_front[0].fitness[m]
            if scale == 0:
                continue
            sorted_front[0].crowding_distance = float('inf')
            sorted_front[-1].crowding_distance = float('inf')
            for i in range(1, num_individuals - 1):
                sorted_front[i].crowding_distance += (sorted_front[i + 1].fitness[m] - sorted_front[i - 1].fitness[m]) / scale
    
    def crowded_comparison_operator(self, individual1, individual2):
        if individual1.rank < individual2.rank:
            return individual1
        elif individual1.rank > individual2.rank:
            return individual2
        elif individual1.crowding_distance > individual2.crowding_distance:
            return individual1
        else:
            return individual2
    
    def generate_offspring(self, population, generation, n_generations, ANNSwap, hybrid=True):
        offspring = []
        while len(offspring) < self.population_size:
            parent1 = self._binary_tournament_selection(population)
            parent2 = self._binary_tournament_selection(population)
            while parent1.chromosome == parent2.chromosome:
                parent2 = self._binary_tournament_selection(population)
            new_offspring = self._pmx_crossover(parent1, parent2)
            self._swap_mutation(new_offspring, generation, n_generations, ANNSwap, hybrid)
            self.problem.evaluate_fitness(new_offspring)
            offspring.append(new_offspring)
        return offspring

    def _binary_tournament_selection(self, population):
        participants = random.sample(population.individuals, self.tournament_size)
        winner = participants[0]
        for ind in participants[1:]:
            winner = self.crowded_comparison_operator(winner, ind)
        return winner

    def _pmx_crossover(self, parent1, parent2):
        cut1, cut2 = sorted(random.sample(range(len(parent1.chromosome)+1), 2))
        offspring = self.problem.generate_individual()
        offspring.set_offspring()
        offspring.chromosome = [None]*len(parent1.chromosome)
        offspring.chromosome[cut1:cut2] = parent1.chromosome[cut1:cut2]
        self._pmx_fill(offspring, parent1, parent2, cut1, cut2)
        return offspring
    
    def _pmx_fill(self, offspring, middle_parent, other_parent, cut1, cut2):
        for i in (*range(0, cut1), *range(cut2, len(offspring.chromosome))):
            gene_candidate = other_parent.chromosome[i]
            while gene_candidate in offspring.chromosome[cut1:cut2]:
                index = middle_parent.chromosome.index(gene_candidate)
                gene_candidate = other_parent.chromosome[index]
            offspring.chromosome[i] = gene_candidate

    def _swap_mutation(self, individual, generation, n_generations, ANNSwap, hybrid):
        individual.save_chromosome()
        if (generation < n_generations * self.ANN_warmup) or not hybrid:
            idx1, idx2 = random.sample(range(len(individual.chromosome)), 2)
        else:
            prob_ann = min(0.7, (generation - n_generations * self.ANN_warmup) / (self.ANN_rampup * n_generations))
            if random.random() < prob_ann:
                idx1, idx2 = predict_swap_indices(ANNSwap, individual)
            else:
                idx1, idx2 = random.sample(range(len(individual.chromosome)), 2)

        individual.swap = (idx1, idx2)
        individual.chromosome[idx1], individual.chromosome[idx2] = individual.chromosome[idx2], individual.chromosome[idx1]

    




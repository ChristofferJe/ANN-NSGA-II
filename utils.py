import matplotlib.pyplot as plt

def plot_population(ax, population, title = "Population", with_rank=True):
    f1_values = []
    f2_values = []
    for individual in population.individuals:
        f1_values.append(individual.fitness[0])
        f2_values.append(individual.fitness[1])
    if with_rank:
        rank = [ind.rank for ind in population.individuals]
        ax.scatter(f1_values, f2_values, c=rank, cmap='viridis')
    else:
        ax.scatter(f1_values, f2_values)
    ax.set_title(title)
    ax.set_xlabel('Objective 1')
    ax.set_ylabel('Objective 2')

def test_valid(population):
    n = len(population.individuals[0].chromosome)
    for individual in population.individuals:
        if sorted(individual.chromosome) != list(range(n)):
            return False
    return True
    
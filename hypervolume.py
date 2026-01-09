from pygmo import hypervolume

def get_reference_point(population1, population2):
    pareto_fitness = [ind.fitness for ind in population1.fronts[0] + population2.fronts[0]]
    nadir1, nadir2 = max(f[0] for f in pareto_fitness), max(f[1] for f in pareto_fitness)
    r1, r2 = nadir1 * 1.05, nadir2 *1.05
    return [r1, r2]

def calculate_hypervolume(population, reference_point):
    pareto_fitness = [ind.fitness for ind in population.fronts[0]]
    hv = hypervolume(pareto_fitness)
    hypervol = hv.compute(reference_point)
    return hypervol
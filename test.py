from NSGAII import NSGAII
from evolution import Evolution
from assignmentProblem import Problem
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path
from hypervolume import get_reference_point, calculate_hypervolume
from utils import plot_population, test_valid

path = Path.cwd() / "data" / "assign200.txt"

# Get the cost matrix from the problem instance
with open(path, "r") as f:
    n = int(f.readline().strip())      # Read matrix dimension
    data = np.fromfile(f, sep=" ")     # Read remaining numbers

cost_matrix = data.reshape((n, n))

# Define the two objectives
def f1(chromosome, cost_matrix):
    return sum(cost_matrix[i][chromosome[i]] for i in range(len(chromosome)))

def f2(chromosome, cost_matrix):
    mean_workload = sum(cost_matrix[i][chromosome[i]] for i in range(len(chromosome))) / len(chromosome)
    var_workload = sum((cost_matrix[i][chromosome[i]] - mean_workload) ** 2 for i in range(len(chromosome))) / len(chromosome)
    return var_workload

# Initialize the problem and the ANN-NSGA-II and NSGA-II algorithms
problem = Problem(objectives = [f1, f2], cost_matrix = cost_matrix)
ga = NSGAII(problem, population_size=50, tournament_size=2)
evolution_non_hybrid = Evolution(problem, ga, hybrid=False, learning_rate=0.001)
evolution_hybrid = Evolution(problem, ga, hybrid=True, learning_rate=0.001)

# Evolve both algorithms for 3000 generations
evolution_hybrid.evolve(3000)
evolution_non_hybrid.evolve(3000)

# Test validity of final populations
print(f"Non-hybrid population is valid: {test_valid(evolution_non_hybrid.population)}")
print(f"Hybrid population is valid: {test_valid(evolution_hybrid.population)}")

# Calculate the hypervolume of both final populations
ref_point = get_reference_point(evolution_non_hybrid.population, evolution_hybrid.population)
hv_non_hybrid = calculate_hypervolume(evolution_non_hybrid.population, ref_point)
hv_hybrid = calculate_hypervolume(evolution_hybrid.population, ref_point)
print(f"Non-hybrid hypervolume: {hv_non_hybrid}")
print(f"Hybrid hypervolume: {hv_hybrid}")

# Plot the final populations
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
ax1 = plot_population(ax1, evolution_non_hybrid.population, title="Non-hybrid Evolution")
ax2 = plot_population(ax2, evolution_hybrid.population, title="Hybrid Evolution")

plt.show()

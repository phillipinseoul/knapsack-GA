### Solving 0-1 Knapsack Problem with Genetic Algorithm ###
import numpy as np
import random
from random import randint
import argparse
import time

# Handle arguments
def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("-p", "--population", default=400, help="Population size", type=int)
  parser.add_argument("-s", "--selection", default="rw", help="Population size", type=str)
  # parser.add_argument("-eli", "--elitism", default=True, help="Population size", type=bool)
  parser.add_argument("-gen", "--max_generation", default=200, help="Maximum number of generation", type=int)
  parser.add_argument("-c", "--crossover", default=0.8, help="Crossover rate", type=float)
  parser.add_argument("-m", "--mutation", default=0.005, help="Mutation rate", type=float)
  parser.add_argument("-t", "--threshold", default=0.7, help="Threshold", type=float)
  args = parser.parse_args()
  return args.population, args.selection, args.max_generation, args.crossover, args.mutation, args.threshold


# Define main()
def main():
    num_items = len(weight_list)

    print ('  Weight    Value')
    for i in range(0, num_items):
      print('{0}    {1}     {2}\n'.format(i+1, weight_list[i], value_list[i]))

    # Get arguments for GA
    population_size, selection_method, max_generation, crossover_rate, mutation_rate, threshold = get_args()

    # Initialize population of given size
    init_pop = population_init(population_size, num_items)
    population = calculate_fitness(init_pop, value_list, weight_list, max_sum)

    # Make a copy initial population
    population_2 = []
    population_3 = []
    for ch in population:
        nch1 = Chromosome(ch.bin_value)
        nch2 = Chromosome(ch.bin_value)
        nch1.fitness = ch.fitness
        nch2.fitness = ch.fitness
        population_2.append(nch1)
        population_3.append(nch2)

    # Repeat genetic algorithm until termination condition is satisfied
    ### First loop: WITH ELITISM ###
    elitism = True
    start_time = time.time()                    # Start recording time
    gen_num = 0
    print("\nWITH ELITISM!\n")

    print("GENERATION: " + str(gen_num))
    print("INITIAL FITNESS: " + str(population[0].fitness))

    while True:
        population = get_next_generation(population, selection_method, elitism, crossover_rate, mutation_rate, value_list, weight_list, max_sum)
        gen_num += 1

        if gen_num % 50 == 0:
            print("GENERATION: " + str(gen_num))
            print("Max fitness: " + str(population[0].fitness))

        if test_terminate(population, max_generation, gen_num, threshold):
            print("\nFinished!")
            print("GENERATION: " + str(gen_num))
            print("Max fitness: " + str(population[0].fitness))
            print("Total time: " + str(time.time() - start_time))
            break
    
    ### Second loop: WITHOUT ELITISM ###
    elitism = False
    start_time = time.time()                    # Start recording time
    gen_num =  0
    print("\nWITHOUT ELITISM!\n")

    print("GENERATION: " + str(gen_num))
    print("INITIAL FITNESS: " + str(population_2[0].fitness))

    while True:
        population_2 = get_next_generation(population_2, selection_method, elitism, crossover_rate, mutation_rate, value_list, weight_list, max_sum)
        gen_num += 1

        if gen_num % 50 == 0:
            print("GENERATION: " + str(gen_num))
            print("Max fitness: " + str(population_2[0].fitness))

        if test_terminate(population_2, max_generation, gen_num, threshold):
            print("\nFinished!")
            print("GENERATION: " + str(gen_num))
            print("Max fitness: " + str(population_2[0].fitness))
            print("Total time: " + str(time.time() - start_time))
            break

    ### Third loop: WITH ELITISM & Overselection ###
    elitism = True
    start_time = time.time()                    # Start recording time
    gen_num =  0
    print("\nOVERSELECTION & WITH ELITISM!\n")

    print("GENERATION: " + str(gen_num))
    print("INITIAL FITNESS: " + str(population_3[0].fitness))

    while True:
        population_3 = get_next_generation(population_3, selection_method, elitism, crossover_rate, mutation_rate, value_list, weight_list, max_sum)
        gen_num += 1

        if gen_num % 50 == 0:
            print("GENERATION: " + str(gen_num))
            print("Max fitness: " + str(population_3[0].fitness))

        if test_terminate(population_3, max_generation, gen_num, threshold):
            print("\nFinished!")
            print("GENERATION: " + str(gen_num))
            print("Max fitness: " + str(population_3[0].fitness))
            print("Total time: " + str(time.time() - start_time))
            break


# Define class Chromosome
# Properties: bin_value (binary value), fitness
# Methods: remove_random(), update_fitness(), mutation()
class Chromosome:
  def __init__(self, bin_value):
    self.bin_value = bin_value
    self.num_items = len(bin_value)
    self.fitness = 0

  def remove_random(self):
    order = [i for i in range(self.num_items)]              # Randomly choose a bit & If it's 1, change it to 0
    random.shuffle(order)
    for i in order:
      if self.bin_value[i] == 1:
        self.bin_value[i] = 0
        break

  def mutation(self, mutation_rate):
    for i in range(len(self.bin_value)):
        if np.random.random() < mutation_rate:
            self.bin_value[i] = (not self.bin_value[i]) * 1       # Change each bit value with probabilty "mutation_rate"

  def update_fitness(self, value_list, weight_list, max_sum):
    while True:
        # Calculate fitness based on current binary value
        v_sum = 0
        w_sum = 0
        for i in range(self.num_items):
            v_sum += self.bin_value[i] * value_list[i]
            w_sum += self.bin_value[i] * weight_list[i]

        if w_sum <= max_sum:      # If w_sum <= max_sum, end loop
            self.fitness = v_sum
            break
        else:                     # If w_sum > max_sum, call remove_random() and repeat
            self.remove_random()


# Randomly declare initial population of size N (Define population_init())
def population_init(pop_size, num_items):
    init_pop = []
    for i in range(pop_size):
        bval = [random.randint(0, 1) for j in range(num_items)]
        new_ch = Chromosome(bval)
        init_pop.append(new_ch)

    return init_pop


# Define calculate_fitness()
# For loop: For each chromosome, execute update_fitness() method
def calculate_fitness(population, value_list, weight_list, max_sum):
    for ch in population:
        ch.update_fitness(value_list, weight_list, max_sum)

    pop_sorted = sorted(population, key=lambda x:x.fitness, reverse=True)       # Sort the list by fitness

    return pop_sorted


# Define selection()
# Choose selection method from console (Roulette-wheel OR Overselection)
# --selection rw/ov
def selection(select_method, population, num):
    if select_method == "rw":
        return roulette_wheel_selection(population, num)
    elif select_method == "ov":
        return over_selection(population, num)


def roulette_wheel_selection(population, num):
    parents = []
    fitness_sum = sum([ch.fitness for ch in population])
    fitness_prob = [ch.fitness / (fitness_sum * 1.0) for ch in population]

    m = num
    while (m > 0):
        select = np.random.choice(len(population), p = fitness_prob)
        parents.append(population[select])
        m -= 1

    return parents

def over_selection(population, num):
    psize = len(population)
    parents = []
    div_num = int(num * 0.2)
    div_idx = int(psize * 0.2)
    m = num
    while (m > 0):  
        if m > div_num:                     # 80% of parents are selected from top 20%
            k = random.randint(0, div_idx)
            parents.append(population[k])
        else:                               # 20% of parents are selected from the rest
            k = random.randint(div_idx + 1, psize - 1)
            parents.append(population[k])
        m -= 1

    return parents

# Define crossover()
# Set crossover ratio: --crossover 0.8
def crossover(ch1, ch2):
    position = np.random.randint(len(ch1.bin_value))
    bv1 = ch1.bin_value                                 # Binary value of Chromosome 1
    bv2 = ch2.bin_value                                 # Binary value of Chromosome 2

    nch1 = Chromosome(bv1)
    nch2 = Chromosome(bv2)
    nch1.bin_value = bv1[0:position] + bv2[position:]
    nch2.bin_value = bv2[0:position] + bv1[position:]

    return nch1, nch2


# Define get_next_generation()
# Choose whether to apply Elitism
def get_next_generation(population, selection_method, elitism, crossover_rate, mutation_rate, value_list, weight_list, max_sum):
    new_population = []
    N = len(population)                               # N: Population size

    if elitism:                                                     # Elitism: Preserve Top 4 chromosomes
        for i in range(4):                                 
            newCh = Chromosome(population[i].bin_value)
            newCh.fitness = population[i].fitness
            new_population.append(newCh)

    k = len(new_population)

    if selection_method == "rw":                            # If Roulette Wheel selection
        parents = roulette_wheel_selection(population, N - k)
    elif selection_method == "ov":                          # If Overselection
        parents = over_selection(population, N - k)
    
    random.shuffle(parents)
    while (k < N):
        i = 0
        if np.random.random() < crossover_rate:
            nch1, nch2 = crossover(parents[i], parents[i+1])                # Perform crossover
        else:
            nch1 = Chromosome(parents[i].bin_value)
            nch2 = Chromosome(parents[i+1].bin_value)
        
        nch1.mutation(mutation_rate)
        nch2.mutation(mutation_rate)
        nch1.update_fitness(value_list, weight_list, max_sum)
        nch2.update_fitness(value_list, weight_list, max_sum)

        if k == N - 1:
            new_population.append(nch1)
            k += 1
        else:
            new_population += [nch1, nch2]                  # Add offsprings to next generation
            k += 2
            i += 2
        
    new_population = sorted(new_population, key=lambda x:x.fitness, reverse=True)

    return new_population


# Define test_terminate()
def test_terminate(population, max_gen, gen_num, threshold):
    if gen_num >= max_gen:                                 # End condition 1: Generation has exceeded the max generation
        return True
    
    fitness_list = [ch.fitness for ch in population]      # End condition 2: If percentage of chromosome that have max fitness is higher than threshold
    max_fitness = max(fitness_list)
    if fitness_list.count(max_fitness) > threshold * len(population):
        return True
    return False

if __name__ == '__main__':
    # Weights, Values for Knapsack Problem
    weight_list = [5, 13, 5, 6, 15, 17, 13, 7, 5, 16, 16, 14, 15, 6, 9, 13, 7, 6, 15, 16]    
    value_list =  [15, 33, 16, 37, 49, 40, 23, 13, 37, 25, 48, 23, 25, 48, 28, 19, 44, 35, 40, 23]
    max_sum = 60

    main()
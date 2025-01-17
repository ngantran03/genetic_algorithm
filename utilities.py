import random 
import function
import math

class Point(object):
    def __init__(self, chromosome_list=None, value_list=None):
        self.fitness = math.inf
        self.individuals = []
        
        if chromosome_list is not None:
            self.individuals = [Individual(chromosome=c) for c in chromosome_list]
        elif value_list is not None:
            self.individuals = [Individual(value=v) for v in value_list]

    def fitness_evaluation(self):
        # Evaluate the fitness of the point
        self.fitness = function.function_evaluation([individual.value for individual in self.individuals])

class Individual(object):
    def __init__(self, chromosome=None, value=None):
        if chromosome is not None:
            self.chromosome = chromosome
            self.value = self.binary_to_float(chromosome)
        elif value is not None:
            self.value = value
            self.chromosome = self.float_to_binary(value)
        self.fitness = 0

    def float_to_binary(self, num: float) -> str:
        if num < 0:
            sign = 1
            num = -num
        else:
            sign = 0

        # Convert integer part to binary
        integer_part = int(num)
        fractional_part = num - integer_part
        binary_integer = bin(integer_part).replace("0b", "")

        # Convert fractional part to binary
        binary_fraction = []
        while fractional_part:
            fractional_part *= 2
            bit = int(fractional_part)
            if bit == 1:
                fractional_part -= bit
                binary_fraction.append('1')
            else:
                binary_fraction.append('0')
            if len(binary_fraction) > 52:  # Limit the length to prevent infinite loop
                break

        binary_fraction = ''.join(binary_fraction)
        binary_representation = f"{binary_integer}.{binary_fraction}"

        # self.chromosome = binary_representation

        return f"{'-' if sign else ''}{binary_representation}"
    
    def binary_to_float(self, binary: str) -> float:
        sign = -1 if binary[0] == '1' else 1
        binary = binary[1:]
        integer_part, fractional_part = binary.split('.')
        integer_part = int(integer_part, 2) if integer_part != '' else 0
        fractional_part = sum(int(bit) * (2 ** -(i + 1)) for i, bit in enumerate(fractional_part)) if fractional_part != '' else 0
        return sign * (integer_part + fractional_part)

def initialize_population(population_size, domain):
    population = []
    for i in range(population_size):
        value_list = []
        for j in range(len(domain)):
            if domain[j][1] == 'int':
                value_list.append(random.randint(domain[j][0][0], domain[j][0][1]))
            elif domain[j][1] == 'float':
                value_list.append(random.uniform(domain[j][0][0], domain[j][0][1]))
        population.append(Point(value_list=value_list))
    return population

def evaluate_population(population):
    for point in population:
        point.fitness_evaluation()

def selection_fitness_ranking(population, num_parents, type):
    # select the best individuals based on their minimum fitness
    if type == 'min':
        return sorted(population, key=lambda x: x.fitness)[:num_parents]
    # select the best individuals based on their maximum fitness
    elif type == 'max':
        return sorted(population, key=lambda x: x.fitness, reverse=True)[:num_parents]

def selection_tournament(population, num_parents, tournament_size):
    # Tournament selection: Select the best individual from a random subset of the population
    parents = []
    for _ in range(num_parents):
        tournament = random.sample(population, tournament_size)
        winner = max(tournament, key=lambda x: x.fitness)
        parents.append(winner)
    return parents

def selection_roulette(population, num_parents):
    # Roulette wheel selection: Select individuals based on their relative fitness
    total_fitness = sum(point.fitness for point in population)
    parents = []
    for _ in range(num_parents):
        pick = random.uniform(0, total_fitness)
        current = 0
        for point in population:
            current += point.fitness
            if current > pick:
                parents.append(point)
                break
    return parents

def mutation(point, prob_mutation=1):
    new_individuals = []
    # Check if mutation should occur for each individual
    if random.random() < prob_mutation:
        for i, individual in enumerate(point.individuals):
            chromosome = list(individual.chromosome)  # Convert to list for mutability
            length = len(chromosome)
            # Pick a random index and flip the bit
            index = random.randint(0, length - 1)
            while chromosome[index] == '.':
                index = random.randint(0, length - 1)
            chromosome[index] = '0' if chromosome[index] == '1' else '1'
            # Update the individual's chromosome
            new_individuals.append(''.join(chromosome))  # Update individual with mutated chromosome
        return Point(chromosome_list=new_individuals)  # Return updated point
    else:
        return None

    
def crossover(point1, point2, crossover_probability = 1):
    if random.random() < crossover_probability:
        new_point_1 = []
        new_point_2 = []
        for ind1, ind2 in zip(point1.individuals, point2.individuals):
            # get "." position in ind1, ind2
            dot_position1 = ind1.chromosome.index(".")
            dot_position2 = ind2.chromosome.index(".")
            left = min(dot_position1, dot_position2)
            right = max(dot_position1, dot_position2)
            # get crossover point   
            crossover_point = random.randint(0, len(ind1.chromosome) - 1)
            while crossover_point >= left and crossover_point <= right:
                crossover_point = random.randint(0, len(ind1.chromosome) - 1)
            # create new chromosome
            new_point_1.append(ind1.chromosome[:crossover_point] + ind2.chromosome[crossover_point:])
            new_point_2.append(ind2.chromosome[:crossover_point] + ind1.chromosome[crossover_point:])
        return Point(chromosome_list=new_point_1), Point(chromosome_list=new_point_2)
    else:
        return None
    
def extrema(population, type):
    if type == 'max':
        return max(population, key=lambda x: x.fitness)
    elif type == 'min':
        return min(population, key=lambda x: x.fitness)


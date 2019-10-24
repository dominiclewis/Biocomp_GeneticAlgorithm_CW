'''
Author: Dominic Lewis
'''

import random
import matplotlib.pyplot as plt

from copy import copy


random.seed(a=None)
NUM_POP = 50 # P
NUM_GENE = 50  # N
NUM_EPOCH = 50
CROSSOVER_PROB = 0.75
MUTATION_PROB = random.uniform(1.0/NUM_POP, 1.0/NUM_GENE)
mean_fit = []
best_fit = []


class Individual:
    def __init__(self, gene):
        self.gene = gene
        self.fitness = 0

def main():
    # Create Population
    popPool = []
    for _ in range(NUM_POP):
        gene = []
        for _ in range(NUM_GENE):
            gene.append(random.randint(0,1))
        popPool.append(Individual(gene))

    # Selection
    for i in range(NUM_EPOCH):
        init = False
        if i == 0:
            init = True
        inspectPop(popPool, init=init)
        popPool = selection(popPool)
    inspectPop(popPool)
    write_csv()
    plot()

def plot():
    x = range(NUM_EPOCH + 1)
    plt.plot(x, best_fit, color='r', label='max')
    plt.plot(x, mean_fit, color='b', label='mean')
    plt.legend(loc="upper left")
    plt.show()


def write_csv():
    with open("fit.csv", 'w') as f:
        f.write("Epoch,Best Candidate,Mean Fitness,\n")
        for i in range(NUM_EPOCH + 1):
            f.write("{e},{bc},{mf},\n".format(
                e=i, bc=best_fit[i], mf=mean_fit[i]))

def assessFitness(individual):
    individual.fitness = 0
    for gene in individual.gene:
        if gene == 1:
            individual.fitness = individual.fitness + 1

def inspectPop(p, init=False):
    meanFitness = 0.0
    fittist = -1
    if init:
        # Generate current population fitness
        for ind in p:
            assessFitness(ind)
    for ind in p:
        # print "Genes:\n{g}\nFitness:\n{f}".format(g=ind.gene, f=ind.fitness)
        if ind.fitness > fittist:
            fittist = ind.fitness
        meanFitness = meanFitness + ind.fitness
    meanFitness = meanFitness / NUM_POP

    print "Fittest Candidate: {f}".format(f=fittist)
    print "Mean Fitness: {mf}".format(mf=meanFitness)
    mean_fit.append(meanFitness)
    best_fit.append(fittist)

def selection(population):
    def crossover(cand_a, cand_b):
        if random.random() <= CROSSOVER_PROB:
            cross_point = random.randint(1, NUM_GENE - 1)
            child_a = Individual(
                cand_a.gene[:cross_point] + cand_b.gene[cross_point:])
            child_b = Individual(
                cand_b.gene[:cross_point] + cand_a.gene[cross_point:])
            return (True, child_a, child_b)
        else:
            return (False, cand_a, cand_b)
    
    def mutate_offspring(child):
        new_gene = []
        for bit in child.gene:
            if random.random() <= MUTATION_PROB:
                # Mutate bit
                new_gene.append(1 - bit)
            else:
                new_gene.append(bit)
        child.gene = new_gene
        return child

    def swap_lowest(pop, prev_fittest):
        # Mutate original list
        pop = sorted(pop, key=lambda x: x.fitness, reverse=True)
        if prev_fittest.fitness > pop[-1].fitness:
            # Remove tail
            pop.pop()
            pop.append(prev_fittest)
        return pop

    shuffle_pop = lambda p : random.shuffle(p)
    offspring = []
    new_pop = []
    # Add the fittest candidate to the offspring
    fittest = copy(sorted(population, key=lambda x: x.fitness, reverse=True)[0])

    for _ in range(NUM_POP):
        parentOne = population[random.randint(0, NUM_POP - 1)]
        parentTwo = population[random.randint(0, NUM_POP - 1)]
        if parentOne.fitness > parentTwo.fitness:
            offspring.append(parentOne)
        else:
            offspring.append(parentTwo)

    shuffle_pop(offspring)
    # Crossover
    for i in range(0, len(offspring), 2):
        temp_children = crossover(offspring[i], offspring[i+1])
        new_pop.append(mutate_offspring(temp_children[1]))
        new_pop.append(mutate_offspring(temp_children[2]))

    for ele in new_pop:
        assessFitness(ele)
    new_pop = swap_lowest(new_pop, fittest)
    return new_pop


if __name__ == "__main__":
    main()
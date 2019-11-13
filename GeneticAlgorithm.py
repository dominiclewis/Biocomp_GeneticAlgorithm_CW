'''
Author: Dominic Lewis
'''

import const
import DataExtract
import random

# Matplotlib virtualenv hack
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

from copy import deepcopy
from bisect import bisect

random.seed(a=None)

mean_fit = []
best_fit = []

class Individual:
    def __init__(self, gene):
        self.gene = gene
        self.fitness = 0
        self.ruleList = []

def main():
    # Create Population
    popPool = []
    max_num = 2
    first = False
    for _ in range(const.NUM_POP):
        count = 1
        gene = []
        for _ in range(const.NUM_GENE):
            if first:
                first = False
            if count == 7:
                max_num = 1
                count = 1
                first = True
            gene.append(random.randint(0, max_num))
            if count == 1:
                max_num = 2
            if not first:
                count += 1
        popPool.append(Individual(gene))

    # Selection
    for i in range(const.NUM_EPOCH):
        init = False
        if i == 0:
            init = True
        inspectPop(popPool, init=init)
        popPool = selection(popPool)
    inspectPop(popPool)
    print "Fittest Candidate"
    for e in fittest_individual.ruleList:
        print e.condition, e.classification
    print "Fittest: {f}".format(
        f=fittest_individual.fitness)

    write_csv()
    plot()

def plot():
    x = range(const.NUM_EPOCH + 1)
    plt.plot(x, best_fit, color='r', label='max')
    plt.plot(x, mean_fit, color='b', label='mean')
    plt.legend(loc="upper left")
    plt.show()


def write_csv():
    with open("fit.csv", 'w') as f:
        f.write("Epoch,Best Candidate,Mean Fitness,\n")
        for i in range(const.NUM_EPOCH + 1):
            f.write("{e},{bc},{mf},\n".format(
                e=i, bc=best_fit[i], mf=mean_fit[i]))

def assessFitness(individual):
    def check_condition(rule, datapoint):
        match = True
        for i, dp in enumerate(datapoint):
            if rule[i] == dp or rule[i] == 2:
                pass
            else:
                match = False
                break
        return match

    # Create Rules
    k = 0
    fitness = 0
    individual.ruleList = []
    for i in range(const.NUM_RULES):
        tempRule = DataExtract.Data()
        for j in range(const.COND_LENGTH):
            tempRule.condition.append(individual.gene[k])
            k = k + 1
        tempRule.classification = individual.gene[k]
        k = k + 1
        individual.ruleList.append(tempRule)
    # Assess the fitness
    for dp in DataExtract.DataHelper.data_list:
        for rule in individual.ruleList:
            if check_condition(rule.condition, dp.condition):
                if rule.classification == dp.classification:
                    fitness = fitness + 1
                break
    individual.fitness = fitness


def inspectPop(p, init=False):
    global fittest_individual
    meanFitness = 0.0
    fittist = -1
    if init:
        # Generate current population fitness
        for ind in p:
            assessFitness(ind)
    for ind in p:
        # print "Genes:\n{g}\nFitness:\n{f}".format(g=ind.gene, f=ind.fitness)
        if ind.fitness > fittist:
            fittest_individual = ind
            fittist = ind.fitness
        meanFitness = meanFitness + ind.fitness
    meanFitness = meanFitness / const.NUM_POP

    print "Fittest Candidate: {f}".format(f=fittist)
    print "Mean Fitness: {mf}".format(mf=meanFitness)
    mean_fit.append(meanFitness)
    best_fit.append(fittist)

def selection(population):
    def crossover(cand_a, cand_b):
        child_a = None
        child_b = None
        if random.random() <= const.CROSSOVER_PROB:
            while True:
                cross_point = random.randint(1, const.NUM_GENE - 1)
                child_a = Individual(
                    cand_a.gene[:cross_point] + cand_b.gene[cross_point:])
                child_b = Individual(
                    cand_b.gene[:cross_point] + cand_a.gene[cross_point:])
                if vali_wc(child_a.gene) and vali_wc(child_b.gene):
                    break
            return (True, child_a, child_b)
        else:
            return (False, cand_a, cand_b)

    def vali_wc(ele):
        count = 1
        for bit in ele:
            if count == 7:
                count = 1
                continue
            if bit != 2:
                return True
            count += 1
        return False

    def mutate_offspring(child):
        first = False
        while True:
            new_gene = []
            count = 1
            b_range = 2
            for bit in child.gene:
                if first:
                    first = False
                if count == 7:
                    # Only allow action bit to be 1 and 0
                    count = 1
                    b_range = 1
                    first = True
                # Mutate bit
                if random.random() <= const.MUTATION_PROB:
                    new_gene.append(random.randint(0, b_range))
                else:
                    new_gene.append(bit)
                if count == 1:
                    b_range = 2
                if not first:
                    count = count + 1
            if vali_wc(new_gene):
                break
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
    population.sort(key=lambda x: x.fitness, reverse=False)
    # print [e.fitness for e in population]
    fittest = deepcopy(population[-1])
    z = 0
    rank_select = []
    max_rel_fit = 0
    rank_indexes = []

    for i in range(len(population)):
        z+= 1
        max_rel_fit += z
        rank_select.append(max_rel_fit)

    for i in range(len(population)):
        rank_indexes.append(bisect(rank_select, random.random() * max_rel_fit))
    [(offspring.append(population[index])) for index in rank_indexes]
    offspring.sort(key=lambda x: x.fitness, reverse=True)
    # print[e.fitness for e in offspring]
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
    DataExtract.DataHelper.load_file_data(
        "/Users/dominiclewis/Repos/BioComp/Biocomp_GeneticAlgorithm_CW/train/"
        "data2.txt")
    main()

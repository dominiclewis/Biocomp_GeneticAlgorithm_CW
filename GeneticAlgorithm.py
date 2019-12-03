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
counter = 0
test_fitness = None

class Individual:
    def __init__(self, gene):
        self.gene = gene
        self.fitness = 0
        self.ruleList = []


def main():
    global counter
    cout = 0
    # Create Population
    popPool = []
    for _ in range(const.NUM_POP):
        count = 1
        gene = []
        for _ in range(const.NUM_GENE):
            action = False
            if count == const.COND_LENGTH + 1:
                # Generating action bit (classification) then reset count for new rule
                count = 1
                action = True
            if action:
                gene.append(random.randint(0, const.MAX_ACTION))
            else:
                # condition
                gene.append(float('%.{p}f'.format(p=const.FLOAT_PRECISION) % random.random()))
                cout += 1
                if count == 2:
                    cout = 0
            if not action:
                count += 1
        popPool.append(Individual(gene))

    # Selection
    for i in range(const.NUM_EPOCH):
        counter += 1
        print("Epoch {c}".format(c=str(counter)))
        init = False
        if i == 0:
            init = True
        inspectPop(popPool, init=init)
        popPool = selection(popPool)
        if fittest_individual.fitness >= (const.MAX_FIT):
            break

    inspectPop(popPool)
    print "Fittest Candidate"
    for e in fittest_individual.ruleList:
        print e.condition, e.classification
    print "Fittest: {f}".format(
        f=fittest_individual.fitness)

    # Run against test set
    global test_fitness
    test_fitness = compare_against_set(fittest_individual, DataExtract.DataHelper.test_data)
    print("Produced a fitness of {f} on test set".format(f=str(test_fitness)))
    write_csv()
    plot()

def plot():
    x = range(counter + 1)
    plt.plot(x, best_fit, color='r', label='max')
    plt.plot(x, mean_fit, color='b', label='mean')
    plt.legend(loc="upper left")
    plt.show()


def write_csv():
    global test_fitness
    with open("fit.csv", 'w') as f:
        f.write("Epoch,Best Candidate,Mean Fitness,\n")
        for i in range(counter + 1):
            f.write("{e},{bc},{mf},\n".format(
                e=i, bc=best_fit[i], mf=mean_fit[i]))
        f.write("\ntest fitness\n{f}".format(f=str(test_fitness)))
    with open("fittest_cand.csv", 'w') as f:
        f.write("Fitness,\n")
        f.write(str(fittest_individual.fitness))
        f.write("\n\n")
        f.write("Condition, Action,\n")
        for e in fittest_individual.ruleList:
            f.write("{cond}, {clas}\n".format(
                cond=e.condition,
                clas=e.classification))


def compare_against_set(individual, dataset):
    # Assess the fitness
    fitness = 0
    for dp_rule in dataset:
        # Get one DP
        for trial_rule in individual.ruleList:
            # Run Trial DP down my gen rules and try ot match
            dp_matched = 0
            match = False
            lookup = None
            for i, dp_gene in enumerate(dp_rule.condition):
                if i == 0:
                    lookup = i
                else:
                    lookup = i * 2
                if trial_rule.condition[lookup] < dp_gene < trial_rule.condition[lookup + 1]:
                    dp_matched += 1
                else:
                    break
                if dp_matched == const.TRAIN_COND_LENGTH:
                    # We've matched every gene in a DP with a generated rule
                    if dp_rule.classification == trial_rule.classification:
                        fitness += 1
                    match = True
            if match:
                # We've matched with a created rule so stop looking for more rules
                break
    return fitness

def assessFitness(individual):
    # Create Rules
    k = 0
    individual.ruleList = []
    for i in range(const.NUM_RULES):
        tempRule = DataExtract.Data()
        for j in range(const.COND_LENGTH):
            tempRule.condition.append(individual.gene[k])
            k = k + 1
        tempRule.classification = individual.gene[k]
        k = k + 1
        individual.ruleList.append(tempRule)

    individual.fitness = compare_against_set(individual, DataExtract.DataHelper.train_data)


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
            cross_point = random.randint(1, const.NUM_GENE - 1)
            child_a = Individual(
                cand_a.gene[:cross_point] + cand_b.gene[cross_point:])
            child_b = Individual(
                cand_b.gene[:cross_point] + cand_a.gene[cross_point:])
            return (True, child_a, child_b)
        else:
            return (False, cand_a, cand_b)

    def mutate_offspring(child):
        get_six_prec = lambda p: float('%.{p}f'.format(p=const.FLOAT_PRECISION) % p)
        action = False
        new_gene = []
        count = 0
        for bit in child.gene:
            count += 1
            if action:
                action = False
            if count == const.COND_LENGTH + 1:
                # Only allow action bit to be 1 and 0
                count = 0
                action = True
            # Mutate bit
            if random.random() <= const.MUTATION_PROB:
                if action:
                    new_gene.append(1 - bit)
                else:
                    mut_val = random.uniform(0, const.MUTATION_AMOUNT)
                    if random.random() > 0.5:
                        # Try to add if possible
                        if get_six_prec(bit + mut_val) < 1:
                            new_gene.append(get_six_prec(bit + mut_val))
                        else:
                            new_gene.append(get_six_prec(bit - mut_val))
                    else:
                        # try to minus if possible
                        if get_six_prec(bit - mut_val > 0):
                            new_gene.append(get_six_prec(bit - mut_val))
                        else:
                            new_gene.append(get_six_prec(bit + mut_val))
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
        "data3.txt")
    main()

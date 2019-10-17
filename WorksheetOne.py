'''
Author: Dominic Lewis
'''

import random

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
    inspectPop(popPool, new=False)

    # Selection
    for epoch_count in range(NUM_EPOCH):
        popPool = selection(popPool)
        inspectPop(popPool, new=True)
    write_csv()

def write_csv():
    with open("fit.csv", 'w') as f:
        f.write("Epoch,Best Candidate,Mean Fitness,\n")
        for i in range(NUM_EPOCH):
            f.write("{e},{bc},{mf},\n".format(
                e=i+1, bc=best_fit[i], mf=mean_fit[i]))

def inspectPop(p, new=False):
    def assessFitness(individual):
        fitness = 0
        for gene in individual.gene:
            if gene == 1:
                fitness = fitness + 1
        individual.fitness = fitness
    meanFitness = 0.0
    fittist = -1
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
    if new:
        mean_fit.append(meanFitness)
        best_fit.append(fittist)

def selection(population):
    def getPoolFitness(p):
        fitness = 0
        for ind in p:
            fitness = fitness + ind.fitness
        return fitness

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

    shuffle_pop = lambda p : random.shuffle(p)
    offspring = [] 
    for _ in range(NUM_POP):
        parentOne = population[random.randint(0, NUM_POP - 1)]
        parentTwo = population[random.randint(0, NUM_POP - 1)]
        if parentOne.fitness >= parentTwo.fitness:
            offspring.append(parentOne)
        else:
            offspring.append(parentTwo)

    shuffle_pop(offspring)
    child_candidate = offspring

    # Check if offspring improves fitness
    if getPoolFitness(population) > getPoolFitness(offspring):
        print "Offspring Fitness:{of}\nLess than\nParent Fitness:{pf}".format(
            of=str(getPoolFitness(offspring)),
            pf=str(getPoolFitness(population))
        )
        print "Swapping"
        shuffle_pop(population)
        child_candidate = population

    new_pop = []
    # Crossover
    for i in range(0, len(child_candidate), 2):
        temp_children = crossover(child_candidate[i], child_candidate[i+1])
        new_pop.append(mutate_offspring(temp_children[1]))
        new_pop.append(mutate_offspring(temp_children[2]))
    return new_pop


if __name__ == "__main__":
    main()
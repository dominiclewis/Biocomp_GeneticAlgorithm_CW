'''
NUM_POP = 50 # P
NUM_EPOCH = 15000
CROSSOVER_PROB = 0.75
NUM_RULES = 55
COND_LENGTH = 6
MUTATION_PROB = 0.02

'''
'''
 Condition Length + 1 action bit
 Rule = (010101  0)
 Num Genes = 1x Rule Length * Number of Rules to be generated to ensure
 enough genes present.
 '''
'''
NUM_GENE = (COND_LENGTH + 1) * NUM_RULES
# MUTATION_PROB = random.uniform(1.0/const.NUM_GENE, 1.0/const.NUM_POP,)
'''
NUM_POP = 500
NUM_EPOCH = 2500
CROSSOVER_PROB = 0.7
NUM_RULES = 10
COND_LENGTH = 12
TRAIN_COND_LENGTH = 6
# DS3 Fitness is assessing if each gene is between a certain range. ie. gene1 < testGene < gene2 (so cond = 6 * 2)
NUM_GENE = (COND_LENGTH + 1) * NUM_RULES
MAX_FIT = 1000
FLOAT_PRECISION = 6
MAX_ACTION = 1
MUTATION_PROB = 0.02
MUTATION_AMOUNT = 0.35

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
NUM_POP = 500 # P
NUM_EPOCH = 15000
CROSSOVER_PROB = 0.4
NUM_RULES = 50
COND_LENGTH = 6
MUTATION_PROB = 0.0021
NUM_GENE = (COND_LENGTH + 1) * NUM_RULES
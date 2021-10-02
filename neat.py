from genome import *
from network import *
from gene import *

def convert(x: str):
    ''' Converts x in a number (if possible) '''
    try:
        return float(x)
    except ValueError:
        return x

class Neat:

    def __init__(self, config_file: str):
        self.load_configs(config_file)
        self.genomes = []
        # a list of all the connection genes, these are the standards so that if a gene
        # points from a -> b it always has the same innovation number
        self.global_conn_genes = []

    def load_configs(self, configs_file):
        lines = []
        with open(configs_file, 'r') as f:
            lines = f.readlines()
        configs = {}
        for line in lines:
            idx = line.index(":")+1
            configs[line[:idx]] = convert(line[idx:].replace(' ', '').replace('\n',''))
        
        self.configs = configs

    def initialize_pop(self):
        ''' 
            Initializes the population based on the given settings (or on the previous genomes)
        '''
        # generate a new population
        if len(self.genomes) == 0:
            self.genomes = [Genome(self.configs['num_inputs'], self.configs['num_outputs'], self.configs['activation'])]
        else:
            species = {}  # map:  species idx ->  list of genomes in that species
            species_tot_fitness = {}  # map:   species idx -> total fitness of that species (sum of all their fitnesses)
            pop_avg_adj_fitness = 0  # average adjusted fitness of the population as a whole
            for genome in self.genomes:
                if genome.species in species:
                    species[genome.species].append(genome)
                    species_tot_fitness[genome.species] += genome.fitness
                else:
                    species[genome.species] = [genome]
                    species_tot_fitness[genome.species] = genome.fitness

            for species_idx in species_tot_fitness:
                # (sum of fitnesses of genomes in current_species) / (number of genomes in current species)
                pop_avg_adj_fitness += species_tot_fitness[species_idx]/len(species[species_idx])
            
            pop_avg_adj_fitness /= self.configs['population'] # divide it by the population again to get the true result

            # order each list of the genomes of a species in an increasing order of their fitness
            new_pop = [] # the new population, which is going to be passed to the next generation 
            for species_idx in species:
                species[species_idx].sort(key=lambda ge: ge.fitness) # sort by fitness
                new_pop.append(species[species_idx][-1]) # save best performing genome of each species for the next generation

                species_avg_adj_fitness = species_tot_fitness[species_idx]/len(species[species_idx])
                new_species_size = species_avg_adj_fitness/pop_avg_adj_fitness # size of the species in the next generation

                # APPLY CROSSOVER
                # number of genomes to be generated with crossovers
                crossover_generated_num = int(self.configs['crossover_rate']*new_species_size)
                # number of the top genomes to use in crossovers
                crossover_genome_pool_size = int(self.configs['top_genomes_rate']*len(species[species_idx]))
                genome_pool = species[species_idx][-crossover_genome_pool_size:] # actual genomes to be used in the crossover
                for _ in range(crossover_generated_num):
                    f, s = None, None
                    while f == s: # make sure there's no crossing over between a genome and itself
                        f, s = np.random.choice(genome_pool), np.random.choice(genome_pool)
                    new_pop.append(crossover(f, s))
                
                # APPLY MUTATIONS 
                # remove the worst performing genome in the species, this won't be mutated and passed to the next gen.
                species[species_idx] = species[species_idx][1:] 

    def run_generation(self):
        self.initialize_pop()
        self.eval_fitness(self.genomes)

    # main part of the process
    def run(self):
        for _ in range(1, self.configs['generations']+1):
            self.run_generation()

if __name__ == '__main__':
    neat = Neat("configs.txt")
    for key in neat.configs:
        print(f"{key} -> {neat.configs[key]}")
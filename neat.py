from genome import *
from network import *
from gene import *
from math import ceil

def convert(x: str):
    ''' Converts x in a number (if possible) '''
    try:
        return float(x)
    except ValueError:
        return x

class Neat:

    def __init__(self, config_file: str, eval_fitness=None):
        self.load_configs(config_file)
        self.genomes = []
        # a list of all the connection genes, each represented as a tuple:
        # (start_idx, end_idx, innovation_number); these are the standards so that if a gene
        # points from a -> b it always has the same innovation number
        self.global_conn_genes = []
        self.eval_fitness = eval_fitness

    def load_configs(self, configs_file):
        lines = []
        with open(configs_file, 'r') as f:
            lines = f.readlines()
        configs = {}
        for line in lines:
            idx = line.index(":")
            configs[line[:idx]] = convert(line[idx+1:].replace(' ', '').replace('\n',''))
        
        self.configs = configs

    def initialize_pop(self, verbosity=1):
        ''' 
            Initializes the population based on the given settings (or on the previous genomes)
        '''
        # generate a new population
        if len(self.genomes) == 0:
            pop = int(self.configs['population'])
            self.genomes = [Genome(int(self.configs['num_inputs']), int(self.configs['num_outputs']), self.configs['activation']) for _ in range(pop)]
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
                crossover_generated_num = ceil(self.configs['crossover_rate']*new_species_size)
                # number of the top genomes to use in crossovers
                crossover_genome_pool_size = int(self.configs['top_genomes_rate']*len(species[species_idx]))
                genome_pool = species[species_idx][-crossover_genome_pool_size:] # actual genomes to be used in the crossover
                if len(genome_pool) > 1:
                    for _ in range(crossover_generated_num):
                        f, s = None, None
                        while f == s: # make sure there's no crossing over between a genome and itself
                            f, s = np.random.choice(genome_pool), np.random.choice(genome_pool)
                        new_pop.append(crossover(f, s))
                

                # APPLY MUTATIONS 
                # remove the worst performing genome in the species, this won't be mutated and passed to the next gen.
                if len(species[species_idx]) > 1:
                    species[species_idx] = species[species_idx][1:] 
                for _ in range(int(new_species_size)-crossover_generated_num):
                    genome = np.random.choice(species[species_idx])
                    new_conns = genome.mutate(self.configs)
                    # lookup innovation numbers and shit
                    for conn in new_conns:
                        found = False
                        for global_conn in self.global_conn_genes: # look to see if the connection has already been created
                            if conn == global_conn[:2]:
                                found = True # if so log it to avoid saving it again
                                conn_gene_idx = genome.network.get_conn_idx(conn[0], conn[1]) # the index of the new connection gene in the network
                                genome.network.conn_genes[conn_gene_idx].innov = global_conn[2] # and apply the right innovation number
                        if not found: # if the connection gene is brand new
                            # save the connection in the "global index" and assign it a new (increasing) innovation number
                            self.global_conn_genes.append((conn[0], conn[1], len(self.global_conn_genes)))
                            # update the innovation number of the actual gene
                            conn_gene_idx = genome.network.get_conn_idx(conn[0], conn[1])
                            genome.network.conn_genes[conn_gene_idx].innov = self.global_conn_genes[-1][2]
                    new_pop.append(genome)

            # ASSIGN A SPECIES TO EACH GENOME
            # reference genomes for each species, the genome in the i-th index is the one representing the i-th species
            ref_genomes = [np.random.choice(species[species_idx]) for species_idx in species]
            for genome in new_pop:
                found = False
                for ref_genome_idx in range(len(ref_genomes)):
                    if genome.are_same_species(ref_genomes[ref_genome_idx], configs=self.configs):
                        genome.species = ref_genome_idx
                        found = True
                        break
                if not found:
                    genome.species = len(ref_genomes)
            
            if verbosity >= 2:
                for species_idx in species_tot_fitness:
                    print(f"Species #{species_idx}: \n\tgenomes: {len(species[species_idx])}")
                    print(f"\n\taverage fitness: {species_tot_fitness[species_idx]/len(species[species_idx])}")
            elif verbosity >= 1:
                print(f"Average fitness: {pop_avg_adj_fitness}")

    def run_generation(self, verbosity=2):
        self.initialize_pop(verbosity=verbosity)
        self.eval_fitness(self.genomes)

    # main part of the process
    def run(self, verbosity=2):
        for gen in range(1, int(self.configs['generations']+1)):
            print("-"*20 + f"Generation #{gen}" + "-"*20)
            self.run_generation(verbosity=verbosity)

def eval_fitness(genomes):
    x = np.array([1,2,3])
    for genome in genomes:
        genome.fitness = int(genome.network._predict(x)+1)

if __name__ == '__main__':
    neat = Neat("configs.txt", eval_fitness)
    neat.run(verbosity=0)
    best_genome = neat.genomes[0]
    for genome in neat.genomes:
        if genome.fitness > best_genome.fitness:
            best_genome = genome
    print(f"Final prediction {best_genome.network._predict(np.array([1,2,3]))}")
    
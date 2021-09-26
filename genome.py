from network import *
from gene import *

class Genome:
    COMPATIBILITY_THRESHOLD = 3.0
    EXCESSIVE_GENES_COEFFICIENT = 0.5
    DISJOINTED_GENES_COEFFICIENT = 0.5
    WEIGHTS_DIFFERENCE_COEFFICIENT = 0.5

    def __init__(self, num_inputs, num_outputs, activation):
        self.species = 0
        self.fitness = 0
        self.network = Network(num_inputs, num_outputs, activation)

    def are_same_species(self, other) -> bool:
        ''' 
            Returns whether the genome belongs to the same species
            as the other one or not
         '''
        e, d = 0, 0 # excessive and disjointed genes   
        # the node genes are in an increasing order of the innovation number 
        # since new ones are always appended at the end of the list, and new ones
        # always have a higher innovation number.
        min_node_innov, max_node_innov = self.network.get_node_innov_interval()

        for gene in other.network.node_genes:
            if min_node_innov <= gene.innov <= max_node_innov:
                # Check if the node is also present in this network...
                found = False
                for node_gene in self.network.node_genes:
                    if node_gene.innov == gene.innov:
                        found = True
                        break
                if not found:
                    d += 1
            else:
                e += 1
        

        # diff -> average difference between two weights of the same connection present in both networks
        # n -> number of shared connection genes
        min_conn_innov, max_conn_innov = self.network.get_weight_innov_interval()

        diff, n = 0, 0
        for j in range(len(other.network.conn_genes)):
            # if the connection gene is not a disjointed gene
            if min_conn_innov < other.network.conn_genes[j].innov < max_conn_innov:
                pass
            else:
                e += 1

            for i in range(len(self.network.conn_genes)):
                if self.network.conn_genes[i].innov == other.network.conn_genes[j].innov:
                    diff += self.network.get_weight(i) - other.network.get_weight(j)
                    n += 1
        diff /= n

        # N -> amount of genes of the parents with the most of them
        self_num_genes = len(self.network.node_genes)+len(self.network.conn_genes)
        other_num_genes = len(other.network.node_genes)+len(other.network.conn_genes)
        N = np.maximum(self_num_genes, other_num_genes)

        # the actual delta representing the difference between the two genomes
        delta = (self.EXCESSIVE_GENES_COEFFICIENT*e+self.DISJOINTED_GENES_COEFFICIENT*d)/N + self.WEIGHTS_DIFFERENCE_COEFFICIENT*diff

        return delta <= self.COMPATIBILITY_THRESHOLD

    # TODO: finish this
    def mutate(self, node_add_prob=0.3, conn_add_prob=0.3, weight_change_prob=0.6, gene_toggle_prob=0.1):
        '''
            Mutates the genome according to the given parameters
            node_add_prob -> probability of adding a new node between a connection\n
            conn_add_prob -> probability of adding a new connection between two nodes\n
            weight_change_prob -> probability that every single weight in the network is changed randomly\n
            gene_toggle_prob -> probability that a gene is toggled (meaning it gets enabled or disabled)
        '''
        if np.random.uniform() <= node_add_prob:
            # Add a node between a given connection already present in the network
            conn_idx = np.random.randint(high=len(self.network.conn_genes)) # pick the connection in which to form the new node
            self.network.disable_conn(conn_idx)

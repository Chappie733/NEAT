from gene import ConnectionGene
from genome import *
from network import Network
from activations import tanh, ReLu
import numpy as np
import sys

if len(sys.argv) == 1:
    sys.argv.append("")

# feed forward test
if sys.argv[1] == 'fft':

    x = np.array([1,2,3])

    net = Network(3,2,tanh)
    net.gen_node()
    net.gen_node()

    net.add_conn(ConnectionGene(3,6,1))
    net.add_conn(ConnectionGene(3,5,1))
    net.add_conn(ConnectionGene(5,0,1))
    net.add_conn(ConnectionGene(6,1,1))

    print(net._predict(x))

elif sys.argv[1] == 'conn_mutation_test':
    gen = Genome(3,1,ReLu)
    gen.network.add_conn(ConnectionGene(1,0,0))
    gen.network.add_conn(ConnectionGene(3,0,1))

    gen.network.gen_node()
    gen.network.add_conn(ConnectionGene(4,0,2))

    print("Before adding a connection:\n")
    print(gen.network, end='\n'*2)

    gen.mutate_add_conn()

    print("After adding a connection:\n")
    print(gen.network)


elif sys.argv[1] == 'node_mutation_test':
    gen = Genome(3,1,ReLu)
    gen.network.add_conn(ConnectionGene(1,0,0))
    gen.network.add_conn(ConnectionGene(3,0,1))

    print("Before adding a node:\n")
    print(gen.network, end='\n'*2)

    gen.mutate_add_node()

    print("After adding a node:\n")
    print(gen.network)

elif sys.argv[1] == 'weight_mutation_test':
    gen = Genome(3,1,ReLu)
    gen.network.add_conn(ConnectionGene(1,0,0))
    gen.network.add_conn(ConnectionGene(3,0,1))
    gen.network.add_conn(ConnectionGene(2,0,1))


    print("Before changing a weight:\n")
    for conn_gene_idx in range(len(gen.network.conn_genes)):
        print(f"Connection gene #{conn_gene_idx}: {gen.network.get_weight(conn_gene_idx)}")

    print("\n")
    gen.mutate_change_weight()

    print("After adding a connection:\n")
    for conn_gene_idx in range(len(gen.network.conn_genes)):
        print(f"Connection gene #{conn_gene_idx}: {gen.network.get_weight(conn_gene_idx)}")
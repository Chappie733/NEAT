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

    net.add_conn(ConnectionGene(5,6,3))
    net.add_conn(ConnectionGene(2,5,1))
    net.add_conn(ConnectionGene(6,1,6))
    net.add_conn(ConnectionGene(3,5,4))
    net.add_conn(ConnectionGene(4,6,2))
    net.add_conn(ConnectionGene(5,0,5))

    for conn_gene in net.conn_genes:
        print(conn_gene, end='\n'*2)

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

    print("After changing a weight:\n")
    for conn_gene_idx in range(len(gen.network.conn_genes)):
        print(f"Connection gene #{conn_gene_idx}: {gen.network.get_weight(conn_gene_idx)}")


elif sys.argv[1] == 'bias_mutation_test':
    gen = Genome(3,1,ReLu)
    gen.network.add_conn(ConnectionGene(1,0,0))
    gen.network.add_conn(ConnectionGene(3,0,1))

    print("Before changing a threshold:\n")
    print(gen.network, end='\n'*2)

    gen.mutate_change_threshold()

    print("After changing a threshold:\n")
    print(gen.network)


elif sys.argv[1] == 'toggle_mutation_test':
    gen = Genome(3,1,ReLu)

    gen.network.add_conn(ConnectionGene(1,0,0))
    gen.network.add_conn(ConnectionGene(3,0,1))
    gen.network.disable_conn()

    print("Before toggling a connection:\n")
    print(gen.network, end='\n'*2)
    print(f"Weights: {gen.network.weights}", end='\n'*2)

    gen.mutate_toggle_gene()

    print("After toggling a connection:\n")
    print(gen.network)
    print(f"Weights: {gen.network.weights}")


elif sys.argv[1] == 'crossover_test':
    A = Genome(3,1,ReLu)
    A.network.gen_node()
    A.network.add_conn(ConnectionGene(1,0,1))
    A.network.add_conn(ConnectionGene(1,4,2))
    A.network.add_conn(ConnectionGene(2,4,3))
    A.network.add_conn(ConnectionGene(3,0,4))
    A.network.add_conn(ConnectionGene(4,0,5))
    A.fitness = 1

    B = Genome(3,1,ReLu)
    B.network.gen_node()
    B.network.gen_node()
    B.network.add_conn(ConnectionGene(1,0,1))
    B.network.add_conn(ConnectionGene(2,4,3))
    B.network.add_conn(ConnectionGene(3,0,4))
    B.network.add_conn(ConnectionGene(1,5,6))
    B.network.add_conn(ConnectionGene(3,4,7))
    B.network.add_conn(ConnectionGene(4,5,8))
    B.network.add_conn(ConnectionGene(5,0,9))

    crossover(A,B)
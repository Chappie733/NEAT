from gene import ConnectionGene
from genome import *
from network import Network
from activations import tanh, ReLu
import numpy as np

x = np.array([1,2,3])

net = Network(3, 2, activation=tanh)

net.gen_node()

for node_gene in net.node_genes:
    print(node_gene, end='\n'*2)

print(net.get_node_innov_interval())

net.add_conn(ConnectionGene(2,0,1))
net.add_conn(ConnectionGene(3,5,2))
net.add_conn(ConnectionGene(5,0,3))
net.add_conn(ConnectionGene(4,1,4))

net.gen_node()

net.add_conn(ConnectionGene(3,6,5))
net.add_conn(ConnectionGene(6,1,6))
print(net._predict(x))


'''
net2 = Network(3, 2, ReLu)

net2.add_conn(ConnectionGene(4,0,1))
net2.add_conn(ConnectionGene(2,1,2))
net2.add_conn(ConnectionGene(3,1,5))

print(net2.get_weight_innov_interval())

for conn_gene in net2.conn_genes:
    print(conn_gene, end='\n'*2)


# test the is_same_species() function
genome1 = Genome(3,2,tanh)
genome1.network = net

genome2 = Genome(3,2,tanh)
genome2.network = net2

print(genome1.are_same_species(genome2))
print(genome2.are_same_species(genome1))
'''
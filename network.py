import numpy as np
from gene import *
from activations import none
from multipledispatch import dispatch

# Important shit
# - np.c_[mat, arr] appends the array as a column in the end
# - The connection genes have to be stored in the array in an increasing order of the innovation number
# - The first nodes are always the output ones (in nodes and node_geens), then come the inputs, and then the
#       hidden nodes

class Network:

    # initializes the most basic structure with no connections
    def __init__(self, num_inputs, num_outputs, activation=none):
        self.num_in, self.num_out = num_inputs, num_outputs
        self.nodes = np.zeros(num_inputs+num_outputs)
        # i-th index -> vector of weights feeding into node i-n (n is the number of input nodes) (they have to be dense I guess)
        # The weights that are not from an actual connection gene are just gonna be 0
        self.weights = np.zeros((num_outputs, num_inputs))
        self.node_genes = [NodeGene(i,0,NodeType.OUTPUT) for i in range(num_outputs)]+[NodeGene(num_outputs+i,0,NodeType.INPUT) for i in range(num_inputs)]
        self.conn_genes = []
        self.activation = activation

    def _predict(self, x) -> np.ndarray:
        if not isinstance(x, np.ndarray) and not isinstance(x, list):
            raise TypeError(f"Expected a numpy array or a python list but received an object of type {type(x)}")
        elif len(x) != self.num_in:
            raise ValueError(f"Expected an input of size {self.num_in} but received one of size {len(x)}")

        # update the hidden nodes in the right order
        self.nodes[self.num_out:self.num_out+self.num_in] = x
        for conn_gene in self.conn_genes:
            if self.node_genes[conn_gene.end]._type != NodeType.OUTPUT:
                act_nodes = self.nodes[self.num_out:] # nodes used in the actual activation
                end_node_mat_idx = conn_gene.end-self.num_in
                self.nodes[conn_gene.end] = np.dot(act_nodes, self.weights[end_node_mat_idx])+self.node_genes[conn_gene.end].bias 

        # calculate the outputs
        for idx in range(self.num_out):
            self.nodes[idx] = np.dot(self.weights[idx], self.nodes[self.num_out:])+self.node_genes[idx].bias

        return self.activation(self.nodes[:self.num_out]) # returns the output nodes

    def predict(self, X) -> np.ndarray:
        return np.array([self._predict(x) for x in X], dtype=np.float32)

    # get the actual structure of the network
    def get_structure(self) -> tuple:
        return (self.node_genes, self.conn_genes)

    # The innovation number for the nodes and for the connections will remain separate
    def gen_node(self, init_bias=True) -> None:
        # add the weights feeding into the other hidden neurons and into the outputs stemming from the new hidden neuron
        self.weights = np.c_[self.weights, np.zeros(len(self.nodes)-self.num_in)]
        # add the weights feeding into the new node from all of the inputs and all of the other hidden neurons
        # maybe I have to add +1 after self.num_out
        self.weights = np.vstack([self.weights, np.zeros(len(self.nodes)-self.num_out+1)])
        self.node_genes.append(NodeGene(len(self.node_genes), self.node_genes[-1].innov+1, NodeType.HIDDEN))
        if init_bias:
            self.node_genes[-1].bias = np.random.normal(scale=1)
        self.nodes = np.append(self.nodes, 0)

    def add_node(self, node: NodeGene) -> None:
        # can't add an input or output node
        if node._type != NodeType.HIDDEN:
            raise TypeError("Cannot add an input or output node to an already initialized network")

        self.weights = np.c_[self.weights, np.zeros(len(self.nodes)-self.num_in)]
        self.weights = np.vstack([self.weights, np.zeros(len(self.nodes)-self.num_out+1)])
        self.node_genes.append(node)
        self.node_genes = np.append(self.nodes, 0)

    def add_conn(self, gene: ConnectionGene, weight=0) -> None:
        # I wanna be clear with error messages
        if gene.end < gene.start and self.node_genes[gene.end]._type == NodeType.HIDDEN:
            raise ValueError("A connection cannot stem from a node with a higher index than the one it points to!")
        elif gene.end == gene.start:
            raise ValueError("A connection should not stem from a node and point to itself!")
        elif self.node_genes[gene.end]._type == NodeType.INPUT:
            raise ValueError("Cannot form a connection feeding into an input node!")

        # make sure the gene is added in its right chronological place, I decided to start checking from the end cause
        # it's more likely that the connection gene has just been generated.
        idx = len(self.conn_genes)-1
        try:
            while self.conn_genes[idx].innov > gene.innov:
                idx -= 1
        except IndexError:
            idx = 0
    
        self.conn_genes.insert(idx+1, gene)
        # if the weight is 0 there is no connection, wouldn't make sense to add this node, I'll
        # just assume that the node wasn't set manually and should therefore be initialized randomly
        end_node_mat_idx = gene.end if self.node_genes[gene.end]._type == NodeType.OUTPUT else gene.end-self.num_in
        self.weights[end_node_mat_idx][gene.start-self.num_out] = weight if weight != 0 else np.random.normal(scale=1)

    @dispatch(ConnectionGene)
    def get_weight(self, gene: ConnectionGene) -> float:
        '''
            Returns the actual value of the weight represented by the given connection gene
        '''
        mat_end_node = gene.end if self.node_genes[gene.end]._type == NodeType.OUTPUT else gene.end-self.num_in
        return self.weights[mat_end_node][gene.start-self.num_out]

    @dispatch(ConnectionGene, float)
    def set_weight(self, gene: ConnectionGene, val: float) -> None:
        '''
            Sets the value of the weight represented by the given connection gene
        '''
        mat_end_node = gene.end if self.node_genes[gene.end]._type == NodeType.OUTPUT else gene.end-self.num_in
        self.weights[mat_end_node][gene.start-self.num_out] = val

    @dispatch(int)
    def get_weight(self, conn_gene_idx: int) -> float:
        '''
            Returns the actual value of the weight represented by the connection gene at index conn_gene_idx
        '''
        gene = self.conn_genes[conn_gene_idx]
        mat_end_node = gene.end if self.node_genes[gene.end]._type == NodeType.OUTPUT else gene.end-self.num_in
        return self.weights[mat_end_node][gene.start-self.num_out]
    
    @dispatch(int, float)
    def set_weight(self, conn_gene_idx: int, val: float) -> None:
        '''
            Sets the value of the weight represented by the connection gene at index conn_gene_idx
        '''
        gene = self.conn_genes[conn_gene_idx]
        mat_end_node = gene.end if self.node_genes[gene.end]._type == NodeType.OUTPUT else gene.end-self.num_in
        self.weights[mat_end_node][gene.start-self.num_out] = val

    @dispatch(int)
    def disable_conn(self, conn_idx: int) -> None:
        '''
            Disables the connection gene at index conn_idx
        '''
        self.conn_genes[conn_idx].enabled = False
        self.set_weight(conn_idx, 0)

    @dispatch(ConnectionGene)
    def disable_conn(self, conn_gene: ConnectionGene) -> None:
        '''
            Disables the connection represented by conn_gene
        '''
        self.set_weight(conn_gene, 0)

    def get_node_innov_interval(self) -> tuple:
        '''
            Returns the interval in which the innovation numbers of the node genes
            can be found in this network, the output is (min_innov, max_innov)
        '''
        try:
            min_innov = self.node_genes[self.num_in+self.num_out].innov
        except IndexError: # if the network doesn't have any hidden nodes
            min_innov = 0
        return min_innov, self.node_genes[-1].innov

    def get_weight_innov_interval(self) -> tuple:
        '''
            Returns the interval in which the innovation numbers of the weight genes
            can be found in this network, the output is (min_innov, max_innov)
        '''
        try:
            min_innov = self.conn_genes[0].innov
            max_innov = self.conn_genes[-1].innov
        except IndexError:
            return 0, 0
        return min_innov, max_innov
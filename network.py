import numpy as np
from gene import *
from activations import ACTIVATIONS, none
from multipledispatch import dispatch
from copy import deepcopy

# Important shit
# - np.c_[mat, arr] appends the array as a column in the end
# - The connection genes have to be stored in the array in an increasing order of the innovation number
# - The first nodes are always the output ones (in nodes and node_geens), then come the inputs, and then the
#       hidden nodes

class Network:
    ''' Class representing a neural network without layers and with a customizable structure '''

    # initializes the most basic structure with no connections
    def __init__(self, num_inputs, num_outputs, activation='none'):
        global ACTIVATIONS

        self.num_in, self.num_out = num_inputs, num_outputs
        self.nodes = np.zeros(num_inputs+num_outputs)
        # i-th index -> vector of weights feeding into node i-n (n is the number of input nodes) (they have to be dense I guess)
        # The weights that are not from an actual connection gene are just gonna be 0
        self.weights = np.zeros((num_outputs, num_inputs))
        self.node_genes = [NodeGene(i, NodeType.OUTPUT) for i in range(num_outputs)]+[NodeGene(num_outputs+i, NodeType.INPUT) for i in range(num_inputs)]
        self.conn_genes = []
        if isinstance(activation, str):
            self.activation = ACTIVATIONS[activation]
        else:
            self.activation = activation

    def _predict(self, x) -> np.ndarray:
        '''
            Returns the values the network predicts when the input x is fed into the network.
        '''
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
    def get_structure(self, make_copy=False) -> tuple:
        '''
            Returns a tuple where the first element is a list of all the node genes in the network, and the second element
            is a list of all the connection genes in the network.\n
            make_copy -> whether to return the very instances of the nodes in the list or a copy
        '''
        return (self.node_genes, self.conn_genes) if not make_copy else (deepcopy(self.node_genes), deepcopy(self.conn_genes))

    # The innovation number for the nodes and for the connections will remain separate
    def gen_node(self, init_bias=True, bias_std=1.0) -> None:
        '''
            Generates a new node and sets its threshold according to a normal distribution with standard deviation bias_std
        '''
        # add the weights feeding into the other hidden neurons and into the outputs stemming from the new hidden neuron
        self.weights = np.c_[self.weights, np.zeros(len(self.nodes)-self.num_in)]
        # add the weights feeding into the new node from all of the inputs and all of the other hidden neurons
        self.weights = np.vstack([self.weights, np.zeros(len(self.nodes)-self.num_out+1)])
        self.node_genes.append(NodeGene(len(self.node_genes), NodeType.HIDDEN))
        if init_bias:
            self.node_genes[-1].bias = np.random.normal(scale=bias_std)
        self.nodes = np.append(self.nodes, 0)

    def gen_nodes(self, num_nodes=2, init_bias=True) -> None:
        '''
            Generates a certain new number of nodes:\nnum_nodes -> the number of nodes generated\n
            init_bias -> whether the bias of the new nodes is initialized (if it isn't its default value is 0)
        '''
        for _ in range(num_nodes):
            self.gen_node(init_bias=init_bias)

    def add_node(self, node: NodeGene) -> None:
        '''
            Adds the node represented by the given gene to the network
        '''
        # can't add an input or output node
        if node._type != NodeType.HIDDEN:
            raise TypeError("Cannot add an input or output node to an already initialized network")

        self.weights = np.c_[self.weights, np.zeros(len(self.nodes)-self.num_in)]
        self.weights = np.vstack([self.weights, np.zeros(len(self.nodes)-self.num_out+1)])
        self.node_genes.append(node)
        self.nodes = np.append(self.nodes, 0)

    def add_conn(self, gene: ConnectionGene, weight_std=1.0) -> int:
        '''
            The function applies the connection gene passed as its argument to the network, if the value for the
            weight isn't already set in the gene object, it's initialized randomly according to a normal
            distribution with standard deviation weight_std
        '''
        # In the end I decided to allow connections from a higher index to a lower one, but other parts
        # of the code (such as in the random mutation part) make checks to avoid them.
        # I wanna be clear with error messages
        if gene.end == gene.start:
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
            idx = -1

        # if the weight is 0 there is no connection, wouldn't make sense to add this node, I'll
        # just assume that the node wasn't set manually and should therefore be initialized randomly    
        gene.weight = 0 if gene.weight != 0 else np.random.normal(scale = weight_std)
        self.conn_genes.insert(idx+1, gene)
        end_node_mat_idx = gene.end if self.node_genes[gene.end]._type == NodeType.OUTPUT else gene.end-self.num_in
        self.weights[end_node_mat_idx][gene.start-self.num_out] = gene.weight

    @dispatch(ConnectionGene)
    def get_weight(self, gene: ConnectionGene) -> float:
        '''
            Returns the actual value of the weight represented by the given connection gene
        '''
        mat_end_node = gene.end if self.node_genes[gene.end]._type == NodeType.OUTPUT else gene.end-self.num_in
        try:
            return self.weights[mat_end_node][gene.start-self.num_out]
        except IndexError:
            return 0
        
    @dispatch(ConnectionGene, float)
    def set_weight(self, gene: ConnectionGene, val: float) -> None:
        '''
            Sets the value of the weight represented by the given connection gene
        '''
        # Update the weight value of the gene
        for conn_gene in self.conn_genes:
            if conn_gene.start == gene.start and conn_gene.end == conn_gene.end:
                conn_gene.weight = val
        mat_end_node = gene.end if self.node_genes[gene.end]._type == NodeType.OUTPUT else gene.end-self.num_in
        self.weights[mat_end_node][gene.start-self.num_out] = val
            
    @dispatch(int)
    def get_weight(self, conn_gene_idx: int) -> float:
        '''
            Returns the actual value of the weight represented by the connection gene at index conn_gene_idx
        '''
        return self.conn_genes[conn_gene_idx].weight

    @dispatch(int, float)
    def set_weight(self, conn_gene_idx: int, val: float) -> None:
        '''
            Sets the value of the weight represented by the connection gene at index conn_gene_idx
        '''
        gene = self.conn_genes[conn_gene_idx]
        mat_end_node = gene.end if self.node_genes[gene.end]._type == NodeType.OUTPUT else gene.end-self.num_in
        # Sometimes this throws an error, I'm not actually sure why, but It happens rarely enough that It doesn't
        # really affect the process as a whole (once every 30 gens with 50 pop), I might look into it at some point
        try:
            self.weights[mat_end_node][gene.start-self.num_out] = val
            self.conn_genes[conn_gene_idx].weight = val
        except IndexError:
            pass

    @dispatch(int)
    def disable_conn(self, conn_idx: int) -> None:
        '''
            Disables the connection gene at index conn_idx
        '''
        self.conn_genes[conn_idx].enabled = False
        self.conn_genes[conn_idx].weight = self.get_weight(conn_idx)
        self.set_weight(conn_idx, 0.0)

    @dispatch(ConnectionGene)
    def disable_conn(self, conn_gene: ConnectionGene) -> None:
        '''
            Disables the connection represented by conn_gene
        '''
        for conn in self.conn_genes:
            if conn.start == conn_gene.start and conn.end == conn_gene.end:
                conn.enabled = False
                conn.weight = self.get_weight(conn)
                break
        self.set_weight(conn_gene, 0.0)

    @dispatch(int)
    def enable_conn(self, conn_idx: int) -> None:
        ''' Enables the connection at index conn_idx '''
        if conn_idx >= len(self.conn_genes):
            raise ValueError("Cannot enable a connection that is not present in the network!")

        self.conn_genes[conn_idx].enabled = True
        self.set_weight(conn_idx, self.conn_genes[conn_idx].weight)

    @dispatch(ConnectionGene)
    def enable_conn(self, conn_gene: ConnectionGene) -> None:
        ''' 
            Enables the connection represented by conn_gene, if it isn't already present in the network 
            this does nothing
        '''
        for gene in self.conn_genes:
            if gene.start == conn_gene.start and gene.end == conn_gene.end:
                gene.enabled = True
                self.set_weight(gene, gene.weight)
                break
        

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

    # TODO: use binary search instead, since the innovation numbers are in an increasing order
    @dispatch(int)
    def has_node(self, node_innov: int) -> bool:
        '''
            Checks whether the node gene with the innovation number passed is also 
            present in this network
        '''
        return node_innov < len(self.node_genes)

    # TODO: use binary search instead, since the innovation numbers are in an increasing order
    @dispatch(NodeGene)
    def has_node(self, node_gene: NodeGene) -> bool:
        '''
            Checks whether the node gene passed is also present in this network
        '''
        return node_gene.index < len(self.node_genes)

    # TODO: use binary search instead, since the innovation numbers are in an increasing order
    @dispatch(ConnectionGene)
    def has_weight(self, conn_gene: ConnectionGene) -> bool:
        '''
            Checks whether the connection gene passed is also present in this network
        '''
        for conn in self.conn_genes:
            if conn.innov == conn_gene.innov:
                return True
        return False
    
    @dispatch(int)
    def has_weight(self, conn_innov: int) -> bool:
        '''
            Checks whether the connection gene with the innovation number passed is also 
            present in this network
        '''
        for conn in self.conn_genes:
            if conn.innov == conn_innov:
                return True
        return False

    def get_num_unmatched_node_genes(self, other) -> int:
        '''
            Returns the number of unmatched node genes with the Network instance other
        '''
        unmatched = 0
        
        s_min, s_max = self.get_node_innov_interval()
        o_min, o_max = other.get_node_innov_interval()

        # the +1 is there simply to include the gene with the highest innov. number
        for node_innov in range(min(s_min, o_min), max(s_max, o_max)+1):
            if self.has_node(node_innov) != other.has_node(node_innov):
                unmatched += 1
        return unmatched

    def has_conn(self, start: int, end: int) -> bool:
        '''
            Returns whether the network has a connection (regardless of whether it's disabled or not)
            between the node with the index start and the node with the index end
        '''
        for conn_gene in self.conn_genes:
            if conn_gene.start == start and conn_gene.end == end:
                return True
        return False

    def get_unmatched_conn_genes(self, other) -> int:
        '''
            Returns the number of unmatched conn genes with the Network instance other
        '''
        unmatched = 0        

        s_min, s_max = self.get_weight_innov_interval()
        o_min, o_max = other.get_weight_innov_interval()

        for conn_innov in range(min(s_min, o_min), max(s_max, o_max)+1):
            if self.has_weight(conn_innov) != other.has_weight(conn_innov):
                unmatched += 1
        return unmatched

    def get_conn_idx(self, start_idx: int, end_idx: int) -> int:
        '''
            Returns the index in this network's connection gene list of the gene with the
            given starting and ending nodes, if the gene isn't present, the function returns
            -1
        '''
        for conn_gene_idx in range(len(self.conn_genes)):
            if self.conn_genes[conn_gene_idx].start == start_idx and self.conn_genes[conn_gene_idx].end == end_idx:
                return conn_gene_idx
        return -1

    # TESTED
    def get_conn_gene(self, conn_innov: int) -> ConnectionGene:
        ''' 
            Returns the connection gene with the given innovation number, and -1 if
            it isn't in the network
         '''
        for conn_gene in self.conn_genes:
            if conn_gene.innov == conn_innov:
                return conn_gene
        return None

    def __str__(self) -> str:
        string = "Network:\n\tNodes:\n"
        for node_gene in self.node_genes:
            string += f"\t\tNode #{node_gene.index} type: {node_gene._type}, bias: {node_gene.bias}\n"
        string += "\tConnections:\n"
        for conn_gene in self.conn_genes:
            string += f"\t\tConnection {conn_gene.start} -> {conn_gene.end}, enabled: {conn_gene.enabled}\n"
        return string

    def load_structure(self, node_genes, conn_genes) -> None:
        '''
            Integrates the network structure represented by the lists of the node and connection genes
            into the instance of the network, applying all the values for the weights and for the thresholds.
        '''
        self.node_genes = node_genes
        self.conn_genes = conn_genes

        amount_hidden = len(node_genes)-self.num_in-self.num_out+1 # number of hidden neurons
        self.weights = np.zeros((self.num_out, self.num_in))
        # GENERATE THE WEIGHTS FOR THE HIDDEN NODES
        for i in range(amount_hidden):
            self.weights = np.c_[self.weights, np.zeros(self.weights.shape[0])]
            self.weights = np.vstack([self.weights, np.zeros(self.weights.shape[1])])

        # update the values of the weight matrix
        for conn_idx in range(len(conn_genes)):
            self.set_weight(conn_idx, conn_genes[conn_idx].weight)

        self.nodes = np.zeros(len(node_genes))
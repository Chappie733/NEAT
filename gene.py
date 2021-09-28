from enum import Enum

class GeneType(Enum):
    CONNECTION_GENE = 0,
    NODE_GENE = 1

class NodeType(Enum):
    INPUT = 0,
    HIDDEN = 1,
    OUTPUT = 2

class Gene:

    def __init__(self, _type, innov=1) -> None:
        self._type = _type
        self.innov = innov

    def __str__(self):
        return f"Gene:\n\tType: {self._type}\n\tInnovation number: {self.innov}"


class ConnectionGene(Gene):

    def __init__(self, start, end, innov, enabled=True, weight=0):
        super().__init__(GeneType.CONNECTION_GENE, innov)
        self.start, self.end = start, end  # indexes of ending and starting node
        self.enabled = enabled
        self.weight = weight # not usually set when the node is created
    
    def __str__(self):
        res = f"Connection Gene: "
        res += f"\n\tIn: {self.start}\n\tOut: {self.end}"
        res += f"\n\tEnabled: {self.enabled}"
        res += f"\n\tInnovation number: {self.innov}"
        return res

    def equals(self, other) -> bool:
        '''
        Checks if the gene is equal to another, this doesn't take into account the innovation number, but only the
        actual connection this gene represents
        '''
        return self.end == other.end and self.start == other.start

    @staticmethod
    def are_equal(f, s) -> bool:
        return f.end == s.end and f.start == s.start

class NodeGene(Gene):

    def __init__(self, index, _type):
        super().__init__(GeneType.NODE_GENE, index)
        self.index = index
        self._type = _type
        self.bias = 0

    def __str__(self):
        return f"Node Gene:\n\tIndex: {self.index}\n\tInnovation number: {self.innov}\n\tBias: {self.bias}\n\tType: {self._type}"
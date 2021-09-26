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
        return f"Gene:\nType: {self._type}\nInnovation number: {self.innov}"


class ConnectionGene(Gene):

    def __init__(self, start, end, innov, enabled=True):
        super().__init__(GeneType.CONNECTION_GENE, innov)
        self.start, self.end = start, end  # indexes of ending and starting node
        self.enabled = enabled
    
    def __str__(self):
        res = f"Connection Gene: "
        res += f"\nIn: {self.start}\nOut: {self.end}"
        res += f"\nEnabled: {self.enabled}"
        res += f"\nInnovation number: {self.innov}"
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

    def __init__(self, index, innov, _type):
        super().__init__(GeneType.NODE_GENE, innov)
        self.index = index
        self._type = _type
        self.bias = 0

    def __str__(self):
        return f"Node Gene:\nIndex: {self.index}\nInnovation number: {self.innov}\nBias: {self.bias}\nType: {self._type}"
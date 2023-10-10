import abc
import typing
import ast

import torch
from torchtext import vocab
from torch.utils.data import Dataset
from torch_geometric.data.data import Data as GeoData
import pandas as pd
import networkx as nx

from data.tokenize import get_source_and_markdown_tokenizers
from data.dataset import SourceGraphDatasetInterface


class SourceGraphDataset(SourceGraphDatasetInterface):
    df: pd.DataFrame

    def __init__(
            self, path: str, src_vocab: vocab.Vocab,
    ):
        self.path = path
        self.src_vocab = src_vocab
    
    def __len__(self):
        if not hasattr(self, 'df'):
            raise NotImplementedError('You must either fill the `df` field or override the `__len__` method')
        return len(self.df)

    @abc.abstractmethod
    def get_nodes_edges_rep(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    @abc.abstractmethod
    def get_edges_rep(self, index) -> torch.Tensor:
        pass
    
    def __getitem__(self, index) -> GeoData:
        nodes, edges = self.get_nodes_edges_rep(index)
        return GeoData(x=nodes, edge_index=edges)


class SourceGraphDatasetWithPreprocess(SourceGraphDataset):

    def ast2graph(self, root_node) -> nx.DiGraph:
        graph = nx.DiGraph()
        stack = [(None, root_node)]

        while stack:
            parent, node = stack.pop()

            if parent is not None:
                graph.add_node(parent)
                graph.add_edge(parent, node)

            for child in ast.iter_child_nodes(node):
                stack.append((node, child))

        return graph

    def src2graph(self, src: str) -> nx.DiGraph:
        parsed_code = ast.parse(src)
        return self.ast2graph(parsed_code)

    def __init__(
            self, path: str, src_vocab: vocab.Vocab,
    ):
        super().__init__(path, src_vocab)
        
        df = pd.read_csv(self.path, delimiter=',')
        self.graphs: typing.List[nx.DiGraph] = []
        for src in df['source']:
            self.graphs.append(self.src2graph(src))

    def node2identifier(self, node):
        possible_identifiers = [
            'asname', 'name', 'attr', 'id', 'arg'
        ]
        for pi in possible_identifiers:
            if hasattr(node, pi):
                return getattr(node, pi)
            return None

    def node2feature(self, node):
        id_ = self.node2identifier(node)
        return self.src_vocab.vocab.get_stoi().get(id_, self.src_vocab.vocab.get_stoi()['<unk>'])
    
    def get_nodes_edges_rep(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        graph = self.graphs[index]
        nodes = list(graph.nodes)
        nodes_rep = torch.Tensor([self.node2feature(n) for n in nodes])
        edges_rep = torch.Tensor([(nodes.index(e[0]), nodes.index(e[1])) for e in graph.edges])
        edges_rep = torch.einsum('ij->ji', edges_rep)
        nodes_rep = nodes_rep.unsqueeze(1)
        return nodes_rep, edges_rep

class SourceGraphDatasetWithoutPreprocess(SourceGraphDataset):
    def __init__(
            self, path: str, src_vocab: vocab.Vocab,
    ):
        super().__init__(path, src_vocab)
        df = pd.read_csv(self.path, delimiter=',', converters={
            'ast_nodes': ast.literal_eval,
            'ast_edges': ast.literal_eval,
        })
        df = df[['ast_nodes', 'ast_edges']]
        self.df = df


    @property
    def nodes(self):
      return self.df['ast_nodes']

    @property
    def edges(self):
      return self.df['ast_edges']

    def get_nodes_edges_rep(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.Tensor(self.nodes.iloc[index]), torch.Tensor(self.edges.iloc[index])

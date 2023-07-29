from __future__ import print_function
import time
from gensim.models import Word2Vec
from .walker import BasicWalker, Walker
import numpy as np
import ipdb
class Node2vec(object):

    def __init__(self, graph, path_length, num_paths, dim, p=1.0, q=1.0, dw=False, **kwargs):
        print(kwargs)
        print(kwargs.get("size", dim))
        kwargs["workers"] = kwargs.get("workers", 1)
        if dw:
            kwargs["hs"] = 1
            p = 1.0
            q = 1.0

        self.graph = graph
        if dw:
            self.walker = BasicWalker(graph, workers=kwargs["workers"])
        else:
            self.walker = Walker(
                graph, p=p, q=q, workers=kwargs["workers"])
            print("Preprocess transition probs...")
            self.walker.preprocess_transition_probs()
        sentences = self.walker.simulate_walks(
            num_walks=num_paths, walk_length=path_length)
        kwargs["sentences"] = sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["vector_size"] = kwargs.get("size", dim)
        kwargs["sg"] = 1

        self.size = kwargs["vector_size"]
        print("Learning representation...")
        word2vec = Word2Vec(**kwargs)

        self.vectors = {}
        # ipdb.set_trace()
        num_nodes = graph.num_nodes
        a = np.sqrt(6. / (num_nodes + 64))
        embedd = np.random.uniform(low=-a, high=a, size=[num_nodes, 64])
        print(len(graph.G.nodes()))
        for word in graph.G.nodes():
            self.vectors[word] = word2vec.wv[word]
            embedd[int(word)] = word2vec.wv[word]
        # for word in range(len(graph.G.nodes())):
        #     self.vectors[word] = word2vec.wv[word]
        #     embedd[word] = word2vec.wv[word]
        print(embedd.shape)
        self.embedd = embedd
        del word2vec

    def save_embeddings(self, filename):
        np.save(filename, self.embedd)

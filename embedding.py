# 处理词的embedding

import numpy as np
import json
import gensim

class embed(object):
    def __init__(self, data_input):
        self.embed = gensim.models.KeyedVectors.load_word2vec_format('vec4.bin', binary = True)
        self.data_input = data_input
        
    def id2embed(self, id):
        # id2word
        word = self.data_input.id2word[id]
        # word2embed
        result = self.embed[word]
        return result
    
    def word2embed(self, word):
        return self.embed[word]
        
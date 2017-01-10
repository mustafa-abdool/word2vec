import numpy as np 
from NeuralNet import SimpleNeuralNet, NegativeSampleNeuralNet
from neuralnetworktrainer import train_embedding_softmax,train_embbedding_neg_sampling
from training_example_helper import get_test_training_data,get_training_data_from_text_file
from util import get_vocab_from_training_examples, get_weights_for_sampling
from visualizationhelper import nearest_neighbor
import math

#Main file for generating neural network embeddings using word2vec
#Creates a simple neural network and outputs the nearest neighbors to some
#common words

#can try get_training_data_from_text_file("text_sample.txt") for a bigger file
training_examples = get_test_training_data(corpus_type="CBOW")
vocab_dict,reverse_vocab_dict,frequency_dict = get_vocab_from_training_examples(training_examples)
vocab_size = len(vocab_dict.keys())

#create neural network 
nn = SimpleNeuralNet(vocab_size,network_type="CBOW",num_context_words = 2.0,hidden_layer_size = 4, learning_rate = 0.005)
embedding_matrix = train_embedding_softmax(nn,training_examples,vocab_dict,num_iterations=100000)

nearest_neighbor('apple',nn.get_embedding_matrix(),vocab_dict, reverse_vocab_dict)
nearest_neighbor('drink',nn.get_embedding_matrix(),vocab_dict, reverse_vocab_dict)
nearest_neighbor('juice',nn.get_embedding_matrix(),vocab_dict, reverse_vocab_dict)

#negative sampling example:

"""
nn = NegativeSampleNeuralNet(vocab_size,num_context_words=5.0,hidden_layer_size = 30, learning_rate = 0.1)
words, probs = get_weights_for_sampling(frequency_dict)
train_embbedding_neg_sampling(nn, training_examples, vocab_dict, words, probs, k=3, num_iterations = 5)
nearest_neighbor('apple',nn.get_embedding_matrix(),vocab_dict, reverse_vocab_dict)
nearest_neighbor('orange',nn.get_embedding_matrix(),vocab_dict, reverse_vocab_dict)
"""
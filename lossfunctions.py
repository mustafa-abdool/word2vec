import math
from util import generate_one_hot_encoding, get_k_neg_samples
import numpy as np


def sigmoid(x):
	return 1/(1 + np.exp(-1.0 * x))


# E = -log(p(output | input)) sum over all training examples (for CBOW)
# E = -log( product of p(output | input) for all output words) (for Skipgram)
def compute_training_set_loss(training_examples, nn, vocab_dict, debug_info=False):
	loss = 0.0
	print '==========STARTING NEW COMPUTATION OF TRAINING LOSS=========='
	for example in training_examples:
		input_words, output_words = example
		input_v, output_v = generate_one_hot_encoding(example, vocab_dict)
		predicted_output = nn.feed_forward(input_v, output_v)

		for w in output_words: 
			output_index = vocab_dict[w]
			if debug_info is True:
				print 'output index is ' + str(output_index)
				print 'output word is ' + w
				print 'output prob is ' + str(predicted_output[output_index])

			#compute p(output | input) from the output vector
			#print predicted_output[output_index]
			loss += -1.0 * math.log(predicted_output[output_index])
	print 'total loss is ' + str(loss)
	return loss

#words = words used in distribution for negative sampling
#probs = define distribution over words
#k = # of negative samples to accquire 
def compute_training_set_loss_negative_sampling(training_examples,nn,vocab_dict,words,probs,k):
	loss = 0.0
	print '==========STARTING NEW COMPUTATION OF TRAINING LOSS WITH NEGATIVE SAMPLES=========='
	for example in training_examples:
		input_words, output_words = example
		input_v, output_v = generate_one_hot_encoding(example, vocab_dict)
		predicted_output = nn.feed_forward(input_v, output_v)

		#generate output words score
		for w in output_words: 
			output_index = vocab_dict[w]
			predicted_val = sigmoid(predicted_output[output_index])
			loss += -1.0 * math.log(predicted_val)

		#generate negative samples score
		neg_words, neg_idx = get_k_neg_samples(words, probs, k, vocab_dict)
		for b in neg_idx:
			predicted_val = sigmoid(-1.0 * predicted_output[b])
			loss += -1.0 * math.log(predicted_val)	
	print 'total loss is ' + str(loss)
	return loss
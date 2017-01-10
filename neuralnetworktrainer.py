from lossfunctions import compute_training_set_loss, compute_training_set_loss_negative_sampling
from util import generate_one_hot_encoding, get_k_neg_samples

#Methods to train a neural network based on either softmax output layer or
#using negative sampling. Negative sampling is faster in general as softmax
#is expensive to compute once the vocabulary gets large

def train_embedding_softmax(nn,training_examples,vocab_dict,num_iterations=20000):
	#print the loss function before you start 
	compute_training_set_loss(training_examples,nn,vocab_dict)

	for i in range(0,num_iterations):
		for example in training_examples:
			input_v, output_v = generate_one_hot_encoding(example, vocab_dict)
			nn.backprop_update(input_v,output_v)
		if (i % 10000 == 0):
			compute_training_set_loss(training_examples,nn,vocab_dict)

	'Final loss is ' + str(compute_training_set_loss(training_examples,nn,vocab_dict))
	return nn.get_embedding_matrix()


#get training examples and call these functions
#return embedding matrix and then you can do stuff with it
def train_embbedding_neg_sampling(nn, training_examples, vocab_dict, words, probs, k=3, num_iterations=20000):
	compute_training_set_loss_negative_sampling(training_examples, nn, vocab_dict, words, probs, k)

	for i in range(0,num_iterations):
		for example in training_examples:
			input_v, output_v = generate_one_hot_encoding(example, vocab_dict)
			context, target = example
			output_indices = []
			for t in target:
				output_indices.append(vocab_dict[t])
			neg_samples, neg_idxs = get_k_neg_samples(words, probs, k, vocab_dict)
			nn.backprop_update(input_v, output_v, output_indices, neg_idxs)
		if (i % 5000 == 0):
			'Loss is' + str(compute_training_set_loss_negative_sampling(training_examples, nn, vocab_dict, words, probs, k)) 
	'Final loss is ' + compute_training_set_loss_negative_sampling(training_examples, nn, vocab_dict, words, probs, k)
	return nn.get_embedding_matrix()


import numpy as np

#returns one hot encoding of (input, output) pairs
#can use skipgram or cbow
def generate_one_hot_encoding(training_example, vocab_dict, type="CBOW"):
	vocab_size = len(vocab_dict.keys())
	input_words, output_words = training_example
	input_vector = np.zeros(vocab_size)
	for w in input_words:
		input_vector[vocab_dict[w]] = 1

	output_vector = np.zeros(vocab_size)
	for w in output_words:
		output_vector[vocab_dict[w]] = 1

	if type=="CBOW":
		input_vector = input_vector/float(len(input_words))
	return (input_vector, output_vector)

#map from word --> raw count
def get_weights_for_sampling(frequency_dict):
	total_count = sum(frequency_dict.values())
	word_list = []
	weights = []
	for k,v in frequency_dict.iteritems():
		word_list.append(k)
		weights.append(v/float(total_count))
	return word_list, weights


def get_k_neg_samples(words, probs, k, vocab_dict):
	samples = []
	samples_idx = []
	for i in range(0,k):
		random_word = np.random.choice(words,p=probs)
		samples.append(random_word)
		samples_idx.append(vocab_dict[random_word])
	return samples, samples_idx

#gets the vocab dictionary, reverse dict and all that
def get_vocab_from_training_examples(training_examples):
	curr_idx = 0
	vocab_dict = {}
	reverse_vocab_dict = {}
	frequency_dict = {}

	for example in training_examples:
		context, target = example
		for c in context:
			if c not in vocab_dict:
				vocab_dict[c] = curr_idx
				curr_idx += 1
			if c not in frequency_dict:
				frequency_dict[c] = 0
			frequency_dict[c] += 1
		for t in target:
			if t not in vocab_dict:
				vocab_dict[t] = curr_idx
				curr_idx += 1
			if t not in frequency_dict:
				frequency_dict[t] = 0
			frequency_dict[t] += 1

	#create a reverse dict for easy idx --> word lookup 
	for k,v in vocab_dict.iteritems():
		reverse_vocab_dict[v] = k	

	return vocab_dict,reverse_vocab_dict,frequency_dict

#corups is of the form input|output, seperated by commas
#if using CBOW, input is a list of words seperated by ^
#if using SkipGram output is a list of words seperated by ^
def get_training_examples_from_corpus(corpus):
	curr_idx = 0
	for example in corpus.split(","):
		context, target = example.split("|")
		for c in context.split("^"):
			if c not in vocab_dict:
				vocab_dict[c] = curr_idx
				curr_idx += 1
			if c not in frequency_dict:
				frequency_dict[c] = 0
			frequency_dict[c] += 1
		for t in target.split("^"):
			if t not in vocab_dict:
				vocab_dict[t] = curr_idx
				curr_idx += 1
			if t not in frequency_dict:
				frequency_dict[t] = 0
			frequency_dict[t] += 1
		training_examples.append((context.split("^"),target.split("^")))

	#create a reverse dict for easy idx --> word lookup 
	for k,v in vocab_dict.iteritems():
		reverse_vocab_dict[v] = k


#form is input|output, seperated by commas
#if using CBOW, input is a list of words seperated by ^
#if using SkipGram output is a list of words seperated by ^
def get_training_examples_from_test_string(corpus):
	training_examples = []
	for example in corpus.split(","):
		context, target = example.split("|")
		input_words = context.split("^")
		output_words = target.split("^")
		training_examples.append((input_words, output_words))

	return training_examples
#Helper file with methods to generate training examples for word2vec

#context | target (for CBOW)
cbow_test_corpus = "drink^juice|apple,eat^apple|orange,drink^juice|rice,drink^milk|juice,drink^rice|milk,drink^milk|water,orange^apple|juice,apple^drink|juice,rice^drink|milk,milk^water|drink,water^juice|drink"

#for skipgram (context | target)
skipgram_test_corpus = "apple|drink^juice,orange|eat^apple,rice|drink^juice,juice|drink^milk,milk|drink^rice,water|drink^milk,juice|orange^apple,juice|apple^drink,milk|rice^drink,drink|milk^water,drink|water^juice,drink|juice^water"

def get_test_training_data(corpus_type="CBOW"):
	if corpus_type=="CBOW":
		return get_training_examples_from_test_string(cbow_test_corpus)
	elif corpus_type=="SKIPGRAM":
		return get_training_examples_from_test_string(skipgram_test_corpus)

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

#load in a text file, do some basic parsing and 
#return list of training examples 
def get_training_data_from_text_file(filename,extract_type="CBOW"):
	f = open(filename)

	raw_text = f.read()
	raw_tokens =  raw_text.split(" ")

	print 'Raw number of tokens in file is ' + str(len(raw_tokens))
	#do some basic filtering 
	filtered_tokens = map(lambda x : x.strip(), raw_tokens)
	cleaned_tokens = map(lambda x : x.rstrip('?:!.,;').replace("\n",""), filtered_tokens)
	examples = get_examples_from_tokens(cleaned_tokens, 5,extract_type)
	return examples


#window size should be odd 
#return a list of ([input_words],[output_words])
def get_examples_from_tokens(token_list,window_size, extract_type):
	training_examples = []
	for window_start in range(0,len(token_list) - window_size):
		window_end = window_start + window_size 
		middle_idx = (window_start + window_end - 1)/2

		outside_window = token_list[window_start:middle_idx] + token_list[middle_idx+1:window_end]
		window_word = [token_list[middle_idx]]
		if extract_type == "CBOW":
			training_examples.append( (outside_window, window_word) )

		elif extract_type == "SKIPGRAM":
			training_examples.append((window_word, outside_window))


	return training_examples
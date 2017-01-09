import numpy as np

def print_dictionary_top_k_helper(mydict, limit):
	k = 1
	for key, value in sorted(mydict.iteritems(), key=lambda (k,v): (v,k), reverse=True):
		print "%s: %s" % (key, mydict[key])
		if k == limit:
			break
		k += 1

#prints out the top 3 nearest neighbors for a given word using cosine distance
def nearest_neighbor(word, hidden_weight_matrix,vocab_dict, reverse_vocab_dict):
	word_vector = hidden_weight_matrix[vocab_dict[word]]
	#go through all the rows and put in a dictionary ?
	#then sort dictionary by value 
	score_dict = {}
	for row_idx in range(0,hidden_weight_matrix.shape[0]):
		if row_idx != vocab_dict[word]:
			score = np.dot(word_vector, hidden_weight_matrix[row_idx])
			score_dict[reverse_vocab_dict[row_idx]] = score/(np.linalg.norm(word_vector) * np.linalg.norm(hidden_weight_matrix[row_idx]))

	#print top 3 closest words
	print 'top 3 closest words to ' + word + '...'
	print_dictionary_top_k_helper(score_dict, 3)
import numpy as np


#works for skipgram and cbow - no need to distinguish between them
#Has one hidden layer
class NegativeSampleNeuralNet:

	 def __init__(self, input_size, network_type="CBOW", num_context_words = 2.0, learning_rate = 0.1, hidden_layer_size = 4):

	 	#contants
		self.LEARNING_RATE = learning_rate
		self.HIDDEN_LAYER_SIZE = hidden_layer_size
		self.NN_TYPE = network_type
		self.NUM_CONTEXT_WORDS = float(num_context_words) #for skipgram

		self.input_size = input_size
		self.layers = []
		self.layers.append(IdentityLayer(input_size, self.HIDDEN_LAYER_SIZE))
		self.layers.append(IdentityLayer(self.HIDDEN_LAYER_SIZE, input_size))


	 def feed_forward(self, input_vector, target_vector):
		first_layer = self.layers[0]
		first_layer_output = first_layer.get_output(input_vector)

		hidden_layer = self.layers[1]
		hidden_layer_output = hidden_layer.get_output(first_layer_output)
		return hidden_layer_output

	 #neg sample indices are a list of positions in the vocab where you chose negative samples 
	 #target index is the index of target word in the vocabs
	 #for skipgram, target index is actually a list where as for CBOW it's just one number
	 def backprop_update(self, input_vector, target_vector, target_indices, neg_sample_indices):
		predicted_vector = self.feed_forward(input_vector, target_vector)

		objective_gradient_wrt_output  = np.zeros(self.input_size)
		#only the gradient for neg samples and target matters 
		for i in neg_sample_indices:
			objective_gradient_wrt_output[i] = self.sigmoid(predicted_vector[i])
		for t in target_indices:
			objective_gradient_wrt_output[t] = self.sigmoid(predicted_vector[t])

		prediction_error = objective_gradient_wrt_output - target_vector

		#todo: make more generic and extend to multiple layers
		#todo: just update individual columns instead of matrix multiplication which results in the same thing anyway
		output_layer = self.layers[1]
		hidden_layer = self.layers[0]

		output_layer_grad = output_layer.backprop(prediction_error)

		output_layer_error = np.dot(prediction_error, output_layer.weights.T)
		hidden_layer_grad = hidden_layer.backprop(output_layer_error)

		#update using derivatives of hidden->output and input-> hidden matrix
		output_layer.weights = output_layer.weights - self.LEARNING_RATE * output_layer_grad
		hidden_layer.weights = hidden_layer.weights - self.LEARNING_RATE * hidden_layer_grad

	 def get_embedding_matrix(self):
	 	return self.layers[0].weights
	 	#could also try: 
		#return (self.layers[0].weights + self.layers[1].weights.T)/2.0

	 def sigmoid(self, x):
		return 1/(1 + np.exp(-1.0 * x))


#simple neural net used for word2vec. Has one hidden layer and softmax output layer
class SimpleNeuralNet:

	 def __init__(self, input_size, num_context_words, network_type="CBOW", learning_rate = 0.1, hidden_layer_size = 4):

	 	#contants
		self.LEARNING_RATE = learning_rate
		self.HIDDEN_LAYER_SIZE = hidden_layer_size
		self.NN_TYPE = network_type
		self.NUM_CONTEXT_WORDS = float(num_context_words) #for skipgram

		self.input_size = input_size
		self.layers = []
		self.layers.append(IdentityLayer(input_size, self.HIDDEN_LAYER_SIZE))
		self.layers.append(SoftMaxLayer(self.HIDDEN_LAYER_SIZE, input_size))


	 def feed_forward(self, input_vector, target_vector):
		first_layer = self.layers[0]
		first_layer_output = first_layer.get_output(input_vector)

		hidden_layer = self.layers[1]
		hidden_layer_output = hidden_layer.get_output(first_layer_output)
		return hidden_layer_output

	 def backprop_update(self, input_vector, target_vector):
		predicted_vector = self.feed_forward(input_vector, target_vector)
		if self.NN_TYPE == "SKIPGRAM":
			prediction_error = self.NUM_CONTEXT_WORDS * predicted_vector - target_vector
		if self.NN_TYPE == "CBOW":
			prediction_error = predicted_vector - target_vector

		#todo: make more generic and extend to multiple layers
		output_layer = self.layers[1]
		hidden_layer = self.layers[0]

		output_layer_grad = output_layer.backprop(prediction_error)

		output_layer_error = np.dot(prediction_error, output_layer.weights.T)
		hidden_layer_grad = hidden_layer.backprop(output_layer_error)

		#update using derivatives of hidden->output and input-> hidden matrix
		output_layer.weights = output_layer.weights - self.LEARNING_RATE * output_layer_grad
		hidden_layer.weights = hidden_layer.weights - self.LEARNING_RATE * hidden_layer_grad

	 def get_embedding_matrix(self):
	 	return self.layers[0].weights
	 	#could also try: 
		#return (self.layers[0].weights + self.layers[1].weights.T)/2.0

#layer that just does the linear transform but nothing else 
class IdentityLayer:

	 def __init__(self, input_size, output_size):
		self.input_size = input_size
		self.output_size = output_size
		self.weights = np.random.random((input_size, output_size))
		self.last_input = None

	 def get_output(self, input_vector):
	 	self.last_input = input_vector
	 	#check order here
		return np.dot(input_vector, self.weights)

	 def backprop(self, error_vector):
		return np.outer(self.last_input, error_vector)

#layer that does the typical linear transform and then computes softmax probabilities 
class SoftMaxLayer:

	 def __init__(self, input_size, output_size):
		self.input_size = input_size
		self.output_size = output_size
		self.weights = np.random.random((input_size, output_size))
		self.last_input = None

	#assume input is column vector
	 def get_output(self, input_vector):
		raw_scores = np.dot(input_vector, self.weights)
		self.last_input = input_vector
		return self.softmax(raw_scores)

	 def softmax(self, x):
	    #helps with preventing overflow 
	    e_x = np.exp(x - np.max(x))
	    return e_x / e_x.sum()

	 def backprop(self, error_vector):
		return np.outer(self.last_input, error_vector)


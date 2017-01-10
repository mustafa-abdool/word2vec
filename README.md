#What is this project ? 

This repo contains a custom implemention of the "word2vec" algorithm for generating word embeddings as described in Mikolov's original paper https://arxiv.org/pdf/1301.3781.pdf . 

#What is word2vec ? 

The purpose of "word2vec" is to find a vector representation of words such that words with similar semantic meaning
are "close" (in the sense of the dot product) in the resulting vector space. For example, you would expect the vector for
"apple" to be close to the vector for "orange". A useful resource is the CS224 lecture where Richard Socher talks about word vectors : https://www.youtube.com/watch?v=T8tQZChniMk

Mostly everything is written from scratch in python except for using the numpy package to compute matrix products and norms.
The NeuralNet class is composed of "layers" which are fairly modular and can easily be extended to make more complex neural networks.

#How is it done ?

The equations used for the loss function and backpropogation were derived in a manner very similar to this paper : https://arxiv.org/pdf/1411.2738v4.pdf

There are also two ways of representing the training examples. Both of these are implemented

1.) Continuous Bag of Words (CBOW) - predict middle word from surrounding words (context)

2.) Skipgram - predict surrounding words (context) from middle word

According to Mikolov, usually the skipgram method is better with smaller amounts of training data whereas CBOW is faster to train and is more accurate for frequent words. 

#How to use it ?

There are two main neural networks that can be used (found in NeuralNet.py)

1.) SimpleNeuralNet - Uses softmax output layer (probability distribution over all the words). This works but is slower.

2.) NegativeSampleNeuralNet - Uses negative sampling to train as explaiend in Mikolov's paper. This is much faster than using the softmax output.

You can see word2vecrunner.py class for examples of using either of the two neural networks to learn embeddings from sample text.
I also impelmented a simple parser in training_example_helper.py which can take in a text file and output a list of training examples from it (either using the skipgram or CBOW methods)

You can use visualizationhelper.py to print out the "closest" words to a given word after the embeddings have been trained. 

#Examples 

top 3 closest words to apple...
orange: 0.696907523086
eat: 0.410704152037
juice: 0.34930571934

top 3 closest words to drink...
orange: 0.33042096028
apple: 0.204528996874
rice: 0.0259775283778

top 3 closest words to juice...
water: 0.303744738162
drink: 0.289949206689
orange: 0.106262570786

#Future work 

-Make clean interface for neural network layers (implement function like backprop, get_output)
-Make clean interface for neural networks (implement functions like feed_forward, backprop)
-Optimize negative sampling backprop (only update a subset of output vectors to speed things up)



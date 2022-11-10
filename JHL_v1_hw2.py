# Default python libralies from HW1
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
# Add

def softmax_forward(vector):
	exps = np.exp(vector)
	out = exps / np.sum(exps)
	print('\tsoftmax_forward : ', out)

	return out


def softmax_forward_stable(vector):
	exps = np.exp( vector - np.max(vector) )
	out = exps / np.sum(exps)
	print('\tsoftmax_forward_stable : ', out)

	return out


class NetworkLayer:
	def __init__(self, X, output_size):
		self.layer = []
		self.loss = None
		self.loss_prime = None
		self.input_size = X.shape[1]
		self.output_size = output_size
		self.weight = None
		# self.bias = None
		self.output = None
		self.weight = np.random.rand(self.input_size, self.output_size)
		self.output = np.zeros(self.output_size)
		# self.bias = np.random.rand(1, self.output_size)


	def fully_connected_layer_forward(X):
		# Initialization
		WtX = np.dot(X, self.weight)

		# Activation function: SoftMax
		self.output = softmax_forward(WtX)
		print( '\tfully_connected_layer_forward: ', self.output)

		# cache = (W, x, b)
		return self.output


	def fully_connected_layer_backwrad():
		###  derivative of Loss w.r.t. w1
		print('\tNot implemented yet..')

	## 
	def step():




def cross_entory(X, y):
	# X: output from FNN(Fully-connected NN), (= num_examples x num_classes)
	# y: label (= num_examples)
	#    y is not one-hot encoded vector

	m = y.shape[0]
	#p = softmax(X)

	print('\trange: ', range(m))
	log_like = -np.log(p[range(m), y])
	loss = np.sum(log_lik) / m

	return loss

'''
def gradient_descent( gradient, x, y, start_point, learn_rate, n_iter, tolerance):
	vector = start_point

	for _ in range(n_iter):
'''


def main():
	# loading the data
	M = loadmat('MNIST_digit_data.mat')

	images_train = M['images_train']
	labels_train = M['labels_train']
	images_test = M['images_test']
	labels_test = M['labels_test']

	# just to make all random sequences on all computers the same.
	np.random.seed(1)

	# randomly permute data points
	inds = np.random.permutation(images_train.shape[0])
	images_train = images_train[inds]
	labels_train = labels_train[inds]

	inds = np.random.permutation(images_test.shape[0])
	images_test = images_test[inds]
	labels_test = labels_test[inds]

	print(images_train.shape)

	BATCH_SIZE = 10

	print( 'Train image size: ', len(images_train) )
	print( 'iteration times: ', len(images_train)//BATCH_SIZE )

	FCL_net = NetworkLayer(images_train, output_size=10)

	for pos in range( len(images_train)//BATCH_SIZE ):
		print( '\tStart point: ', start_pos )
		batch_X = images_train[pos:pos+BATCH_SIZE]
		batch_Y = lables_train[pos:pos+BATCH_SIZE]

		predict = FCL_net.fully_connected_layer_forward(batch_X)
		print( '\tPrediction: ', predict)

		pos += BATCH_SIZE

	'''
	# ============================= Hoemework 7-c ===================================
	# 10 different dataset using logspace 
	#data_size = [ 29, 57, 109, 208, 396, 756, 1442, 2750, 5244, 10000 ]
	data_size = [int(x) for x in np.logspace(math.log10(30), 4, num=10, endpoint=True)]
	
	accuracy_avg_list = []
	acc_avg = 0

	# to speed up, use only first 1000 testing images
	images_test_temp = images_test[:1000, :]
	labels_test_temp = labels_test[:1000, :]

	for pos in data_size:
		# get traiding data based on data size value
		images_train_temp = images_train[:pos, :]
		labels_train_temp = labels_train[:pos, :]

		accuracy, acc_avg = kNN(images_train_temp, labels_train_temp, images_test_temp, labels_test_temp, k=1)
		print( 'Accuracy : ', accuracy, '\t acc_avg: ', acc_avg )
		accuracy_avg_list.append( round(acc_avg, 4) )
	
	# make plots
	plt.figure(figsize=(10,6))
	plt.plot(data_size, accuracy_avg_list, 'o-', markersize=3)
	plt.grid(alpha=0.3)
	plt.title('k=1 accuracy avg')
	plt.xlabel('Data size')
	plt.ylabel('Average accuracy')
	plt.tight_layout()
	plt.savefig('hw1_7_c.png')           
	# ============================= Hoemework 7-c ===================================
	'''
	'''
	# ============================= Hoemework 7-d ===================================
	# list for k values
	k_list = [1, 2, 3, 5, 10]
	#data_size = [ 29, 57, 109, 208, 396, 756, 1442, 2750, 5244, 10000 ]
	data_size = [int(x) for x in np.logspace(math.log10(30), 4, num=10, endpoint=True)]

	accuracy_avg_list = []
	acc_avg = 0

	# to speed up, use only first 1000 testing images
	images_test_temp = images_test[:1000, :]
	labels_test_temp = labels_test[:1000, :]

	# test and make plots
	plt.figure(figsize=(10,6))
	for k_value in k_list:
		# initialize average accuracy list on each k-values
		accuracy_avg_list = [] 
		for pos in data_size:
			# get traiding data based on data size value
			images_train_temp = images_train[:pos, :]
			labels_train_temp = labels_train[:pos, :]

			accuracy, acc_avg = kNN(images_train_temp, labels_train_temp, images_test_temp, labels_test_temp, k=k_value)
			accuracy_avg_list.append( round(acc_avg, 4) )

			print( 'K: ', k_value, '\nData size: ', pos, '\nAccuracy : ', accuracy, '\nAcc_avg: ', acc_avg )
		plt.plot(data_size, accuracy_avg_list, markersize=3)
		plt.legend(k_list)

	plt.grid(alpha=0.3)
	plt.title('Accuracy avg')
	plt.xlabel('Data size')
	plt.ylabel('Average accuracy')
	plt.tight_layout()

	plt.savefig('hw1_7_d.png')           
	# ============================= Hoemework 7-d ===================================
	'''
	'''
	# ============================= Hoemework 7-e ===================================
	# list for k values
	k_list = [1, 2, 3, 5, 10]
	accuracy_avg_list = []
	acc_avg = 0

	# to speed up, use only first 1000 testing images
	images_test_temp = images_test[:1000, :]
	labels_test_temp = labels_test[:1000, :]

	# get trained data set
	images_train_temp = images_train[:1000, :]
	labels_train_temp = labels_train[:1000, :]

	# get validation data set
	images_valid = images_train[1000:2000, :]
	labels_valid = labels_train[1000:2000, :]

	for k_value in k_list:
		accuracy, acc_avg = kNN(images_train_temp, labels_train_temp, images_valid, labels_valid, k=k_value)
		accuracy_avg_list.append( round(acc_avg, 4) )
		print( 'K: ', k_value, '\nAccuracy : ', accuracy, '\nAcc_avg: ', acc_avg )

	plt.figure(figsize=(10,6))
	plt.plot(k_list, accuracy_avg_list)
	plt.grid(alpha=0.3)
	plt.title('Accuracy avg')
	plt.xlabel('K')
	plt.ylabel('Average accuracy')
	plt.xticks(k_list)
	plt.tight_layout()

	plt.savefig('hw1_7_e.png')           
	# ============================= Hoemework 7-e ===================================
	'''
	#show the 10'th train image
	#i=10
	#im = images_train[i,:].reshape((28,28),order='F')
	#plt.imshow(im)
	#plt.title('Class Label:'+str(labels_train[i][0]))
	#plt.show()


if __name__ == "__main__":
	main()

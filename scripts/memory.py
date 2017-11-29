#!/usr/bin/env python

import rospy
import roboy_dep.msg
import roboy_communication_middleware.msg

import pickle
import numpy as np
import time

import tensorflow as tf

# Implementation for asynchronous Hopfield neural network
# Note: memory capacity approx. 0.14*nodes
'''
class Hopfield_Neural_Network:
	def __init__(self,nodes,iterations=100,weights=None):
		self.nodes = nodes
		self.iterations = iterations
		try:
			if weights == None:
				self.weights = np.zeros((nodes,nodes))
		except ValueError:
			self.weights = weights
	
	def store(self,input):
		dW = np.outer(input,input)
		np.fill_diagonal(dW,0)
		self.weights += dW
		
	def recall(self,input,range_=None):
		# Can specify range of nodes to iterate over (i.e. nodes that are "input" may be known as correct)
		if type(range_) == tuple:
			a = range(range_[0],range_[1])
		else:
			a = self.nodes
		update_sequence = np.random.choice(a, self.iterations)
		for node in update_sequence:
			input[node] = np.sign(np.inner(input,self.weights[:,node]))
			if input[node] == 0: # this was missing
				input[node] = -1
		return input
	
	def setIter(self,iter_):
		self.iterations = iter_
	
	def save_weights(self,filename):
		np.save(filename, self.weights)
		
	def load_weights(self,filename):
		weights = np.load(filename)
		if weights.shape == (self.nodes, self.nodes):
			self.weights = weights
		else:
			raise ValueError("Dimensions of specified weight array does not match network weight dimensions!")
'''

'''
def callback_hnn(msg):
	print "Brain id msg: %f \n" %msg.brain_id
	nodes = hnns[hnns.keys()[0]].nodes

	matrix_shape = (1,)
	m_encoder = scalar_sdr(80,44,-0.25,0.25,matrix_shape)
	b_encoder = scalar_sdr(92,40,0.0,1.0)
	#rospy.loginfo("Brain id: %f", msg.brain_id)

	data = np.zeros(nodes)-1
	brain_id = msg.brain_id
	brain_sig = b_encoder.encode(brain_id)
	data[-b_encoder.n:] = brain_sig

	matrix = np.zeros((6,12))
	
	for index in hnns:
		mem = hnns[index].recall(data,(0,m_encoder.n))
		matrix_out = mem[0:m_encoder.nodes]
		matrix_decoded = m_encoder.decode_ndarray(matrix_out)
		#rospy.loginfo("%s, %f", index,matrix_decoded)
		index = list(index)
		matrix[index[0],index[1]] = matrix_decoded

	# expand matrix to full matrix dimensions
	flat = matrix.flatten()
	active_motors = [1,3,4,5,10,12]
	active_sensors = np.array(active_motors*2)
	active_sensors[len(active_motors):] += 14
	matrix = np.zeros((14,28))
	k = 0
	for i in active_motors:
		for j in active_sensors:
			matrix[i,j] = flat[k]
			k += 1

	msg = roboy_dep.msg.depMatrix()
	for i in range(matrix.shape[0]):
		cArray = roboy_dep.msg.cArray()
		cArray.cArray = matrix[i,:]
		cArray.size = matrix.shape[1]
		msg.depMatrix.append(cArray)
	msg.size = matrix.shape[0]
	
	#print matrix
	#print msg, "\n"
	global trigger
	global dep_matrix_pub
	if trigger == 1:
		dep_matrix_pub.publish(msg)
		trigger = 0
'''

# HTM SDR Scalar Encoder
# Input: Scalar
# Parameters: n - number of units, w - bits used to represent signal (width), b - buckets (i.e. resolution), 
#             min - minimum value of input (inclusive), max - maximum input value (inclusive)
class scalar_sdr:
	
	def __init__(self, b, w, min_, max_, shape=(1,1), neg=True):
		if type(b) != int or type(w) != int or type(min_) != float or type(max_) != float:
			raise TypeError("b - buckets must be int, w - width must be int, min_ must be float and max_ must be float")
		self.b = b # must be int
		self.w = w # must be int
		self.min = min_ # must be float
		self.max = max_ # must be float
		self.n = b+w-1 # number of units for encoding
		self.ndarray_shape = shape
		self.nodes = self.n*reduce(lambda x, y: x*y, self.ndarray_shape)
		self.neg = neg
		
	def encode(self,input_):
		if input_ > self.max or input_ < self.min:
			raise ValueError("Input outside encoder range!")
		if type(input_) != float:
			raise TypeError("Input must be float!")
		if self.neg:
			output = np.zeros(self.n)-1
		else:
			output = np.zeros(self.n)
		index = int((input_-self.min)/(self.max-self.min)*self.b)
		output[index:index+self.w] = 1
		return output
	
	def encode_ndarray(self,input_):
		if input_.shape != self.ndarray_shape:
			raise ValueError("Input dimensions do not match specified encoder dimensions!")
		output = []
		for i in np.nditer(input_, order='K'):
			output.append(self.encode(float(i)))
		return np.array(output)

	def decode(self,input_):
		if len(input_) != self.n: 
			raise TypeError("Input length does not match encoder length!")
		if len(np.nonzero(input_+1)[0]) == 0:
			return np.nan
		max_ = 0
		output = 0.0
		for i in range(self.b):
			x = np.zeros(self.n)-1
			x[i:i+self.w] = 1
			if x.shape != input_.shape:
				input_ = input_.reshape(x.shape)
			score = np.inner(x,input_) # this was broken for some input formats 
			if score > max_:
				max_ = score
				output = float(i)/float(self.b)*(self.max-self.min)+self.min
		return output
			
	def decode_ndarray(self,input_):
		if input_.shape != (reduce(lambda x, y: x*y, self.ndarray_shape)*self.n,): 
			raise ValueError("Input dimensions do not match specified encoder dimensions!")
		input_ = input_.reshape(self.ndarray_shape+(self.n,))
		output = []
		for i in np.ndindex(self.ndarray_shape):
			output.append(self.decode(input_[i]))
		output = np.array(output).reshape(self.ndarray_shape)
		return output
	
	def set_ndarray_shape(self,shape):
		if type(shape) != tuple:
			raise TypeError("Must provide tuple of array dimensions!")
		self.ndarray_shape = shape

'''
def trigger_callback(msg):

	try:
		sess
	except NameError:
		global sess
		sess = tf.Session()
		init = tf.global_variables_initializer()
		sess.run(init)

	#print msg
	active_motors = [1,3,4,5,10,12]
	pos = np.array(msg.position)[active_motors].reshape(6,1)
	#print "Pos: ", pos
	#print 

	pos_encoder = scalar_sdr(11,1,-100000.0,100000.0,shape=(6,1),neg=False)
	pos_ = pos_encoder.encode_ndarray(np.array(pos).astype('float32')).reshape(1,6,pos_encoder.n)
	#print "Pos_: ", pos_[0]
	#print 

	with tf.name_scope("input"):
		pos_nodes = tf.placeholder(tf.float32, [None,6,pos_encoder.n], name="motor_pos")
		trigger_bias = tf.placeholder(tf.float32, [None,6], name = "trigger_bias")

	with tf.name_scope("weights"):
		w_pos = tf.placeholder(tf.float32, [None,6,pos_encoder.n], name="pos_weights")
		w_trig = tf.placeholder(tf.float32, [None,6], name = "trigger_weights")

	with tf.name_scope("trigger"):
		pos_input = tf.multiply(pos_nodes,w_pos)
		motor_activation = tf.nn.relu(tf.sign(pos_input))
		trigger_sum_0 = tf.reduce_sum(motor_activation, axis = 2)
		bias = tf.multiply(trigger_bias,w_trig)
		trigger_sum_1 = tf.reduce_sum(trigger_sum_0 + bias,axis=1)
		trigger_activation = tf.nn.relu(tf.sign(trigger_sum_1))


	#global neurons_pos
	weights_pos = neurons_pos["weights_pos"]
	trig_b = neurons_pos["trig_b"].reshape(1,6)
	trig_w = neurons_pos["trig_w"].reshape(1,6)

	#print weights_pos
	#print pos_.shape, vel_.shape, weights_pos.shape, weights_vel.shape, trig_b.shape, trig_w.shape

	trigger.append(sess.run(trigger_activation,{pos_nodes: pos_, w_pos: weights_pos, trigger_bias: trig_b, w_trig: trig_w}))
	if len(trigger)>1:
		trigger.pop(0)
	if np.array(trigger).any():
		#global dep_matrix_pub
		#global dep_matrix_msg
		print 1
		
		try:
			print "Published"
			dep_matrix_pub.publish(dep_matrix_msg)
		except NameError:
			pass		
	else:
		print 0
	#print "Trigger: ", np.array(trigger).any()
'''

def trigger_callback_ind(msg):
	#t0 = time.time()
	pos_encoder = scalar_sdr(41,3,-100000.0,100000.0,shape=(6,1),neg=False)
	vel_encoder = scalar_sdr(5,1,-70.0,70.0,shape=(6,1),neg=False)

	try: 
		sess
	except NameError:
		global sess
		sess = tf.Session()
		init = tf.global_variables_initializer()
		sess.run(init)
		
		global pos_nodes
		global vel_nodes
		global trigger_bias
		global w_pos
		global w_vel
		global w_trig

		with tf.name_scope("input"):
			pos_nodes = tf.placeholder(tf.float32, [None,6,pos_encoder.n], name="motor_pos")
			vel_nodes = tf.placeholder(tf.float32, [None,6,pos_encoder.n,vel_encoder.n], name="motor_vel")
			trigger_bias = tf.placeholder(tf.float32, [None,6], name = "trigger_bias")

		with tf.name_scope("weights"):
			w_pos = tf.placeholder(tf.float32, [None,6,pos_encoder.n], name="pos_weights")
			w_vel = tf.placeholder(tf.float32, [None,6,pos_encoder.n,vel_encoder.n], name="vel_weights")
			w_trig = tf.placeholder(tf.float32, [None,6], name = "trigger_weights")

		with tf.name_scope("trigger"):
			vel_input = tf.multiply(w_vel,vel_nodes)
			vel_inhibitor = tf.reduce_sum(tf.nn.relu(tf.sign(vel_input)),axis=3)
			pos_input = tf.multiply(pos_nodes,w_pos)
			motor_sum = pos_input + vel_inhibitor*-1
			motor_activation = tf.nn.relu(tf.sign(motor_sum))
			trigger_sum_0 = tf.reduce_sum(motor_activation, axis = 2)
			bias = tf.multiply(trigger_bias,w_trig)
			trigger_sum_1 = tf.reduce_sum(trigger_sum_0 + bias,axis=1)
			global trigger_activation
			trigger_activation = tf.nn.relu(tf.sign(trigger_sum_1))

	brain_id_to_behv_id = {0.125: "zero", 0.375: "fb", 0.625: "fs", 0.875: "sd"}

	global old_brain_id
	id_ = brain_id_to_behv_id[old_brain_id]
	#print id_
	#print msg
	active_motors = [1,3,4,5,10,12]
	pos = np.array(msg.position)[active_motors].reshape(6,1)
	vel = np.array(msg.velocity)[active_motors].reshape(6,1)
	#print "Pos: ", pos
	#print "Vel: ", vel
	#print 
	pos_ = pos_encoder.encode_ndarray(np.array(pos).astype('float32')).reshape(1,6,pos_encoder.n)
	vel_ = vel_encoder.encode_ndarray(np.array(vel).astype('float32')).reshape(1,6,vel_encoder.n)
	#print "Pos_: ", pos_[0]
	#print "Vel_: ", vel_[0]
	#print 
	a = np.zeros((1,6,pos_encoder.n,vel_encoder.n))
	for k in range(1):
		for i in range(6):
			for j in range(pos_encoder.n):
				a[k][i][j] = vel_[k][i].reshape(vel_encoder.n)
	vel_ = a

	global neurons_ind
	weights_pos = neurons_ind[id_]["weights_pos"]
	weights_vel = neurons_ind[id_]["weights_vel"]
	trig_b = neurons_ind[id_]["trig_b"].reshape(1,6)
	trig_w = neurons_ind[id_]["trig_w"].reshape(1,6)

	#print weights_pos
	#print pos_.shape, vel_.shape, weights_pos.shape, weights_vel.shape, trig_b.shape, trig_w.shape
	#global trigger
	trigger.append(sess.run(trigger_activation,{pos_nodes: pos_.reshape(1,6,pos_encoder.n), vel_nodes: vel_.reshape(1,6,pos_encoder.n,vel_encoder.n), w_pos: weights_pos.reshape(1,6,pos_encoder.n), w_vel: weights_vel.reshape(1,6,pos_encoder.n,vel_encoder.n), trigger_bias: trig_b, w_trig: trig_w}))

	global dep_matrix_msg
	global dep_matrix_pub
	global new_brain_id
	#print old_brain_id, new_brain_id
	
	if len(trigger)>1:
		trigger.pop(0)
	if np.array(trigger).any():
		try:
			dep_matrix_pub.publish(dep_matrix_msg)
			old_brain_id = new_brain_id
			del dep_matrix_msg
			#print 1
		except NameError:
			pass
			#print "NameError: dep_matrix_msg not defined!"
	else:
		pass
		#print 0
	
	#print time.time()-t0
	#print "Trigger: ", np.array(trigger).any()

class LAM:
	def __init__(self,shape,weights=None):
		self.shape = shape
		try:
			if weights == None:
				self.weights = np.zeros(shape)
		except ValueError:
			self.weights = weights
	
	def store(self,input,output):
		dW = np.outer(input,output)
		self.weights += dW
		
	def recall(self,input,threshold=0,print_=False):
		output = input.dot(self.weights)-threshold
		if print_ == True:
			print output
		output[output > 0] = 1
		output[output < 0] = 0
		return output

	def save_weights(self,filename):
		np.save(filename, self.weights)
		
	def load_weights(self,filename):
		weights = np.load(filename)
		if weights.shape == self.shape:
			self.weights = weights
		else:
			raise ValueError("Dimensions of specified weight array does not match network weight dimensions!")

def callback_lam(msg):
	#print "Brain id msg: %f \n" %msg.brain_id
	#global lam

	m_encoder = scalar_sdr(100,21,-0.25,0.25,(6,12),neg=False)
	b_encoder = scalar_sdr(100,21,0.0,1.0,neg=False)

	global new_brain_id
	new_brain_id = msg.brain_id
	brain_sig = b_encoder.encode(new_brain_id)
	mem = lam.recall(np.array(brain_sig))
	matrix = m_encoder.decode_ndarray(np.array(mem).reshape((reduce(lambda x, y: x*y, m_encoder.ndarray_shape)*m_encoder.n,)))
	flat = matrix.flatten()
	active_motors = [1,3,4,5,10,12]
	active_sensors = np.array(active_motors*2)
	active_sensors[len(active_motors):] += 14
	matrix = np.zeros((14,28))
	k = 0
	for i in active_motors:
		for j in active_sensors:
			matrix[i,j] = flat[k]
			k += 1

	msg = roboy_dep.msg.depMatrix()
	for i in range(matrix.shape[0]):
		cArray = roboy_dep.msg.cArray()
		cArray.cArray = matrix[i,:]
		cArray.size = matrix.shape[1]
		msg.depMatrix.append(cArray)
	msg.size = matrix.shape[0]

	global dep_matrix_msg
	dep_matrix_msg = roboy_dep.msg.depMatrix()
	dep_matrix_msg = msg

def main():
	rospy.init_node('dep_memory', anonymous=True)
	
	'''
	rospy.Subscriber("roboy_dep/brain_id", roboy_dep.msg.brain_id, callback_hnn)
	global hnns
	pickle_in = open("/home/roboy/dep/dep_memory/hnns_80.pickle")
	hnns = pickle.load(pickle_in)
	for index in hnns:
		hnns[index].setIter(600)
	'''

	rospy.Subscriber("roboy_dep/brain_id", roboy_dep.msg.brain_id, callback_lam, queue_size=1)
	global lam
	pickle_in = open("/home/roboy/dep/dep_memory/lam.pickle")
	lam = pickle.load(pickle_in)
	
	'''
	global neurons_pos
	pickle_in = open("/home/roboy/dep/dep_memory/neurons_pos.pickle")
	neurons_pos = pickle.load(pickle_in)
	
	rospy.Subscriber("/roboy/middleware/MotorStatus", roboy_communication_middleware.msg.MotorStatus, trigger_callback, queue_size=1)
	'''
	global dep_matrix_pub
	dep_matrix_pub = rospy.Publisher('/roboy_dep/depLoadMatrix', roboy_dep.msg.depMatrix,queue_size=1)

	#global dep_matrix_msg

	global neurons_ind
	pickle_in = open("/home/roboy/dep/dep_memory/neurons_individual_behvs.pickle")
	neurons_ind = pickle.load(pickle_in)
	rospy.Subscriber("/roboy/middleware/MotorStatus", roboy_communication_middleware.msg.MotorStatus, trigger_callback_ind, queue_size=1)

	global trigger
	trigger = []

	global old_brain_id
	old_brain_id = 0.125

	global new_brain_id
	new_brain_id = 0.125

	rospy.spin()

if __name__ == '__main__':
	main()
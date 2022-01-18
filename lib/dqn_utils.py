import numpy as np
import os
import random
import tensorflow as tf
import itertools

class History:
    def __init__(self):
        self.history = np.zeros(shape=(4, 84, 84), dtype=np.uint8)

    def add(self, screen):
        """
        screen: (84, 84)
        """
        self.history[:-1] = self.history[1:]
        self.history[-1] = screen

    def get(self):
        """
        return: (4, 84, 84)
        """
        return self.history

    def reset(self):
        self.history *= 0

class DdqnValue():
	"""
		Processes a raw Atari images. Resizes it and converts it to grayscale.
	"""
	def __init__(self): 
		# Bulid the Tensorflow Graph
		with tf.variable_scope( "doubledqnerror" ):
			self.q_values = tf.placeholder( shape=[32, 4], dtype=tf.float32 )
			self.t_values = tf.placeholder( shape=[32, 4], dtype=tf.float32 )
			gather_indices = tf.range(32) * 4 + tf.argmax(self.q_values, axis=1, output_type=tf.int32)
			self.output = tf.gather(tf.reshape(self.t_values, [-1]), gather_indices)

	def process( self, sess, q_values, t_values ):
		"""
		Args:
		    sess: A Tensorflow session object
		    state: A [210, 160, 3] Atari RGB State

		Returns:
		    A processed [84, 84, 1] state representing grayscale values.
		"""
		return sess.run( self.output, { self.q_values : q_values, self.t_values : t_values } )

# Atari Actions: 0 (noop), 1 (fire), 2 (left) and 3 (right) are valid actions

class StateProcessor():
	"""
		Processes a raw Atari images. Resizes it and converts it to grayscale.
	"""
	def __init__(self): 
		# Bulid the Tensorflow Graph
		with tf.variable_scope( "state_processor" ):
			self.input_state = tf.placeholder( shape=[210, 160, 3], dtype=tf.uint8 )
			self.output = tf.image.rgb_to_grayscale( self.input_state )
			self.output = tf.image.crop_to_bounding_box( self.output, 34, 0, 160, 160 )
			self.output = tf.image.resize_images(
			 		self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR )
			self.output = tf.squeeze( self.output )

	def process( self, sess, state ):
		"""
		Args:
		    sess: A Tensorflow session object
		    state: A [210, 160, 3] Atari RGB State

		Returns:
		    A processed [84, 84, 1] state representing grayscale values.
		"""
		return sess.run( self.output, { self.input_state : state } )

def state_process( sess, state_processor, state ):
	"""
	state process, let [210, 160, 3] to [84, 84, 4]
	Args:
		sess : Tensorflow session
		state_processor : a class
		state: a rgb state
	Returns:
		 A processed [84, 84, 4] state representing grayscale values.
	"""
	state = state_processor.process( sess, state )
	# state = np.stack( [state] * 4, axis=2 )
	return state

def copy_model_parameters(sess, estimator1, estimator2):
	"""
	Copies the model parameters of one estimator to another.

	Args:
	  sess: Tensorflow session instance
	  estimator1: Estimator to copy the paramters from
	  estimator2: Estimator to copy the parameters to
	"""
	e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
	e1_params = sorted(e1_params, key=lambda v: v.name)
	e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
	e2_params = sorted(e2_params, key=lambda v: v.name)

	update_ops = []
	for e1_v, e2_v in zip(e1_params, e2_params):
	    op = e2_v.assign(e1_v)
	    update_ops.append(op)

	sess.run(update_ops)

def make_epsilon_greedy_policy(estimator, nA):
	"""
	Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

	Args:
	    estimator: An estimator that returns q values for a given state
	    nA: Number of actions in the environment.

	Returns:
	    A function that takes the (sess, observation, epsilon) as an argument and returns
	    the probabilities for each action in the form of a numpy array of length nA.

	"""
	def policy_fn(sess, observation, epsilon):
	    A = np.ones(nA, dtype=float) * epsilon / nA
	    q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
	    best_action = np.argmax(q_values)
	    A[best_action] += (1.0 - epsilon)
	    return A
	return policy_fn

def populate_replay_buffer( sess, env, state_processor, replay_memory_init_size, VALID_ACTIONS, Transition, policy ):
	"""
	populate replay buffer at first

	Args:
		sess:
		env:
		state_processor:
		eplay_memory_init_size:
		VALID_ACTIONS:
		Transition:
		policy:

	Return:
		replay_memory : has 'eplay_memory_init_size' steps experience.
	"""
	replay_memory = []
	history = History()
	state = env.reset()
	state_proc = state_process( sess, state_processor, state )
	for _ in range(4):
		history.add(state_proc)

	for i in range( replay_memory_init_size ):
		action = np.random.choice( len(VALID_ACTIONS), p=policy( sess, history.get(), 1 ) )
		next_state, reward, done, _ = env.step( action )
		next_state_proc = state_process( sess, state_processor, next_state )
		history.add(next_state_proc)
		experience = Transition( action, next_state_proc, reward, done )
		replay_memory.append( experience )

		if done:
			state = env.reset()
			state_proc = state_process( sess, state_processor, state )
			for _ in range(4):
				history.add(state_proc)

	return replay_memory
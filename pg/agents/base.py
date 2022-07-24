import numpy as np
import tensorflow as tf
import os

def flat(var_list):
    return tf.concat([tf.reshape(v, (-1,)) for v in var_list], axis=0)

def setfromflat(var_list, theta):
    shapes = [v.get_shape().as_list() for v in var_list]
    assigns = []
    start = 0
    for (v, shape) in zip(var_list, shapes):
        size = int(np.prod(shape))
        new = theta[start:start+size]
        assigns.append(tf.assign(v, tf.reshape(new, shape)))
        start += size
    return assigns


class BaseAgent():
    def __init__(self):
        self.saver = self._build_saver()
        self.actor_assign = self.setactorfromflat()
        self.critic_assign = self.setcriticfromflat()
        self.pi_flatted = flat(self._get_var_list('pi'))
        self.vf_flatted = flat(self._get_var_list('vf'))

    def _get_var_list(self, name=None):
        scope = tf.compat.v1.get_default_graph().get_name_scope()
        vars = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
            scope=os.path.join(scope, name)+'/')
        return vars

    def _build_saver(self):
        pi_params = self._get_var_list('pi')
        vf_params = self._get_var_list('vf')
        return tf.compat.v1.train.Saver(var_list=pi_params+vf_params,
                                        max_to_keep=4)

    def save_weight(self, checkpoint_dir, epoch):
        if not os.path.exists(checkpoint_dir):
            raise
        self.saver.save(
            self.sess,
            os.path.join(checkpoint_dir, 'tf_ckpt'),
            global_step=epoch)

    def load_weight(self, checkpoint_dir, epoch=None):
        if not os.path.exists(checkpoint_dir):
            raise
        self.saver.restore(
            self.sess,
            os.path.join(checkpoint_dir, f'tf_ckpt-{epoch}'))

    def setactorfromflat(self):
        var_list = self._get_var_list('pi')
        x = flat(var_list)
        self.actor_param_ph = tf.compat.v1.placeholder(
            shape=x.shape, dtype=tf.float32, name="actor_param_ph")
        assigns = setfromflat(var_list, self.actor_param_ph)
        return assigns

    def setcriticfromflat(self):
        var_list = self._get_var_list('vf')
        x = flat(var_list)
        self.critic_param_ph = tf.compat.v1.placeholder(
            shape=x.shape, dtype=tf.float32, name="critic_param_ph")
        assigns = setfromflat(var_list, self.critic_param_ph)
        return assigns

    def assign_actor_weights(self, param):
        self.sess.run(self.actor_assign, feed_dict={self.actor_param_ph: param})

    def assign_critic_weights(self, param):
        self.sess.run(self.critic_assign, feed_dict={self.critic_param_ph: param})

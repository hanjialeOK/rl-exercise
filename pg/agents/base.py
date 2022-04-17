import tensorflow as tf
import os


class BaseAgent():
    def __init__(self):
        pass

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

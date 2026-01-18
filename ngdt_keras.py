# ngdt_keras.py
# Requires: tensorflow
import tensorflow as tf
import numpy as np

class NGDT(tf.keras.optimizers.Optimizer):
    """
    NGD-T optimizer for tf.keras.
    Usage:
        opt = NGDTKeras(learning_rate=1.0, Q_budget=1e-3, fisher_method='diag', name='NGDT')
    Key args similar to PyTorch version.
    """
    def __init__(self, learning_rate=1.0, Q_budget=1e-3, fisher_method='diag',
                 damping=1e-3, eta_min=1e-6, eta_max=1.0, beta_f=0.95,
                 beta_mom=0.9, eta_null_ratio=1e-3, eps=1e-8, name='NGDT', **kwargs):
        super().__init__(name=name, **kwargs)
        self._set_hyper('learning_rate', learning_rate)
        self.Q_budget = Q_budget
        self.fisher_method = fisher_method
        self.damping = damping
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.beta_f = beta_f
        self.beta_mom = beta_mom
        self.eta_null_ratio = eta_null_ratio
        self.eps = eps

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'fisher_ema', initializer='zeros')
            self.add_slot(var, 'momentum_nat', initializer='zeros')

    @tf.function
    def _resource_apply_dense(self, grad, var):
        # update fisher EMA
        fisher = self.get_slot(var, 'fisher_ema')
        if self.fisher_method == 'diag':
            fisher_update = tf.square(grad)
        else:
            # block diag: scalar mean expanded
            mean_sq = tf.reduce_mean(tf.square(grad))
            fisher_update = tf.fill(tf.shape(var), mean_sq)
        fisher.assign(self.beta_f * fisher + (1.0 - self.beta_f) * fisher_update)

        # compute natural gradient
        invF = 1.0 / (fisher + self.damping + 1e-30)
        g_nat = invF * grad

        # compute Delta_F contribution for this var (we'll sum externally)
        # but Keras optimizer API expects per-var apply; we return update
        m = self.get_slot(var, 'momentum_nat')
        m.assign(self.beta_mom * m + (1.0 - self.beta_mom) * g_nat)

        # compute eta_T from global Delta_F: we approximate by using current g^T F^{-1} g
        # Note: to compute global Delta_F we would need to aggregate across vars.
        # Here we compute a local eta_T proxy; for exact regulator compute Delta_F externally and set as hyper.
        local_Delta = tf.reduce_sum(grad * g_nat)
        # safe scalar conversion
        local_Delta = tf.maximum(local_Delta, 1e-30)
        eta0 = self._get_hyper('learning_rate')
        eta_T = eta0 * (self.Q_budget / (local_Delta + self.eps))
        eta_T = tf.clip_by_value(eta_T, self.eta_min, self.eta_max)

        eta_null = self.eta_null_ratio * eta_T
        var.assign_sub(eta_T * m + eta_null * grad)

    def _resource_apply_sparse(self, grad, var, indices):
        # sparse updates: fallback to dense behavior
        dense_grad = tf.convert_to_tensor(tf.IndexedSlices(grad, indices, tf.shape(var)))
        return self._resource_apply_dense(dense_grad, var)

    def get_config(self):
        config = super().get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'Q_budget': self.Q_budget,
            'fisher_method': self.fisher_method,
            'damping': self.damping,
            'eta_min': self.eta_min,
            'eta_max': self.eta_max,
            'beta_f': self.beta_f,
            'beta_mom': self.beta_mom,
            'eta_null_ratio': self.eta_null_ratio,
            'eps': self.eps,
        })
        return config


def example():
    """
    Example to use NGDT for Tensorflow Keras
    """
    opt = NGDT(learning_rate=1.0, Q_budget=1e-3, fisher_method='diag', damping=1e-3)
    # For simple use (local proxy):
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(dataset, epochs=10)
    # For exact Delta_F regulator: implement custom train_step that computes global Delta_F then applies updates.
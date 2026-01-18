# ngdt_keras_kfac_integration.py
import tensorflow as tf

# --- Minimal K-FAC adapter for Keras ---
class KFACAdapterTF:
    def __init__(self, model, ema_decay=0.95, damping=1e-3, eig_freq=20):
        self.model = model
        self.ema_decay = ema_decay
        self.damping = damping
        self.eig_freq = eig_freq
        self.step = tf.Variable(0, dtype=tf.int32)
        # store per-layer factors keyed by layer object
        self.factors = {}
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Dense) or isinstance(layer, tf.keras.layers.Conv2D):
                self.factors[layer] = {'A': None, 'G': None, 'A_eig': None, 'G_eig': None}

    def capture_activations_and_grads(self, layer, inputs, outputs, grad_output):
        # user should call this helper inside custom training loop after forward/backward
        layer._kfac_input = inputs  # tensor
        layer._kfac_grad_output = grad_output  # tensor

    def step_batch(self):
        self.step.assign_add(1)
        for layer, info in self.factors.items():
            x = getattr(layer, '_kfac_input', None)
            grad_out = getattr(layer, '_kfac_grad_output', None)
            if x is None or grad_out is None:
                continue
            if isinstance(layer, tf.keras.layers.Dense):
                # x: (N, in), grad_out: (N, out)
                A = tf.matmul(x, x, transpose_a=True) / tf.cast(tf.shape(x)[0], tf.float32)
                G = tf.matmul(grad_out, grad_out, transpose_a=True) / tf.cast(tf.shape(grad_out)[0], tf.float32)
            else:
                # Conv2D: extract patches
                patches = tf.image.extract_patches(images=x,
                                                   sizes=[1, layer.kernel_size[0], layer.kernel_size[1], 1],
                                                   strides=[1, layer.strides[0], layer.strides[1], 1],
                                                   rates=[1,1,1,1],
                                                   padding=layer.padding.upper())
                N = tf.shape(patches)[0]
                K = patches.shape[-1]
                patches2 = tf.reshape(patches, [N, -1, K])  # (N, L, K)
                patches2 = tf.reshape(patches2, [-1, K])    # (N*L, K)
                A = tf.matmul(patches2, patches2, transpose_a=True) / tf.cast(tf.shape(patches2)[0], tf.float32)
                go = tf.reshape(grad_out, [-1, tf.shape(grad_out)[-1]])
                G = tf.matmul(go, go, transpose_a=True) / tf.cast(tf.shape(go)[0], tf.float32)
            if info['A'] is None:
                info['A'] = A
                info['G'] = G
            else:
                info['A'] = self.ema_decay * info['A'] + (1.0 - self.ema_decay) * A
                info['G'] = self.ema_decay * info['G'] + (1.0 - self.ema_decay) * G
            if tf.equal(self.step % self.eig_freq, 0):
                a_vals, a_vecs = tf.linalg.eigh(info['A'])
                g_vals, g_vecs = tf.linalg.eigh(info['G'])
                a_vals = tf.maximum(a_vals, 1e-6)
                g_vals = tf.maximum(g_vals, 1e-6)
                info['A_eig'] = (a_vals, a_vecs)
                info['G_eig'] = (g_vals, g_vecs)

    def apply_preconditioner(self, grads_and_vars):
        """
        grads_and_vars: list of (grad, var) pairs
        returns list of g_nat aligned with grads_and_vars
        """
        g_nat_list = []
        for grad, var in grads_and_vars:
            if grad is None:
                g_nat_list.append(None)
                continue
            # find owning layer by matching var shape to layer weights
            owner = None
            for layer, info in self.factors.items():
                w = layer.kernel if hasattr(layer, 'kernel') else None
                if w is not None and w.shape == var.shape:
                    owner = layer
                    break
            if owner is None:
                g_nat_list.append(grad / (self.damping + 1e-6))
                continue
            info = self.factors[owner]
            if info.get('A_eig') is None or info.get('G_eig') is None:
                g_nat_list.append(grad / (self.damping + 1e-6))
                continue
            a_vals, a_vecs = info['A_eig']
            g_vals, g_vecs = info['G_eig']
            # compute inverse action: reshape grad to matrix and apply G^{-1} @ grad @ A^{-1}
            if isinstance(owner, tf.keras.layers.Dense):
                out, inn = var.shape
                grad_mat = tf.reshape(grad, [out, inn])
                G_inv = g_vecs @ tf.linalg.diag(1.0 / (g_vals + self.damping)) @ tf.transpose(g_vecs)
                A_inv = a_vecs @ tf.linalg.diag(1.0 / (a_vals + self.damping)) @ tf.transpose(a_vecs)
                nat = G_inv @ grad_mat @ A_inv
                g_nat_list.append(tf.reshape(nat, tf.shape(var)))
            else:
                out, inn, kh, kw = var.shape
                grad_mat = tf.reshape(grad, [out, inn * kh * kw])
                G_inv = g_vecs @ tf.linalg.diag(1.0 / (g_vals + self.damping)) @ tf.transpose(g_vecs)
                A_inv = a_vecs @ tf.linalg.diag(1.0 / (a_vals + self.damping)) @ tf.transpose(a_vecs)
                nat = G_inv @ grad_mat @ A_inv
                g_nat_list.append(tf.reshape(nat, tf.shape(var)))
        return g_nat_list

# --- Example train_step using KFACAdapterTF and NGDTKerasExact optimizer ---
@tf.function
def train_step_kfac(model, optimizer, adapter, x, y, loss_fn):
    """
    model: tf.keras.Model
    optimizer: NGDTKerasExact instance (stores fisher_ema and momentum slots)
    adapter: KFACAdapterTF instance
    """
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = loss_fn(y, logits)
    grads = tape.gradient(loss, model.trainable_variables)

    # capture activations and output grads for each layer
    # user must set these on layers during forward/backward; here we assume adapter hooks or manual capture
    # For this example, we assume adapter.step_batch will read layer._kfac_input and layer._kfac_grad_output
    # Update adapter factors
    adapter.step_batch()

    # compute preconditioned natural gradients using adapter
    grads_and_vars = list(zip(grads, model.trainable_variables))
    g_nat_list = adapter.apply_preconditioner(grads_and_vars)

    # compute global Delta_F = sum g^T g_nat
    contribs = []
    for (g, _), g_nat in zip(grads_and_vars, g_nat_list):
        if g is None or g_nat is None:
            contribs.append(tf.constant(0.0, dtype=tf.float32))
        else:
            contribs.append(tf.reduce_sum(g * g_nat))
    Delta_F = tf.add_n(contribs)

    # compute eta_T
    eta0 = optimizer._get_hyper('learning_rate')
    Q_budget = optimizer.Q_budget
    eps = optimizer.eps
    eta_T = eta0 * (Q_budget / (Delta_F + eps))
    eta_T = tf.clip_by_value(eta_T, 1e-6, 1.0)

    # apply updates: compute natural-space momentum and update variables
    for var, g, g_nat in zip(model.trainable_variables, grads, g_nat_list):
        if g is None or g_nat is None:
            continue
        fisher = optimizer.get_slot(var, 'fisher_ema')
        # update fisher EMA in optimizer slots for bookkeeping (optional)
        fisher.assign(optimizer.beta_f * fisher + (1.0 - optimizer.beta_f) * tf.square(g))
        invF = 1.0 / (fisher + optimizer.damping + 1e-30)
        nat = invF * g  # fallback if adapter not used for this var
        m = optimizer.get_slot(var, 'momentum_nat')
        # prefer adapter g_nat if available
        g_nat_used = g_nat if g_nat is not None else nat
        m.assign(optimizer.beta_mom * m + (1.0 - optimizer.beta_mom) * g_nat_used)
        eta_null = optimizer.eta_null_ratio * eta_T
        var.assign_sub(eta_T * m + eta_null * g)

    return loss, {'Delta_F': Delta_F, 'eta_T': eta_T}

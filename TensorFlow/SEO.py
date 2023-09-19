"""SEO optimizer implementation."""
import tensorflow as tf


class SEO(tf.keras.optimizers.Optimizer):
    r"""A Simple and Efficient Optimizer for Deep Learning.
        author:gaozhihan
        email:gaozhihan@vip.qq.com"""

    def __init__(
            self,
            learning_rate: float = 0.001,
            beta_1: float = 0.9,
            beta_2: float = 0.999,
            belief: float = 0.5,
            weight_decay: float = 1e-4,
            epsilon: float = 1e-15,
            clipnorm=None,
            clipvalue=None,
            global_clipnorm=None,
            use_ema: bool = False,
            ema_momentum: float = 0.5,
            ema_overwrite_frequency=1,
            jit_compile: bool = True,
            name="SEO",
            **kwargs
    ):
        super().__init__(
            name=name,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            jit_compile=jit_compile,
            **kwargs
        )
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.belief = belief
        self.weight_decay = weight_decay

    def build(self, var_list):
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self._built = True
        self._m = []
        self._e = []
        self._u = []
        self._v = []
        for var in var_list:
            self._e.append(self.add_variable_from_reference(model_variable=var, variable_name="e", shape=()))
            self._m.append(self.add_variable_from_reference(model_variable=var, variable_name="m"))
            self._u.append(self.add_variable_from_reference(model_variable=var, variable_name="u"))
            self._v.append(self.add_variable_from_reference(model_variable=var, variable_name="v"))

    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable."""
        var_dtype = variable.dtype
        weight_decay = tf.cast(self.weight_decay, var_dtype)
        epsilon = tf.cast(self.epsilon, var_dtype)
        belief = tf.cast(self.belief, var_dtype)
        step = tf.cast(self.iterations + 1, var_dtype)
        beta_1 = tf.cast(self.beta_1, var_dtype)
        beta_2 = tf.cast(self.beta_2, var_dtype)
        index = self._index_dict[self._var_key(variable)]
        u = self._u[index]
        m = self._m[index]
        e = self._e[index]
        v = self._v[index]
        lr = tf.cast(self.learning_rate, var_dtype)
        init_iter = self.iterations == 0
        bias_correction1 = 1. - tf.pow(beta_1, step)
        bias_correction2 = 1. - tf.pow(beta_2, step)
        alpha = tf.sqrt(bias_correction2) / bias_correction1
        one_minus_beta_1 = belief * (1. - beta_1)
        one_minus_beta_2 = 1. - beta_2
        alpha_beta_1 = one_minus_beta_1 * alpha
        decay_beta_1 = 1. + alpha_beta_1
        decay = 1. + lr * weight_decay
        if isinstance(gradient, tf.IndexedSlices):
            # Sparse gradients.
            indices = gradient.indices
            grad = gradient.values
            e.assign(tf.divide(e + tf.linalg.norm(grad) * alpha_beta_1, decay_beta_1), read_value=False)
            grad = grad * e
            delta = grad - m
            m.scatter_add(tf.IndexedSlices(delta * one_minus_beta_1, indices))
            v.scatter_add(tf.IndexedSlices((tf.math.abs(delta * (grad - m)) - v) * one_minus_beta_2, indices))
        else:
            # Dense gradients.
            grad = gradient
            e.assign(tf.divide(e + tf.linalg.norm(grad) * alpha_beta_1, decay_beta_1), read_value=False)
            grad = grad * e
            delta = grad - m
            m.assign_add(delta * one_minus_beta_1, read_value=False)
            v.assign_add((tf.math.abs(delta * (grad - m)) - v) * one_minus_beta_2, read_value=False)
        origin_var = tf.cond(init_iter, lambda: variable, lambda: (variable * decay) + (lr * u))
        u.assign(tf.divide(u + alpha * m * tf.math.rsqrt(v + epsilon), decay), read_value=False)
        variable.assign(tf.divide(origin_var - lr * u, decay), read_value=False)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(self._learning_rate),
                "weight_decay": self.weight_decay,
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "belief": self.belief,
            }
        )
        return config

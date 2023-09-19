"""SEO optimizer implementation."""
import math
import torch
from torch.optim.optimizer import Optimizer


class SEO(Optimizer):
    r"""A Simple and Efficient Optimizer for Deep Learning.
        author: gaozhihan
        email: gaozhihan@vip.qq.com"""

    def __init__(
            self,
            params,
            lr: float = 1e-3,
            betas=(0.9, 0.999, 0.5),
            eps: float = 1e-15,
            weight_decay: float = 1e-4,
    ) -> None:
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if eps < 0.0:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 0: {}'.format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 1: {}'.format(betas[1])
            )
        if weight_decay < 0:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )
        defaults = dict(
            lr=lr,
            betas=betas,
            epsilon=eps,
            weight_decay=weight_decay,
        )
        super(SEO, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SEO, self).__setstate__(state)

    def step(self, closure=None):
        r"""Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'SEO does not support sparse gradients, '
                        'please consider SparseAdam instead'
                    )
                state = self.state[p]
                beta_1, beta_2, belief = group['betas']
                weight_decay = group['weight_decay']
                epsilon = group['epsilon']
                lr = group['lr']

                # State initialization
                init_iter = len(state) == 0
                if init_iter:
                    state['step'] = torch.tensor(0.0, dtype=torch.float, device=p.device)
                    state['m'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    state['e'] = torch.tensor(0.0, dtype=p.dtype, device=p.device)
                    state['u'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    state['v'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                # get current state variable
                m = state['m']
                v = state['v']
                e = state['e']
                u = state['u']
                state['step'] += 1.
                step = state['step']
                bias_correction1 = 1. - beta_1 ** step
                bias_correction2 = 1. - beta_2 ** step
                alpha = math.sqrt(bias_correction2) / bias_correction1
                one_minus_beta_1 = belief * (1. - beta_1)
                one_minus_beta_2 = 1. - beta_2
                alpha_beta_1 = one_minus_beta_1 * alpha
                decay_beta_1 = 1. + alpha_beta_1
                decay = 1. + weight_decay * lr
                e.add_(torch.divide(e + torch.norm(p.grad, p=2) * alpha_beta_1, decay_beta_1) - e)
                grad = p.grad.data * e
                delta = grad - m.data
                m.data.add_(delta * one_minus_beta_1)
                v.data.add_((torch.abs(delta * (grad - m.data)) - v.data) * one_minus_beta_2)
                origin_var = p.data if init_iter else (p.data * decay + lr * u.data)
                u.data.add_(torch.divide(u.data + alpha * m.data * torch.rsqrt(v.data + epsilon), decay) - u.data)
                p.data.add_(torch.divide(origin_var - lr * u.data, decay) - p.data)
        return loss

import torch
from torch.optim import Optimizer

class PerGodGradientDescent(Optimizer):
    def __init__(self, params, learning_rate=0.001, mu=0.01, name="PGD"):
        defaults = dict(learning_rate=learning_rate, mu=mu)
        super(PerGodGradientDescent, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['learning_rate']
            mu = group['mu']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]

                if 'vstar' not in state:
                    state['vstar'] = torch.zeros_like(p.data)
                if 'gold' not in state:
                    state['gold'] = torch.zeros_like(p.data)

                vstar = state['vstar']
                gold = state['gold']

                with torch.no_grad():
                    v_diff = vstar - mu * (p.data - vstar)
                    vstar.copy_(p.data)
                    grad_diff = grad + gold + mu * v_diff
                    p.data.add_(-lr, grad_diff)

        return loss

    def set_params(self, cog, avg_gradient, client):
        all_params = list(self.param_groups[0]['params'])
        for param, value in zip(all_params, cog):
            state = self.state[param]
            vstar = state['vstar']
            vstar.data.copy_(torch.tensor(value, dtype=param.dtype))

        gprev = client.get_grads()

        gdiff = [g1 - g2 for g1, g2 in zip(avg_gradient, gprev)]

        all_params = list(self.param_groups[0]['params'])
        for param, grad in zip(all_params, gdiff):
            state = self.state[param]
            gold = state['gold']
            gold.data.copy_(torch.tensor(grad, dtype=param.dtype))

import torch
from torch.optim import Optimizer

class PerturbedGradientDescent(Optimizer):
    def __init__(self, params, lr=0.001, mu=0.01, name="PGD"):
        defaults = dict(lr=lr, mu=mu)
        super(PerturbedGradientDescent, self).__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            mu = group['mu']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]

                if 'vstar' not in state:
                    state['vstar'] = torch.zeros_like(p.data)

                vstar = state['vstar']

                with torch.no_grad():
                    v_diff = vstar - mu * (p.data - vstar)
                    vstar.copy_(p.data)
                    scaled_grad = grad + mu * v_diff
                    p.data.add_(-lr, scaled_grad)

        return loss

    def set_params(self, cog, client):
        all_params = list(self.param_groups[0]['params'])
        for param, value in zip(all_params, cog):
            state = self.state[param]
            vstar = state['vstar']
            vstar.data.copy_(torch.tensor(value, dtype=param.dtype))

# Sample Usage
# optimizer = PerturbedGradientDescent(model.parameters(), lr=0.001, mu=0.01)

# optimizer.zero_grad()
# loss = compute_loss()
# loss.backward()
# optimizer.step()

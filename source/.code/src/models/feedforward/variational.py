"""Variational neurons: every weight is a Gaussian random variable."""
import torch
import torch.nn.functional as F


class VariationalLinear(torch.nn.Module):
    """A linear layer with Gaussian weights ``w ~ N(mu, sigma^2)``.

    ``sigma = softplus(rho)`` keeps it positive. Instead of sampling the
    weights, the layer samples its outputs (the *local reparameterization
    trick*): for an input ``x`` the output ``x W^T + b`` is Gaussian with

    - mean ``x mu^T + b``;
    - variance ``x^2 (sigma^2)^T`` (squares element-wise),

    so ``output = mean + sqrt(variance) * eps`` with ``eps ~ N(0, 1)``. The
    layer samples in training *and* in evaluation; repeated passes give
    different outputs, and their spread is the uncertainty of the network.

    ``kl()`` is ``KL(N(mu, sigma^2) || N(0, prior_std^2))`` summed over all
    weights.
    """

    def __init__(self, in_features, out_features, prior_std=1.0, initial_rho=-5.0):
        super().__init__()
        self.prior_std = prior_std
        self.mu = torch.nn.Parameter(torch.randn(out_features, in_features) / in_features ** 0.5)
        self.rho = torch.nn.Parameter(torch.full((out_features, in_features), initial_rho))
        self.bias = torch.nn.Parameter(torch.zeros(out_features))

    def sigma(self):
        return F.softplus(self.rho)

    def forward(self, signal):
        output = F.linear(signal, self.mu, self.bias)
        ## Add the noise: sqrt(variance) * eps, eps from torch.randn_like
        ## YOUR CODE HERE
        # -- start --
        variance = F.linear(signal * signal, self.sigma() ** 2)
        output = output + variance.sqrt() * torch.randn_like(output)
        # -- end --
        return output

    def kl(self):
        result = self.mu.sum() * 0
        ## KL(N(mu, sigma^2) || N(0, s^2)) = log(s / sigma)
        ##     + (sigma^2 + mu^2) / (2 s^2) - 1/2, summed over the weights
        ## YOUR CODE HERE
        # -- start --
        sigma = self.sigma()
        result = (torch.log(self.prior_std / sigma)
                  + (sigma ** 2 + self.mu ** 2) / (2 * self.prior_std ** 2) - 0.5).sum()
        # -- end --
        return result


class VariationalNet(torch.nn.Module):
    """``784 -> dim_hidden -> 10`` of variational layers with ReLU.

    The images (in ``[-1, 1]``) are first moved to ``[0, 1]``: the output
    variance of a variational layer is ``x^2 sigma^2`` summed over the
    inputs, so a black background pixel then adds no noise.

    Follows the course batch protocol: reads ``batch['data']['image']``,
    writes ``batch['signals']['output']`` and ``batch['postprocessed']['class']``.
    """

    def __init__(self, dim_input=28 * 28, dim_hidden=256, dim_output=10, prior_std=1.0):
        super().__init__()
        self.hidden = VariationalLinear(dim_input, dim_hidden, prior_std=prior_std)
        self.output = VariationalLinear(dim_hidden, dim_output, prior_std=prior_std)

    def forward(self, batch):
        signal = (batch['data']['image'].reshape(len(batch['data']['image']), -1) + 1) / 2
        signal = self.output(torch.relu(self.hidden(signal)))
        batch['signals'] = {'output': signal}
        batch['postprocessed'] = {'class': signal.argmax(dim=1)}
        return batch

    def kl(self):
        return self.hidden.kl() + self.output.kl()

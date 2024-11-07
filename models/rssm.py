import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent

class RSSM(nn.Module):
    def __init__(
        self,
        deter=4096,
        stoch=32,
        unroll=False,
        initial="learned",
        unimix=0.01,
        action_clip=1.0,
        action_dim=2,
        device="cuda" if torch.cuda.is_available() else "cpu",
        **kw,
    ):
        super(RSSM, self).__init__()
        self.device = device  # 将模型放在指定设备上
        self.deter = deter
        self.stoch = stoch
        self.unroll = unroll
        self.initial = initial
        self.unimix = unimix
        self.action_clip = action_clip
        self.kw = kw

        # Initial hidden state for learned initialization
        if initial == "learned":
            self.initial_deter = nn.Parameter(torch.zeros(deter, device=self.device))

        # Fully connected layers for RSSM
        self.obs_out = nn.Linear(deter + 4096, kw.get("units", 1024)).to(self.device)
        self.img_in = nn.Linear(stoch + action_dim, kw.get("units", 1024)).to(self.device)
        self.img_out = nn.Linear(self.deter, kw.get("units", 1024)).to(self.device)

        # GRU module for state update
        self.gru = nn.GRUCell(input_size=kw.get("units", 1024), hidden_size=deter).to(self.device)

    def initial_state(self, batch_size):
        state = {
            'deter': torch.zeros(batch_size, self.deter, device=self.device),
            'mean': torch.zeros(batch_size, self.stoch, device=self.device),
            'std': torch.ones(batch_size, self.stoch, device=self.device),
            'stoch': torch.zeros(batch_size, self.stoch, device=self.device),
        }

        if self.initial == "learned":
            state['deter'] = torch.tanh(self.initial_deter).repeat(batch_size, 1).to(self.device)
            state['stoch'] = self.get_stoch(state['deter'])
        return state

    def observe(self, embed, action, is_first, state=None):
        embed, action, is_first = embed.to(self.device), action.to(self.device), is_first.to(self.device)
        if state is None:
            state = self.initial_state(action.size(0))

        post, prior = [], []
        for t in range(action.size(1)):
            state, p = self.obs_step(state, action[:, t], embed[:, t], is_first[:, t])
            post.append(state)
            prior.append(p)

        post = {k: torch.stack([s[k] for s in post], dim=1) for k in post[0]}
        prior = {k: torch.stack([s[k] for s in prior], dim=1) for k in prior[0]}
        return post, prior

    def imagine(self, action, state=None):
        action = action.to(self.device)
        if state is None:
            state = self.initial_state(action.size(0))

        prior = []
        for t in range(action.size(1)):
            state = self.img_step(state, action[:, t])
            prior.append(state)

        prior = {k: torch.stack([s[k] for s in prior], dim=1) for k in prior[0]}
        return prior

    def get_dist(self, state, sg=False):
        mean, std = state["mean"], state["std"]
        std = torch.clamp(std, min=1e-4)  # Ensure non-zero standard deviation
        
        # If sg (stop_gradient) is True, detach mean and std to stop gradient flow
        if sg:
            mean = mean.detach()
            std = std.detach()
            
        dist = torch.distributions.Independent(torch.distributions.Normal(mean, std), 1)
        return dist

    def obs_step(self, prev_state, prev_action, embed, is_first):
        if self.action_clip > 0.0:
            prev_action = prev_action * (self.action_clip / torch.clamp(prev_action.abs(), min=self.action_clip))

        # Masking for the first step
        mask = 1.0 - is_first.float()
        prev_state = {k: v * mask.unsqueeze(-1) for k, v in prev_state.items()}
        prev_action = prev_action * mask.unsqueeze(-1)

        prior = self.img_step(prev_state, prev_action)
        x = torch.cat([prior["deter"], embed], dim=-1)
        x = self.obs_out(x)
        stats = self._stats(x)

        dist = self.get_dist(stats)
        stoch = dist.rsample()  # Sample with reparameterization
        post = {"stoch": stoch, "deter": prior["deter"], **stats}
        return post, prior

    def img_step(self, prev_state, prev_action):
        prev_stoch = prev_state["stoch"]
        prev_action = prev_action.view(prev_action.size(0), -1)

        x = torch.cat([prev_stoch, prev_action], dim=-1)
        x = self.img_in(x)
        deter = self.gru(x, prev_state["deter"])
        x = self.img_out(deter)
        stats = self._stats(x)

        dist = self.get_dist(stats)
        stoch = dist.rsample()  # Sample with reparameterization
        return {"stoch": stoch, "deter": deter, **stats}

    def get_stoch(self, deter):
        x = self.img_out(deter)
        stats = self._stats(x)
        dist = self.get_dist(stats)
        return dist.mean

    def _stats(self, x):
        mean, std = x.chunk(2, dim=-1)
        std = 2 * torch.sigmoid(std / 2) + 0.1  # Ensure non-zero std deviation
        return {"mean": mean, "std": std}

    def dyn_loss(self, post, prior, impl="kl", free=1.0):
        if impl == "kl":
            # Use stop gradient on the posterior
            kl_div = torch.distributions.kl_divergence(
                self.get_dist(post, sg=True), 
                self.get_dist(prior)
            )
        elif impl == "logprob":
            # Compute negative log probability of post stochastic states under the prior distribution
            kl_div = -self.get_dist(prior).log_prob(self.stop_gradient(post["stoch"]))
        else:
            raise NotImplementedError(f"Dynamics loss implementation '{impl}' is not supported.")

        # Apply free bit threshold to avoid vanishing gradients
        if free:
            kl_div = torch.clamp(kl_div, min=free)

        return kl_div

    def rep_loss(self, post, prior, impl="kl", free=1.0):
        if impl == "kl":
            # Use stop gradient on the prior
            kl_div = torch.distributions.kl_divergence(
                self.get_dist(post), 
                self.get_dist(prior, sg=True)
            )
        elif impl == "uniform":
            # KL divergence between the posterior and a uniform distribution
            uniform_prior = {k: torch.zeros_like(v) for k, v in prior.items()}
            kl_div = torch.distributions.kl_divergence(
                self.get_dist(post), 
                self.get_dist(uniform_prior)
            )
        elif impl == "entropy":
            # Negative entropy of the posterior distribution
            kl_div = -self.get_dist(post).entropy()
        elif impl == "none":
            # No representation loss
            kl_div = torch.zeros(post["deter"].shape[:-1], device=post["deter"].device)
        else:
            raise NotImplementedError(f"Representation loss implementation '{impl}' is not supported.")

        # Apply free bit threshold
        if free:
            kl_div = torch.clamp(kl_div, min=free)

        return kl_div

    def stop_gradient(self, tensor):
        # Utility function to stop gradients (similar to JAX's stop_gradient)
        return tensor.detach()
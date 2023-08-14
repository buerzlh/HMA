import torch
import torch.nn as nn
import numpy as np


class BasicFullyConnectedNet(nn.Module):
    def __init__(self, dim, depth, hidden_dim=256, use_tanh=False, use_bn=False, out_dim=None, use_an=False):
        super(BasicFullyConnectedNet, self).__init__()
        layers = []
        layers.append(nn.Linear(dim, hidden_dim))
        if use_bn:
            assert not use_an
            layers.append(nn.BatchNorm1d(hidden_dim))
        if use_an:
            assert not use_bn
            layers.append(ActNorm(hidden_dim))
        layers.append(nn.LeakyReLU())
        for d in range(depth):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(hidden_dim, dim if out_dim is None else out_dim))
        if use_tanh:
            layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class DoubleVectorCouplingBlock(nn.Module):
    """Support uneven inputs"""
    def __init__(self, in_channels, hidden_dim, hidden_depth=2):
        super().__init__()
        dim1 = (in_channels // 2) + (in_channels % 2)
        dim2 = in_channels // 2
        self.s = nn.ModuleList([
            BasicFullyConnectedNet(dim=dim1, out_dim=dim2, depth=hidden_depth,
                                   hidden_dim=hidden_dim, use_tanh=True),
            BasicFullyConnectedNet(dim=dim1, out_dim=dim2, depth=hidden_depth,
                                   hidden_dim=hidden_dim, use_tanh=True),
        ])
        self.t = nn.ModuleList([
            BasicFullyConnectedNet(dim=dim1, out_dim=dim2, depth=hidden_depth,
                                   hidden_dim=hidden_dim, use_tanh=False),
            BasicFullyConnectedNet(dim=dim1, out_dim=dim2, depth=hidden_depth,
                                   hidden_dim=hidden_dim, use_tanh=False),
        ])

    def forward(self, x, reverse=False):
        assert len(x.shape) == 4
        x = x.squeeze(-1).squeeze(-1)
        if not reverse:
            logdet = 0
            for i in range(len(self.s)):
                idx_apply, idx_keep = 0, 1
                if i % 2 != 0:
                    x = torch.cat(torch.chunk(x, 2, dim=1)[::-1], dim=1)
                x = torch.chunk(x, 2, dim=1)
                scale = self.s[i](x[idx_apply])
                x_ = x[idx_keep] * (scale.exp()) + self.t[i](x[idx_apply])
                x = torch.cat((x[idx_apply], x_), dim=1)
                logdet_ = torch.sum(scale.view(x.size(0), -1), dim=1)
                logdet = logdet + logdet_
            return x[:,:,None,None], logdet
        else:
            idx_apply, idx_keep = 0, 1
            for i in reversed(range(len(self.s))):
                if i % 2 == 0:
                    x = torch.cat(torch.chunk(x, 2, dim=1)[::-1], dim=1)
                x = torch.chunk(x, 2, dim=1)
                x_ = (x[idx_keep] - self.t[i](x[idx_apply])) * (self.s[i](x[idx_apply]).neg().exp())
                x = torch.cat((x[idx_apply], x_), dim=1)
            return x[:,:,None,None]



class NormalizingFlow(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        # return transformed, logdet
        raise NotImplementedError

    def reverse(self, *args, **kwargs):
        # return transformed_reverse
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        # return sample
        raise NotImplementedError

class ActNorm(nn.Module):
    def __init__(self, num_features, logdet=False, affine=True,
                 allow_reverse_init=True):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height*width*torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            else:
                self.initialize(output)
                self.initialized.fill_(1)

        if len(output.shape) == 2:
            output = output[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h

        
        
class Shuffle(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(Shuffle, self).__init__()
        self.in_channels = in_channels
        idx = torch.randperm(in_channels)
        self.register_buffer('forward_shuffle_idx', nn.Parameter(idx, requires_grad=False))
        self.register_buffer('backward_shuffle_idx', nn.Parameter(torch.argsort(idx), requires_grad=False))

    def forward(self, x, reverse=False, conditioning=None):
        if not reverse:
            return x[:, self.forward_shuffle_idx, ...], 0
        else:
            return x[:, self.backward_shuffle_idx, ...]
        


class UnconditionalFlatDoubleCouplingFlowBlock(nn.Module):
    def __init__(self, in_channels, hidden_dim, hidden_depth):
        super().__init__()
        self.norm_layer = ActNorm(in_channels, logdet=True)
        self.coupling = DoubleVectorCouplingBlock(in_channels,
                                                   hidden_dim,
                                                   hidden_depth)
        self.shuffle = Shuffle(in_channels)

    def forward(self, x, reverse=False):
        if not reverse:
            h = x
            logdet = 0.0
            h, ld = self.norm_layer(h)
            logdet += ld
            h, ld = self.coupling(h)
            logdet += ld
            h, ld = self.shuffle(h)
            logdet += ld
            return h, logdet
        else:
            h = x
            h = self.shuffle(h, reverse=True)
            h = self.coupling(h, reverse=True)
            h = self.norm_layer(h, reverse=True)
            return h

    def reverse(self, out):
        return self.forward(out, reverse=True)
        

class Flow(NormalizingFlow):
    """Flat, multiple blocks of ActNorm, DoubleAffineCoupling, Shuffle"""
    def __init__(self, in_channels, n_flows, hidden_dim, hidden_depth):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = hidden_dim
        self.num_blocks = hidden_depth
        self.n_flows = n_flows
        self.sub_layers = nn.ModuleList()

        for flow in range(self.n_flows):
            self.sub_layers.append(UnconditionalFlatDoubleCouplingFlowBlock(
                                   self.in_channels, self.mid_channels,
                                   self.num_blocks)
                                   )

    def forward(self, x, reverse=False):
        if len(x.shape) == 2:
            x = x[:,:,None,None]
        self.last_outs = []
        self.last_logdets = []
        if not reverse:
            logdet = 0.0
            for i in range(self.n_flows):
                x, logdet_ = self.sub_layers[i](x)
                logdet = logdet + logdet_
                self.last_outs.append(x)
                self.last_logdets.append(logdet)
            return x.squeeze(-1).squeeze(-1), logdet
        else:
            for i in reversed(range(self.n_flows)):
                x = self.sub_layers[i](x, reverse=True)
            return x.squeeze(-1).squeeze(-1)

    def reverse(self, out):
        if len(out.shape) == 2:
            out = out[:,:,None,None]
        return self(out, reverse=True)

    def sample(self, num_samples, device="cpu"):
        zz = torch.randn(num_samples, self.in_channels, 1, 1).to(device)
        return self.reverse(zz)

    def get_last_layer(self):
        return getattr(self.sub_layers[-1].coupling.t[-1].main[-1], 'weight')
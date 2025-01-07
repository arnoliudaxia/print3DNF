import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from activation import trunc_exp
from .renderer import NeRFRenderer

class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 encoding="hashgrid",
                 encoding_dir="sphere_harmonics",
                 encoding_bg="hashgrid",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 bound=1,
                 density_max_scale = 100.,
                 print_density = [[2.883636819, 0.302264858, 0.193614904], [0.043354038, 8.749335941, 0.950262213], [0.0253694, 0.020829506, 2.294867001], [2.846568562, 3.071851834, 3.899442581], [2.807944551, 3.380290754, 5.646479876],  [0, 0, 0]],
                 print_color = [[0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 0, 0], [1, 1, 1], [0, 0, 0]],
                 **kwargs,
                 ):
        super().__init__(bound, **kwargs)

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * bound)
        self.density_max_scale = torch.tensor(density_max_scale, dtype=torch.float32, requires_grad=False).cuda()
        self.print_density = torch.tensor(print_density, dtype=torch.float32, requires_grad=False).cuda()
        self.print_color = torch.tensor(print_color, dtype=torch.float32, requires_grad=False).cuda()
        # self.density_scale = nn.Parameter(torch.ones(1))

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 4
            else:
                out_dim = hidden_dim
            
            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color
        self.encoder_dir, self.in_dim_dir = get_encoder(encoding_dir)
        
        # color_net =  []
        # for l in range(num_layers_color):
        #     if l == 0:
        #         in_dim = self.in_dim_dir + self.geo_feat_dim
        #     else:
        #         in_dim = hidden_dim_color
            
        #     if l == num_layers_color - 1:
        #         out_dim = 3 # 3 rgb
        #     else:
        #         out_dim = hidden_dim_color
            
        #     color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        # self.color_net = nn.ModuleList(color_net)

        # background network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg        
            self.hidden_dim_bg = hidden_dim_bg
            self.encoder_bg, self.in_dim_bg = get_encoder(encoding_bg, input_dim=2, num_levels=4, log2_hashmap_size=19, desired_resolution=2048) # much smaller hashgrid 
            
            bg_net = []
            for l in range(num_layers_bg):
                if l == 0:
                    in_dim = self.in_dim_bg + self.in_dim_dir
                else:
                    in_dim = hidden_dim_bg
                
                if l == num_layers_bg - 1:
                    out_dim = 3 # 3 rgb
                else:
                    out_dim = hidden_dim_bg
                
                bg_net.append(nn.Linear(in_dim, out_dim, bias=False))

            self.bg_net = nn.ModuleList(bg_net)
        else:
            self.bg_net = None


    def forward(self, x, d): # 测试的时候
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        # sigma
        x = self.encoder(x, bound=self.bound)

        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        # h = F.softmax(h, dim=1)

        #sigma = F.relu(h[..., 0])
        # sigma = (h.unsqueeze(-1) * self.print_density).sum(dim=1) * self.density_max_scale

        # color = (h.unsqueeze(-1) * self.print_color).sum(dim=1)
        sigma = trunc_exp(h[..., 0])
        if self.density_max_scale!=-1:
            sigma = torch.clamp(sigma, 0, self.density_max_scale)
        color = torch.clamp(h[..., [1, 2, 3]], 0, 1)
        
        # bp()
        return sigma, color

    def density(self, x): # 训练的时候
        # x: [N, 3], in [-bound, bound]

        x = self.encoder(x, bound=self.bound)
        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        # sigma = F.relu(h[..., 0])
        
        # h = torch.clamp(h, 0, 1)
        # h = F.relu(h)
        # h = F.softmax(h, dim=1)
        # print(h[0])
        # density_scale = torch.clamp(self.density_scale, 0, self.density_max_scale)
        # print(density_scale, self.density_max_scale)
        # sigma = (h.unsqueeze(-1) * self.print_density).sum(dim=1) * density_scale
        # geo_feat = (h.unsqueeze(-1) * self.print_color).sum(dim=1) * density_scale / self.density_max_scale
        # sigma = trunc_exp(h[..., 0:3])
        # geo_feat = torch.clamp(h[..., 3:6], 0, 1)

        sigma = trunc_exp(h[..., 0])
        # print(sigma.max().item())
        # 打印红色文本
        # print("\033[91m" + str(sigma.max().item()) + "\033[0m")
        
        if self.density_max_scale!=-1:
            sigma = torch.clamp(sigma, 0, self.density_max_scale)
        # print("\033[91m" + str(sigma.max().item()) + "\033[0m")
        geo_feat = torch.clamp(h[..., [1, 2, 3]], 0, 1)
        
        
        # geo_feat = geo_feat * sigma / self.density_max_scale

        
        # geo_feat = sigma * geo_feat
        # geo_feat = geo_feat / (geo_feat.max(dim=1)[0].unsqueeze(-1) + 1e-5)

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
        }

    def background(self, x, d):
        # x: [N, 2], in [-1, 1]
        d = torch.zeros_like(d)
        d[:, 0] = 1
        h = self.encoder_bg(x) # [N, C]
        d = self.encoder_dir(d)

        h = torch.cat([d, h], dim=-1)
        for l in range(self.num_layers_bg):
            h = self.bg_net[l](h)
            if l != self.num_layers_bg - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs

    # allow masked inference
    def color(self, x, d, mask=None, geo_feat=None, **kwargs):
        # # x: [N, 3] in [-bound, bound]
        # # mask: [N,], bool, indicates where we actually needs to compute rgb.

        # if mask is not None:
        #     rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device) # [N, 3]
        #     # in case of empty mask
        #     if not mask.any():
        #         return rgbs
        #     x = x[mask]
        #     d = d[mask]
        #     geo_feat = geo_feat[mask]

        # d = torch.zeros_like(d)
        # d[:, 0] = 1
        # d = self.encoder_dir(d)
        # h = torch.cat([d, geo_feat], dim=-1)
        # for l in range(self.num_layers_color):
        #     h = self.color_net[l](h)
        #     if l != self.num_layers_color - 1:
        #         h = F.relu(h, inplace=True)
        
        # # sigmoid activation for rgb
        # h = torch.sigmoid(h)

        # if mask is not None:
        #     rgbs[mask] = h.to(rgbs.dtype) # fp16 --> fp32
        # else:
        #     rgbs = h

        return geo_feat

    # optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
            # {'params': [self.density_scale], 'lr': 10*lr},
        ]
        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})
        
        return params

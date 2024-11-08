import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from maskrcnn_benchmark.modeling.roi_heads.relation_head.attention_blocks import Trans_block
from maskrcnn_benchmark.modeling.make_layers import make_fc


class CouplingLayer(nn.Module):

    def __init__(self, d, intermediate_dim, swap=False):
        nn.Module.__init__(self)
        self.d = d - (d // 2)
        self.swap = swap
        self.net_s_t = nn.Sequential(
            nn.Linear(self.d, intermediate_dim),
            nn.ReLU(inplace=True),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.ReLU(inplace=True),
            nn.Linear(intermediate_dim, (d - self.d) * 2),
        )

    def forward(self, x, logpx=None, reverse=False):

        if self.swap:
            x = torch.cat([x[:, self.d:], x[:, :self.d]], 1)

        in_dim = self.d
        out_dim = x.shape[1] - self.d

        s_t = self.net_s_t(x[:, :in_dim])
        scale = torch.sigmoid(s_t[:, :out_dim] + 2.)
        shift = s_t[:, out_dim:]

        logdetjac = torch.sum(torch.log(scale).view(scale.shape[0], -1), 1, keepdim=True)

        if not reverse:
            y1 = x[:, self.d:] * scale + shift
            delta_logp = -logdetjac
        else:
            y1 = (x[:, self.d:] - shift) / scale
            delta_logp = logdetjac

        y = torch.cat([x[:, :self.d], y1], 1) if not self.swap else torch.cat([y1, x[:, :self.d]], 1)

        if logpx is None:
            return y
        else:
            return y, logpx + delta_logp
        
        
class VarianceSchedule(nn.Module):

    def __init__(self, num_steps, beta_1, beta_T, mode='linear'):
        super().__init__()
        assert mode in ('linear', )
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas

class flow_model(nn.Module):
    def __init__(self,in_dim,latent_dim,depth) -> None:
        super().__init__()
        
        self.flow_modules=nn.ModuleList([])
        for dep_idx in range(depth):
            self.flow_modules.append(CouplingLayer(in_dim,latent_dim,swap=(dep_idx % 2 == 0)))
        
    def forward(self, x, logpx=None, reverse=False, inds=None):
        if inds is None:
            if reverse:
                inds = range(len(self.flow_modules) - 1, -1, -1)
            else:
                inds = range(len(self.flow_modules))

        if logpx is None:
            for i in inds:
                x = self.flow_modules[i](x, reverse=reverse)
            return x
        else:
            for i in inds:
                x, logpx = self.flow_modules[i](x, logpx, reverse=reverse)
            return x, logpx

class diffusion_model(nn.Module):
    def __init__(self,in_dim,out_dims=[128,256,512,256,128],residual=True):
        super().__init__()
        # ********* init parameter *********
        self.num_steps=50
        beta_1,beta_T,sched_mode=1e-4,0.02,'linear'
        self.in_dim=in_dim
        
        self.net=diffusion_bloack(in_dim,self.num_steps,out_dims,residual)
        
        self.var_sched=VarianceSchedule(self.num_steps,beta_1,beta_T,mode=sched_mode)
        
    def forward(self, x_0, context, condition_reps,rel_proto,rel_nums, t=None):
        """
        Args:
            x_0:  Input proto representation, (B, d).
            context:  Shape latent, (B, F).
            condition_reps: condition rel representation, (B, d)
        """
        batch_size, reps_dim = x_0.size()
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)
        else:
            t=[t]*batch_size
        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t]

        c0 = torch.sqrt(alpha_bar).view(-1, 1)       # (B, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1)   # (B, 1)

        e_rand = torch.randn_like(x_0)  # (B, d)
        e_theta,ctx_emb = self.net(c0 * x_0 + c1 * e_rand, beta=beta, context=context, condition_reps=condition_reps,rel_proto=rel_proto,t=t,rel_nums=rel_nums)

        return e_theta,e_rand,ctx_emb

    def sample(self,context,condition_reps,rel_proto,rel_nums,flexibility=0.0, ret_traj=False):
        batch_size = context.size(0)
        x_T = torch.randn([batch_size, self.in_dim]).to(context.device)
        traj = {self.num_steps: x_T}
        for t in range(self.num_steps, 0, -1):
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            sigma = self.var_sched.get_sigmas(t, flexibility)

            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

            x_t = traj[t]
            beta = self.var_sched.betas[[t]*batch_size]
            e_theta,_ = self.net(x_t, beta=beta, context=context, condition_reps=condition_reps, rel_proto=rel_proto, t=[t]*batch_size,rel_nums=rel_nums)
            x_next = c0 * (x_t - c1 * e_theta) + sigma * z
            traj[t-1] = x_next.detach()     # Stop gradient and save trajectory.
            traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
            if not ret_traj:
                del traj[t]
        
        if ret_traj:
            return traj
        else:
            return traj[0]

    

class diffusion_bloack(nn.Module):
    def __init__(self,in_dim,num_steps,out_dims=[128,256,512,256,128],residual=True) -> None:
        super().__init__()
        
        self.act = F.leaky_relu
        self.residual = residual
        self.layers = nn.ModuleList([
            ConcatSquashLinear(in_dim if idx==0 else out_dims[idx-1], out_dim, in_dim+3) for idx,out_dim in enumerate(out_dims)
        ])
        self.layers.append(ConcatSquashLinear(out_dims[-1],in_dim,in_dim+3))
        
        
        """
        self.time_embedding=nn.Embedding(num_steps+1,512)
        nn.init.normal_(self.time_embedding.weight, mean=0, std=1)
        
        self.gate_condition=nn.Sequential(
            nn.Linear(in_dim*2,in_dim),
            nn.Sigmoid()
        )
        
        self.ctx_proj=make_fc(in_dim,in_dim//2)
        self.condition_gate=make_fc(in_dim,in_dim//2)
        self.condition_bias=make_fc(in_dim,in_dim//2)
        self.proto_proj=make_fc(in_dim,in_dim//2)
        self.refine_ctx_condition=nn.ModuleList([    
                 # refine context by rel_proto
            Trans_block(1,8,64,64,in_dim//2,in_dim//2)
            for _ in range(1)
        ])
        self.gate_fusion_time=nn.Sequential(
            nn.Linear(512,in_dim//2),
            nn.Sigmoid()
        )
        """
        
    def forward(self, x, beta, context, condition_reps,rel_proto,t,rel_nums):
        """
        Args:
            x:  prototype representation at some timestep t, (B, d).
            beta:     Time. (B, ).
            context:  Shape latents. (B, F).
            condition_reps: condition reps, (B, 3d)
        """
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1)          # (B, 1)
        context = context.view(batch_size, -1)   # (B, F)

        # time_emb=self.time_embedding(torch.tensor(t,device=context.device))
        # context=context+self.gate_condition(torch.cat([context,condition_reps],dim=-1))*condition_reps
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 3)
        ctx=torch.cat([time_emb,context],dim=-1)
        """
        rel_proto=self.proto_proj(rel_proto)
        ctx=self.ctx_proj(context)*self.condition_gate(condition_reps)+self.condition_bias(condition_reps)
        for refine_ctx in self.refine_ctx_condition:
            ctx=refine_ctx(ctx,rel_proto.unsqueeze(0).expand(len(rel_nums),-1,-1),rel_nums)
        ctx=ctx*self.gate_fusion_time(time_emb)
        """
        
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx,x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            return x + out,ctx
        else:
            return out,ctx
        
class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = nn.Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate + bias
        return ret
import time
import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

def selective_scan_easy_v2(us, dts, As, Bs, Cs, Ds, delta_bias=None, delta_softplus=False, return_last_state=False, chunksize=3):
    """
    # B: batch_size, G: groups, D: dim, N: state dim, L: seqlen
    us: B, G * D, L 
    dts: B, G * D, L
    As: G * D, N
    Bs: B, G, N, L
    Cs: B, G, N, L
    Ds: G * D
    delta_bias: G * D
    # chunksize can be any as you like. But as the chunksize raises, hs may get None, as exp(sum(delta) A) is really small
    """
    def selective_scan_chunk(us, dts, As, Bs, Cs, hprefix, Mask):
        """
        partial(h) / partial(t) = Ah + Bu; y = Ch + Du;
        => partial(h*exp(-At)) / partial(t) = Bu*exp(-At);
        => h_t = h_0 + sum_{0}_{t}_{Bu*exp(A(t-v)) dv};
        => h_b = exp(A(dt_a + ... + dt_{b-1})) * (h_a + sum_{a}_{b-1}_{Bu*exp(-A(dt_a + ... + dt_i)) dt_i});
           y_i = C_i*h_i + D*u_i
        """
        """
        us, dts: (L, B, G, D) # L is chunk_size
        As: (G, D, N)
        Bs, Cs: (L, B, G, N)
        Ds: (G, D)
        hprefix: (B, G, D, N)
        """
        LChunk = us.shape[0]
        ts = dts.cumsum(dim=0)
        K_chunk_tmp = torch.einsum("gdn,lbgd->lbgdn", -As, ts)
        K_chunk = K_chunk_tmp.exp()
        Ats = (-K_chunk_tmp).exp()
        Q_chunk = torch.einsum("lbgd,lbgn->lbgdn", dts * us, Bs)
        if LChunk < chunksize:
            Qm_chunk = Mask[:LChunk, :LChunk, None, None, None, None] * Q_chunk.unsqueeze(0).repeat(LChunk, 1, 1, 1, 1, 1)
        else:
            Qm_chunk = Mask[:, :, None, None, None, None] * Q_chunk.unsqueeze(0).repeat(LChunk, 1, 1, 1, 1, 1)
        H_tmp = Ats * torch.einsum("rlbgdn,lbgdn->rbgdn", Qm_chunk, K_chunk)
        hs = H_tmp + Ats * hprefix.unsqueeze(0)
        ys = torch.einsum("lbgn,lbgdn->lbgd", Cs, hs)
        return ys, hs

    dtype = torch.float32
    # dtype = torch.float16
    inp_dtype = us.dtype
    has_D = Ds is not None
    if chunksize < 1:
        chunksize = Bs.shape[-1]
    Mask = torch.tril(us.new_ones((chunksize, chunksize)))

class SelectiveScanEasy(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, us, dts, As, Bs, Cs, Ds, delta_bias=None, delta_softplus=False, return_last_state=False, chunksize=64):
        has_D = Ds is not None
        dtype = torch.float32
        
        dts = dts.to(dtype)
        if delta_bias is not None:
            dts = dts + delta_bias.view(1, -1, 1).to(dtype)
        if delta_softplus:
            dts = torch.nn.functional.softplus(dts)

        B_squeeze = (len(Bs.shape) == 3)
        C_squeeze = (len(Cs.shape) == 3)
        if B_squeeze:
            Bs = Bs.unsqueeze(1)
        if C_squeeze:
            Cs = Cs.unsqueeze(1)
        B, G, N, L = Bs.shape
        us = us.view(B, G, -1, L).permute(3, 0, 1, 2).to(dtype)
        dts = dts.view(B, G, -1, L).permute(3, 0, 1, 2).to(dtype)
        As = As.view(G, -1, N).to(dtype)
        Bs = Bs.permute(3, 0, 1, 2).to(dtype)
        Cs = Cs.permute(3, 0, 1, 2).to(dtype)
        Ds = Ds.view(G, -1).to(dtype) if has_D else None
        D = As.shape[1]
        
        ctx.shape = (B, G, D, N, L)
        ctx.delta_softplus = delta_softplus
        ctx.return_last_state = return_last_state
        ctx.chunksize = chunksize
        ctx.BC_squeeze = (B_squeeze, C_squeeze)
        save_for_backward = [us, dts, As, Bs, Cs, Ds, delta_bias]

        chunks = list(range(0, L, chunksize))
        oys = []
        ohs = []        
        hprefix = us.new_zeros((B, G, D, N), dtype=torch.float)
        for i in chunks:
            ts = dts[i:i+chunksize].cumsum(dim=0)
            Ats = torch.einsum("gdn,lbgd->lbgdn", As, ts).exp()
            # scale = Ats[-1:].detach()
            scale = 1
            rAts = Ats / scale
            duts = dts[i:i + chunksize] * us[i:i + chunksize]
            dtBus = torch.einsum("lbgd,lbgn->lbgdn", duts, Bs[i:i + chunksize])
            tmp_dtBus_div_rAts = (dtBus / rAts)
            tmp_dtBus_div_rAts_cumsum = tmp_dtBus_div_rAts.cumsum(dim=0) 
            hs = rAts * tmp_dtBus_div_rAts_cumsum + Ats * hprefix.unsqueeze(0)
            ys = torch.einsum("lbgn,lbgdn->lbgd", Cs[i:i + chunksize], hs) 
            oys.append(ys)
            ohs.append(hs)
            hprefix = hs[-1]

        oys = torch.cat(oys, dim=0)
        ohs = torch.cat(ohs, dim=0)
        if has_D:
            oys = oys + Ds * us

        save_for_backward.extend([ohs])

        ctx.save_for_backward(*save_for_backward)
        
        oys = oys.permute(1, 2, 3, 0).view(B, -1, L)
  
        return oys if not return_last_state else (oys, hprefix.view(B, G * D, N))

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, doys: torch.Tensor, *args):
        us, dts, As, Bs, Cs, Ds, delta_bias, ohs = ctx.saved_tensors
        B, G, D, N, L = ctx.shape
        chunksize = ctx.chunksize
        delta_softplus = ctx.delta_softplus
        doys = doys.view(B, G, D, L).permute(3, 0, 1, 2)

        def rev_comsum(x):
            cum_sum = torch.cumsum(x, dim=0)
            return (x - cum_sum + cum_sum[-1:None])

        dus = None
        dDs = None
        if Ds is not None:
            dDs = torch.einsum("lbgd,lbgd->gd", doys, us).view(-1)
            dus = torch.einsum("lbgd,gd->lbgd", doys, Ds)

        chunks = list(range(0, L, chunksize))
        dAs = us.new_zeros((G, D, N), dtype=torch.float)
        dus = us.new_zeros((L, B, G, D), dtype=torch.float) if dus is None else dus
        ddts = us.new_zeros((L, B, G, D), dtype=torch.float)
        dBs = us.new_zeros((L, B, G, N), dtype=torch.float)
        dCs = us.new_zeros((L, B, G, N), dtype=torch.float)
        dhprefix = us.new_zeros((B, G, D, N), dtype=torch.float)
        for i in chunks[::-1]:
            # forward procedure ================
            tmp_dts = dts[i:i+chunksize]
            ts = tmp_dts.cumsum(dim=0)
            Ats = torch.einsum("gdn,lbgd->lbgdn", As, ts).exp()
            scale = Ats[-1].detach()
            scale = 1
            rAts = Ats / scale
            duts = dts[i:i + chunksize] * us[i:i + chunksize]
            dtBus = torch.einsum("lbgd,lbgn->lbgdn", duts, Bs[i:i + chunksize])
            dtBus_div_rAts = (dtBus / rAts)
            hs_minus_prefix_div_rAts = dtBus_div_rAts.cumsum(dim=0)

            # backward procedure ================
            _hs = ohs[i:i+chunksize]
            _hprefix = ohs[i - 1] if i > 0 else None
            dCs[i:i + chunksize] = torch.einsum("lbgd,lbgdn->lbgn", doys[i:i + chunksize], _hs)
            dhs = doys[i:i + chunksize].unsqueeze(4) * Cs[i:i + chunksize].unsqueeze(3) # lbgd,lbgn->lbgdn
            dhs[-1] = dhs[-1] + dhprefix
            dhprefix = torch.einsum("lbgdn,lbgdn -> bgdn", dhs, Ats)
            dAts_hprefix = dhs * _hprefix.unsqueeze_(0) if i > 0 else None # lbgdn,bgdn->lbgdn
            drAts_hs_minus_prefix = dhs * hs_minus_prefix_div_rAts
            dhs_minus_prefix_div_rAts = dhs * rAts

            d_dtBus_div_rAts = rev_comsum(dhs_minus_prefix_div_rAts)
            
            ddtBus = d_dtBus_div_rAts / rAts
            dBs[i:i + chunksize] = torch.einsum("lbgdn,lbgd->lbgn", ddtBus, duts)
            dduts = torch.einsum("lbgdn,lbgn->lbgd", ddtBus, Bs[i:i + chunksize])
            dus[i:i + chunksize] = dus[i:i + chunksize] + dduts * dts[i:i + chunksize]

            drAts_dtBus_div_rAts = d_dtBus_div_rAts * (-dtBus_div_rAts / rAts)

            ddts[i:i + chunksize] = dduts * us[i:i + chunksize]
            dAts = drAts_dtBus_div_rAts / scale + drAts_hs_minus_prefix / scale + (dAts_hprefix if i > 0 else 0)
       
            dAts_noexp = dAts * Ats # d(e^x) = e^x * dx
            dAs = dAs + torch.einsum("lbgdn,lbgd->gdn", dAts_noexp, ts)
            d_ts = torch.einsum("lbgdn,gdn->lbgd", dAts_noexp, As)
            
            _part_ddts = rev_comsum(d_ts) # the precision is enough
            
            ddts[i:i + chunksize] = ddts[i:i + chunksize] + _part_ddts
        
        if delta_softplus:
            # softplus = log(1 + e^x); dsoftplus = e^x /(1+e^-x) = 1 - 1 / (1+e^x) = 1 - (e^-softplus)
            ddts = ddts - ddts * (-dts).exp()

        ddelta_bias = None 
        if delta_bias is not None:
            ddelta_bias = ddts.sum([0, 1]).view(-1)
           
        dus = dus.permute(1, 2, 3, 0).view(B, -1, L)
        ddts = ddts.permute(1, 2, 3, 0).view(B, -1, L)
        dAs = dAs.view(-1, N)
        dBs = dBs.permute(1, 2, 3, 0)
        dCs = dCs.permute(1, 2, 3, 0)
        if ctx.BC_squeeze[0]:
            dBs = dBs.flatten(1, 2)
        if ctx.BC_squeeze[1]:
            dCs = dCs.flatten(1, 2)

        return (dus, ddts, dAs, dBs, dCs, dDs, ddelta_bias, None, None, None)

# 只使用了一次，将数据使用4*4的卷积和LN产生patch，然后转化为b*w*h*c
class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x

class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
        
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H//2, W//2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x
    
class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim*2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
        x= self.norm(x)

        return x
    
class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim  # 96
        self.dim_scale = dim_scale  # 4
                          #        96             384
        self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)
        #                          24  
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
        x= self.norm(x)

        return x

class SS2D(nn.Module):
    def __init__(
        self,
        d_model,   # 96
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
    ):
        super().__init__()
        self.d_model = d_model  # 96
        self.d_state = d_state  # 16
        self.d_conv = d_conv   # 3
        self.expand = expand   # 2
        self.d_inner = int(self.expand * self.d_model)  # 192
        self.dt_rank = math.ceil(self.d_model / 16)  # 6

        #                           96                 384
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,  # 192
            out_channels=self.d_inner,  # 192
            kernel_size=d_conv,  # 3
            padding=(d_conv - 1) // 2,    # 1
            bias=conv_bias,  
            groups=self.d_inner,  # 192  
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False), 
        )
        # 4*38*192的数据 初始化x的数据
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        # 初始化dt的数据吧
        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        # 初始化A和D
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        self.selective_scan = SelectiveScanEasy()
        
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan.apply(
            xs, dts, 
            As, Bs, Cs, Ds,
            dt_projs_bias,
            True,
            False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)  # x走的是ss2d的路径

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)  
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)   # 这里的z忘记了一个Linear吧
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,  # 96
        drop_path: float = 0,  # 0.2
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),  # nn.LN
        mamba_drop_rate: float = 0,  # 0
        d_state: int = 16,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)# 96             0.2                   16
        self.self_attention = SS2D(d_model=hidden_dim, dropout=mamba_drop_rate, d_state=d_state)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        # print(input.shape, "传入模块的大小")
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x

class VSSLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(  # 以第一个为例
        self, 
        dim,  # # 96
        depth,  # 2
        d_state=16,
        drop = 0.,
        attn_drop=0.,
        drop_path=0.,   # 每一个模块都有一个drop
        norm_layer=nn.LayerNorm, 
        downsample=None,  # PatchMergin2D
        use_checkpoint=False,  
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,   # 96
                drop_path=drop_path[i],  # 0.2
                norm_layer=norm_layer,  # nn.LN
                mamba_drop_rate=attn_drop, # 0 
                d_state=d_state,  # 16
            )
            for i in range(depth)])
        
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
                x = blk(x)
        
        if self.downsample is not None:
            x = self.downsample(x)

        return x
    

class VSSLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        upsample=None, 
        use_checkpoint=False, 
        d_state=16,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                mamba_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])
        
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x
    
class VSSM(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000, depths=[2, 2, 9, 2], depths_decoder=[2, 9, 2, 2],
                 dims=[96, 192, 384, 768], dims_decoder=[768, 384, 192, 96], d_state=16, drop_rate=0., mamba_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False):
        super().__init__()
        self.num_classes = num_classes # 1
        self.num_layers = len(depths)  # 4
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]   # 96
        self.num_features = dims[-1]  # 768 
        self.dims = dims   # [96, 192, 384, 768]

        # 4*4+LN-> b*w*h*c
        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None)
        
        
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 生成对应的sum(depths)随机深度衰减数值 dpr是正序，dpr_decoder是倒序（用到了[start:end:-1] 反向步长）
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):  # 以第一个为例 num_layers = 4
            layer = VSSLayer(
                dim=dims[i_layer],  # 96
                depth=depths[i_layer],  # 2
                d_state=d_state,  # 16
                drop=drop_rate,   # 0
                attn_drop=mamba_drop_rate, # 0
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], # ，每一个模块传一个概率值
                norm_layer=norm_layer, # nn.LN
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            
            self.layers.append(layer)

        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):  # 以第一个为例，num_layers=2
            layer = VSSLayer_up(
                dim=dims_decoder[i_layer],  # 768
                depth=depths_decoder[i_layer], # 2
                d_state=d_state,  # 16
                drop=drop_rate,  # 0
                attn_drop=mamba_drop_rate, # 0
                drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
                norm_layer=norm_layer, # nn.LN
                upsample=PatchExpand2D if (i_layer != 0) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers_up.append(layer)

        #  输入 64*64*96 ->linear+LN b*256*256*24                          96                             nn.LN
        self.final_up = Final_PatchExpand2D(dim=dims_decoder[-1], dim_scale=4, norm_layer=norm_layer)
        #     维度变换 输出b*1*256*256         24                 1 
        self.final_conv = nn.Conv2d(dims_decoder[-1]//4, num_classes, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless
        
        Conv2D is not intialized !!!
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
   
    def forward_features(self, x):
        skip_list = []
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            skip_list.append(x)
            x = layer(x)
        return x, skip_list
    
    def forward_features_up(self, x, skip_list):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = layer_up(x+skip_list[-inx])

        return x
    
    def forward_final(self, x):
        # input 3*64*64*96   out=3 256 256 24
        x = self.final_up(x)   
        x = x.permute(0,3,1,2)
        # out=3 24 256 256 
        x = self.final_conv(x)
        return x

    def forward(self, x):
        x, skip_list = self.forward_features(x)
        x = self.forward_features_up(x, skip_list)
        x = self.forward_final(x)  
        return x

class VMUNet(nn.Module):
    def __init__(self, 
                 input_channels=3, 
                 num_classes=1,
                 depths=[2, 2, 9, 2], 
                 depths_decoder=[2, 9, 2, 2],
                 drop_path_rate=0.2,
                 load_ckpt_path=None,
                ):
        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes

        self.vmunet = VSSM(in_chans=input_channels, # 3
                           num_classes=num_classes,  # 1
                           depths=depths,  # [2,2,9,2]
                           depths_decoder=depths_decoder, # [2,9,2,2]
                           drop_path_rate=drop_path_rate, # 0.2
                        )
    
    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.vmunet(x)
        if self.num_classes == 1: return torch.sigmoid(logits)
        else: return logits
    

# x = torch.randn(3, 3, 256, 256).to("cuda:0")
# net = VMUNet(3,1).to("cuda:0")
# print(net(x).shape)

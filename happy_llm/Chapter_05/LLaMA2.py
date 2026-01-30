import math
from typing import Tuple

import torch
from numpy import reshape
from transformers import PretrainedConfig
import torch.nn as nn
import torch.nn.functional as F
class ModelConfig(PretrainedConfig):
    model_type = "Tiny_K"
    def __init__(
            self,
            dim: int = 768, #模型维度
            n_layers: int = 12, # Transformer 层数
            n_heads: int = 16, # 注意力机制的头数
            n_kv_heads: int = 8, # 键值头的数量
            vocab_size: int = 6144, # 词汇表大小
            hiden_dim: int = None, # 隐藏层的维度
            multiple_of: int = 64,
            norm_eps: float = 1e-5, #归一化层的eps
            max_seq_len: int = 512, # 最大序列长度
            dropout: float = 0.0, # dropout 概率
            flash_attn: bool = True, # 是否使用FlashAttention
            **kwargs,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hiden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.flash_attn = flash_attn
        super().__init__(**kwargs)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        # eps 防止除0
        # weight 是一个可学习的参数， 初始化为1
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # RMSNorm 核心
        # x.pow(2).mean(-1, keepdim=True) 计算了输入x的平方的均值
        # torch.rsqrt 是平方根的倒数，这样就得到了RMSNorm的分母部分 eps防止分母为0
        # 最后乘以x，得到RMSNorm的结果
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # forward函数是模型的前向传播
        # 首先将输入x转为float类型，然后进行RMSNorm，最后转换回去
        # 最后乘以weight,这是RMSNorm的一个可学习的缩放因子
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

args = ModelConfig()
norm = RMSNorm(args.dim, args.norm_eps)
x = torch.randn(1, 50, args.dim)
output = norm(x)
print(output.shape)
# out: torch.Size([1, 50, 768])


'''将键和值的维度扩展到和查询的维度一样'''
'''实际上就是将一个K/V头对应多个Q头 比如Q有16个头 K/V有4个头 那么就让四个Q头对应一个K/V头'''
'''现在维度上进行扩展 然后再合并'''
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # 获取输入张量的形状： 批量大小、序列长度、键/值对头的数量、每个头的维度大小
    bs, slen, n_kv_heads, head_dim = x.shape
    # 如果重复次数为1 则不需要重复，直接返回原始张量
    if n_rep == 1:
        return x
    # 对张量进行扩展和重塑操作以重复键值对
    return (
        x[:, :, :, None, :] # 在第四个维度（头的维度前）添加一个新的维度
        .expand(bs, slen, n_kv_heads, n_rep, head_dim) # 将新添加的维度扩展到n_rep大小，实现重复的操作
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim) # 重新塑形，合并键/值对头的数量和重复次数的维度
    )


'''旋转嵌入'''
# 此处dim应为 dim//n_head, 因为是对每个head进行旋转嵌入的
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    # torch.arange(0, dim, 2)[: (dim // 2)].float()生成了一个从0开始，步长wei2的序列，长度为dim的一半
    # 然后每个元素除以dim，再取theta的倒数，得到频率
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成一个从0到end的序列，长度为end
    t = torch.arange(end, device=freqs.device)
    # 计算外积，得到一个二维矩阵，每一行是t的元素乘以freqs的元素
    freqs = torch.outer(t, freqs).float()
    # 计算频率的余弦值，得到实部
    freqs_cos = torch.cos(freqs)
    # 计算频率的正弦值，得到虚部
    freqs_sin = torch.sin(freqs)

    return freqs_cos, freqs_sin

def reshape_for_broadcast(freqs_cis: torch.Tensor, x:torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freq_cos: torch.Tensor,
        freq_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_r, xq_i = xq.float(),reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    freq_cos = reshape_for_broadcast(freq_cos, xq_r)
    freq_sin = reshape_for_broadcast(freq_sin, xq_r)

    xq_out_r = xq_r * freq_cos - xq_i * freq_sin
    xq_out_i = xq_r * freq_sin + xq_i * freq_cos
    xk_out_r = xk_r * freq_cos - xk_i * freq_sin
    xk_out_i = xk_r * freq_sin + xk_i * freq_cos

    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0

        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = args.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kvheads
        self.head_dim = args.dim // args.n_heads

        # 定义权重矩阵
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)

        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # 定义dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        self.attn_dropout = nn.Dropout(args.dropout)
        # 保存dropout概率
        self.dropout = args.dropout

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            # 若不支持flash Attention， 则使用手动实现的注意力机制， 并设置mask
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # 创建一个上三角矩阵，用于遮蔽未来信息
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
        # 获取批次大小和序列长度,[batch_size, seq_len, dim]
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # 调整形状以适应头的维度。
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # 应用旋转位置嵌入
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # 对k/v 进行扩展
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # 将头作为批次维度处理
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 根据是否支持Flash Attention 选择实现方式
        if self.flash:
            # 使用Flash Attention
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)

        else:
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert hasattr(self, 'mask')
            scores = scores + self.mask[:, :, :seqlen, :seqlen]
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)


        # 恢复时间维度合并头
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # 最终投影回残差流
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output
























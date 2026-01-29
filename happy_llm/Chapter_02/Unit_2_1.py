
import math

import torch
import torch.nn as nn
# 注意力机制的实现，query、key 词向量 相乘 -> 放缩 -> softmax -> 与V相乘 得到注意力分数

'''注意力计算函数'''
def attention(query, key, value, dropout=None):
    # 获取key词向量的维度，key响亮的维度和Value向量维度相同
    d_k = query.size(-1)
    # 计算Q与K的内积并作放缩
    # transpose--相当于转置
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # Softmax
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn

"""
自注意力机制：
对序列中每个 token，用它作为 Query，
与所有 token 的 Key 计算相似度，
经 softmax 得到权重，
再对所有 Value 加权求和，
从而生成该 token 的新表示。 我要用别人的信息，重新定义我自己

"自":
Q、K、V 全部来自同一个序列
token 既是“提问者”，也是“信息提供者”
"""

"""掩码自注意力"""
# 先通过 full 函数创建一个1 * seq_len * seq_len 的矩阵
# triu 函数的功能是创建一个上三角矩阵
mask = torch.full((1, args.max_seq_len, args.max_seq_len), float("-inf"))
mask = torch.triu(mask, diagonal=1)
# 将注意力分数与掩码作和
scores = scores + mask[:, :seqlen, :seqlen]
scores = F.softmax(socres.float(), dim=-1).type_as(xq)


"""多头自注意力机制"""
# dim 每个token 用长度为多少的向量表示
# 多头指每个头负责一部分特征 dim=512 n_heads=8 则每个头只看64维
"""
事实上，所谓的多头注意力机制其实就是将原始的输入序列进行多组的自注意力处理；然后再将每一组得到的自注意力结果拼接起来，再通过一个线性层进行处理，得到最终的输出。
我们可以通过矩阵运算巧妙地实现并行的多头计算，其核心逻辑在于使用三个组合矩阵来代替了n个参数矩阵的组合，也就是矩阵内积再拼接其实等同于拼接矩阵再内积。
一个头一个Linear -> 一次性算出所有头的Q 拼接在一个Q中
"""
class MultiHeadAttention(nn.Module):
    def __init__(self, args: ModelArgs, is_causal=False):
        super().__init__()
        assert args.dim % args.n_heads == 0
        self.head_dim = args.dim // args.n_heads
        self.n_heads = args.n_heads

        self.wq = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)

        self.wo = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.is_causal = is_causal

        # 多头注意力Mask矩阵比之前定义的Mask矩阵多一个维度(头)
        if is_causal:
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)

    def forword(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        bsz, seqlen, _ = q.shape
        xq, xk, xv = self.wq(q), self.wk(k), self.wv(v)

        # 将Q、K、V 拆分成多头， 维度为(B, T, n_head, dim // n_head)，然后交换维度，变成(B, n_head, T, dim // n_head)
        # 因为在注意力计算中我们是为了取后两个维度参与计算
        # 为什么要先按照B*T*n_head*C//n_head展开在呼唤1、2维度为不是直接按注意力输入展开，是因为view的展开方式是直接把输入全部排开
        # 然后按照要求构造，可以发现只有上述操作能够实现我们将每个头对应部分取出来的目的
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 注意力计算
        # 计算QK^T / sqrt(d_k)，维度为(B, nh, T, hs) × (B, nh, hs, T) -> (B, nh, T, T)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if self.is_causal:
            assert hasattr(self, "mask")
            # 这里截取到叙利厄长度，因为有些序列可能比max_seq_len 短
            scores = scores + self.mask[:, :, :seqlen, :seqlen]

        scores = F.softmax(scores.float(), dim=1).type_as(xq)
        scores = self.attn_dropout(scores)
        output = torch.matmul(scores, xv)

        # 恢复时间维度并合并头
        # 将多头的结果拼接起来，先交换维度为(B, T, n_head, dim // n_head). 再拼接成 (B, T, n_head * dim // n_head)
        # contiguous 函数用于重新开辟一块新内存存储， 因为Pytorch设置先transpose再view会报错，
        # 因为view直接基于底层存储得到，然而transpose并不会该边底层存储， 因此需要额外存储
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # 最终投影回残差流
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output

"""前馈神经网络"""
class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        """低维->高维->低维 一般是dim -> dim * 4 -> dim -> dim*4 -> dim (在高维空间更容易表达复杂模式)"""
        # 第一层线性变换，从输入维度到隐藏维度
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        # 第二层线性变换， 从隐藏维度到输入维度
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        # 定义dropout层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 前向传播函数
        # 首先，输入想通过第一层线性变换和RELU激活函数->非线性
        # 最后，通过第二层线性变换和dropout层防止过拟合
        return self.dropout(self.w2(F.relu(self.w1(x))))

"""Layer Norm层归一化"""
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class EncoderLayer(nn.Module):
    """Encoder层"""
    def __init__(self, args):
        super().__init__()
        # 一个Layer中有两个LayerNorm 分别在Attention 之前和 MLP之前
        self.attention_norm = LayerNorm(args.n_embd)
        # Encoder 不需要掩码
        self.attention = MultiHeadAttention(args, is_causal=False)
        self.fnn_norm = LayerNorm(args.n_embd)
        self.feed_forward = MLP(args.dim, args.dim, args.dropout)

    def forward(self, x):
        # Layer Norm
        norm_x = self.attention_norm(x)
        # 自注意力
        h = x + self.attention.forward(norm_x, norm_x, norm_x)
        # 经过前馈神经网络
        out = h + self.feed_forward.forward(self.fnn_norm(h))
        return out

class Encoder(nn.Module):
    '''Encoder 块'''
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.n_layer)])
        self.norm = LayerNorm(args.n_embd)

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return self.norm(x)



"""Decoder"""

class DecoderLayer(nn.Module):
    '''解码层'''
    def __init__(self, args):
        super().__init__()
        # 一个Layer 中有三个LayerNorm，分别在Mask Attention 之前、Self Attention 之前和MLP之前
        self.attention_norm_1 = LayerNorm(args.n_embd)
        # Decoder 的第一个部分市 Mask Attention， 传入 is_causal=True
        self.mask_attention = MultiHeadAttention(args, is_causal=True)
        self.attention_norm_2 = LayerNorm(args.n_embd)
        # Decoder 的第二个部分市 类似于Encoder 的Attention，传入 is_causal=False
        self.attention = MultiHeadAttention(args, is_causal=False)
        self.ffn_norm = LayerNorm(args.n_embd)
        # 第三个部分是MLP
        self.feed_forward = MLP(args.dim, args.dim, args.dropout)

    def forward(self, x, enc_out):
        # Layer Norm
        norm_x = self.attention_norm_1(x)
        # 掩码自注意力
        x = x + self.mask_attention.forward(norm_x, norm_x, norm_x)
        # 多头注意力
        norm_x = self.attention_norm_2(x)
        h = x + self.attention.forward(norm_x, enc_out, enc_out)
        # 经过前馈神经网络
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out

class Decoder(nn.Module):
    '''解码器'''
    def __init__(self, args):
        super(Decoder, self).__init__()
        # 一个Decoder由N个DecoderLayer 组成
        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.n_layer)])
        self.norm = LayerNorm(args.n_embd)

    def forward(self, x, enc_out):
        for layer in self.layers:
            x = layer(x, enc_out)

        return self.norm(x)




































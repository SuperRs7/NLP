import math
from typing import Tuple, Optional

import torch
from numpy import reshape
from transformers import PretrainedConfig, PreTrainedModel
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast


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
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
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
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        # 定义权重矩阵
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)

        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # 定义dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
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

'''构建MLP模块'''

class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        # 如果没有指定隐藏层的维度，我们将其设置为输入维度的4倍
        # 然后将其减少到2/3，最后确保它是multiple_of的倍数
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        # 定义第一层线性变换，从输入维度到隐藏维度
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        # 定义第二层线性变换，从隐藏维度到输入维度
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        # 定义第三层线性变换，从输入维度到隐藏维度
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        # 定义dropout层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        # 前向传播函数
        # 首先，输入x通过第一层线性变换和SILU激活函数
        # 然后，结果乘以输入x通过第三层线性变化的结果
        # 最后，通过第二层线性变换和dropout层
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

'''LLaMA2 Decoder Layer'''

class DecoderLayer(nn.Module):
    def __init__(self, layer_id: int, args: ModelConfig):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = MLP(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        # 定义层ID
        self.layer_id = layer_id
        #定义注意力计算的归一化层
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # 定义前馈神经网络计算的归一化层
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin):
        # 前向传播函数
        # 首先，输入x经过注意力归一化层，然后进行注意力计算，结果与输入x相加得到h
        # 然后，h经过前馈神经网络归一化层，然后进行前馈神经网络计算，结果与h相加得到输出
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin)
        out = h + self.feed_forward.forward(self.ffn_norm(h))

        return out

class Transformer(PreTrainedModel):
    config_class = ModelConfig # 配置类
    last_loss: Optional[torch.Tensor] # 记录最后一次计算的损失

    def __init__(self, args: ModelConfig = None):
        super().__init__(args)
        # 初始化 模型参数
        self.args = args
        # 词汇表大小
        self.vocab_size = args.vocab_size
        # 层数
        self.n_layer = args.n_layers

        # 词嵌入层
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

        # Dropout层
        self.dropout = nn.Dropout(args.dropout)

        # Decoder 层
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(DecoderLayer(layer_id, args))

        # 归一化层
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        # 输出层
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # 将词嵌入层的权重与输出层的权重共享
        self.tok_embeddings.weight = self.output.weight

        # 预计算相对位置嵌入的频率
        freqs_cos, freqs_sin = precompute_freqs_cis(self.args.dim // self.args.n_heads, self.args.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # 初始化所有权重
        self.apply(self._init_weights)
        # 对残差投影进行特殊的缩放初始化
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * args.n_layers))

        # 初始化最后一次前向传播的损失属性
        self.last_loss = None
        self.OUT = CausalLMOutputWithPast() # 输入容器
        self._no_split_modules = [name for name, _ in self.named_modules()]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:

        if 'input_ids' in kwargs:
            tokens = kwargs['input_ids']
        if 'attention_mask' in kwargs:
            targets = kwargs['attention_mask']

        # 前向传播函数
        _bsz, seqlen = tokens.shape
        # 通过词嵌入和Dropout层
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        # 获取相对位置嵌入的频率
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        # 通过Decoder层
        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)

        # 通过归一化层
        h = self.norm(h)

        if targets is not None:
            # 如果给定了目标，计算损失
            logits = self.output(h)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0, reduction='none')

        else:
            # 推理时的小优化: 只对最后一个位置的输出进行前向传播
            logits = self.output(h[:, [-1], :])
            self.last_loss = None

        # 设置输出
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('last_loss', self.last_loss)
        return self.OUT

    @torch.inference_mode()
    def generate(self, idx, stop_id=None, max_new_tokens=256, temperature=1.0, top_k=None):
        index = idx.shape[1]
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.args.max_seq_len else idx[:, -self.args.max_seq_len:]

            logits = self(idx_cond).logits
            logits = logits[:, -1, :]

            if temperature == 0.0:
                # 选择最有可能的索引
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[: [-1]]] = -float('Inf')

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            if idx_next == stop_id:
                break

            idx = torch.cat((idx, idx_next), dim=1)


        return idx[:, index:]
    def _greedy_decode(self, logits: torch.Tensor) -> torch.Tensor:
        '''贪婪解码： 选择概率最大的token'''
        _, idx_next = torch.topk(logits, k = 1, dim=-1)
        return idx_next

    def _random_sample(self, logits: torch.Tensor, temperature: float = 1.0, top_k : int = None) -> torch.Tensor:
        '''随机采样： 基于概率分布随机选择token'''
        logits = logits / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        return idx_next

    def _beam_search(self, idx: torch.Tensor, max_new_tokens: int, num_beams: int, temperature: float = 1.0, top_k: int = None, stop_id: int = None) -> torch.Tensor:
        '''束搜索 维护多个候选序列，选择最优路径'''
        '''在每一步生成时， 不止选择一个最佳token，而是保留多个候选路径，最终选择累计概率最高的完整序列'''
        """
        Args:
            idx: 输入序列 (batch_size, seq_len)
            max_new_tokens: 最大生成token数量
            num_beams: 束宽度，表示保留的候选路径的数量
            temperature: 温度参数，控制分布的平滑程度
            top_k: top-k过滤参数，限制候选token范围
            stop_id: 停止生成的token ID, 遇到则停止
            
        Returns:
            生成的token序列，形状为(batch_size,generated_length)
            只返回新生成的部分，不包含原始输入序列
        
        """
        # 获取输入序列的基本信息
        batch_size = idx.shape[0]
        seq_len = idx.shape[1]
        # 初始化束: 创建 num_beams 个候选序列
        beams = [idx.clone() for _ in range(num_beams)]
        # 初始化每个候选序列的累积对数概率分布
        beam_scores = torch.zeros(num_beams, device=idx.device)
        # 第一个候选是原始输入序列，分数为0
        beam_scores[0] = 0.0
        # 其他候选初始分数设为负无穷，表示尚未生成
        beam_scores[1:] = float('inf')

        # 主循环：逐步生成新的token，最多生成max_new_tokens个
        for step in range(max_new_tokens):
            # 每轮迭代收集新的候选序列和分数
            new_beams = [] # 新的候选序列列表
            new_scores = [] # 对应的分数列表

            # 遍历当前所有的候选序列
            for beam_idx, beam in enumerate(beams):
                # 跳过无效候选(分数为负无穷的序列)
                if beam_scores[beam_idx] == float('-inf'):
                    continue
                # 序列长度检查: 如果炒股共最大长度，就截取最后的部分
                beam_cond = beam if beam.size(1) <= self.args.max_seq_len else beam[:, -self.args.max_seq_len:]

                # 前向传播：获取模型对当前序列的预测
                output = self(beam_cond)
                # 提取最后一个位置的logits，用于预测下一个token
                logits = output.logits[:, -1, :] # 形状(1, vocab_size)
                # 温度缩放，调整logits的分布
                if temperature != 1.0:
                    logits = logits / temperature
                    # 温度 > 1：分布更平滑，增加随机性
                    # 温度 < 1: 分布更尖锐，更确定
                # Top-k 过滤：限制候选token范围
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    # 将不在前top_k内的logits设为负无穷
                    logits[logits < v[:, [-1]]] = -float('Inf')
                    # 这样采样时只会考虑前top_k个token

                # 计算对数概率：使用log_softmax避免数值不稳定
                log_probs = F.log_softmax(logits, dim=-1)

                # 获取前 num_beams 个最可能选择的token
                # 这里的top-k与上面的不同 上面的是全局过滤，这里的是束搜索的分支选择
                top_log_probs, top_indices = torch.topk(log_probs, k=num_beams, dim=-1)
                # 为当前候选序列生成num_beams个扩展序列
                for k in range(num_beams):
                    # 选择第k个候选token
                    token = top_indices[:, k:k+1] # token ID
                    log_prob = top_log_probs[:, k] # 对应的对数概率
                    # 扩展序列：将新token添加到当前序列的末尾
                    new_beam = torch.cat([beam, token], dim=1)
                    # 更新累积分数： 原序列分数 + 新token的对数概率
                    new_scores = beam_scores[beam_idx] + log_prob.item()
                    # 保存新的候选序列和分数
                    new_beams.append(new_beam)
                    new_scores.append(new_scores)
            if not new_beams:
                break

            # 筛选最佳候选，从所有新生成的候选中选择分数最高的num_beams个
            # 按分数降序排序， 获取索引
            sorted_indices = sorted(range(len(new_scores)), key=lambda i: new_scores[i], reverse=True)
            # 选择前num_beams 个最佳候选
            beams = [new_beams[i] for i in sorted_indices[:new_beams]]
            beam_scores = [new_scores[i] for i in sorted_indices[:new_beams]]

            # 停止条件减产：检查最佳序列是否以停止token结尾
            if stop_id is not None and beams[0][0, -1] == stop_id:
                break

        # 返回得分最高的序列，只返回新生成的部分
        # beams[0] 是最终得分最高的完整序列
        return beams[0][:, seq_len:]
    @torch.inference_mode()
    def generate_supe(self,
                      idx,
                      stop_id=None,
                      max_new_tokens=256,
                      temperature=1.0,
                      top_k=None,
                      do_sample=False,
                      num_beams=1
                      ):

        """
                高级文本生成函数，支持三种解码策略：

                1. 贪婪解码（Greedy Search）：
                   - 参数：do_sample=False, num_beams=1
                   - 特点：每步选择概率最大的token，速度快、结果确定

                2. 随机采样（Random Sampling）：
                   - 参数：do_sample=True, num_beams=1
                   - 特点：基于概率分布随机采样，可配合temperature和top-k控制多样性

                3. 束搜索（Beam Search）：
                   - 参数：do_sample=False, num_beams>1
                   - 特点：维护多条候选路径，选择总概率最高的序列，质量更高但速度较慢

                Args:
                    idx: 输入序列张量，形状为 (batch_size, seq_len)
                    stop_id: 停止生成的token ID
                    max_new_tokens: 最大生成token数量
                    temperature: 温度参数，控制随机性，越高越随机
                    top_k: 只考虑概率最高的k个token，None表示不考虑
                    do_sample: 是否使用随机采样，False时使用确定性解码
                    num_beams: 束搜索的束宽度，1表示不使用束搜索

                Returns:
                    生成的token序列，形状为 (batch_size, generated_length)
        """

        # 参数验证
        if temperature <= 0:
            temperature = 0.001
        if num_beams < 1:
            num_beams = 1
        if top_k is not None and top_k < 1:
            top_k = None

        # 束搜索逻辑
        if not do_sample and num_beams > 1:
            return self._beam_search(idx, max_new_tokens, num_beams, temperature, top_k, stop_id)
        # 贪婪解码和随机采样逻辑
        index = idx.shape[1]
        for _ in range(max_new_tokens):
            # 如果序列上下文过长，截断它到最大长度
            idx_cond = idx if idx.size(1) <= self.args.max_seq_len else idx[:, -self.args.max_seq_len:]

            # 前向传播获取序列中最后一个位置的logits
            logits = self(idx_cond).logits
            logits = logits[:, -1, :]
            # 根据参数选择解码策略
            if do_sample:
                idx_next = self._random_sample(logits, temperature, top_k)
            else:
                # 当temperature=0时使用贪婪解码
                if temperature < 0.1:
                    idx_next = self._greedy_decode(logits)
                else:
                    idx_next = self._random_sample(logits, temperature, top_k)
            # 停止检查条件
            if stop_id is not None and idx_next[0, 0] == stop_id:
                break

            # 将选择的token添加到序列中
            idx = torch.cat((idx, idx_next), dim=1)

        return idx[:, index:] # 只返回生成的token




# LLaMA2Model.forward 接受两个参数，tokens和targets，其中tokens是输入的张量, 应为int类型
x = torch.randint(0, 6144, (1, 50)) # [bs, seq_len]
# 实例化LLaMA2Model
model = Transformer(args=args)
# 计算model的全部参数
num_params = sum(p.numel() for p in model.parameters())
print('Number of parameters:', num_params)

out = model(x)
print(out.logits.shape) # [batch_size, 1, vocab_size]







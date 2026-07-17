import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

class TransformerEncoderLayer(nn.Module):

    def __init__(self, embed_dim=1936, nhead=4, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)
        # 通过多个“头”并行地计算注意力分数，能够更好地捕捉输入序列中不同部分之间的关系。
        # nhead=8，那么每个输入向量就会被划分成 8 个子向量，每个子向量会进行独立的注意力计算，最终将 8 个结果拼接起来作为输出。

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)  # 1936 2048
        self.dropout = nn.Dropout(dropout)  # 训练过程中会随机丢弃 10% 的神经元，以防止过拟合
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)  # 2048 1936

        self.norm1 = nn.LayerNorm(embed_dim)  # 对输出进行标准化，帮助稳定训练过程，确保每一层的输入分布一致，防止梯度爆炸或梯度消失
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, input_key_padding_mask):
        # local attention
        src2, local_attention_weights = self.self_attn(src, src, src, key_padding_mask=input_key_padding_mask)

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(nn.functional.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src, local_attention_weights


class TransformerDecoderLayer(nn.Module):

    def __init__(self, embed_dim=1936, nhead=4, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        self.multihead2 = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)


        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, global_input, input_key_padding_mask, position_embed):

        tgt2, global_attention_weights = self.multihead2(query=global_input+position_embed, key=global_input+position_embed,
                                                         value=global_input, key_padding_mask=input_key_padding_mask)
        tgt = global_input + self.dropout2(tgt2)
        tgt = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(nn.functional.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt, global_attention_weights


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)  # num_layers=1
        self.num_layers = num_layers

    def forward(self, input, input_key_padding_mask):
        output = input
        weights = torch.zeros([self.num_layers, output.shape[1], output.shape[0], output.shape[0]]).to(output.device)

        for i, layer in enumerate(self.layers):
            output, local_attention_weights = layer(output, input_key_padding_mask)
            weights[i] = local_attention_weights
        if self.num_layers > 0:
            return output, weights
        else:
            return output, None


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, embed_dim):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers


    def forward(self, global_input, input_key_padding_mask, position_embed):

        output = global_input
        weights = torch.zeros([self.num_layers, output.shape[1], output.shape[0], output.shape[0]]).to(output.device)

        for i, layer in enumerate(self.layers):
            output, global_attention_weights = layer(output, input_key_padding_mask, position_embed)
            weights[i] = global_attention_weights

        if self.num_layers>0:
            return output, weights
        else:
            return output, None

class FreqGate(nn.Module):
    """
    将“类频率先验”摘要成一个条件向量，再映射为通道级门控 g∈(0,1)^D
    - 不依赖实例类别后验
    - 仅用 freq_tensor 的统计特征作为条件
    """
    def __init__(self, d_model: int, hidden: int = 64, channel_wise: bool = True, tau: float = 1.0):
        super().__init__()
        self.out_dim = d_model if channel_wise else 1
        self.tau = tau  # 温度（越小 gate 趋近 0/1，越大更平滑）
        # 6维统计: mean/min/max/std/entropy/tailness
        self.mlp = nn.Sequential(
            nn.Linear(6, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.out_dim)
        )

    def forward(self, freq_tensor: torch.Tensor) -> torch.Tensor:
        """
        freq_tensor: [C]，类频率（原始计数或归一化概率都可）
        return: g ∈ (0,1)^{D} 或 标量(1)
        """
        eps = 1e-6
        pi = freq_tensor.float()
        # 归一化到概率
        pi = (pi + eps) / (pi.sum() + eps * pi.numel())

        # 统计量
        mean = pi.mean()  # 平均值
        minv = pi.min()  # 最小值
        maxv = pi.max()  # 最大值
        std  = pi.std(unbiased=False)  # 标准差（无偏）
        entropy  = -(pi * torch.log(pi + eps)).sum()  # 熵（信息量）
        tailness = torch.log(mean / (pi + eps)).mean()  # 尾部性（tailness），衡量尾类的稀疏性

        cond = torch.stack([mean, minv, maxv, std, entropy, tailness], dim=0)  # [6]

        # MLP → 门控，带温度
        logits = self.mlp(cond) / self.tau                          # [D] 或 [1]
        g = torch.sigmoid(logits)                                   # (0,1)
        return g


class FrequencyAwareTransformerEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=1, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.freq_gate = FreqGate(d_model=d_model, hidden=64, channel_wise=True, tau=1.0)


    def forward(self, src, key_padding_mask=None, pos=None, freq_tensor=None):
        """
        src: [L, B, D]
        key_padding_mask: [B, L] with bool dtype
        pos: [B, L, D] (optional)
        freq_tensor: [B, L] or [N], low value = tail class (will be emphasized)
        """
        x = src
        if pos is not None:
            x = x + pos

        # Transformer Encoder
        x = self.encoder(x, src_key_padding_mask=key_padding_mask.T)  # [L, B, D] nn.TransformerEncoder

        # === Frequency-aware scaling ===
        if freq_tensor is not None:
            # 1) 计算门控（通道级或标量）
            g = self.freq_gate(freq_tensor).to(x.device)  # [D] 或 [1]
            g = g.view(1, 1, -1)                          # 便于广播到 [L,B,D]

            # 2) 备选分支：无参 LayerNorm 做“尾类友好”的稳定化表示，对张量 x 的最后一维做归一化
            x_tail = torch.nn.functional.layer_norm(x, normalized_shape=(x.size(-1),))

            # 3) 仅在有效 token 上进行融合；padding 位置保持不变
            if key_padding_mask is not None:
                valid = (~key_padding_mask).transpose(0, 1).unsqueeze(-1)  # [L,B,1]
                x_new = g * x_tail + (1.0 - g) * x
                x = torch.where(valid, x_new, x)  # True 的位置用 x_new，否则用原来的 x
            else:
                x = g * x_tail + (1.0 - g) * x

        return x


class FrequencyAwareTransformerDecoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=1, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.freq_gate = FreqGate(d_model=d_model, hidden=64, channel_wise=True, tau=1.0)

    def forward(self, tgt, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None,
                pos=None, memory_pos=None, freq_tensor=None):
        """
        tgt: [L, B, D] - decoder input
        memory: [M, B, D] - encoder output
        pos: positional encoding for tgt
        memory_pos: positional encoding for memory
        freq_tensor: [B, L] - frequency weight on tgt side
        """
        if pos is not None:
            tgt = tgt + pos
        if memory_pos is not None:
            memory = memory + memory_pos

        x = self.decoder(tgt=tgt, memory=memory,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        # === Frequency-aware scaling ===
        if freq_tensor is not None:
            # 1) 计算门控（通道级或标量）
            g = self.freq_gate(freq_tensor).to(x.device)  # [D] 或 [1]
            g = g.view(1, 1, -1)                          # 便于广播到 [L,B,D]

            # 2) 备选分支：无参 LayerNorm 做“尾类友好”的稳定化表示
            x_tail = torch.nn.functional.layer_norm(x, normalized_shape=(x.size(-1),))

            # 3) 仅在有效 token 上进行融合；padding 位置保持不变
            if tgt_key_padding_mask is not None:
                valid = (~tgt_key_padding_mask).transpose(0, 1).unsqueeze(-1)  # [L,B,1]
                x_new = g * x_tail + (1.0 - g) * x
                x = torch.where(valid, x_new, x)
            else:
                x = g * x_tail + (1.0 - g) * x

        return x

class transformer(nn.Module):
    ''' Spatial Temporal Transformer
        local_attention: spatial encoder
        global_attention: temporal decoder
        position_embedding: frame encoding (window_size*dim)
        mode: both--use the features from both frames in the window
              latter--use the features from the latter frame in the window
    '''
    def __init__(self, enc_layer_num=1, dec_layer_num=3, embed_dim=1936, nhead=8, dim_feedforward=2048,
                 dropout=0.1, mode=None, mem_compute=True, mem_fusion=None, selection=None, selection_lambda=0.5):
        super(transformer, self).__init__()
        self.mode = mode
        self.mem_fusion = mem_fusion
        self.mem_compute = mem_compute
        self.selection = selection
        
        encoder_layer = TransformerEncoderLayer(embed_dim=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.local_attention = TransformerEncoder(encoder_layer, enc_layer_num)
        self.local_attention_tail = FrequencyAwareTransformerEncoder(d_model=embed_dim, nhead=nhead, num_layers=enc_layer_num, dropout=dropout)

        if mem_compute:
            if mem_compute == 'seperate':
                self.mem_attention = nn.ModuleDict()

                for rel in ['attention','contacting','spatial']:
                    self.mem_attention.update({rel: nn.MultiheadAttention(embed_dim, 1, 0.0, bias=False)})

            else:
                self.mem_attention = nn.MultiheadAttention(embed_dim, 1, 0.0, bias=False)
            if selection == 'manual':
                self.selector = selection_lambda  # 0.5
            else:
                self.selector = nn.Linear(embed_dim,1)
            

        decoder_layer = TransformerDecoderLayer(embed_dim=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.global_attention = TransformerDecoder(decoder_layer, dec_layer_num, embed_dim)
        self.global_attention_tail = FrequencyAwareTransformerDecoder(d_model=embed_dim, nhead=nhead, num_layers=dec_layer_num, dropout=dropout)

        self.position_embedding = nn.Embedding(2, embed_dim)  # present and next frame
        nn.init.uniform_(self.position_embedding.weight)  # 对层的权重进行均匀分布初始化

        # === Fusion Gate ===
        self.fusion_gate_local = nn.Sequential(nn.Linear(embed_dim * 2 + 64, embed_dim), nn.Sigmoid())
        self.fusion_gate_global = nn.Sequential(nn.Linear(embed_dim * 2 + 64, embed_dim), nn.Sigmoid())
        self.freq_mapper = nn.ModuleDict({'attention': nn.Linear(3, 64),
                                          'spatial': nn.Linear(6, 64),
                                          'contact': nn.Linear(17, 64),})

    def memory_hallucinator(self, memory, feat, freq_tensor=None, rel_type=None):
        if len(memory) != 0:
            e = self.mem_selection(feat)
            q = feat.unsqueeze(1)
            
            if self.mem_compute == 'seperate':
                mem_features = {}
                for rel in ['attention','contacting','spatial']:
                    k = v = memory[rel].unsqueeze(1)
                    mem_features[rel],_ = self.mem_attention[rel](q,k,v)
                mem_features = torch.cat([v for k,v in mem_features.items()], dim=1)
                mem_features = mem_features.mean(dim=1)
            else:
                memory = torch.cat([v for k,v in memory.items()], dim=0)
                k = v = memory.unsqueeze(1)
                mem_features,_ = self.mem_attention(q,k,v)

            # new
            mem_features = mem_features.squeeze(1)
            # === freq_tensor modulation ===
            if freq_tensor is not None and rel_type is not None:
                if freq_tensor.dim() == 1:
                    if freq_tensor.shape[0] != feat.shape[0]:
                        freq_tensor = F.pad(freq_tensor, (0, feat.shape[0] - freq_tensor.shape[0]))[:feat.shape[0]]
                    freq_tensor = freq_tensor.unsqueeze(-1)
                freq_bias = torch.log(1.0 / (freq_tensor + 1e-6))
                mem_features = mem_features * (1.0 + freq_bias)
            # new end
           
            if e is not None:
                mem_encoded_features = e*feat + (1-e)*mem_features.squeeze(1)
            else:
                mem_encoded_features = feat + mem_features.squeeze(1)
            # mem_encoded_features = feat + e*mem_features.squeeze(1)
        else:
            mem_encoded_features = feat

        return mem_encoded_features

    def mem_selection(self, feat):
        if self.selection == 'manual':
            return self.selector
        else:
            return self.selector(feat).sigmoid()

    def build_global_inputs(self, mem_encoder_features, im_idx, position_embedding_weight):
        """
        构建 global_input、position_embed、global_masks、idx
        和【1】语义对齐，且更健壮：
          - b 用 im_idx.max()+1 计算，不要求排序
          - position_embed 逐行按 im_idx 映射到 pos[0]/pos[1]，不依赖“先 j 后 j+1”的顺序假设
          - global_masks 直接由 global_input 是否为全零行得到（True=padding）
        """
        device = mem_encoder_features.device
        im_idx = im_idx.to(torch.long)
        feature_dim = mem_encoder_features.shape[1]

        # 计算帧数（不依赖 im_idx 排序）
        b = int(im_idx.max().item() + 1)
        if b <= 1:
            # 没有可滑动的 pair，返回空形状但类型一致的张量
            empty = mem_encoder_features.new_zeros((0, 0, feature_dim))
            empty_mask = torch.ones((0, 0), dtype=torch.bool, device=device)
            empty_idx = torch.full((0, 0), -1, device=device, dtype=torch.long)
            return empty, empty, empty_mask, empty_idx

        # 所有滑动窗口 pair: (j, j+1)
        frame_pairs = [(j, j + 1) for j in range(b - 1)]

        # 为每个 pair 收集索引（保持原顺序）
        pair_indices = []
        pair_lens = []
        for fj, fk in frame_pairs:
            idx_pair_mask = (im_idx == fj) | (im_idx == fk)
            idxs = torch.nonzero(idx_pair_mask, as_tuple=True)[0]  # 按原序
            pair_indices.append(idxs)
            pair_lens.append(int(idxs.numel()))

        # 构造 global_input：按最长序列 pad 到同一长度（padding_value=0 与【1】一致）
        global_seqs = [
            mem_encoder_features[idxs] if len(idxs) > 0
            else mem_encoder_features.new_zeros((0, feature_dim))
            for idxs in pair_indices
        ]
        global_input = pad_sequence(global_seqs, batch_first=False, padding_value=0.0)  # [max_N, num_pairs, D]

        # 构造 position_embed：逐行根据 im_idx[idxs] 判断属于 fj 还是 fk
        position_seqs = []
        pos0 = position_embedding_weight[0].unsqueeze(0)  # [1, D]
        pos1 = position_embedding_weight[1].unsqueeze(0)  # [1, D]
        for p, idxs in enumerate(pair_indices):
            fj, fk = frame_pairs[p]
            if len(idxs) == 0:
                pos_rows = mem_encoder_features.new_zeros((0, feature_dim))
            else:
                im_this = im_idx[idxs]  # [Lp]
                # True -> pos0，False -> pos1
                is_fj = (im_this == fj).unsqueeze(1)             # [Lp, 1]
                pos_rows = torch.where(
                    is_fj,
                    pos0.expand(im_this.shape[0], -1),           # [Lp, D]
                    pos1.expand(im_this.shape[0], -1)            # [Lp, D]
                )
            position_seqs.append(pos_rows)
        position_embed = pad_sequence(position_seqs, batch_first=False, padding_value=0.0)  # [max_N, num_pairs, D]

        # 构造 idx：缺省为 -1，前 Lp 行写入对应的 im_idx
        max_N = global_input.shape[0]
        num_pairs = global_input.shape[1]
        idx = torch.full((max_N, num_pairs), -1, device=device, dtype=torch.long)
        for p, idxs in enumerate(pair_indices):
            Lp = len(idxs)
            if Lp > 0:
                idx[:Lp, p] = im_idx[idxs]

        # 构造 mask（与【1】等价：True=padding, False=有效）
        # 【1】做法：对每行特征求和==0 判定 padding；这里用绝对值和更稳妥
        global_masks = (global_input.abs().sum(dim=2) == 0).permute(1, 0)  # [num_pairs, max_N]

        return global_input, position_embed, global_masks, idx

    def forward(self, features, im_idx, memory=None, rel_type=None, freq_tensor=None):
        if memory is None:
            memory = []
        rel_idx = torch.arange(im_idx.shape[0])
        l = torch.sum(im_idx == torch.mode(im_idx)[0])  # 计算 im_idx 中出现次数最多的元素（众数）的数量
        b = int(im_idx[-1] + 1) # 计算 im_idx 中的不同元素的数量，这里应该是video中frame的数量，因为索引是从 0 开始，所以需要加 1
        rel_input = torch.zeros([l, b, features.shape[1]]).to(features.device)  # 全0，[每帧中最大pair对儿max_N, num_frames, D(1936)]
        masks = torch.zeros([b, l], dtype=torch.bool).to(features.device)  # [num_frames, 每帧中最大pair对儿max_N], masks 用于标记哪些位置是有效的 box，哪些是填充（padding），在 transformer 的注意力机制中用于屏蔽无效位置
        if freq_tensor is not None:
            mapped_freq = self.freq_mapper[rel_type](freq_tensor.unsqueeze(0))  #  将freq统一成1 * 64维

        for i in range(b):
            rel_input[:torch.sum(im_idx == i), i, :] = features[im_idx == i]  # 将每对儿关系特征按照帧的顺序填充到 rel_input 中，少于 max_N 的部分用 0 填充
            masks[i, torch.sum(im_idx == i):] = 1  # 将 masks 中对应的填充部分标记为 1（有效为 0），用于 transformer 的注意力机制

        # spatial encoder
        local_output, local_attention_weights = self.local_attention(rel_input, masks)  # 进行局部注意力计算
        local_output = (local_output.permute(1, 0, 2)).contiguous().view(-1, features.shape[1])[masks.view(-1) == 0]  # 将 local_head 转置并展平为 [有效的pair对儿数量, D(1936)]
        if freq_tensor is not None:
            local_head = local_output
            local_tail = self.local_attention_tail(rel_input, masks, freq_tensor=freq_tensor)
            local_tail = local_tail.permute(1, 0, 2).contiguous().view(-1, features.shape[1])[masks.view(-1) == 0]
            # 归一化对齐
            local_h = F.layer_norm(local_head, (local_head.size(-1),))
            local_t = F.layer_norm(local_tail, (local_tail.size(-1),))
            freq_tensor_expanded = mapped_freq.view(1, -1).expand(local_head.shape[0], -1)
            # 频率条件标准化
            freq_feat = F.layer_norm(freq_tensor_expanded, (freq_tensor_expanded.size(-1),))
            # 门控输入
            fusion_input_local = torch.cat([local_h, local_t, freq_feat], dim=-1)
            # 门控（带温度与夹紧）
            logits = self.fusion_gate_local[0](fusion_input_local)   # Linear(2D+64 → D)
            T = 1.5
            gate = torch.sigmoid(logits / T)
            gate = torch.clamp(gate, 0.02, 0.98)
            # 稳定融合（残差式）
            local_output = local_h + (1 - gate) * (local_t - local_h)

        if self.mem_compute and self.mem_fusion == 'early':
            mem_encoder_features = self.memory_hallucinator(memory=memory, feat=local_output)
        else:
            mem_encoder_features = local_output

        # --- Global Attention ---
        global_input = torch.zeros([l * 2, b - 1, features.shape[1]]).to(features.device)
        position_embed = torch.zeros([l * 2, b - 1, features.shape[1]]).to(features.device)
        idx = -torch.ones([l * 2, b - 1]).to(features.device)
        idx_plus = -torch.ones([l * 2, b - 1], dtype=torch.long).to(features.device) #TODO

        # sliding window size = 2
        for j in range(b - 1):
            global_input[:torch.sum((im_idx == j) + (im_idx == j + 1)), j, :] = mem_encoder_features[(im_idx == j) + (im_idx == j + 1)]
            idx[:torch.sum((im_idx == j) + (im_idx == j + 1)), j] = im_idx[(im_idx == j) + (im_idx == j + 1)]
            idx_plus[:torch.sum((im_idx == j) + (im_idx == j + 1)), j] = rel_idx[(im_idx == j) + (im_idx == j + 1)] #TODO

            position_embed[:torch.sum(im_idx == j), j, :] = self.position_embedding.weight[0]
            position_embed[torch.sum(im_idx == j):torch.sum(im_idx == j)+torch.sum(im_idx == j+1), j, :] = self.position_embedding.weight[1]

        global_masks = (torch.sum(global_input.view(-1, features.shape[1]),dim=1) == 0).view(l * 2, b - 1).permute(1, 0)
        # temporal decoder
        global_output, global_attention_weights = self.global_attention(global_input, global_masks, position_embed)
#         print(global_output.shape)
        output = torch.zeros_like(features)

        if self.mode == 'both':
            # both
            for j in range(b - 1):
                if j == 0:
                    output[im_idx == j] = global_output[:, j][idx[:, j] == j]
                if j == b - 2:
                    output[im_idx == j+1] = global_output[:, j][idx[:, j] == j+1]
                else:
                    output[im_idx == j + 1] = (global_output[:, j][idx[:, j] == j + 1] + global_output[:, j + 1][idx[:, j + 1] == j + 1]) / 2

        elif self.mode == 'latter':
            # later
            for j in range(b - 1):
                if j == 0:
                    output[im_idx == j] = global_output[:, j][idx[:, j] == j]
                output[im_idx == j + 1] = global_output[:, j][idx[:, j] == j + 1]

        if self.mem_compute and self.mem_fusion == 'late':
            local_output = output
            output = self.memory_hallucinator(memory=memory, feat=output)
            mem_encoder_features = output
         
        return output,local_output,mem_encoder_features, global_attention_weights, local_attention_weights


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


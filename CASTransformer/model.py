import random
import math
from kan import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

MIN_SEQ_LEN = 5

class CASTransformer(nn.Module):
    def __init__(
        self,
        n_questions,
        n_pid=0,
        d_model=128,
        d_fc=256,
        n_heads=8,
        n_layers=1,
        dropout=0.05,
        lambda_fcl=0.1,
        lambda_bcl=0.1,
        proj=False,
        hard_neg=True,
        window=1,
        shortcut=False,
        use_d_correct=False,
        use_d_skill_correct=False,
        kan_output_dim=2,
    ):
        super().__init__()
        self.n_questions = n_questions
        self.n_pid = n_pid
        self.d_model = d_model

        self.q_embed = nn.Embedding(n_questions + 1, d_model)
        self.s_embed = nn.Embedding(2, d_model)

        self.d_correct_projection = nn.Linear(kan_output_dim, d_model)
        self.use_d_correct = use_d_correct
        self.use_d_skill_correct = use_d_skill_correct

        if n_pid > 0:
            self.q_diff_embed = nn.Embedding(n_questions + 1, d_model)
            self.s_diff_embed = nn.Embedding(2, d_model)
            self.p_diff_embed = nn.Embedding(n_pid + 1, d_model)

        self.kan_kt = KAN([2, 5, 2], grid=20, k=3, grid_eps=1.0)

        if use_d_correct:
            self.d_correct_fc = nn.Linear(1, d_model)
        if use_d_skill_correct:
            self.d_skill_correct_fc = nn.Linear(1, d_model)

        self.n_heads = n_heads
        self.block1 = CASTransformerLayer(d_model, n_heads, dropout)
        self.block2 = CASTransformerLayer(d_model, n_heads, dropout)
        self.block3 = CASTransformerLayer(d_model, n_heads, dropout)
        self.block4 = CASTransformerLayer(d_model, n_heads, dropout, kq_same=False)
        self.block5 = CASTransformerLayer(d_model, n_heads, dropout)

        self.know_params = nn.Parameter(torch.empty(n_questions, d_model))
        torch.nn.init.uniform_(self.know_params, -1.0, 1.0)

        self.out = nn.Sequential(
            nn.Linear(d_model * 2, d_fc),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_fc, d_fc // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_fc // 2, 1)
        )

        if proj:
            self.proj = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
        else:
            self.proj = None

        self.dropout_rate = dropout
        self.lambda_fcl = lambda_fcl
        self.lambda_bcl = lambda_bcl
        self.hard_neg = hard_neg
        self.shortcut = shortcut
        self.n_layers = n_layers
        self.window = window

        self.bilinear_skill_correct = BilinearInteraction(d_model)
        if use_d_correct:
            self.bilinear_d_correct = BilinearInteraction(d_model)

        # 增加用于对比学习的投影头 投影头用于将原始的表示映射到对比学习空间
        self.fcl_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        self.bcl_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, q_emb, s_emb, lens):
        if self.shortcut:
            hq, _ = self.block1(q_emb, q_emb, q_emb, lens, peek_cur=True)
            hs, scores = self.block2(s_emb, s_emb, s_emb, lens, peek_cur=True)
            output, _ = self.block3(hq, hq, hs, lens, peek_cur=False)
            bs, seqlen, d_model = output.size()
            z = output
            q_scores = scores
            k_scores = None
            return z, q_scores, k_scores

        if self.n_layers == 1:
            hq = q_emb
            p, q_scores = self.block1(q_emb, q_emb, s_emb, lens, peek_cur=True)
        elif self.n_layers == 2:
            hq = q_emb
            hs, _ = self.block1(s_emb, s_emb, s_emb, lens, peek_cur=True)
            p, q_scores = self.block2(hq, hq, hs, lens, peek_cur=True)
        else:
            hq, _ = self.block1(q_emb, q_emb, q_emb, lens, peek_cur=True)
            hs, _ = self.block2(s_emb, s_emb, s_emb, lens, peek_cur=True)
            p, q_scores = self.block3(hq, hq, hs, lens, peek_cur=True)

        bs, seqlen, d_model = p.size()
        n_questions = self.n_questions

        query = self.know_params[None, :, None, :].expand(bs, -1, seqlen, -1).contiguous()


        query = query.view(bs * n_questions, seqlen, d_model)


        hq_expanded = hq.unsqueeze(1).expand(-1, n_questions, -1, -1).reshape_as(query)
        p_expanded = p.unsqueeze(1).expand(-1, n_questions, -1, -1).reshape_as(query)


        z, k_scores = self.block4(
            query, hq_expanded, p_expanded, torch.repeat_interleave(lens, n_questions), peek_cur=False
        )


        if z.numel() == 0:
            return None, None, None

        # z = z.view(bs, n_questions, seqlen, d_model).transpose(1, 2).contiguous().view(bs, seqlen, -1)
        z = z.view(bs, n_questions, seqlen, d_model).transpose(1, 2).contiguous()

        k_scores = (
            k_scores.reshape(bs, n_questions, self.n_heads, seqlen, -1)
            .permute(0, 2, 3, 1, 4).contiguous()
        )

        return z, q_scores, k_scores

    def embedding_kan(self, q, s, pid=None, d_correct=None, d_skill_correct=None, n=1):
        device = next(self.parameters()).device
        q = q.to(device)
        s = s.to(device)
        if pid is not None:
            pid = pid.to(device)
        if d_correct is not None:
            d_correct = d_correct.to(device)
        if d_skill_correct is not None:
            d_skill_correct = d_skill_correct.to(device)

        q_emb, s_emb, lens, p_diff_emb = self.embedding(q, s, pid)

        if not isinstance(p_diff_emb, torch.Tensor):
            p_diff_emb = torch.tensor(p_diff_emb, device=device)

        if self.use_d_skill_correct and d_skill_correct is not None:
            d_skill_correct = d_skill_correct.unsqueeze(-1)
            d_skill_correct_emb = self.d_skill_correct_fc(d_skill_correct)
            batch_size, seq_len, features = d_skill_correct_emb.shape
            d_skill_correct_emb_2d = d_skill_correct_emb.view(batch_size * seq_len, features)
            d_skill_correct_emb = self.kan_kt(d_skill_correct_emb_2d)

            d_skill_correct_emb = self.d_correct_projection(d_skill_correct_emb)
            q_emb = self.bilinear_skill_correct(q_emb, d_skill_correct_emb)

        if self.n_pid > 0 and pid is not None:
            if self.use_d_correct and d_correct is not None:
                d_correct_emb = self.d_correct_fc(d_correct.unsqueeze(-1))
                batch_size, seq_len, features = d_correct_emb.shape
                d_correct_emb_2d = d_correct_emb.view(batch_size * seq_len, features)
                d_correct_emb = self.kan_kt(d_correct_emb_2d)

                d_correct_emb = self.d_correct_projection(d_correct_emb)
                p_diff_emb = self.bilinear_d_correct(p_diff_emb, d_correct_emb)

        return q_emb, s_emb, lens, p_diff_emb

    def embedding(self, q, s, pid=None):
        q = q.long()
        s = s.long()

        lens = (s >= 0).sum(dim=1)
        q = q.masked_fill(q < 0, 0)
        s = s.masked_fill(s < 0, 0)

        q_emb = self.q_embed(q)
        s_emb = self.s_embed(s) + q_emb

        p_diff = torch.tensor(0.0, device=q.device)

        if self.n_pid > 0 and pid is not None:
            pid = pid.long()
            pid = pid.masked_fill(pid < 0, 0)
            p_diff = self.p_diff_embed(pid)
            q_diff_emb = self.q_diff_embed(q)
            q_emb += q_diff_emb * p_diff
            s_diff_emb = self.s_diff_embed(s) + q_diff_emb
            s_emb += s_diff_emb * p_diff

        return q_emb, s_emb, lens, p_diff

    def readout(self, z, query):
        bs, seqlen, _ = query.size()
        key = (
            self.know_params[None, None, :, :]
            .expand(bs, seqlen, -1, -1)
            .view(bs * seqlen, self.n_questions, -1)
        )
        value = z.reshape(bs * seqlen, self.n_questions, -1)

        beta = torch.matmul(
            key,
            query.reshape(bs * seqlen, -1, 1),
        ).view(bs * seqlen, 1, self.n_questions)

        alpha = torch.softmax(beta, -1)
        result = torch.matmul(alpha, value).view(bs, seqlen, -1)

        return result

    def predict(self, q, s, pid=None, d_correct=None, d_skill_correct=None, n=1):
        q_emb, s_emb, lens, p_diff_emb = self.embedding_kan(q, s, pid, d_correct, d_skill_correct, n)

        z, q_scores, k_scores = self(q_emb, s_emb, lens)

        if self.shortcut:
            assert n == 1
            h = z
            query = q_emb[:, n - 1:, :]
        else:
            query = q_emb[:, n - 1:, :]
            h = self.readout(z[:, : query.size(1), :], query)

        y = self.out(torch.cat([query, h], dim=-1)).squeeze(-1)

        if y.dim() == 0:
            y = y.unsqueeze(0)

        if pid is not None:
            p_diff = (p_diff_emb ** 2).mean() * 1e-3

            return y, z, q_emb, p_diff, (q_scores, k_scores)
        else:
            p_diff = torch.tensor(0.0, device=y.device)

        if y.size(1) >= 2:
            with torch.no_grad():
                # 计算当前时间步的预测概率
                y_prob = torch.sigmoid(y)

                # 获取前一个时间步的预测概率和实际标签
                y_prev = y_prob[:, :-1]  # [batch_size, seq_len - 1]
                s_prev = s[:, :-1].float()  # [batch_size, seq_len - 1]

                # 计算前一个时间步的预测是否正确
                prev_correct = ((y_prev >= 0.5) == s_prev).float()  # [batch_size, seq_len - 1]


            return y, z, q_emb, p_diff, (q_scores, k_scores)

    def get_loss(self, q, s, pid=None, d_correct=None, d_skill_correct=None):
        logits, _, _, reg_loss, _ = self.predict(q, s, pid, d_correct, d_skill_correct)
        masked_labels = s[s >= 0].float()
        masked_logits = logits[s >= 0]
        return (
            F.binary_cross_entropy_with_logits(
                masked_logits, masked_labels, reduction="mean"
            )
            + reg_loss
        )

    def get_cl_loss(self, q, s, pid=None, n=1, d_correct=None, d_skill_correct=None):
        lens = (s >= 0).sum(dim=1)
        minlen = lens.min().item()

        if minlen < MIN_SEQ_LEN:
            return self.get_loss(q, s, pid, d_correct, d_skill_correct)

        # 获取原始样本的输出
        logits, z_1, q_emb, reg_loss, _ = self.predict(q, s, pid, d_correct, d_skill_correct)
        masked_logits = logits[s >= 0]

        # 数据增强，生成正样本和负样本
        q_aug, s_aug = self.data_augmentation(q, s)
        _, z_2, _, _, _ = self.predict(q_aug, s_aug, pid, d_correct, d_skill_correct)

        # 特征级对比学习损失
        fcl_loss = self.feature_contrastive_loss(z_1, z_2)

        # 批次级对比学习损失
        bcl_loss = self.batch_contrastive_loss(z_1)

        # 预测损失
        masked_labels = s[s >= 0].float()
        pred_loss = F.binary_cross_entropy_with_logits(
            masked_logits, masked_labels, reduction="mean"
        )

        # 总损失
        total_loss = pred_loss + self.lambda_fcl * fcl_loss + self.lambda_bcl * bcl_loss + reg_loss

        lam_fcl_loss = fcl_loss / (fcl_loss + bcl_loss)
        lam_bcl_loss = bcl_loss / (fcl_loss + bcl_loss)

        cl_loss = fcl_loss * lam_fcl_loss + bcl_loss * lam_bcl_loss

        return total_loss, pred_loss, cl_loss

    # 数据增强
    def data_augmentation(self, q, s):
        q_aug = q.clone()
        s_aug = s.clone()

        bs, seqlen = q.size()
        for b in range(bs):
            idx = random.sample(
                range(seqlen), max(1, int(seqlen * self.dropout_rate))
            )
            for i in idx:
                q_aug[b, i] = random.randint(0, self.n_questions)
                s_aug[b, i] = random.randint(0, 1)
        return q_aug, s_aug

    # 特征级对比学习
    def feature_contrastive_loss(self, z1, z2):
        bs, seqlen, n_questions, d_model = z1.size()
        N = bs * seqlen * n_questions
        temperature = 0.07

        # 将 z1 和 z2 展平成 [N, d_model]
        z1_flat = z1.view(N, d_model)
        z2_flat = z2.view(N, d_model)

        # 投影并归一化
        z1_proj = F.normalize(self.fcl_projection(z1_flat), dim=-1)
        z2_proj = F.normalize(self.fcl_projection(z2_flat), dim=-1)

        # 创建相似度矩阵占位符
        sim_matrix = torch.empty(N, N, device=z1.device)

        # 定义分批计算的 batch_size
        batch_size = 64  # 根据显存容量调整

        # 分批计算相似度矩阵
        for i in range(0, N, batch_size):
            end_i = min(i + batch_size, N)
            z1_batch = z1_proj[i:end_i]  # [batch_size, d_model]

            # 计算当前批次的相似度
            sim_batch = torch.matmul(z1_batch, z2_proj.T) / temperature  # [batch_size, N]
            sim_matrix[i:end_i] = sim_batch  # 将结果放入相似度矩阵

        # 创建标签，正样本为对角线元素
        labels = torch.arange(N).to(z1.device)

        # 排除自身（正样本）
        sim_matrix.fill_diagonal_(-float('inf'))

        # 为每个样本选取 k 个困难负样本
        k = 16  # 困难负样本的数量，可根据显存容量调整
        _, hard_neg_indices = torch.topk(sim_matrix, k, dim=1)  # [N, k]

        # 构造 logits 和标签
        pos_logits = (z1_proj * z2_proj).sum(dim=-1, keepdim=True) / temperature  # [N, 1]
        neg_logits = sim_matrix[
            torch.arange(N).unsqueeze(1), hard_neg_indices
        ]  # [N, k]

        logits = torch.cat([pos_logits, neg_logits], dim=1)  # [N, k + 1]
        labels = torch.zeros(N, dtype=torch.long).to(z1.device)  # 正样本标签为 0

        # 计算损失
        loss = F.cross_entropy(logits, labels, reduction='mean')

        return loss

    def batch_contrastive_loss(self, z):
        bs, seqlen, n_questions, d_model = z.size()
        N = bs * seqlen * n_questions
        temperature = 0.07

        z_flat = z.view(N, d_model)

        # 投影并归一化
        z_proj = F.normalize(self.bcl_projection(z_flat), dim=-1)  # [N, d_model]

        # 创建相似度矩阵占位符
        sim_matrix = torch.empty(N, N, device=z.device)

        # 定义分批计算的 batch_size
        batch_size = 64  # 根据显存容量调整

        # 分批计算相似度矩阵
        for i in range(0, N, batch_size):
            end_i = min(i + batch_size, N)
            z_batch = z_proj[i:end_i]  # [batch_size, d_model]

            # 计算当前批次的相似度
            sim_batch = torch.matmul(z_batch, z_proj.T) / temperature  # [batch_size, N]
            sim_matrix[i:end_i] = sim_batch  # 将结果放入相似度矩阵

        # 创建标签，正样本为自身
        labels = torch.arange(N).to(z.device)

        # 排除自身（正样本）
        sim_matrix.fill_diagonal_(-float('inf'))

        # 为每个样本选取 k 个困难负样本
        k = 16  # 困难负样本的数量
        _, hard_neg_indices = torch.topk(sim_matrix, k, dim=1)  # [N, k]

        # 构造 logits 和标签
        pos_logits = (z_proj * z_proj).sum(dim=-1, keepdim=True) / temperature  # [N, 1]
        neg_logits = sim_matrix[
            torch.arange(N).unsqueeze(1), hard_neg_indices
        ]  # [N, k]

        logits = torch.cat([pos_logits, neg_logits], dim=1)  # [N, k + 1]
        labels = torch.zeros(N, dtype=torch.long).to(z.device)  # 正样本标签为 0

        # 计算损失
        loss = F.cross_entropy(logits, labels, reduction='mean')

        return loss

    def tracing(self, q, s, pid=None, d_correct=None, d_skill_correct=None):
        pad = torch.zeros_like(q[:1]).to(self.know_params.device)
        q = torch.cat([q, pad], dim=0).unsqueeze(0)
        s = torch.cat([s, pad], dim=0).unsqueeze(0)
        if pid is not None:
            pid = torch.cat([pid, pad], dim=0).unsqueeze(0)

        with torch.no_grad():
            q_emb, s_emb, _, p_diff = self.embedding(q, s, pid)
            q_emb, s_emb, lens, _ = self.embedding_kan(q_emb, s_emb, p_diff, d_correct, d_skill_correct)
            z, _, _ = self(q_emb, s_emb, lens)
            query = self.know_params.unsqueeze(1).expand(-1, z.size(1), -1).contiguous()
            z = z.expand(self.n_questions, -1, -1).contiguous()
            h = self.readout(z, query)
            y = self.out(torch.cat([query, h], dim=-1)).squeeze(-1)
            y = torch.sigmoid(y)
        return y

class BilinearInteraction(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn(input_dim, input_dim))
        self.linear_x = nn.Linear(input_dim, input_dim)
        self.linear_y = nn.Linear(input_dim, input_dim)
        self.batch_norm = nn.BatchNorm1d(input_dim)

    def forward(self, x1, x2):

        batch_size, seq_len, feature_dim = x1.shape

        x1_flat = x1.view(-1, feature_dim)
        x2_flat = x2.view(-1, feature_dim)

        interaction = torch.matmul(x1_flat, self.W)

        interaction = interaction * x2_flat
        interaction = interaction.view(batch_size, seq_len, -1)


        interaction_flat = interaction.view(-1, feature_dim)
        interaction_flat = self.batch_norm(interaction_flat)
        interaction = interaction_flat.view(batch_size, seq_len, feature_dim)

        return interaction

class CASTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout, kq_same=False):
        super().__init__()
        self.masked_attn_head = DilatedMultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            kernel_size=5,
            dilation=[3, 1],
            kq_same=kq_same,
            attn_drop=dropout,
            proj_drop=dropout
        )

        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def device(self):
        return next(self.parameters()).device

    def forward(self, query, key, values, lens, peek_cur=False):
        seqlen = query.size(1)

        if seqlen < MIN_SEQ_LEN:
            return query, None

        mask = torch.ones(seqlen, seqlen).tril(0 if peek_cur else -1)
        mask = mask.bool()[None, None, :, :].to(self.device())

        if self.training:
            mask = mask.expand(query.size(0), -1, -1, -1).contiguous()

            for b in range(query.size(0)):
                if lens[b] < MIN_SEQ_LEN:
                    continue
                idx = random.sample(
                    range(lens[b] - 1), max(1, int(lens[b] * self.dropout_rate))
                )
                for i in idx:
                    mask[b, :, i + 1 :, i] = 0

        # [bs, seq_len, d_model]
        query_, scores = self.masked_attn_head(query, key, values, mask)

        query = query + self.dropout(query_)
        return self.layer_norm(query), scores

class DilatedMultiHeadAttention(nn.Module):
    def __init__(
            self,
            d_model,
            n_heads,
            kernel_size=3,
            dilation=[3, 1],
            bias=True,
            qk_scale=None,
            attn_drop=0.05,
            proj_drop=0.05,
            kq_same=False,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.num_dilation = len(dilation)
        # 尺度分割以后的头数， 8/2 = 4
        self.heads_per_dilation = n_heads // self.num_dilation
        self.kq_same = kq_same

        # 为每个 dilation 创建独立的线性层
        self.linears = nn.ModuleList(
            [nn.Linear(d_model // self.num_dilation, d_model // self.num_dilation, bias=bias) for _ in
             range(self.num_dilation)])

        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.scale = qk_scale or (self.d_k ** -0.5)

        # 为每个 dilation 计算 padding
        self.paddings = [d * (kernel_size - 1) // 2 for d in self.dilation]

    def unfold1d(self, x, dilation, padding):
        # [bs, d_k, heads_per_dilation, seq_len]
        bs, d_k, h_d, seq_len = x.shape
        x = x.reshape(bs * d_k, h_d, seq_len)  # [bs * d_k, h_d, seq_len]

        if seq_len + 2 * padding < self.kernel_size:
            return torch.zeros(bs * d_k, h_d, seq_len, self.kernel_size).to(x.device)

        x = F.pad(x, (padding, padding))
        effective_kernel_size = self.kernel_size + (self.kernel_size - 1) * (dilation - 1)

        x_unfold = x.unfold(dimension=2, size=effective_kernel_size, step=1)

        if dilation > 1:
            x_unfold = x_unfold[..., ::dilation]

        return x_unfold  # [bs, head, seq_len, kernel_size]

    def forward(self, q, k, v, mask=None):
        # [batch_size, seq_len, d_model] ——> [1, 91, 128]
        bs, seq_len, dmodel = q.shape

        split_size = dmodel // self.num_dilation  # 128 / 2 = 64

        q_splits = torch.split(q, split_size, dim=-1)  # [bs, seq_len, split_size] --> [1, 91, 64]
        k_splits = torch.split(k, split_size, dim=-1)
        v_splits = torch.split(v, split_size, dim=-1)

        attn_outputs = []
        attn_probs_all = []

        for i, dil in enumerate(self.dilation):

            q_linear = self.linears[i]
            k_linear = q_linear if self.kq_same else self.linears[i]
            v_linear = self.linears[i]

            q_i = q_linear(q_splits[i])  # [1, 91, 64]
            k_i = k_linear(k_splits[i])  # [1, 91, 64]
            v_i = v_linear(v_splits[i])  # [1, 91, 64]

            padding = self.paddings[i]

            # [bs, seq_len, head, d_k] head:分割以后的头数8/2=4。 d_k:分割以后的模型维度128/8=16。
            q_i = q_i.reshape(bs, seq_len, self.heads_per_dilation, self.d_k)
            k_i = k_i.reshape(bs, seq_len, self.heads_per_dilation, self.d_k)
            v_i = v_i.reshape(bs, seq_len, self.heads_per_dilation, self.d_k)

            # 转置和调整形状
            q_i = q_i.transpose(1, 2).permute(0, 3, 1, 2)  # [bs, d_k, heads_per_dilation, seq_len]
            k_i = k_i.transpose(1, 2).permute(0, 3, 1, 2)  # [bs, d_k, heads_per_dilation, seq_len]
            v_i = v_i.transpose(1, 2).permute(0, 3, 1, 2)  # [bs, d_k, heads_per_dilation, seq_len]


            # 调用 unfold1d
            k_patches = self.unfold1d(k_i, dilation=dil, padding=padding)   # [bs * d_k, head, seq_len, kernel_size]
            v_patches = self.unfold1d(v_i, dilation=dil, padding=padding)

            # 调整 q_i 形状

            q_i = q_i.reshape(bs * self.heads_per_dilation, self.d_k, seq_len)  # [bs * heads_per_dilation, d_k, seq_len]
            q_i = q_i.unsqueeze(-1)  # [bs * heads_per_dilation, d_k, seq_len, 1]

            # 计算注意力得分
            # [bs * heads_per_dilation, d_k, seq_len, kernel_size]
            k_patches_reshaped = k_patches.reshape(bs * self.heads_per_dilation, self.d_k, seq_len, -1)
            # [bs * heads_per_dilation, seq_len, kernel_size]
            attn_scores = (q_i * k_patches_reshaped).sum(dim=1) * self.scale

            # 计算注意力概率
            attn_probs = F.softmax(attn_scores, dim=-1)     # [bs * heads_per_dilation, seq_len, kernel_size]
            attn_probs = self.attn_drop(attn_probs)     # [bs * heads_per_dilation, seq_len, kernel_size]

            # 计算注意力输出
            # [bs * heads_per_dilation, d_k, seq_len, kernel_size]
            v_patches_reshaped = v_patches.reshape(bs * self.heads_per_dilation, self.d_k, seq_len, -1)
            attn_output = (attn_probs.unsqueeze(1) * v_patches_reshaped).sum(dim=-1)    # [bs * heads_per_dilation, d_k, seq_len]

            # 恢复形状
            # [bs, heads_per_dilation, d_k, seq_len]
            attn_output = attn_output.reshape(bs, self.heads_per_dilation, self.d_k, seq_len)

            # [bs, seq_len, heads_per_dilation, d_k]
            attn_output = attn_output.permute(0, 3, 1, 2).contiguous()

            # [bs, seq_len, heads_per_dilation * d_k]
            attn_output = attn_output.reshape(bs, seq_len, self.heads_per_dilation * self.d_k)
            attn_outputs.append(attn_output)
            attn_probs_all.append(attn_probs.reshape(bs, self.heads_per_dilation, seq_len, -1))

        # 拼接所有 dilation 组的输出
        # [bs, seq_len, d_model]
        attn_output = torch.cat(attn_outputs, dim=-1)
        # [bs, head, seq_len, kernel_size]
        attn_probs = torch.cat(attn_probs_all, dim=1)

        # 最终线性投影
        # [bs, seq_len, d_model]
        output = self.out_proj(attn_output)
        output = self.proj_drop(output)

        return output, attn_probs
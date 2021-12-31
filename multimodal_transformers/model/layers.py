import torch.nn as nn
import torch
from torch.nn.parameter import Parameter

# pytorch layer
class KeyAttention(nn.Module):
    """
    Compute attention between two sentences (S1(w1, e), S2(w2, e)) on word
    level (W(w1, w2), where w1 and w2 are the number of words in each sentence)
    Return: (att(scarlar, ), att_softmax(word_num, ))

    # Arguments
        op: The way to compute the word-level-attention.
            dp: Dot product. No weight for this approach.
                W = dot_product(S1, S2^T)
            sdp: Dot product with normalization (scaled dot product),
                [Vaswani, 2017]: W = dot_product(S1, S2^T)/sqrt(e)
            gen: General [Luong, 2015], W = dot_product(S1, M, S2^T),
                M is the weights to learn
            con: Concat [Bahdanau, 2015], W = dot_product(
                                                v,
                                                tanh(dot_product(M, [S1; S2]))
                                                )
                 where v and M are weights to learn.
        seed: random seed for initializing weights when it's needed.
              If seed = -1, then a identity matrix will be used
              for initialization.
        emb_dim: Dimension of word embeddings.
        word_att_pool: {max|sum|mean}, the pooling operation for
                       word-level attention.
        merge_ans_key: {concat|mean}
        beta: Bool.
    """
    def __init__(self,
                 op='dp',
                 seed=-1,
                 emb_dim=300,
                 word_att_pool='max',
                 merge_ans_key='concat',
                 beta=False,
                 **kwargs):
        super(KeyAttention, self).__init__(**kwargs)
        self.op = op
        self.seed = seed
        self.emb_dim = emb_dim
        self.word_att_pool = word_att_pool
        self.merge_ans_key = merge_ans_key
        self.beta = beta
        self.W = None
        self.M = None
        self.v = None
        self.bias = None
        self.init_weights()

    def init_weights(self, input_shape):
        self.token_num_ans = input_shape[0][1]
        self.token_num_key = input_shape[1][1]
        if self.seed != -1:
            torch.manual_seed(self.seed)
        if self.op == 'dp':
            self.W = Parameter(torch.Tensor(self.emb_dim, self.emb_dim))
            self.bias = Parameter(torch.Tensor(self.emb_dim))
            torch.nn.init.xavier_uniform_(self.W)
            torch.nn.init.constant_(self.bias, 0)
        elif self.op == 'sdp':
            self.W = Parameter(torch.Tensor(self.emb_dim, self.emb_dim))
            self.bias = Parameter(torch.Tensor(self.emb_dim))
            torch.nn.init.xavier_uniform_(self.W)
            torch.nn.init.constant_(self.bias, 0)
        elif self.op == 'gen':
            self.M = Parameter(torch.Tensor(self.emb_dim, self.emb_dim))
            self.bias = Parameter(torch.Tensor(self.emb_dim))
            torch.nn.init.xavier_uniform_(self.M)
            torch.nn.init.constant_(self.bias, 0)
        super(KeyAttention, self).init_weights(input_shape)
    def compute_mask(self, inputs, mask):
        return None

    def bdot(a, b):
        B = a.shape[0]
        S = a.shape[1]
        return torch.bmm(a.view(B, 1, S), b.view(B, S, 1)).reshape(-1)

    def softmax(self, x, mask):
        y = torch.exp(x - torch.max(x, axis=1, keepdim=True))
        sum_y = self.bdot(y, torch.permute(mask, (0, 2, 1)))
        return y/sum_y

    def forward(self, inputs):
        ans, mask_ans, key, mask_key = inputs
        mask_ans_inf = torch.abs(mask_ans - 1) * -10000
        mask_key_inf = torch.abs(mask_key - 1) * -10000

        mask_ans_inf_1 = torch.unsqueeze(mask_ans_inf, 1)
        mask_key_inf_1 = torch.unsqueeze(mask_key_inf, 1)

        mask_ans_2 = torch.unsqueeze(mask_ans, 2)
        mask_key_2 = torch.unsqueeze(mask_key, 2)

        ans = ans * mask_ans_2
        key = key * mask_key_2

        Z_dp = self.bdot(key, torch.permute(key, (0, 2, 1)))

        norm_ans = torch.sqrt(torch.maximum(torch.sum(torch.square(ans), -1), 1e-8))
        norm_key = torch.sqrt(torch.maximum(torch.sum(torch.square(key), -1), 1e-8))

        norm_repeat_ans = torch.repeat_interleave(norm_ans, self.token_num_key)
        norm_repeat_key = torch.repeat_interleave(norm_key, self.token_num_ans)
        norm_repeat_key = torch.permute(norm_repeat_key, (0, 2, 1))

        Z_cos = Z_dp / (norm_repeat_key * norm_repeat_ans)

        if self.op == "dp":
            Z = Z_dp
        elif self.op == "sdp":
            Z = Z_dp / torch.sqrt(self.bias)
        elif self.op == "gen":
            Z = torch.dot(key, self._M)
            Z = self.bdot(Z, torch.permute(key, (0, 2, 1)))
        elif self.op == "cos":
            Z = Z_cos

        Z_key = torch.permute(Z, (0, 2, 1))
        if self.mask_pad:
            Z_softmax_key = self.softmax(Z_key + mask_key_inf_1, axis=2)
        else:
            Z_softmax_key = self.softmax(Z_key, axis=2)

        V = self.bdot(Z_softmax_key, key)
        V = V * mask_ans_2

        Z_ans = Z
        if self.mask_pad:
            Z_softmax_ans = self.softmax(Z_ans + mask_ans_inf_1, axis=1)
        else:
            Z_softmax_ans = self.softmax(Z_ans, axis=1)

        U = self.bdot(Z_softmax_ans, ans)
        U = U * mask_key_2

        beta_key = torch.sigmoid(torch.max(Z_cos + mask_ans_inf_1, axis=2) * 5)
        beta_key = torch.unsqueeze(beta_key, 2)

        Z_cos = torch.permute(Z_cos, (0, 2, 1))
        beta_ans = torch.sigmoid(torch.max(Z_cos + mask_key_inf_1, axis=2) * 5)

        beta_ans = torch.unsqueeze(beta_ans, 2)

        if self.beta:
            U = U * beta_key
            V = V * beta_ans

        if self.word_att_pool == "sum":
            v = torch.sum(V, 1, keepdims=False)
            u = torch.sum(U, 1, keepdims=False)
        elif self.word_att_pool == "max":
            v = torch.max(V, 1, keepdims=False)
            u = torch.max(U, 1, keepdims=False)
        elif self.word_att_pool == "mean":
            v = torch.sum(V, 1, keepdims=False) / torch.sum(mask_ans_2, 1)
            u = torch.sum(U, 1, keepdims=False) / torch.sum(mask_key_2, 1)
        else:
            raise TypeError(
                "The pooling method need to be 'max', 'sum' or 'mean'!"
            )

        if self.merge_ans_key == 'concat':
            f = torch.cat([u, v], 1)
        elif self.merge_ans_key == 'mean':
            f = (u + v) / 2
        elif self.merge_ans_key == 'ans':
            f = u
        elif self.merge_ans_key == 'key':
            f = v

        Z_softmax_key = torch.permute(Z_softmax_key, (0, 2, 1))

        beta_ans = torch.unsqueeze(torch.squeeze(beta_ans, 2), 1)
        beta_key = torch.unsqueeze(torch.squeeze(beta_key, 2), 1)
        rtn_list = [f, Z, Z_softmax_ans, Z_softmax_key, beta_ans, beta_key]
        return rtn_list

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)
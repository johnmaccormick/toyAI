import time
import numpy as np

# import lightning as L  # Lightning makes it easier to write, optimize and scale our code
# We'll store our data in DataLoaders
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam  # We will use the Adam optimizer, which is, essentially,
import torch.nn as nn  # torch.nn gives us nn.Module(), nn.Embedding() and nn.Linear()
import torch  # torch let's us create tensors and also provides helper functions
import torch.nn.functional as F  # This gives us the softmax() and argmax()

# use Saver() for debgging if necessary
# from jm_util import Saver
# SAVER = Saver()


class Data(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.len = len(X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.len


class Corpus:
    def __init__(self):
        self.token_to_id = {'what': 0,
                            'is': 1,
                            'statquest': 2,
                            'awesome': 3,
                            '<EOS>': 4,  # <EOS> = end of sequence
                            'apple': 5,
                            'banana': 6,
                            'grape': 7,
                            'pear': 8,
                            'fruit': 9,
                            '<PAD>': 10,
                            }

        self.vocab_size = len(self.token_to_id)
        self.id_to_token = dict(map(reversed, self.token_to_id.items()))

        self.pad_idx = self.token_to_id['<PAD>']

        self.input_strings = ['what is statquest <EOS> awesome',
                              'what is statquest <EOS> awesome',
                              'what is statquest <EOS> awesome',
                              'statquest is what <EOS> awesome',
                              'what is apple <EOS> fruit',
                              'what is banana <EOS> fruit',
                              'fruit <EOS> pear grape',
                              'pear <EOS> pear',
                              'grape <EOS> grape',
                              ]

        # input_strings = ['what is']

        self.inputs = [self.input_to_tensor(input_string)
                       for input_string in self.input_strings]
        advanced_inputs = [self.advance_input(
            input_str) for input_str in self.input_strings]
        self.labels = [self.input_to_tensor(advanced_input)
                       for advanced_input in advanced_inputs]

        self.add_padding(self.inputs)
        self.add_padding(self.labels)
        self.DATASET = Data(self.inputs, self.labels)

    def input_to_IDs(self, input_str):
        return [self.token_to_id[w] for w in input_str.split()]

    def input_to_tensor(self, input_str):
        return torch.tensor(self.input_to_IDs(input_str))

    def inputs_to_tensor(self, input_strs):
        return torch.tensor([self.input_to_IDs(input_str) for input_str in input_strs])

    def advance_input(self, input_str):
        words = input_str.split()
        del words[0]
        words.append('<EOS>')
        return ' '.join(words)

    def ids_to_string(self, ids):
        # ids is a 1D  tensor containing token IDs
        tokens = [self.id_to_token[ids[i].item()] for i in range(len(ids))]
        return ' '.join(tokens)

    def add_padding(self, ids_list):
        # ids_list is list of tensors
        max_len = max(map(len, ids_list))
        for i, ids in enumerate(ids_list):
            pad_len = max_len - len(ids)
            if pad_len > 0:
                padding = torch.full((pad_len,), self.pad_idx)
                padded_tensor = torch.cat((ids, padding))
                ids_list[i] = padded_tensor


# global singleton
CORPUS = Corpus()


class BasicTransformerParams():

    def __init__(self):
        '''The so-called model dimension, also sometimes called the embedding dimension. 
        This is the length of the vectors output by the token embedding. 
        It is also the length of the vectors transmitted between major layers of 
        the transformer neural network.'''
        self.d_model = 2

        '''Also known as the context window, this is the maximum number of 
        tokens the model can ingest.'''
        self.max_input_tokens = 6

        '''Does the transformer use multiple layers of self attention?'''
        self.use_attn_layers = False

        '''Assuming multiple self attention layers are used, how many layers are there?'''
        self.num_attn_layers = 3

        '''In each of self attention layer, 
        there can be multiple attention heads. 
        This is the number of attention heads in each layer.'''
        self.num_attn_heads = 1

        '''Final dimension of weight matrices W_q, W_k in attention heads
        -- we can use these to project into a lower-dimensional space'''
        self.d_qk = 3

        '''Final dimension of weight matrix W_v in attention heads and first dim of W_o
        ("values" and "outputs" matrices)'''
        self.d_vo = 4

        '''This specifies the type of attention head(s) to be used: 
        'single', 'multi', or 'multicompact'.
        - 'single' employs a single head in each layer; 
        - 'multi' employs multiple heads in each layer implemented in a somewhat 
          inefficient but simple manner; 
        - 'multicompact' employs multiple heads in each layer implemented 
          so that the weights for all heads are compacted into one set of matrices 
          and can thus be computed more efficiently.'''
        self.attn_head_config = 'single'

        '''True if the so-called 'FFN' sublayer in each attention layer 
        is a genuine feedforward network
        with two layers separated by a nonlinear activation function.
        False if the 'FFN' layer is just a single linear module.'''
        self.use_2ffn = False

        '''Number of nodes in the middle part of the two-layer FFN (if used)'''
        self.d_ffn = 20

        '''Size of minibatches when performing learning'''
        self.batch_size = 5

        '''Learning rate used by the optimizer'''
        self.learning_rate = 0.02

        '''Number of epochs that the optimizer will run for. 
        An epoch is one iteration through the training set.'''
        self.num_epochs = 300

        '''While training is taking place, the current loss can be printed every time 
        a fixed number of epochs are completed.'''
        self.loss_print_freq = 50

        '''PyTorch should be manually seeded with this value before training.'''
        self.seed = 42

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")


def prepend_singleton_dims(tensor, target_dims):
    # Written by Colab AI
    """Prepends singleton dimensions to a tensor until it has exactly target_dims dimensions.

    Args:
      tensor: A PyTorch tensor.

    Returns:
      A PyTorch tensor with exactly target_dims dimensions.
    """
    num_dims_to_add = target_dims - len(tensor.shape)
    if num_dims_to_add > 0:
        return tensor.reshape((1,) * num_dims_to_add + tensor.shape)
    else:
        return tensor


class DimInfo:
    """This class carries around information about dimensions in the model. 
    It should only be used for debugging and assertions. It is not thread safe. 
    It is used as a singleton class, but I can imagine situations where batches 
    with different numbers of inputs are running simultaneously in different threads, 
    and various other similar problems."""

    def __init__(self, this_batch_size=-1, num_inputs=-1, d_model=-1, d_ffn=-1, d_qk=-1, d_vo=-1):
        # Similar comments to self.num_inputs apply here (see below).
        # This records the batch size of the latest batch, which might not always
        # be equal to the requested batch size (e.g. the last few instances in a data set).
        self.this_batch_size = this_batch_size
        # On any given forward or backward pass, this records the number of
        # tokens that were in the input. (Typically, if there is a batch of inputs,
        # they are padded so that the length of all inputs is the same as the maximum.)
        # This value is initialized to -1, but typically it will store the number of
        # tokens input in the most recent (or current) call to forward().
        self.num_inputs = num_inputs
        # number of dims in token embeddings
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.d_qk = d_qk
        self.d_vo = d_vo

    def check_encoding_shape(self, t: torch.Tensor):
        assert t.shape == (self.this_batch_size, self.num_inputs, self.d_model)

    def check_qk_shape(self, t: torch.Tensor):
        assert t.shape == (self.this_batch_size, self.num_inputs, self.d_qk)

    def check_v_shape(self, t: torch.Tensor):
        assert t.shape == (self.this_batch_size, self.num_inputs, self.d_vo)

    def check_attn_shape(self, t: torch.Tensor):
        assert t.shape == (self.this_batch_size,
                           self.num_inputs, self.num_inputs)

    def check_qk_shape_multi(self, t: torch.Tensor, num_heads):
        assert t.shape == (self.this_batch_size,
                           self.num_inputs, num_heads * self.d_qk)

    def check_v_shape_multi(self, t: torch.Tensor, num_heads):
        assert t.shape == (self.this_batch_size,
                           self.num_inputs, num_heads * self.d_vo)

    def check_qk_shape_unflattened(self, t: torch.Tensor, num_heads):
        assert t.shape == (self.this_batch_size, num_heads,
                           self.num_inputs,  self.d_qk)

    def check_v_shape_unflattened(self, t: torch.Tensor, num_heads):
        assert t.shape == (self.this_batch_size, num_heads,
                           self.num_inputs, self.d_vo)

    def check_attn_unflattened(self, t: torch.Tensor, num_heads):
        assert t.shape == (self.this_batch_size, num_heads,
                           self.num_inputs, self.num_inputs)


class PositionEncoding(nn.Module):

    def __init__(self, d_model, max_len=6):

        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(start=0, end=max_len,
                                step=1).float().unsqueeze(1)
        embedding_index = torch.arange(start=0, end=d_model, step=2).float()

        div_term = 1/torch.tensor(10000.0)**(embedding_index / d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, word_embeddings):
        # word_embeddings dims are: batch, token_pos, embed_dim
        # assert word_embeddings.ndim == 3
        # crop to length of input
        pe_crop = self.pe[:word_embeddings.size(-2), :]
        # print(f'pe.shape {self.pe.shape}')
        # print(f'pe_crop.shape {pe_crop.shape}')
        # print(f'word_embeddings.shape {word_embeddings.shape}')
        # this addition uses broadcast semantics to copy pe to each
        # element of the batch
        return word_embeddings + pe_crop


# AttentionHead is just the part that calculates similarity
# based on keys and queries, applies mask,
# and (at present, but this may change)
# also applies the value weights.
# MultiHeadAttention combines several AttentionHeads additively.
# The AttentionLayer combines a MultiHeadAttention with FFN
# (typically 2-layer FC with activation in between).

class AttentionHead(nn.Module):

    def __init__(self, d_model, d_qk, d_vo, dim_info: DimInfo):
        super().__init__()

        self.dim_info = dim_info
        self.dim_info.d_qk = d_qk
        self.dim_info.d_vo = d_vo

        # There are always 3 dims for input. dim 0 is for batch. row is 1, col is 2.
        self.row_dim = 1
        self.col_dim = 2

        self.W_q = nn.Linear(in_features=d_model,
                             out_features=d_qk, bias=False)
        self.W_k = nn.Linear(in_features=d_model,
                             out_features=d_qk, bias=False)
        self.W_v = nn.Linear(in_features=d_model,
                             out_features=d_vo, bias=False)
        self.W_o = nn.Linear(in_features=d_vo,
                             out_features=d_model, bias=False)

        self.printed_details = False
        self.saved_attention = torch.Tensor()

    def forward(self, encodings, mask):
        self.dim_info.check_encoding_shape(encodings)

        q = self.W_q(encodings)
        k = self.W_k(encodings)
        v = self.W_v(encodings)

        self.dim_info.check_qk_shape(q)
        self.dim_info.check_qk_shape(k)
        self.dim_info.check_v_shape(v)

        sims = torch.matmul(q, k.transpose(
            dim0=self.row_dim, dim1=self.col_dim))
        self.dim_info.check_attn_shape(sims)
        scaled_sims = sims / torch.tensor(k.size(self.col_dim)**0.5)
        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(mask=mask, value=-1e9)

        # We need *row-wise* softmax, because matrix will be *post*-muliplied by v.
        # The row is the last dim, hence dim=-1.
        # Note dims are B,n,n where B=batch, n=inputs.
        attention_percents = F.softmax(scaled_sims, dim=-1)
        # We can check that the softmax was applied along the correct dimension,
        # because the mask should cause everything above the leading diagonal to be zero.
        # So when softmax is applied to the first row, there is only one nonzero element
        # -- the top left element. Therefore, the top left element should be normalized
        # to 1.0 (or extremely close to it).
        topleft = attention_percents[0, 0, 0].item()
        assert (abs(topleft-1.0) < 1e-5)
        attention_scores = torch.matmul(attention_percents, v)
        self.dim_info.check_v_shape(attention_scores)
        attention_outputs = self.W_o(attention_scores)
        self.dim_info.check_encoding_shape(attention_outputs)
        return attention_outputs


class AttentionLayer(nn.Module):
    def __init__(self,
                 btp: BasicTransformerParams,
                 dim_info: DimInfo):
        super().__init__()
        self.dim_info = dim_info
        self.dim_info.d_ffn = btp.d_ffn
        self.dim_info.d_qk = btp.d_qk
        self.dim_info.d_vo = btp.d_vo
        # attn_seed = 1234

        # torch.manual_seed(attn_seed)
        # print(f'torch.manual_seed with attn_seed in AttentionLayer(): {
        #       attn_seed}')

        # attention head(s)
        # 'single', 'multi', 'multicompact'
        if btp.attn_head_config == 'single':
            print('single head')
            self.self_attention = AttentionHead(
                d_model=btp.d_model, d_qk=btp.d_qk, d_vo=btp.d_vo, dim_info=dim_info)
        elif btp.attn_head_config == 'multi':
            print(f'multihead, num_heads={btp.num_attn_heads}')
            self.self_attention = MultiHeadAttention(
                d_model=btp.d_model, num_heads=btp.num_attn_heads, d_qk=btp.d_qk, d_vo=btp.d_vo, dim_info=dim_info)
        elif btp.attn_head_config == 'multicompact':
            print(f'compact multihead, num_heads={btp.num_attn_heads}')
            self.self_attention = MultiAttentionHeadCompact(
                d_model=btp.d_model, num_heads=btp.num_attn_heads, d_qk=btp.d_qk, d_vo=btp.d_vo, dim_info=dim_info)
        else:
            assert False
        # FFN layer
        if btp.use_2ffn:
            self.ffn = FFN_2_layer(btp.d_model, btp.d_ffn, btp.d_model)
        else:
            self.ffn = nn.Linear(
                in_features=btp.d_model, out_features=btp.d_model)

    def forward(self, encodings, mask):
        self.dim_info.check_encoding_shape(encodings)
        attention_scores = self.self_attention(encodings, mask)
        self.dim_info.check_encoding_shape(attention_scores)
        residual_attention_values = encodings + attention_scores
        self.dim_info.check_encoding_shape(residual_attention_values)
        # print(f'residual_attention_values:\n{residual_attention_values}')
        ffn_outputs = self.ffn(residual_attention_values)
        self.dim_info.check_encoding_shape(ffn_outputs)
        # print(f'ffn_outputs:\n{ffn_outputs}')
        residual_ffn_values = residual_attention_values + ffn_outputs
        self.dim_info.check_encoding_shape(residual_ffn_values)
        # the above residuals follow equation (1) of Feng2023NeurIPS-chain-of-thought.
        # In the notation of that equation,
        # residual_attention_values is X^{l-1} + Attn^l(X^{l-1})
        # and
        # residual_ffn_values is X^{l}
        self.dim_info.check_encoding_shape(residual_ffn_values)
        return residual_ffn_values


class MultiAttentionHeadCompact(nn.Module):

    def __init__(self, d_model, num_heads, d_qk, d_vo, dim_info: DimInfo):

        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dim_info = dim_info
        self.dim_info.d_qk = d_qk
        self.dim_info.d_vo = d_vo
        self.d_qk = d_qk
        self.d_vo = d_vo
        self.width_qk = num_heads*d_qk
        self.height_vo = num_heads*d_vo

        self.W_q = nn.Linear(in_features=d_model,
                             out_features=self.width_qk, bias=False)
        self.W_k = nn.Linear(in_features=d_model,
                             out_features=self.width_qk, bias=False)
        self.W_v = nn.Linear(in_features=d_model,
                             out_features=num_heads*d_vo, bias=False)
        self.W_o = nn.Linear(in_features=self.height_vo,
                             out_features=d_model, bias=False)

        # e.g. when 4 dims, dim 0 is for batch, dim 1 for head_ID. row is 2, col is 3.
        # But there could be even more dims?? -3, -2 and -1 work for this?
        self.head_dim = -3
        self.row_dim = -2
        self.col_dim = -1

        self.printed_details = False
        self.saved_attention = torch.Tensor()

    def unflatten_head_vals(self, vals: torch.Tensor, n, H, d):
        return vals.view(-1, n, H, d).transpose(dim0=1, dim1=2)

    def flatten_head_vals(self, vals: torch.Tensor, n, H, d):
        return vals.transpose(dim0=1, dim1=2).reshape(-1, n, H*d)

    def forward(self, encodings: torch.Tensor, mask: torch.Tensor):
        self.dim_info.check_encoding_shape(encodings)

        q = self.W_q(encodings)
        k = self.W_k(encodings)
        v = self.W_v(encodings)
        # print(f'created v, shape is {v.shape}')

        self.dim_info.check_qk_shape_multi(q, self.num_heads)
        self.dim_info.check_qk_shape_multi(k, self.num_heads)
        self.dim_info.check_v_shape_multi(v, self.num_heads)

        num_inputs = encodings.shape[1]
        q = self.unflatten_head_vals(q, num_inputs, self.num_heads, self.d_qk)
        k = self.unflatten_head_vals(k, num_inputs, self.num_heads, self.d_qk)
        v = self.unflatten_head_vals(v, num_inputs, self.num_heads, self.d_vo)

        self.dim_info.check_qk_shape_unflattened(q, self.num_heads)
        self.dim_info.check_qk_shape_unflattened(k, self.num_heads)
        self.dim_info.check_v_shape_unflattened(v, self.num_heads)

        sims = torch.matmul(q, k.transpose(
            dim0=self.row_dim, dim1=self.col_dim))
        self.dim_info.check_attn_unflattened(sims, self.num_heads)

        scaled_sims = sims / torch.tensor(self.d_qk**0.5)

        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(mask=mask, value=-1e9)

        # We need *row-wise* softmax, because matrix will be *post*-muliplied by v.
        # The row is the last dim, hence dim=-1.
        # Note dims are B,H,n,n where B=batch, H=heads, n=inputs.
        attention_percents = F.softmax(scaled_sims, dim=-1)
        # We can check that the softmax was applied along the correct dimension,
        # because the mask should cause everything above the leading diagonal to be zero.
        # So when softmax is applied to the first row, there is only one nonzero element
        # -- the top left element. Therefore, the top left element should be normalized
        # to 1.0 (or extremely close to it).
        topleft = attention_percents[0, 0, 0, 0].item()
        assert (abs(topleft-1.0) < 1e-5)

        attention_scores = torch.matmul(attention_percents, v)
        self.dim_info.check_v_shape_unflattened(
            attention_scores, self.num_heads)

        attention_scores_unstacked = self.flatten_head_vals(
            attention_scores, num_inputs, self.num_heads, self.d_vo)
        self.dim_info.check_v_shape_multi(
            attention_scores_unstacked, self.num_heads)
        # *********** difference observed here
        attention_outputs = self.W_o(attention_scores_unstacked)
        self.dim_info.check_encoding_shape(attention_outputs)

        return attention_outputs


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads, d_qk, d_vo, dim_info: DimInfo):
        super().__init__()
        self.attention_heads = nn.ModuleList()
        self.num_heads = num_heads
        self.dim_info = dim_info
        for _ in range(num_heads):
            attention_head = AttentionHead(
                d_model=d_model, d_qk=d_qk, d_vo=d_vo, dim_info=dim_info)
            self.attention_heads.append(attention_head)
        # e.g. when 3 dims, dim 0 is for batch. row is 1, col is 2.
        # But there could be even more dims?? -2 and -1 work for this?
        # self.row_dim = -2
        # self.col_dim = -1
        self.printed_details = False
        self.saved_attention = torch.Tensor()

    def forward(self, encodings, mask):
        self.dim_info.check_encoding_shape(encodings)
        attention_scores_tot = torch.zeros_like(encodings)

        # todo: Presumably there is a way to run these heads in parallel
        for head in self.attention_heads:
            attention_scores = head.forward(encodings, mask)
            self.dim_info.check_encoding_shape(attention_scores)
            attention_scores_tot += attention_scores
        self.dim_info.check_encoding_shape(attention_scores_tot)
        return attention_scores_tot


class FFN_2_layer(nn.Module):
    def __init__(self, d_in, d_mid, d_out):
        super().__init__()
        self.fc1 = nn.Linear(in_features=d_in, out_features=d_mid)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=d_mid, out_features=d_out)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x


class AttentionLayers(nn.Module):
    def __init__(self,
                 btp: BasicTransformerParams,
                 dim_info: DimInfo):
        super().__init__()
        self.dim_info = dim_info
        self.dim_info.d_ffn = btp.d_ffn
        self.dim_info.d_qk = btp.d_qk
        self.dim_info.d_vo = btp.d_vo
        self.attn_layers = nn.ModuleList()
        self.num_layers = btp.num_attn_layers
        for _ in range(btp.num_attn_layers):
            attention_layer = AttentionLayer(btp, dim_info)
            self.attn_layers.append(attention_layer)

    def forward(self, encodings, mask):
        output = encodings
        for attn_layer in self.attn_layers:
            output = attn_layer(output, mask)
        return output


def compare_attn_outputs(attn_output: torch.Tensor, attn_output2: torch.Tensor):
    with torch.no_grad():
        diff = attn_output2 - attn_output
        if torch.any(diff != 0.0) or attn_output.shape != attn_output2.shape:
            print('attns differ')
            print(f'attn_output: {attn_output}')
            print(f'attn_output2: {attn_output2}')
            pass
        # else:
        #     print('SAME attns')


class DecoderOnlyTransformer(nn.Module):
    # class DecoderOnlyTransformer(L.LightningModule):

    def __init__(self, btp: BasicTransformerParams, num_tokens):

        super().__init__()

        self.btp = btp
        self.dim_info = DimInfo(d_model=btp.d_model)

        self.vocab_size = num_tokens
        self.max_num_inputs = btp.max_input_tokens

        # token embedding
        self.we = nn.Embedding(num_embeddings=num_tokens,
                               embedding_dim=btp.d_model)
        # print(f'self.we.weight.data:\n{self.we.weight.data}')

        # position encoding
        self.pe = PositionEncoding(d_model=btp.d_model,
                                   max_len=btp.max_input_tokens)

        if not btp.use_attn_layers:
            self.attn_layer = AttentionLayer(btp, self.dim_info)
        else:
            self.attn_layer = AttentionLayers(btp, self.dim_info)

        # final FC for token classification (outputs logits)
        self.tok_classifier = nn.Linear(
            in_features=btp.d_model, out_features=num_tokens)

        self.loss = nn.CrossEntropyLoss()

        self.saved_token_IDs = torch.Tensor()

        self.printed_mask = False

        # count number of forward() calls, for debugging
        self.forward_counter = 0

    def forward(self, token_ids):

        assert token_ids.ndim == 2
        # A batch produced by a DataLoader could be smaller than the specified
        # batch size, for example when drawing from the last few instances at
        # the end of the dataset.
        self.dim_info.this_batch_size = token_ids.shape[0]
        self.dim_info.num_inputs = token_ids.shape[1]

        word_embeddings = self.we(token_ids)
        self.dim_info.check_encoding_shape(word_embeddings)

        position_encoded = self.pe(word_embeddings)
        self.dim_info.check_encoding_shape(position_encoded)

        num_inputs = token_ids.shape[1]
        mask = torch.tril(torch.ones(
            (num_inputs, num_inputs), device=self.btp.device))
        mask = mask == 0

        attn_output = self.attn_layer(position_encoded, mask)
        self.dim_info.check_encoding_shape(attn_output)

        tok_class_output = self.tok_classifier(attn_output)

        return tok_class_output

    def get_saved_attention(self):
        return self.self_attention.saved_attention

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.btp.learning_rate)

    def training_step(self, batch, batch_idx):
        input_tokens, labels = batch  # collect input
        output = self.forward(input_tokens[0])
        loss = self.loss(output, labels[0])

        return loss


def predict_top(model, model_input, num_top):
    # last row (final token) only
    logits = model(model_input.to(model.btp.device))[-1, :]
    probs = F.softmax(logits, dim=0)
    top_probs, top_indices = torch.topk(probs, num_top)
    for idx, prob in zip(top_indices, top_probs):
        token = CORPUS.id_to_token[idx.item()]
        print(f'{token}: {prob:.2f}')


def predict(model, model_input, max_len):
    input_length = model_input.size(dim=0)

    # get predictions (logits) from the model
    predictions = model(model_input.to(model.btp.device))
    # Since we only want the prediction from the
    # last row (the most recent prediction) we use reverse index for the
    # row, -1.
    predicted_id = torch.tensor([torch.argmax(predictions[-1, :])])
    # We'll store predicted_id in an array, predicted_ids, that
    # we'll add to each time we predict a new output token.
    predicted_ids = predicted_id

    # Now use a loop to predict output tokens until we get an
    # <EOS> token.

    for i in range(input_length, max_len):
        # if the prediction is <EOS>, then we are done
        if (predicted_id == CORPUS.token_to_id["<EOS>"]):
            break

        model_input = torch.cat((model_input, predicted_id))

        predictions = model(model_input.to(model.btp.device))
        predicted_id = torch.tensor([torch.argmax(predictions[-1, :])])
        predicted_ids = torch.cat((predicted_ids, predicted_id))

    return [CORPUS.id_to_token[id.item()] for id in predicted_ids]


def compare_model_params(models):
    model_params = []
    for model in models:
        this_model_params = []
        for name, param in model.named_parameters():
            this_model_params.append((name, param))
            # print(f"{name}: {param.data}")
        model_params.append(this_model_params)
    for p1, p2 in zip(*model_params):
        name1, param1 = p1
        name2, param2 = p2
        diff = param1.data - param2.data
        print(name1, 'diff\n', diff)


def evaluate_gradient(model, x, y):
    model.train()
    model.zero_grad()
    y_pred = model(x.to(model.btp.device))  # Perform a forward pass
    loss = model.loss(y_pred, y.to(model.btp.device))
    # print(f'loss {loss}')
    loss.backward()


def calc_pred_and_loss(model: nn.Module, X: torch.Tensor, y: torch.Tensor):
    y_pred = model(X.to(model.btp.device))

    # Ensure y is 2d and y_pred 3d
    y = prepend_singleton_dims(y, 2)
    y_pred = prepend_singleton_dims(y_pred, 3)
    if y_pred.ndim == 4:
        y_pred = y_pred.squeeze(0)
    # See my diary entry for 10/23/2024 for detailed explanation of this transpose.
    # Basically, the first dimension of y_pred should be the batches (which is correct already),
    # but the second should be the different possible classes
    # (i.e. ranging over the whole vocabulary of tokens),
    # but this is currently in the wrong place so we need to swap
    # it from the third dimension.
    # e.g. batch size 2, vocab size 10, input len 5 gives y_pred
    # with shape 2,5,10 but gets transposed to 2,10,5.
    y_pred.transpose_(dim0=-2, dim1=-1)
    loss = model.loss(y_pred, y.to(model.btp.device))
    return y_pred, loss


def do_training_step(model: torch.nn.Module, optimizer, X: torch.Tensor, y: torch.Tensor, print_loss=False):
    # X could be a batch or a single instance.
    # If it's a single instance, convert it to a batch of size 1.
    # This guarantees the input will have a known dimensionality.
    # Same for y.
    assert X.ndim == 1 or X.ndim == 2
    if X.ndim == 1:
        X.unsqueeze_(-1)
    assert y.ndim == 1 or y.ndim == 2
    if y.ndim == 1:
        y.unsqueeze_(-1)

    X = X.to(model.btp.device)
    y = y.to(model.btp.device)
    y_pred = model(X)
    y_pred, loss = calc_pred_and_loss(model, X, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # I believe the cross entropy loss will compute the average loss,
    # so we need to multiply by the number of training samples in
    # the batch.
    batch_size = y_pred.shape[0]
    aggregate_loss = loss.item() * batch_size
    if print_loss:
        print(f'batch_size {batch_size}, ' +
              f'aggregate_loss {aggregate_loss}, ' +
              f'avg loss {loss.item()}')
    return aggregate_loss


def do_epoch(model, optimizer, dataloader, print_loss=False):
    total_loss = 0.0
    for batch, (X, y) in enumerate(dataloader):
        loss = do_training_step(model, optimizer, X, y)
        total_loss += loss
    if print_loss:
        print(f'total epoch loss: {total_loss}')
    return total_loss


def do_epochs(model, optimizer, dataloader):
    model.train()
    start_time = time.time()
    for epoch in range(model.btp.num_epochs):
        total_loss = do_epoch(model, optimizer, dataloader)
        if (epoch+1) % model.btp.loss_print_freq == 0 or epoch == 0 or epoch == model.btp.num_epochs-1:
            avg_loss = total_loss / len(dataloader.dataset)
            print(f'epoch: {epoch}, avg_loss: {avg_loss:.5f}')
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training time: {training_time:.2f} seconds")
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


def create_model(btp: BasicTransformerParams):
    torch.manual_seed(btp.seed)
    print(f'torch.manual_seed: {btp.seed}')

    model = DecoderOnlyTransformer(btp, num_tokens=len(CORPUS.token_to_id), )
    model.to(btp.device)
    model.train()
    optimizer = Adam(model.parameters(), lr=btp.learning_rate)
    dataloader = DataLoader(CORPUS.DATASET, batch_size=btp.batch_size)
    return model, optimizer, dataloader


def find_val(tensor, value):
    """Finds the index of the first element equal to a given value in a 1D PyTorch tensor.
       -- gen my Colab AI

    Args:
      tensor: The 1D PyTorch tensor.
      value: The value to search for.

    Returns:
      The index of the first element equal to the given value, or -1 if the value
      is not found.
    """
    indices = (tensor == value).nonzero(as_tuple=True)[0]
    if len(indices) > 0:
        return indices[0].item()
    else:
        return -1


def count_errors(model, dataloader, print_errs=False, response_errs_only=False):
    num_errs = 0
    num_response_errs = 0
    for batch, (X, y) in enumerate(dataloader):
        y_pred = model(X.to(model.btp.device))
        predicted_ids = torch.argmax(y_pred, dim=-1)
        mispredictions = (y - predicted_ids).bool().squeeze(0)
        this_num_errs = torch.count_nonzero(mispredictions).item()
        num_errs += this_num_errs
        if this_num_errs > 0:
            msg = '\nerrors:' if not response_errs_only else '\nresponse errors:'
            printed_msg = False

            this_batch_size = X.shape[0]
            if mispredictions.ndim == 1:
                mispredictions.unsqueeze_(0)
            for i in range(this_batch_size):
                mispreds = mispredictions[i]
                if response_errs_only:
                    idx = find_val(y[i], CORPUS.token_to_id['<EOS>'])
                    if idx >= 0:
                        relevant_mispreds = mispreds[idx+1:]
                    else:
                        relevant_mispreds = torch.Tensor()
                else:
                    relevant_mispreds = mispreds
                if relevant_mispreds.any():
                    num_response_errs += 1
                    if print_errs:
                        if not printed_msg:
                            print(msg)
                            printed_msg = True
                        input_IDs = X[i]
                        pred_IDs = predicted_ids[i]
                        true_IDs = y[i]
                        print()
                        print(f'input_IDs: {CORPUS.ids_to_string(input_IDs)}')
                        print(f'pred_IDs: {CORPUS.ids_to_string(pred_IDs)}')
                        print(f'true_IDs: {CORPUS.ids_to_string(true_IDs)}')

    print(f'num_errs: {num_errs}')
    if response_errs_only:
        print(f'num_response_errs: {num_response_errs}')
    return num_response_errs if response_errs_only else num_errs


def print_individual_losses(model, dataloader):
    model.eval()
    losses = []
    for batch_idx, (X, y) in enumerate(dataloader):
        this_batch_size = X.shape[0]
        print(f'batch_idx {batch_idx} contains {this_batch_size} samples')
        for i in range(this_batch_size):
            _, loss = calc_pred_and_loss(model, X[i], y[i])
            losses.append(loss.item())
    print(losses)
    print(f'total {np.sum(losses)}')


def rounded_tensor_to_str(tensor, precision=2):
    return np.array_str(
        tensor.cpu().detach().numpy(), precision=precision, suppress_small=True)


def print_gradients(model: nn.Module):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"Gradient of {name}:\n{param.grad}")
        else:
            print(f"Gradient of {name}:\nNot available")


def print_params(model: nn.Module):
    for name, param in model.named_parameters():
        print(f"{name}:\n{param.data}")


def copy_attn_multi_to_compact(btp: BasicTransformerParams, multi_model, compact_model):
    multi_attn = multi_model.attn_layer.self_attention
    compact_attn = compact_model.attn_layer.self_attention

    c = compact_attn
    m = multi_attn.attention_heads
    d_qk = btp.d_qk
    d_vo = btp.d_vo

    with torch.no_grad():
        for h in range(btp.num_attn_heads):
            c.W_q.weight[h*d_qk:(h+1)*d_qk, :] = m[h].W_q.weight
            c.W_k.weight[h*d_qk:(h+1)*d_qk, :] = m[h].W_k.weight
            c.W_v.weight[h*d_vo:(h+1)*d_vo, :] = m[h].W_v.weight
            c.W_o.weight[:, h*d_vo:(h+1)*d_vo] = m[h].W_o.weight


def main1():
    btp = BasicTransformerParams()
    btp.num_attn_heads = 2
    btp.attn_head_config = 'multi'
    multi, _, _ = create_model(btp, batch_size=btp.batch_size)
    btp.attn_head_config = 'multicompact'
    compact, _, _ = create_model(btp, batch_size=btp.batch_size)
    copy_attn_multi_to_compact(btp, multi, compact)


def main2():
    btp = BasicTransformerParams()

    btp.num_attn_heads = 3
    for btp.batch_size in [1, 5]:
        for btp.use_attn_layers in [False, ]:
            for btp.use_2ffn in [False, True]:
                models_etc = dict()
                for btp.attn_head_config in ['multi', 'multicompact']:
                    model, optimizer, dataloader = create_model(btp)
                    models_etc[btp.attn_head_config] = (
                        model, optimizer, dataloader)
                copy_attn_multi_to_compact(models_etc['multi'][0],
                                           models_etc['multicompact'][0],
                                           btp.num_attn_heads, btp.d_model)
                for btp.attn_head_config in ['multi', 'multicompact']:
                    max_steps = 3
                    model, optimizer, dataloader = models_etc[btp.attn_head_config]
                    for step, (X, y) in enumerate(dataloader):
                        # print(f"starting step {step}, params:")
                        # print_params(model)
                        # print_gradients(model)
                        if step >= max_steps:
                            break
                        # print(f'training step {step}...')
                        do_training_step(model, optimizer,
                                         X, y, print_loss=True)

    return


def main3():
    btp = BasicTransformerParams()

    for btp.attn_head_config in ['single', 'multi', 'multicompact']:
        for btp.batch_size in [1, 5]:
            for btp.use_attn_layers in [False, True]:
                for btp.use_2ffn in [False, True]:
                    max_steps = 3
                    model, optimizer, dataloader = create_model(btp)
                    for step, (X, y) in enumerate(dataloader):
                        # print(f"starting step {step}, params:")
                        # print_params(model)
                        # print_gradients(model)
                        if step >= max_steps:
                            break
                        # print(f'training step {step}...')
                        do_training_step(model, optimizer,
                                         X, y, print_loss=False)

    return


def main4():

    btp = BasicTransformerParams()

    batch_sizes = [5]
    models = []
    optimizers = []
    dataloaders = []

    for batch_size in batch_sizes:
        model, optimizer, dataloader = create_model(btp, batch_size=batch_size)
        models.append(model)
        optimizers.append(optimizer)
        dataloaders.append(dataloader)

    for model, optimizer, dataloader in zip(models, optimizers, dataloaders):
        do_epochs(model, optimizer, dataloader)
    # compare_model_params(models)

    in_strs = ['what is statquest <EOS>',
               'statquest is what <EOS>',
               'statquest is',
               'what is',
               'what',
               'what is apple <EOS>',
               'what is banana <EOS>',
               'fruit <EOS>',
               'pear <EOS>',
               'grape <EOS>',
               ]

    with torch.no_grad():
        for model, dataloader in zip(models, dataloaders):
            model.eval()
            print(f'\nmodel type {type(model)}, '
                  f'batch size {dataloader.batch_size}')
            for in_str in in_strs:
                model_input = CORPUS.input_to_tensor(in_str)
                pred_tokens = predict(model, model_input, btp.max_input_tokens)
                print(in_str, ' -> ', ' '.join(pred_tokens))
            count_errors(model, dataloader, print_errs=True,
                         response_errs_only=True)

    with torch.no_grad():
        model = models[0]
        in_str = 'what is apple <EOS>'
        num_top = 3
        predict_top(model, CORPUS.input_to_tensor(in_str), num_top=num_top)
        attention = model.get_saved_attention()
        token_ids = model.saved_token_IDs
        tokens = [CORPUS.id_to_token[t.item()] for t in token_ids]
        print(f'attention\n{rounded_tensor_to_str(attention)}')
        print(f'tokens {tokens}')


def main():
    btp = BasicTransformerParams()
    btp.d_model = 6

    for btp.attn_head_config in ('multi', 'multicompact', ):
        model, optimizer, dataloader = create_model(btp)
        do_epochs(model, optimizer, dataloader)
        response_errs = count_errors(
            model, dataloader, response_errs_only=True)
        print(f'response_errs: {response_errs}')


if __name__ == "__main__":
    main()

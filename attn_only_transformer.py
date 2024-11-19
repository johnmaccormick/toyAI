import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import basic_transformer as bt
import corpus
import alphabet


class MinimalAttnHead(nn.Module):

    def __init__(self, btp: bt.BasicTransformerParams, dim_info: bt.DimInfo):
        super().__init__()

        self.btp = btp
        self.dim_info = dim_info
        self.dim_info.d_qk = btp.d_qk

        # There are always 3 dims for input: batch, tokens, embedding.
        # dim 0 is for batch. row is 1, col is 2.
        self.row_dim = 1
        self.col_dim = 2

        self.W_q = nn.Linear(in_features=btp.d_model,
                             out_features=btp.d_qk, bias=False)
        self.W_k = nn.Linear(in_features=btp.d_model,
                             out_features=btp.d_qk, bias=False)

    def forward(self, encodings):

        self.dim_info.check_encoding_shape(encodings)
        num_inputs = encodings.shape[1]

        q = self.W_q(encodings)
        k = self.W_k(encodings)
        self.dim_info.check_qk_shape(q)
        self.dim_info.check_qk_shape(k)

        sims = torch.matmul(q, k.transpose(
            dim0=self.row_dim, dim1=self.col_dim))
        self.dim_info.check_attn_shape(sims)

        scaled_sims = sims / torch.tensor(k.size(self.col_dim)**0.5)
        mask = bt.make_mask(num_inputs, self.btp.device)
        scaled_sims = scaled_sims.masked_fill(mask=mask, value=-1e9)

        attention_percents = nn.functional.softmax(scaled_sims, dim=-1)

        topleft = attention_percents[0, 0, 0].item()
        assert (abs(topleft-1.0) < 1e-5)

        attention_outputs = torch.matmul(attention_percents, encodings)
        self.dim_info.check_encoding_shape(attention_outputs)
        return attention_outputs

    def zero_out_W_bil(self):
        with torch.no_grad():
            self.W_bil.weight.data.zero_()

    def mask_pad_token(self, corp: corpus.Corpus):
        mask_val = -1e9
        with torch.no_grad():
            self.W_bil.weight.data[:, corp.pad_idx] = mask_val


class AttnOnlyTransformer(nn.Module):
    def __init__(self, btp: bt.BasicTransformerParams):

        super().__init__()

        self.btp = btp
        self.dim_info = bt.DimInfo(d_model=btp.d_model)

        assert btp.d_model == btp.vocab_size

        if btp.use_position_encoding:
            self.pe = bt.PositionEncoding(d_model=btp.d_model,
                                          max_len=btp.max_input_tokens)

        if not btp.use_attn_layers:
            self.head = MinimalAttnHead(btp, self.dim_info)
        else:
            assert btp.num_attn_layers > 0
            self.heads = nn.ModuleList(
                [MinimalAttnHead(btp, self.dim_info) for _ in range(btp.num_attn_layers)])

        self.loss = nn.CrossEntropyLoss()

    def forward(self, token_ids):

        assert token_ids.ndim == 2
        # A batch produced by a DataLoader could be smaller than the specified
        # batch size, for example when drawing from the last few instances at
        # the end of the dataset.
        self.dim_info.this_batch_size = token_ids.shape[0]
        self.dim_info.num_inputs = token_ids.shape[1]

        encoding = nn.functional.one_hot(
            token_ids, num_classes=self.btp.vocab_size).float()

        if self.btp.use_position_encoding:
            encoding = self.pe(encoding)
            self.dim_info.check_encoding_shape(encoding)

        # num_inputs = token_ids.shape[1]
        # mask = torch.tril(torch.ones(
        #     (num_inputs, num_inputs), device=self.btp.device))
        # mask = mask == 0

        if not self.btp.use_attn_layers:
            encoding = self.head(encoding)
        else:
            for h in self.heads:
                encoding = h(encoding)

        self.dim_info.check_encoding_shape(encoding)

        return encoding

    # def create_model(btp: bt.BasicTransformerParams, corp: corpus.Corpus):
    #     torch.manual_seed(btp.seed)
    #     print(f'torch.manual_seed: {btp.seed}')

    #     model = AttnOnlyTransformer(btp, num_tokens=len(corp.token_to_id), )
    #     model.to(btp.device)
    #     model.train()
    #     optimizer = torch.optim.Adam(model.parameters(), lr=btp.learning_rate)
    #     dataloader = DataLoader(corp.dataset, batch_size=btp.batch_size)
    #     return model, optimizer, dataloader


class AttnAndUnembedTransformer(nn.Module):
    def __init__(self, btp: bt.BasicTransformerParams):

        super().__init__()

        assert btp.vocab_size > 0
        self.btp = btp
        self.dim_info = bt.DimInfo(d_model=btp.d_model)

        self.attn_only = AttnOnlyTransformer(btp)

        # Should we allow bias? Probably, because it allows prior probabilities
        #  to be represented. I think. However, my pencil and paper work does
        #  not have a bias in the unembed module, so for now we do not allow it.
        self.unembed = nn.Linear(
            in_features=btp.d_model, out_features=btp.vocab_size, bias=False)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, token_ids):

        assert token_ids.ndim == 2
        # A batch produced by a DataLoader could be smaller than the specified
        # batch size, for example when drawing from the last few instances at
        # the end of the dataset.
        self.dim_info.this_batch_size = token_ids.shape[0]
        self.dim_info.num_inputs = token_ids.shape[1]

        attn_encoded = self.attn_only(token_ids)
        self.dim_info.check_encoding_shape(attn_encoded)

        unembedded = self.unembed(attn_encoded)
        self.dim_info.check_encoding_shape(unembedded)

        return unembedded


def learn_attn_only():
    corp = alphabet.get_Alphabet_corpus()
    btp = bt.BasicTransformerParams()
    btp.batch_size = 1
    btp.num_epochs = 500
    vocab_size = len(corp.id_to_token)
    btp.d_model = vocab_size
    btp.d_qk = vocab_size
    btp.d_vo = vocab_size
    model, optimizer, dataloader = bt.create_model(
        btp, corp, AttnOnlyTransformer)
    avg_loss = bt.do_epochs(model, optimizer, dataloader)
    response_errs = bt.count_errors(
        model, dataloader, response_errs_only=True, corp=corp)
    print(f'avg_loss {avg_loss}, response_errs {response_errs}')
    bt.print_some_predictions(corp, model)
    bt.print_params(model)


def learn_attnUnembed():
    corp = alphabet.get_Alphabet_corpus()
    btp = bt.BasicTransformerParams()
    btp.batch_size = 1
    vocab_size = len(corp.id_to_token)
    btp.d_model = vocab_size
    btp.d_qk = 3
    btp.d_vo = vocab_size
    btp.use_attn_layers = True
    btp.num_attn_layers = 3
    model, optimizer, dataloader = bt.create_model(
        btp, corp, AttnAndUnembedTransformer)
    avg_loss = bt.do_epochs(model, optimizer, dataloader)
    response_errs = bt.count_errors(
        model, dataloader, response_errs_only=True, corp=corp)
    print(f'avg_loss {avg_loss}, response_errs {response_errs}')
    # bt.print_some_predictions(corp, model)
    # bt.print_params(model)


def main():
    # learn_attn_only()
    learn_attnUnembed()


if __name__ == "__main__":
    main()

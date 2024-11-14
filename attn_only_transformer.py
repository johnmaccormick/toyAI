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

        # There are always 3 dims for input: batch, tokens, embedding.
        # dim 0 is for batch. row is 1, col is 2.
        self.row_dim = 1
        self.col_dim = 2

        # bil for bilinear form
        self.W_bil = nn.Linear(in_features=btp.d_model,
                               out_features=btp.d_model, bias=False)

    def forward(self, encodings):

        self.dim_info.check_encoding_shape(encodings)

        enc_mult_W = self.W_bil(encodings)
        self.dim_info.check_encoding_shape(enc_mult_W)
        sims = torch.matmul(encodings, enc_mult_W.transpose(
            dim0=self.row_dim, dim1=self.col_dim))
        self.dim_info.check_attn_shape(sims)
        scaled_sims = sims / torch.tensor(enc_mult_W.size(self.col_dim)**0.5)
        # print(f'scaled_sims:\n{scaled_sims}')

        num_inputs = encodings.shape[1]
        mask = bt.make_mask(num_inputs, self.btp.device)
        scaled_sims = scaled_sims.masked_fill(mask=mask, value=-1e9)
        # print(f'masked scaled_sims:\n{scaled_sims}')

        # We need *row-wise* softmax, because matrix will be *post*-muliplied by v.
        # The row is the last dim, hence dim=-1.
        # Note dims are B,n,n where B=batch, n=inputs.
        attention_percents = nn.functional.softmax(scaled_sims, dim=-1)
        # print(f'attention_percents:\n{attention_percents}')
        # We can check that the softmax was applied along the correct dimension,
        # because the mask should cause everything above the leading diagonal to be zero.
        # So when softmax is applied to the first row, there is only one nonzero element
        # -- the top left element. Therefore, the top left element should be normalized
        # to 1.0 (or extremely close to it).
        topleft = attention_percents[0, 0, 0].item()
        assert (abs(topleft-1.0) < 1e-5)
        attention_outputs = torch.matmul(attention_percents, encodings)
        self.dim_info.check_encoding_shape(attention_outputs)
        return attention_outputs


class AttnOnlyTransformer(nn.Module):
    # class DecoderOnlyTransformer(L.LightningModule):

    def __init__(self, btp: bt.BasicTransformerParams, num_tokens):

        super().__init__()

        self.btp = btp
        self.dim_info = bt.DimInfo(d_model=btp.d_model)

        self.vocab_size = num_tokens
        self.max_num_inputs = btp.max_input_tokens

        assert btp.d_model == self.vocab_size

        # position encoding
        self.pe = bt.PositionEncoding(d_model=btp.d_model,
                                      max_len=btp.max_input_tokens)

        # self.head = bt.AttentionHead(
        #     btp.d_model, btp.d_qk, btp.d_vo, self.dim_info)
        self.head = MinimalAttnHead(btp, self.dim_info)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, token_ids):

        assert token_ids.ndim == 2
        # A batch produced by a DataLoader could be smaller than the specified
        # batch size, for example when drawing from the last few instances at
        # the end of the dataset.
        self.dim_info.this_batch_size = token_ids.shape[0]
        self.dim_info.num_inputs = token_ids.shape[1]

        one_hot = nn.functional.one_hot(token_ids, num_classes=self.vocab_size)

        position_encoded = self.pe(one_hot)
        self.dim_info.check_encoding_shape(position_encoded)

        num_inputs = token_ids.shape[1]
        mask = torch.tril(torch.ones(
            (num_inputs, num_inputs), device=self.btp.device))
        mask = mask == 0

        attn_output = self.head(position_encoded)
        self.dim_info.check_encoding_shape(attn_output)

        return attn_output

    def create_model(btp: bt.BasicTransformerParams, corp: corpus.Corpus):
        torch.manual_seed(btp.seed)
        print(f'torch.manual_seed: {btp.seed}')

        model = AttnOnlyTransformer(btp, num_tokens=len(corp.token_to_id), )
        model.to(btp.device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=btp.learning_rate)
        dataloader = DataLoader(corp.dataset, batch_size=btp.batch_size)
        return model, optimizer, dataloader


def learn_attn_only():
    corp = alphabet.get_Alphabet_corpus()
    btp = bt.BasicTransformerParams()
    btp.batch_size = 1
    vocab_size = len(corp.id_to_token)
    btp.d_model = vocab_size
    btp.d_qk = vocab_size
    btp.d_vo = vocab_size
    model, optimizer, dataloader = AttnOnlyTransformer.create_model(btp, corp)
    avg_loss = bt.do_epochs(model, optimizer, dataloader)
    response_errs = bt.count_errors(
        model, dataloader, response_errs_only=True, corp=corp)
    print(f'avg_loss {avg_loss}, response_errs {response_errs}')
    bt.print_some_predictions(corp, model)
    bt.print_params(model)


def main():
    learn_attn_only()


if __name__ == "__main__":
    main()

import time
import numpy as np

# import lightning as L  # Lightning makes it easier to write, optimize and scale our code
# We'll store our data in DataLoaders
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam  # We will use the Adam optimizer, which is, essentially,
import torch.nn as nn  # torch.nn gives us nn.Module(), nn.Embedding() and nn.Linear()
import torch  # torch let's us create tensors and also provides helper functions
import torch.nn.functional as F  # This gives us the softmax() and argmax()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


USE_ATTN_LAYERS = True
VERBOSE = False
DEBUG_ATTN = False
MAX_INPUT_TOKENS = 6  # i.e. context window
D_MODEL = 4  # 4
# Final dimension of weight matrices W_q, W_k in attention heads
# -- we can use these to project into a lower-dimensional space
D_QK = 1
NUM_ATTN_HEADS = 5
NUM_ATTN_LAYERS = 3
# True if the so-called 'FFN' layer is a genuine feedforward network
# with two layers separated by a nonlinear activation function.
# False if the 'FFN' layer is just a single linear module.
USE_2FFN = True
# Number of nodes in the middle part of the two-layer FFN (if used)
D_FFN = 20


LEARNING_RATE = 0.02
NUM_EPOCHS = 300
LOSS_PRINT_FREQ = 50
BATCH_SIZE = 5
THE_SEED = 42
SAVE_ATTENTION = False
MULTI_HEAD = True


token_to_id = {'what': 0,
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
id_to_token = dict(map(reversed, token_to_id.items()))

pad_idx = token_to_id['<PAD>']


def input_to_IDs(input_str):
    return [token_to_id[w] for w in input_str.split()]


def input_to_tensor(input_str):
    return torch.tensor(input_to_IDs(input_str))


def inputs_to_tensor(input_strs):
    return torch.tensor([input_to_IDs(input_str) for input_str in input_strs])


def advance_input(input_str):
    words = input_str.split()
    del words[0]
    words.append('<EOS>')
    return ' '.join(words)


def ids_to_string(ids):
    # ids is a 1D  tensor containing token IDs
    tokens = [id_to_token[ids[i].item()] for i in range(len(ids))]
    return ' '.join(tokens)


def add_padding(ids_list):
    # ids_list is list of tensors
    max_len = max(map(len, ids_list))
    for i, ids in enumerate(ids_list):
        pad_len = max_len - len(ids)
        if pad_len > 0:
            padding = torch.full((pad_len,), pad_idx)
            padded_tensor = torch.cat((ids, padding))
            ids_list[i] = padded_tensor


input_strings = ['what is statquest <EOS> awesome',
                 'what is statquest <EOS> awesome',
                 'what is statquest <EOS> awesome',
                 'statquest is what <EOS> awesome',
                 'what is apple <EOS> fruit',
                 'what is banana <EOS> fruit',
                 'fruit <EOS> pear grape',
                 'pear <EOS> pear',
                 'grape <EOS> grape',
                 ]

inputs = [input_to_tensor(input_string) for input_string in input_strings]
advanced_inputs = [advance_input(input_str) for input_str in input_strings]
labels = [input_to_tensor(advanced_input)
          for advanced_input in advanced_inputs]

add_padding(inputs)
add_padding(labels)


class Data(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.len = len(X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.len


dataset = Data(inputs, labels)
# dataloader = DataLoader(dataset, batch_size=batch_size)


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
        if VERBOSE:
            print('pos encoding:', pe.data)
            print('pos encoding shape:', pe.shape)

    def forward(self, word_embeddings):
        # word_embeddings dims are: batch, token_pos, embed_dim
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

    def __init__(self, d_model, d_qk):

        super().__init__()

        self.W_q = nn.Linear(in_features=d_model,
                             out_features=d_qk, bias=False)
        self.W_k = nn.Linear(in_features=d_model,
                             out_features=d_qk, bias=False)
        self.W_v = nn.Linear(in_features=d_model,
                             out_features=d_model, bias=False)

        # e.g. when 3 dims, dim 0 is for batch. row is 1, col is 2.
        # But there could be even more dims?? -2 and -1 work for this?
        self.row_dim = -2
        self.col_dim = -1

        self.printed_details = False
        self.saved_attention = torch.Tensor()

        if VERBOSE:
            print('W_q:', self.W_q.weight.shape)
            print('W_k:', self.W_k.weight.shape)
            print('W_v:', self.W_v.weight.shape)

        # print('W_q vals:', self.W_q.weight)

    def forward(self, encodings_for_q, encodings_for_k, encodings_for_v, mask):

        q = self.W_q(encodings_for_q)
        k = self.W_k(encodings_for_k)
        v = self.W_v(encodings_for_v)

        if DEBUG_ATTN:
            print(f'q: {q}')
            print(f'k.size(self.col_dim): {k.size(self.col_dim)}')

        sims = torch.matmul(q, k.transpose(
            dim0=self.row_dim, dim1=self.col_dim))

        scaled_sims = sims / torch.tensor(k.size(self.col_dim)**0.5)

        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(mask=mask, value=-1e9)

        attention_percents = F.softmax(scaled_sims, dim=self.col_dim)
        attention_scores = torch.matmul(attention_percents, v)

        # if DEBUG_ATTN:
        #     print(f'attention_percents: {attention_percents}')

        if VERBOSE and not self.printed_details:
            print('encodings_for_q shape:', encodings_for_q.shape)
            print('q:', q.shape)

            print('k:', k.shape)
            print('v:', v.shape)
            print('sims:', sims)
            print('sims shape:', sims.data.shape)
            print('scaled_sims:', scaled_sims)
            print('scaled sims shape:', scaled_sims.data.shape)
            print('attention_percents:', attention_percents.data)
            print('attention_percents shape:', attention_percents.data.shape)
            print('attention_scores:', attention_scores.data)
            print('attention_scores shape:', attention_scores.data.shape)
            self.printed_details = True

        if SAVE_ATTENTION:
            self.saved_attention = attention_percents

        # if torch.rand(1).item() < 0.002:
        #     print(rounded_tensor_to_str(attention_percents))

        return attention_scores


class AttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ffn, d_qk):
        super().__init__()

        # attn_seed = 1234
        # torch.manual_seed(attn_seed)
        # print(f'torch.manual_seed with attn_seed in AttentionLayer(): {
        #       attn_seed}')

        # attention head(s)
        if not MULTI_HEAD:
            print('single head')
            self.self_attention = AttentionHead(d_model=d_model, d_qk=d_qk)
        else:
            print(f'multihead, num_heads={NUM_ATTN_HEADS}')
            self.self_attention = MultiHeadAttention(
                d_model=d_model, num_heads=num_heads, d_qk=d_qk)

        # FFN layer
        if USE_2FFN:
            self.ffn = FFN_2_layer(d_model, d_ffn, d_model)
        else:
            self.ffn = nn.Linear(
                in_features=d_model, out_features=d_model)

    def forward(self, encodings, mask):
        attention_scores = self.self_attention(
            encodings, encodings, encodings, mask)
        residual_attention_values = encodings + attention_scores
        # print(f'residual_attention_values:\n{residual_attention_values}')
        ffn_outputs = self.ffn(residual_attention_values)
        # print(f'ffn_outputs:\n{ffn_outputs}')
        residual_ffn_values = residual_attention_values + ffn_outputs
        # the above residuals follow equation (1) of Feng2023NeurIPS-chain-of-thought.
        # In the notation of that equation,
        # residual_attention_values is X^{l-1} + Attn^l(X^{l-1})
        # and
        # residual_ffn_values is X^{l}
        return residual_ffn_values


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads, d_qk):
        super().__init__()
        self.attention_heads = nn.ModuleList()
        self.num_heads = num_heads
        for _ in range(num_heads):
            attention_head = AttentionHead(d_model=d_model, d_qk=d_qk)
            self.attention_heads.append(attention_head)
        # e.g. when 3 dims, dim 0 is for batch. row is 1, col is 2.
        # But there could be even more dims?? -2 and -1 work for this?
        # self.row_dim = -2
        # self.col_dim = -1
        self.printed_details = False
        self.saved_attention = torch.Tensor()

    def forward(self, encodings_for_q, encodings_for_k, encodings_for_v, mask):
        # return self.attention_heads[0].forward(encodings_for_q, encodings_for_k, encodings_for_v, mask)
        attention_scores_tot = torch.zeros_like(encodings_for_q)
        if SAVE_ATTENTION:
            self.saved_attention = torch.zeros_like(encodings_for_q)

        # todo: Presumably there is a way to run these heads in parallel
        for head in self.attention_heads:
            attention_scores = head.forward(
                encodings_for_q, encodings_for_k, encodings_for_v, mask)
            attention_scores_tot += attention_scores
            if SAVE_ATTENTION:
                self.saved_attention += head.saved_attention
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
    def __init__(self, d_model, num_heads, d_ffn, num_attn_layers, d_qk):
        super().__init__()
        self.attn_layers = nn.ModuleList()
        self.num_layers = num_attn_layers
        for _ in range(num_attn_layers):
            attention_layer = AttentionLayer(
                d_model=d_model, num_heads=num_heads, d_ffn=d_ffn, d_qk=d_qk)
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

    def __init__(self, num_tokens, d_model, max_len):

        super().__init__()

        # token embedding
        self.we = nn.Embedding(num_embeddings=num_tokens,
                               embedding_dim=d_model)

        # position encoding
        self.pe = PositionEncoding(d_model=d_model,
                                   max_len=max_len)

        # # attention head(s)
        # if not MULTI_HEAD:
        #     print('single head')
        #     self.self_attention = AttentionHead(d_model=d_model)
        # else:
        #     print(f'multihead, num_heads={NUM_ATTN_HEADS}')
        #     self.self_attention = MultiHeadAttention(
        #         d_model=d_model, num_heads=NUM_ATTN_HEADS)

        # # FFN layer
        # if USE_2FFN:
        #     self.fc_layer = FFN_2_layer(d_model, D_FFN, d_model)
        # else:
        #     self.fc_layer = nn.Linear(
        #         in_features=d_model, out_features=d_model)

        if not USE_ATTN_LAYERS:
            self.attn_layer = AttentionLayer(
                d_model=d_model, num_heads=NUM_ATTN_HEADS, d_ffn=D_FFN, d_qk=D_QK)
        else:
            self.attn_layer = AttentionLayers(
                d_model=d_model, num_heads=NUM_ATTN_HEADS, d_ffn=D_FFN,
                num_attn_layers=NUM_ATTN_LAYERS, d_qk=D_QK)

        # final FC for token classification (outputs logits)
        self.tok_classifier = nn.Linear(
            in_features=d_model, out_features=num_tokens)

        self.loss = nn.CrossEntropyLoss()

        self.saved_token_IDs = torch.Tensor()

        self.printed_mask = False
        if VERBOSE:
            print('we:', self.we.weight.shape)
            print('pe:', self.pe.pe.shape)
            # print('fc_layer:', self.fc_layer.weight.shape)

        # count number of forward() calls, for debugging
        self.forward_counter = 0

    def forward(self, token_ids):
        # global DEBUG_ATTN
        # self.forward_counter += 1
        # if self.forward_counter == 3:
        #     DEBUG_ATTN = True
        if SAVE_ATTENTION:
            self.saved_token_IDs = token_ids

        word_embeddings = self.we(token_ids)
        position_encoded = self.pe(word_embeddings)

        mask = torch.tril(torch.ones(
            (token_ids.size(dim=-1), token_ids.size(dim=-1)), device=device))
        mask = mask == 0
        if VERBOSE and not self.printed_mask:
            print('mask:', mask.data)
            print('mask shape:', mask.shape)
            self.printed_mask = True

        # attention_input = position_encoded

        # self_attention_values = self.self_attention(attention_input,
        #                                             attention_input,
        #                                             attention_input,
        #                                             mask=mask)

        # residual_connection_values = position_encoded + self_attention_values
        # fc_layer_output = self.fc_layer(residual_connection_values)

        # print(f'\nDecoderOnlyTransformer.forward(), ' +
        #       f'execution {self.forward_counter}')
        # print('\n\nWithout attn_layers...')
        attn_output = self.attn_layer(position_encoded, mask)
        # print(f'attn_output: {attn_output}')
        # print('\n\nWith attn_layers...')
        # attn_output2 = self.attn_layers(position_encoded, mask)
        # print(f'attn_output2: {attn_output2}')

        # compare_attn_outputs(attn_output, attn_output2)

        # residual_output = position_encoded + attn_output
        # tok_class_output = self.tok_classifier(residual_output)
        tok_class_output = self.tok_classifier(attn_output)

        return tok_class_output

    def get_saved_attention(self):
        return self.self_attention.saved_attention

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=LEARNING_RATE)

    def training_step(self, batch, batch_idx):
        input_tokens, labels = batch  # collect input
        output = self.forward(input_tokens[0])
        loss = self.loss(output, labels[0])

        return loss


def predict_top(model, model_input, num_top):
    # last row (final token) only
    logits = model(model_input.to(device))[-1, :]
    probs = F.softmax(logits, dim=0)
    top_probs, top_indices = torch.topk(probs, num_top)
    for idx, prob in zip(top_indices, top_probs):
        token = id_to_token[idx.item()]
        print(f'{token}: {prob:.2f}')


def predict(model, model_input, max_len):
    input_length = model_input.size(dim=0)

    # get predictions (logits) from the model
    predictions = model(model_input.to(device))
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
        if (predicted_id == token_to_id["<EOS>"]):
            break

        model_input = torch.cat((model_input, predicted_id))

        predictions = model(model_input.to(device))
        predicted_id = torch.tensor([torch.argmax(predictions[-1, :])])
        predicted_ids = torch.cat((predicted_ids, predicted_id))

    return [id_to_token[id.item()] for id in predicted_ids]


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
    y_pred = model(x.to(device))  # Perform a forward pass
    loss = model.loss(y_pred, y.to(device))
    # print(f'loss {loss}')
    loss.backward()


def calc_pred_and_loss(model, X, y):
    y_pred = model(X.to(device))
    # See my diary entry for 10/23/2024 for detailed explanation of this transpose.
    # Basically, when we have a batch size greater than one,
    # then the first dimension of y_pred should be the batches (which is correct already),
    # but the second should be the different possible classes
    # (i.e. ranging over the whole vocabulary of tokens),
    # but this is currently in the wrong place so we need to swap
    # it from the third dimension.
    # e.g. batch size 2, vocab size 10, input len 5 gives y_pred
    # with shape 2,5,10 but gets transposed to 2,10,5.
    if y_pred.ndim > 2:
        y_pred.transpose_(dim0=1, dim1=2)
    loss = model.loss(y_pred, y.to(device))
    return y_pred, loss


def do_training_step(model, optimizer, X, y, print_loss=False):
    # X could be a batch or a single instance
    X_orig = X
    y_orig = y
    X = X.squeeze(0)
    y = y.squeeze(0).to(device)
    y_pred = model(X.to(device))
    # print(f'y_pred {y_pred}')
    y_pred, loss = calc_pred_and_loss(model, X, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # print(loss.item())
    # I believe the cross entropy loss will compute the average loss,
    # so we need to multiply by the number of training samples in
    # the batch.
    batch_size = y_pred.shape[0] if y_pred.ndim > 2 else 1
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
    for epoch in range(NUM_EPOCHS):
        total_loss = do_epoch(model, optimizer, dataloader)
        if (epoch+1) % LOSS_PRINT_FREQ == 0 or epoch == 0 or epoch == NUM_EPOCHS-1:
            avg_loss = total_loss / len(dataloader.dataset)
            print(f'epoch: {epoch}, avg_loss: {avg_loss:.5f}')
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training time: {training_time:.2f} seconds")


def create_model(batch_size):
    # L.seed_everything(seed=THE_SEED)
    torch.manual_seed(THE_SEED)
    print(f'torch.manual_seed: {THE_SEED}')

    model = DecoderOnlyTransformer(num_tokens=len(
        token_to_id), d_model=D_MODEL, max_len=MAX_INPUT_TOKENS)
    model.to(device)
    model.train()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    dataloader = DataLoader(dataset, batch_size=batch_size)
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
        y_pred = model(X.to(device))
        predicted_ids = torch.argmax(y_pred, dim=-1)
        # print(f'X: {X}')
        # print(f'y_pred: {y_pred}')
        # print(f'y: {y}')
        # print(f'predicted_ids: {predicted_ids}')
        mispredictions = (y - predicted_ids).bool()
        this_num_errs = torch.count_nonzero(mispredictions).item()
        num_errs += this_num_errs
        # print(f'this_num_errs: {this_num_errs}')
        if this_num_errs > 0:
            msg = '\nerrors:' if not response_errs_only else '\nresponse errors:'
            printed_msg = False

            this_batch_size = X.shape[0]
            for i in range(this_batch_size):
                mispreds = mispredictions[i]
                if response_errs_only:
                    idx = find_val(y[i], token_to_id['<EOS>'])
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
                        print(f'input_IDs: {ids_to_string(input_IDs)}')
                        print(f'pred_IDs: {ids_to_string(pred_IDs)}')
                        print(f'true_IDs: {ids_to_string(true_IDs)}')

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


def multi_count_response_errors(batch_size, num_trials):
    global THE_SEED
    start_seed = 48  # 4242
    total_errs = 0
    for trial in range(num_trials):
        THE_SEED = start_seed + trial
        model, optimizer, dataloader = create_model(batch_size=batch_size)
        do_epochs(model, optimizer, dataloader)
        response_errs = count_errors(
            model, dataloader, response_errs_only=True)
        print(f'response_errs: {response_errs}')
        total_errs += response_errs
    avg_errs = total_errs / num_trials
    print(f'avg_errs: {avg_errs}')
    # print_params(model)


def print_gradients(model: nn.Module):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"Gradient of {name}:\n{param.grad}")
        else:
            print(f"Gradient of {name}:\nNot available")


def print_params(model: nn.Module):
    for name, param in model.named_parameters():
        print(f"{name}:\n{param.data}")


def main():

    global USE_ATTN_LAYERS
    for USE_ATTN_LAYERS in (False, True):
        model, optimizer, dataloader = create_model(batch_size=BATCH_SIZE)
        do_epochs(model, optimizer, dataloader)
        response_errs = count_errors(
            model, dataloader, response_errs_only=True)
        print(f'response_errs: {response_errs}')
    #    print_params(model)

    # print('single training step')
    # for step, (X, y) in enumerate(dataloader):
    #     print(f'training step {step}')
    #     do_training_step(model, optimizer, X, y, print_loss=True)
    #     print_gradients(model)
    #     if step > 10:
    #         break

    return

    # batch_size = 1
    # global MULTI_HEAD
    # for MULTI_HEAD in True, False:
    #     model, optimizer, dataloader = create_model(batch_size=batch_size)
    #     do_epochs(model, optimizer, dataloader)
    #     response_errs = count_errors(
    #         model, dataloader, response_errs_only=True)
    #     print(f'response_errs: {response_errs}')
    # for batch, (X, y) in enumerate(dataloader):
    #     loss = do_training_step(model, optimizer, X, y)
    #     print(f'batch {batch}, loss {loss}')
    # print_gradients(model)
    # if batch >= 1:
    #     break

    # return

    # global MULTI_HEAD
    # for MULTI_HEAD in (True, ):
    #     multi_count_response_errors(batch_size=5, num_trials=2)
    # return

    multi_count_response_errors(batch_size=BATCH_SIZE, num_trials=2)
    return

    batch_sizes = [5]
    models = []
    optimizers = []
    dataloaders = []

    for batch_size in batch_sizes:
        model, optimizer, dataloader = create_model(batch_size=batch_size)
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
                model_input = input_to_tensor(in_str)
                pred_tokens = predict(model, model_input, MAX_INPUT_TOKENS)
                print(in_str, ' -> ', ' '.join(pred_tokens))
            count_errors(model, dataloader, print_errs=True,
                         response_errs_only=True)

    with torch.no_grad():
        model = models[0]
        in_str = 'what is apple <EOS>'
        num_top = 3
        global SAVE_ATTENTION
        SAVE_ATTENTION = True
        predict_top(model, input_to_tensor(in_str), num_top=num_top)
        attention = model.get_saved_attention()
        token_ids = model.saved_token_IDs
        tokens = [id_to_token[t.item()] for t in token_ids]
        print(f'attention\n{rounded_tensor_to_str(attention)}')
        print(f'tokens {tokens}')

    # with torch.no_grad():
    #     model = models[0]
    #     SAVE_ATTENTION = True
    #     for in_str in input_strings:
    #         logits = model(input_to_tensor(in_str).to(device))
    #         attention = model.get_saved_attention()
    #         token_ids = model.saved_token_IDs
    #         print('\n' + in_str)
    #         print(rounded_tensor_to_str(attention))

    # print('individual losses')
    # for model, dataloader in zip(models, dataloaders):
    #     print_individual_losses(model, dataloader)

    # print('single training step')
    # for model, optimizer, dataloader in zip(models, optimizers, dataloaders):
    #     for X, y in dataloader:
    #         do_training_step(model, optimizer, X, y, print_loss=True)
    #         break

    # print('single epoch')
    # for model, optimizer, dataloader in zip(models, optimizers, dataloaders):
    #     for X, y in dataloader:
    #         do_epoch(model, optimizer, dataloader, print_loss=True)
    #         break

    # return


#####################################################################
#####################################################################
#####################################################################
#####################################################################

    # # Now create the input for the transformer...
    # model_input = input_to_tensor('what is statquest <EOS>')
    # pred_tokens = predict(model, model_input, max_length)
    # print('Predictions before training:')
    # print(pred_tokens)

    # trainer = L.Trainer(max_epochs=30)
    # trainer.fit(model, train_dataloaders=dataloader)

    # pred_tokens = predict(model, model_input, max_length)
    # print('Predictions after training:')
    # print(pred_tokens)

    # print('Additional tests:')
    # in_strs = ['statquest is what <EOS>',
    #            'statquest is',
    #            'what is',
    #            'what',
    #            'what is apple <EOS>',
    #            'what is banana <EOS>',
    #            'fruit <EOS>',
    #            ]
    # for in_str in in_strs:
    #     model_input = input_to_tensor(in_str)
    #     pred_tokens = predict(model, model_input, max_length)
    #     print(in_str, ' -> ', ' '.join(pred_tokens))

if __name__ == "__main__":
    main()

import lightning as L  # Lightning makes it easier to write, optimize and scale our code
# We'll store our data in DataLoaders
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.optim import Adam  # We will use the Adam optimizer, which is, essentially,
import torch.nn as nn  # torch.nn gives us nn.Module(), nn.Embedding() and nn.Linear()
import torch  # torch let's us create tensors and also provides helper functions
import torch.nn.functional as F  # This gives us the softmax() and argmax()
import time
import numpy as np

verbose = False
max_length = 6  # max tokens -- i.e. context window
d_model = 4  # 4
learning_rate = 0.02
num_epochs = 300
loss_print_freq = 50
# batch_size = 1
the_seed = 44


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

    def __init__(self, d_model=2, max_len=6):

        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(start=0, end=max_len,
                                step=1).float().unsqueeze(1)
        embedding_index = torch.arange(start=0, end=d_model, step=2).float()

        div_term = 1/torch.tensor(10000.0)**(embedding_index / d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)
        if verbose:
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


class Attention(nn.Module):

    def __init__(self, d_model=2):

        super().__init__()

        self.W_q = nn.Linear(in_features=d_model,
                             out_features=d_model, bias=False)
        self.W_k = nn.Linear(in_features=d_model,
                             out_features=d_model, bias=False)
        self.W_v = nn.Linear(in_features=d_model,
                             out_features=d_model, bias=False)

        # e.g. when 3 dims, dim 0 is for batch. row is 1, col is 2. But there could be even more dims?? -2 and -1 work for this?
        self.row_dim = -2
        self.col_dim = -1

        self.printed_details = False
        if verbose:
            print('W_q:', self.W_q.weight.shape)
            print('W_k:', self.W_k.weight.shape)
            print('W_v:', self.W_v.weight.shape)

    def forward(self, encodings_for_q, encodings_for_k, encodings_for_v, mask=None):

        q = self.W_q(encodings_for_q)
        k = self.W_k(encodings_for_k)
        v = self.W_v(encodings_for_v)

        sims = torch.matmul(q, k.transpose(
            dim0=self.row_dim, dim1=self.col_dim))

        scaled_sims = sims / torch.tensor(k.size(self.col_dim)**0.5)

        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(mask=mask, value=-1e9)

        attention_percents = F.softmax(scaled_sims, dim=self.col_dim)
        attention_scores = torch.matmul(attention_percents, v)

        if verbose and not self.printed_details:
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

        return attention_scores


# class DecoderOnlyTransformer(nn.Module):
class DecoderOnlyTransformer(L.LightningModule):

    def __init__(self, num_tokens=4, d_model=2, max_len=6):

        super().__init__()

        self.we = nn.Embedding(num_embeddings=num_tokens,
                               embedding_dim=d_model)
        self.pe = PositionEncoding(d_model=d_model,
                                   max_len=max_len)
        self.self_attention = Attention(d_model=d_model)
        self.fc_layer = nn.Linear(in_features=d_model, out_features=num_tokens)

        self.loss = nn.CrossEntropyLoss()

        self.printed_mask = False
        if verbose:
            print('we:', self.we.weight.shape)
            print('pe:', self.pe.pe.shape)
            print('fc_layer:', self.fc_layer.weight.shape)

    def forward(self, token_ids):

        word_embeddings = self.we(token_ids)
        position_encoded = self.pe(word_embeddings)

        mask = torch.tril(torch.ones(
            (token_ids.size(dim=-1), token_ids.size(dim=-1))))
        mask = mask == 0
        if verbose and not self.printed_mask:
            print('mask:', mask.data)
            print('mask shape:', mask.shape)
            self.printed_mask = True

        self_attention_values = self.self_attention(position_encoded,
                                                    position_encoded,
                                                    position_encoded,
                                                    mask=mask)

        residual_connection_values = position_encoded + self_attention_values
        fc_layer_output = self.fc_layer(residual_connection_values)

        return fc_layer_output

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=learning_rate)

    def training_step(self, batch, batch_idx):
        input_tokens, labels = batch  # collect input
        output = self.forward(input_tokens[0])
        loss = self.loss(output, labels[0])

        return loss


def predict_top(model, model_input, num_top):
    # last row (final token) only
    logits = model(model_input)[-1, :]
    probs = F.softmax(logits, dim=0)
    top_probs, top_indices = torch.topk(probs, num_top)
    for idx, prob in zip(top_indices, top_probs):
        token = id_to_token[idx.item()]
        print(f'{token}: {prob:.2f}')


def predict(model, model_input, max_len):
    input_length = model_input.size(dim=0)

    # get predictions (logits) from the model
    predictions = model(model_input)
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

        predictions = model(model_input)
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
    y_pred = model(x)  # Perform a forward pass
    loss = model.loss(y_pred, y)
    # print(f'loss {loss}')
    loss.backward()


def calc_pred_and_loss(model, X, y):
    y_pred = model(X)
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
    loss = model.loss(y_pred, y)
    return y_pred, loss


def do_training_step(model, optimizer, X, y, print_loss=False):
    # X could be a batch or a single instance
    X_orig = X
    y_orig = y
    X = X.squeeze(0)
    y = y.squeeze(0)
    y_pred = model(X)
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
        print(f'batch_size {batch_size}, aggregate_loss {
              aggregate_loss}, avg loss {loss.item()}')
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
    for epoch in range(num_epochs):
        total_loss = do_epoch(model, optimizer, dataloader)
        if (epoch+1) % loss_print_freq == 0 or epoch == 0 or epoch == num_epochs-1:
            avg_loss = total_loss / len(dataloader.dataset)
            print(f'epoch: {epoch}, avg_loss: {avg_loss:.5f}')


def create_model(batch_size):
    L.seed_everything(seed=the_seed)
    model = DecoderOnlyTransformer(num_tokens=len(
        token_to_id), d_model=d_model, max_len=max_length)
    model.train()
    optimizer = Adam(model.parameters(), lr=learning_rate)
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
        y_pred = model(X)
        predicted_ids = torch.argmax(y_pred, dim=-1)
        # print(f'X: {X}')
        # print(f'y_pred: {y_pred}')
        # print(f'y: {y}')
        # print(f'predicted_ids: {predicted_ids}')
        mispredictions = (y - predicted_ids).bool()
        this_num_errs = torch.count_nonzero(mispredictions).item()
        num_errs += this_num_errs
        # print(f'this_num_errs: {this_num_errs}')
        if this_num_errs > 0 and print_errs:
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


def main():
    batch_sizes = [5]
    models = []
    optimizers = []
    dataloaders = []

    for i in range(len(batch_sizes)):
        model, optimizer, dataloader = create_model(batch_size=batch_sizes[i])
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

    # for model, dataloader in zip(models, dataloaders):
    #     model.eval()
    #     print(f'\nmodel type {type(model)}, '
    #           f'batch size {dataloader.batch_size}')
    #     for in_str in in_strs:
    #         model_input = input_to_tensor(in_str)
    #         pred_tokens = predict(model, model_input, max_length)
    #         print(in_str, ' -> ', ' '.join(pred_tokens))
    #     count_errors(model, dataloader, print_errs=True,
    #                  response_errs_only=True)

    model = models[0]
    in_str = 'what is apple <EOS>'
    num_top = 3
    predict_top(model, input_to_tensor(in_str), num_top=num_top)

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

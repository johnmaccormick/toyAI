verbose = True

import torch ## torch let's us create tensors and also provides helper functions
import torch.nn as nn ## torch.nn gives us nn.Module(), nn.Embedding() and nn.Linear()
import torch.nn.functional as F # This gives us the softmax() and argmax()
from torch.optim import Adam ## We will use the Adam optimizer, which is, essentially,
                             ## a slightly less stochastic version of stochastic gradient descent.
from torch.utils.data import TensorDataset, DataLoader ## We'll store our data in DataLoaders

import lightning as L ## Lightning makes it easier to write, optimize and scale our code

token_to_id = {'what' : 0,
               'is' : 1,
               'statquest' : 2,
               'awesome': 3,
               '<EOS>' : 4, ## <EOS> = end of sequence
               'apple' : 5,
               'banana' : 6,
               'grape' : 7,
               'pear' : 8
              }
id_to_token = dict(map(reversed, token_to_id.items()))

inputs = torch.tensor([[token_to_id["what"], ## input #1: what is statquest <EOS> awesome
                        token_to_id["is"],
                        token_to_id["statquest"],
                        token_to_id["<EOS>"],
                        token_to_id["awesome"]],

                       [token_to_id["statquest"], # input #2: statquest is what <EOS> awesome
                        token_to_id["is"],
                        token_to_id["what"],
                        token_to_id["<EOS>"],
                        token_to_id["awesome"]]])

labels = torch.tensor([[token_to_id["is"],
                        token_to_id["statquest"],
                        token_to_id["<EOS>"],
                        token_to_id["awesome"],
                        token_to_id["<EOS>"]],

                       [token_to_id["is"],
                        token_to_id["what"],
                        token_to_id["<EOS>"],
                        token_to_id["awesome"],
                        token_to_id["<EOS>"]]])

dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset)

class PositionEncoding(nn.Module):

    def __init__(self, d_model=2, max_len=6):

        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(start=0, end=max_len, step=1).float().unsqueeze(1)
        embedding_index = torch.arange(start=0, end=d_model, step=2).float()

        div_term = 1/torch.tensor(10000.0)**(embedding_index / d_model)


        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)
        if verbose:
          print('pos encoding:', pe.data)
          print('pos encoding shape:', pe.shape)

    def forward(self, word_embeddings):

        return word_embeddings + self.pe[:word_embeddings.size(0), :]

class Attention(nn.Module):

    def __init__(self, d_model=2):

        super().__init__()

        self.W_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

        self.row_dim = 0
        self.col_dim = 1

        self.printed_details = False
        if verbose:
          print('W_q:', self.W_q.weight.shape)
          print('W_k:', self.W_k.weight.shape)
          print('W_v:', self.W_v.weight.shape)


    def forward(self, encodings_for_q, encodings_for_k, encodings_for_v, mask=None):

        q = self.W_q(encodings_for_q)
        k = self.W_k(encodings_for_k)
        v = self.W_v(encodings_for_v)

        sims = torch.matmul(q, k.transpose(dim0=self.row_dim, dim1=self.col_dim))

        scaled_sims = sims / torch.tensor(k.size(self.col_dim)**0.5)

        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(mask=mask, value=-1e9)

        attention_percents = F.softmax(scaled_sims, dim=self.col_dim)
        attention_scores = torch.matmul(attention_percents, v)

        if verbose and not self.printed_details:
          print('encodings_for_q:', encodings_for_q.shape)
          print('q:', q.shape)
          print('k:', k.shape)
          print('v:', v.shape)
          print('scaled_sims:', scaled_sims)
          print('scaled sims shape:', scaled_sims.data.shape)
          print('attention_percents:', attention_percents.data)
          print('attention_percents shape:', attention_percents.data.shape)
          print('attention_scores:', attention_scores.data)
          print('attention_scores shape:', attention_scores.data.shape)
          self.printed_details = True

        return attention_scores

class DecoderOnlyTransformer(L.LightningModule):

    def __init__(self, num_tokens=4, d_model=2, max_len=6):

        super().__init__()

        L.seed_everything(seed=42)

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

        mask = torch.tril(torch.ones((token_ids.size(dim=0), token_ids.size(dim=0))))
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
        return Adam(self.parameters(), lr=0.1)


    def training_step(self, batch, batch_idx):
        input_tokens, labels = batch # collect input
        output = self.forward(input_tokens[0])
        loss = self.loss(output, labels[0])

        return loss

def predict(model, model_input):
  input_length = model_input.size(dim=0)

  ## Now get get predictions from the model
  predictions = model(model_input)
  ## NOTE: "predictions" is the output from the fully connected layer,
  ##      not a softmax() function. We could, if we wanted to,
  ##      Run "predictions" through a softmax() function, but
  ##      since we're going to select the item with the largest value
  ##      we can just use argmax instead...
  ## ALSO NOTE: "predictions" is a matrix, with one row of predicted values
  ##      per input token. Since we only want the prediction from the
  ##      last row (the most recent prediction) we use reverse index for the
  ##      row, -1.
  predicted_id = torch.tensor([torch.argmax(predictions[-1,:])])
  ## We'll store predicted_id in an array, predicted_ids, that
  ## we'll add to each time we predict a new output token.
  predicted_ids = predicted_id

  ## Now use a loop to predict output tokens until we get an
  ## <EOS> token.
  max_length = 6
  for i in range(input_length, max_length):
      if (predicted_id == token_to_id["<EOS>"]): # if the prediction is <EOS>, then we are done
          break

      model_input = torch.cat((model_input, predicted_id))

      predictions = model(model_input)
      predicted_id = torch.tensor([torch.argmax(predictions[-1,:])])
      predicted_ids = torch.cat((predicted_ids, predicted_id))

  ## Now printout the predicted output phrase.
  print("Predicted Tokens:\n")
  for id in predicted_ids:
      print("\t", id_to_token[id.item()])

## First, create a model from DecoderOnlyTransformer()
model = DecoderOnlyTransformer(num_tokens=len(token_to_id), d_model=4, max_len=6)

## Now create the input for the transformer...
model_input = torch.tensor([    token_to_id["grape"],
                            token_to_id["what"],
                            token_to_id["is"],
                            token_to_id["statquest"],
                            token_to_id["<EOS>"]])

predict(model, model_input)



trainer = L.Trainer(max_epochs=30)
trainer.fit(model, train_dataloaders=dataloader)

predict(model, model_input)

model_input2 = torch.tensor([token_to_id["statquest"],
                            token_to_id["is"],
                            token_to_id["what"],
                            token_to_id["<EOS>"]])

predict(model, model_input2)

model_input3 = torch.tensor([token_to_id["statquest"],
                            token_to_id["is"]])

predict(model, model_input3)

model_input4 = torch.tensor([token_to_id["what"],
                            token_to_id["is"]])

predict(model, model_input4)

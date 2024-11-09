from torch.utils.data import Dataset
import torch


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
    def __init__(self, token_to_id: dict[str, int], input_strings: list[str]):
        self.create_dataset(token_to_id, input_strings)

        # self.token_to_id = {'what': 0,
        #                     'is': 1,
        #                     'statquest': 2,
        #                     'awesome': 3,
        #                     '<EOS>': 4,  # <EOS> = end of sequence
        #                     'apple': 5,
        #                     'banana': 6,
        #                     'grape': 7,
        #                     'pear': 8,
        #                     'fruit': 9,
        #                     '<PAD>': 10,
        #                     }

        # self.input_strings = ['what is statquest <EOS> awesome',
        #                       'what is statquest <EOS> awesome',
        #                       'what is statquest <EOS> awesome',
        #                       'statquest is what <EOS> awesome',
        #                       'what is apple <EOS> fruit',
        #                       'what is banana <EOS> fruit',
        #                       'fruit <EOS> pear grape',
        #                       'pear <EOS> pear',
        #                       'grape <EOS> grape',
        #                       ]

        # input_strings = ['what is']

    def create_dataset(self, token_to_id: dict[str, int], input_strings: list[str]):
        self.token_to_id = token_to_id
        self.input_strings = input_strings
        self.vocab_size = len(self.token_to_id)
        self.id_to_token = dict(map(reversed, self.token_to_id.items()))
        self.pad_idx = self.token_to_id['<PAD>']
        self.inputs = [self.input_to_tensor(input_string)
                       for input_string in self.input_strings]
        advanced_inputs = [self.advance_input(
            input_str) for input_str in self.input_strings]
        self.labels = [self.input_to_tensor(advanced_input)
                       for advanced_input in advanced_inputs]
        self.add_padding(self.inputs)
        self.add_padding(self.labels)
        self.dataset = Data(self.inputs, self.labels)

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

    def token_dict_from_tokens(tokens: list[str]) -> dict[str, int]:
        token_to_id = dict()
        for i, tok in enumerate(tokens):
            token_to_id[tok] = i
        for tok in ['<PAD>', '<EOS>']:
            if tok not in token_to_id:
                token_to_id[tok] = i + 1
        return token_to_id

    def token_dict_from_string(token_str: str) -> dict[str, int]:
        tokens = token_str.split()
        return Corpus.token_dict_from_tokens(tokens)


class Statquest_inputs:
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

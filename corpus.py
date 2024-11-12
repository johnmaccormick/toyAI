from torch.utils.data import Dataset
import torch
import random


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
    def __init__(self, token_to_id=None, input_strings=None, token_str=None):

        self.create_dataset(token_to_id, input_strings, token_str)

    def create_dataset(self, token_to_id: dict[str, int], input_strings: list[str],
                       token_str: str):
        assert token_to_id is None or token_str is None
        assert input_strings is not None
        if token_str is not None:
            token_to_id = Corpus.token_dict_from_string(token_str)
        elif token_to_id is None:
            token_to_id = Corpus.token_dict_from_inputs(input_strings)

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
        assert ids.ndim == 1
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
        id = -1
        tokens.extend(['<PAD>', '<EOS>'])
        for tok in tokens:
            if tok not in token_to_id:
                id += 1
                token_to_id[tok] = id
        return token_to_id

    def token_dict_from_string(token_str: str) -> dict[str, int]:
        tokens = token_str.split()
        return Corpus.token_dict_from_tokens(tokens)

    def token_dict_from_inputs(input_strings: list[str]) -> dict[str, int]:
        return Corpus.token_dict_from_string(' '.join(input_strings))


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


class Synonym_inputs:
    def __init__(self, seed, num_syn_lists, fixed_order):
        self.num_syn_lists = num_syn_lists
        self.fixed_order = fixed_order
        self.rng = random.Random(seed)

        self.syn_lists = [
            ["happy", "joyful", "content"],
            ["fast", "quick", "speedy"],
            ["smart", "intelligent", "clever"]
        ]
        self.syn_lists = self.syn_lists[:self.num_syn_lists]
        self.syn_sets = [frozenset(l) for l in self.syn_lists]

        # Flatten syn_list to get all words in a single list
        self.all_words = frozenset(
            [word for group in self.syn_lists for word in group])
        self.all_words_list = sorted(list(self.all_words))

        self.syn_indexes = dict()  # str->int
        for i, this_list in enumerate(self.syn_lists):
            for word in this_list:
                self.syn_indexes[word] = i

    def find_synonym(self, input_str: str):
        words = input_str.split()
        return self.find_synonym_in_list(words)

    def find_synonym_in_list(self, words: list[str]):
        target = words[-1]
        syns = self.syn_sets[self.syn_indexes[target]]
        for word in words[:-1]:
            if word in syns:
                return word
        assert False, 'No synonym found'

    def are_synonyms(self, word1, word2):
        return self.syn_indexes[word1] == self.syn_indexes[word2]

    # def make_choice(self, vals):
    #     # We sort the list so that,
    #     # for a given rng state, the choice is deterministic.
    #     # If the param vals is something like a set or frozen set,
    #     # the order in which it is converted to a list seems to be
    #     # nondeterministic.
    #     return self.rng.choice(sorted(list(vals)))

    def make_input(self):
        # Randomly choose one word from each inner list
        chosen_words = [self.rng.choice(group)
                        for group in self.syn_lists]
        if not self.fixed_order:
            self.rng.shuffle(chosen_words)
        # Filter out words already chosen
        remaining_words = [
            word for word in self.all_words_list if word not in chosen_words]
        # Choose one additional word that wasn't already chosen
        additional_word = self.rng.choice(remaining_words)
        query = chosen_words + [additional_word]
        syn = self.find_synonym_in_list(query)
        # Combine the chosen words into a single string
        result_string = ' '.join(query + ['<EOS>'] + [syn])
        return result_string

    def make_inputs(self, num_inputs):
        inputs = []
        labels = []
        for _ in range(num_inputs):
            input_str = self.make_input()
            label = self.find_synonym(input_str)
            inputs.append(input_str)
            labels.append(label)
        return inputs, labels


def main():
    seed = 54545
    num_syn_lists = 2
    fixed_order = True
    si = Synonym_inputs(seed, num_syn_lists, fixed_order)
    inputs, labels = si.make_inputs(num_inputs=4)
    print([x for x in zip(inputs, labels)])
    # for input, label in zip(inputs, labels):


if __name__ == "__main__":
    main()

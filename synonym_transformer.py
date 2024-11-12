import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics.pairwise import cosine_distances
import basic_transformer as bt
import corpus


# written by cloab AI
def cosine_distance_matrix(embedding_module):
    """
    Calculates the cosine distance matrix between all embeddings in a PyTorch Embedding module.

    Args:
      embedding_module: A PyTorch nn.Embedding module.

    Returns:
      A NumPy array representing the cosine distance matrix.
    """

    # Get all embeddings from the module
    embeddings = embedding_module.weight.detach().cpu().numpy()

    # Calculate cosine distances between all pairs of embeddings
    distance_matrix = cosine_distances(embeddings)

    return distance_matrix


def learn_syns_fixed_order():
    # Succeeded learning synonyms for fixed order of inputs of the categories,
    # 3 syn lists.
    btp = bt.BasicTransformerParams()
    btp.num_epochs = 200
    btp.d_model = 16
    btp.num_attn_heads = 8
    btp.attn_head_config = 'multicompact'
    btp.use_2ffn = True
    btp.d_qk = btp.d_model
    btp.d_vo = btp.d_model
    btp.d_ffn = 32

    btp.use_attn_layers = False
    btp.batch_size = 10
    btp.learning_rate = 0.002
    btp.loss_print_freq = 50

    num_syn_lists = 3
    num_in_strs = 300
    fixed_order = True

    si = corpus.Synonym_inputs(seed=btp.seed, num_syn_lists=num_syn_lists,
                               fixed_order=fixed_order)
    inputs, labels = si.make_inputs(num_inputs=num_in_strs)
    corp = corpus.Corpus(input_strings=inputs)
    print(f'num_in_strs {num_in_strs}, vocab size {corp.vocab_size}')
    start_seed = 123
    response_err_vals = []
    for i in range(5):
        btp.seed = start_seed + i
        model, optimizer, dataloader = bt.create_model(btp, corp)
        avg_loss = bt.do_epochs(model, optimizer, dataloader)
        response_errs = bt.count_errors(
            model, dataloader, response_errs_only=True, corp=corp)
        response_err_vals.append(response_errs)
    print(f'response_err_vals: {response_err_vals}')


def learn_syns_random_order():
    btp = bt.BasicTransformerParams()
    btp.num_epochs = 500
    btp.d_model = 16
    btp.num_attn_heads = 16
    btp.attn_head_config = 'multicompact'
    btp.use_2ffn = True
    btp.d_qk = btp.d_model
    btp.d_vo = btp.d_model
    btp.d_ffn = 32

    btp.use_attn_layers = False
    btp.batch_size = 10
    btp.learning_rate = 0.002
    btp.loss_print_freq = 100

    num_syn_lists = 3
    num_in_strs = 300
    fixed_order = False

    si = corpus.Synonym_inputs(seed=btp.seed, num_syn_lists=num_syn_lists,
                               fixed_order=fixed_order)
    inputs, labels = si.make_inputs(num_inputs=num_in_strs)
    corp = corpus.Corpus(input_strings=inputs)
    print(f'num_in_strs {num_in_strs}, vocab size {corp.vocab_size}')
    start_seed = 123
    response_err_vals = []
    for i in range(5):
        btp.seed = start_seed + i
        model, optimizer, dataloader = bt.create_model(btp, corp)
        avg_loss = bt.do_epochs(model, optimizer, dataloader)
        response_errs = bt.count_errors(
            model, dataloader, response_errs_only=True, corp=corp)
        response_err_vals.append(response_errs)
    print(f'response_err_vals: {response_err_vals}')


def create_and_save_syn_model():
    btp = bt.BasicTransformerParams()
    btp.num_epochs = 200
    btp.d_model = 16
    btp.num_attn_heads = 16
    btp.attn_head_config = 'multicompact'
    btp.use_2ffn = True
    btp.d_qk = btp.d_model
    btp.d_vo = btp.d_model
    btp.d_ffn = 32

    btp.use_attn_layers = False
    btp.batch_size = 10
    btp.learning_rate = 0.002
    btp.loss_print_freq = 100

    num_syn_lists = 3
    num_in_strs = 300
    fixed_order = False

    si = corpus.Synonym_inputs(seed=btp.seed, num_syn_lists=num_syn_lists,
                               fixed_order=fixed_order)
    inputs, labels = si.make_inputs(num_inputs=num_in_strs)
    corp = corpus.Corpus(input_strings=inputs)
    print(f'num_in_strs {num_in_strs}, vocab size {corp.vocab_size}')
    start_seed = 123
    response_err_vals = []
    btp.seed = start_seed
    model, optimizer, dataloader = bt.create_model(btp, corp)
    avg_loss = bt.do_epochs(model, optimizer, dataloader)
    response_errs = bt.count_errors(
        model, dataloader, response_errs_only=True, corp=corp)
    response_err_vals.append(response_errs)
    print(f'response_err_vals: {response_err_vals}')
    torch.save(model.state_dict(), 'syn-model.pth')


def investigate_syn_model():
    btp = bt.BasicTransformerParams()
    btp.num_epochs = 50
    btp.d_model = 16
    btp.num_attn_heads = 16
    btp.attn_head_config = 'multicompact'
    btp.use_2ffn = True
    btp.d_qk = btp.d_model
    btp.d_vo = btp.d_model
    btp.d_ffn = 32

    btp.use_attn_layers = False
    btp.batch_size = 10
    btp.learning_rate = 0.002
    btp.loss_print_freq = 100

    num_syn_lists = 3
    num_in_strs = 300
    fixed_order = False

    si = corpus.Synonym_inputs(seed=btp.seed, num_syn_lists=num_syn_lists,
                               fixed_order=fixed_order)
    inputs, labels = si.make_inputs(num_inputs=num_in_strs)
    corp = corpus.Corpus(input_strings=inputs)
    start_seed = 123
    response_err_vals = []
    btp.seed = start_seed
    model, optimizer, dataloader = bt.create_model(btp, corp)
    model.load_state_dict(torch.load('syn-model.pth', weights_only=True))
    response_errs = bt.count_errors(
        model, dataloader, response_errs_only=True, corp=corp)
    response_err_vals.append(response_errs)
    print(f'response_err_vals: {response_err_vals}')

    print_some_predictions(corp, model)

    # cosines = cosine_distance_matrix(model.we)
    # print(f'cosine dists:\n {cosines}')
    return model, corp, si


def print_some_predictions(corp: corpus.Corpus, model: bt.DecoderOnlyTransformer):
    dloader_batch1 = DataLoader(corp.dataset, batch_size=1)
    num_to_print = 15
    for instance, (X, y) in enumerate(dloader_batch1):
        if instance >= num_to_print:
            break
        y_pred = model(X.to(model.btp.device))
        predicted_ids = torch.argmax(y_pred, dim=-1)
        print(f'{instance} : ' +
              f'{corp.ids_to_string(X.squeeze())} --> ' +
              f'{corp.ids_to_string(predicted_ids.squeeze())}')


def analyze_cosine_distances(model: bt.DecoderOnlyTransformer, corp: corpus.Corpus, si: corpus.Synonym_inputs):
    cosines = cosine_distance_matrix(model.we)
    for group in si.syn_lists:
        for i, word1 in enumerate(group):
            for word2 in group[i+1:]:
                assert si.are_synonyms(word1, word2)
                dist = cosines[corp.token_to_id[word1],
                               corp.token_to_id[word2]]
                print(word1, word2, dist)
        print()
    intra_tot, num_intra = 0.0, 0
    inter_tot, num_inter = 0.0, 0
    intra_vals = []
    inter_vals = []
    for i, word1 in enumerate(si.all_words_list):
        for word2 in si.all_words_list[i+1:]:
            id1 = corp.token_to_id[word1]
            id2 = corp.token_to_id[word2]
            if si.are_synonyms(word1, word2):
                num_intra += 1
                intra_tot += cosines[id1, id2]
                intra_vals.append(cosines[id1, id2])
            else:
                num_inter += 1
                inter_tot += cosines[id1, id2]
                inter_vals.append(cosines[id1, id2])
    intra_avg = intra_tot / num_intra
    inter_avg = inter_tot / num_inter
    print(f'intra_avg {intra_avg}, num_intra {num_intra}')
    print(f'inter_avg {inter_avg}, num_inter {num_inter}')
    t_statistic, p_value = stats.ttest_ind(intra_vals, inter_vals)
    print(f'p-value of T test for difference of two means: {p_value}')

    plt.plot(intra_vals, 'o', label='intra_vals')
    plt.plot(inter_vals, 'x', label='inter_vals')

    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Visualization of synonym cosine distances')
    plt.legend()

    plt.show()


def main():
    model, corp, si = investigate_syn_model()
    analyze_cosine_distances(model, corp, si)
    # create_and_save_syn_model()


# torch.save(model.state_dict(), 'syn-model.pth')

if __name__ == "__main__":
    main()

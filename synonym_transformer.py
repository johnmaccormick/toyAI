import torch
from torch.utils.data import Dataset, DataLoader
import basic_transformer as bt
import corpus


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


def main():
    investigate_syn_model()
    # create_and_save_syn_model()


# torch.save(model.state_dict(), 'syn-model.pth')

if __name__ == "__main__":
    main()

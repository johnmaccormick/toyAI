import basic_transformer as bt
import corpus


def get_Alphabet_corpus():
    alph = corpus.Alphabet_inputs
    corp = corpus.Corpus(token_to_id=alph.token_to_id,
                         input_strings=alph.input_strings)
    return corp


def learn_alphabet_fixed_emb():
    corp = get_Alphabet_corpus()
    btp = bt.BasicTransformerParams()
    btp.batch_size = 1
    btp.num_attn_heads = 1
    btp.attn_head_config = 'single'
    vocab_size = len(corp.id_to_token)
    btp.d_model = vocab_size
    btp.d_qk = vocab_size
    btp.d_vo = vocab_size
    btp.use_2ffn = False
    btp.use_fixed_embedding = True
    model, optimizer, dataloader = bt.create_model(btp, corp)
    avg_loss = bt.do_epochs(model, optimizer, dataloader)
    response_errs = bt.count_errors(
        model, dataloader, response_errs_only=True, corp=corp)
    print(f'avg_loss {avg_loss}, response_errs {response_errs}')
    bt.print_some_predictions(corp, model)
    bt.print_params(model)


def main():
    learn_alphabet_fixed_emb()


# torch.save(model.state_dict(), 'syn-model.pth')

if __name__ == "__main__":
    main()

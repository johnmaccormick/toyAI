import torch
from torch.utils.data import Dataset, DataLoader
import basic_transformer as bt
import corpus


def main():
    btp = bt.BasicTransformerParams()
    btp.num_epochs = 150
    btp.d_model = 4
    btp.num_attn_heads = 3
    btp.attn_head_config = 'multicompact'
    btp.use_2ffn = True

    num_in_strs = 100
    inputs, labels = corpus.Synonym_inputs.make_inputs(
        num_inputs=num_in_strs, seed=54545)
    corp = corpus.Corpus(input_strings=inputs)
    print(f'num_in_strs {num_in_strs}, vocab size {corp.vocab_size}')
    model, optimizer, dataloader = bt.create_model(btp, corp)
    avg_loss = bt.do_epochs(model, optimizer, dataloader)
    response_errs = bt.count_errors(
        model, dataloader, response_errs_only=True, corp=corp)
    dataloader = DataLoader(corp.dataset, batch_size=1)
    num_instances_to_print = 20
    for instance, (X, y) in enumerate(dataloader):
        if instance >= num_instances_to_print:
            break
        num_in_tokens = X.shape[1]
        y_pred = model(X.to(model.btp.device))
        predicted_ids = torch.argmax(y_pred, dim=-1)
        assert predicted_ids.shape == (1, num_in_tokens)
        predicted_str = corp.ids_to_string(predicted_ids.squeeze())
        assert X.shape == (1, num_in_tokens)
        in_str = corp.ids_to_string(X.squeeze())
        print(f'in_str {in_str}, predicted_str {predicted_str}')


if __name__ == "__main__":
    main()

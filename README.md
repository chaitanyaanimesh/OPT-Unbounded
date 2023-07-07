# OPT-Unbounded
An implementation in Pytorch showing how to replace the learned position embedding with positional encoding in Meta's OPT models to make the context length unbounded. 
It also shows how to re(learn) the token embeddings with the new position encoding scheme by fine-tuning on a different dataset.

# How to Run

Specify the model, dataset, batch size and the context (max sequence) length for fine-tuning.

```
python3 main.py --model facebook/opt-iml-max-1.3b --dataset OpenAssistant/oasst1 --batch_size 64 --max_seq_length 512
```
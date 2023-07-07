import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, DataCollatorWithPadding
from datasets import load_dataset
from transformers import AdamW, get_scheduler
from tqdm.auto import tqdm
from positional_encoding import PositionalEncoding
from data import DataTokenizer
import constants
from positional_encoding import PositionalEncoding

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="facebook/opt-125m",
        choices=[
            "facebook/opt-iml-1.3b",
            "facebook/opt-iml-max-1.3b",
            "facebook/opt-125m",
            "facebook/opt-350m"
        ],
        help="Model to use",
    )
    parser.add_argument(
        "--dataset",
        default="OpenAssistant/oasst1",
        choices=["OpenAssistant/oasst1", "wikitext"],
        help="Dataset Name",
    )
    
    parser.add_argument(
        "--dataset_config",
        default="wikitext-2-raw-v1",
        choices=["wikitext-2-raw-v1", "wikitext-103-raw-v1"],
        help="dataset config if any. To be used for wikitext.",
    )
    
    parser.add_argument(
        "--add_labels",
        default=True,
        type=bool,
        help="Specify if you want to add labels for Causal Language Modelling",
    )
    
    parser.add_argument("--freeze_layers", default=False, type=bool, help="If set true, only embeddings well be learned.")
    
    parser.add_argument("--batch_size", default=64, type=int, help="Batch Size for training")
    parser.add_argument(
        "--max_seq_length",
        default=256,
        type=int,
        help="Max sequence length for training beyond which truncation will be done"
    )

    parser.add_argument("--n_epochs", default=5, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--lr_scheduler_type", type=str, default='linear', help="LR Scheduler Type")
    
    parser.add_argument("--checkpoint_path", default='./checkpoint/', type=str, help="path where dataset would be downloaded")
    
    args = parser.parse_args(args=[])
    return args

def checkpoint(args, model):
    state = {
            "net": model.state_dict(),
            "dataset": args.dataset,
            "epochs": args.n_epochs,
            "lr_contrastive": args.lr,   
        }
            
    path=args.checkpoint_path+args.model+".pth"
    torch.save(state, path)


def build_unbounded_model(args):
    model_name=args.model
    model_config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.model.decoder.embed_positions = PositionalEncoding(embedding_dim=model_config.word_embed_proj_dim)
    if args.freeze_layers:
        for param in model.parameters():
            param.requires_grad=False
        
    model.model.decoder.embed_tokens.weight.requires_grad=True
    
    return model
    
    
def train(args, data_tokenizer):
    
    
    tokenized_datasets = data_tokenizer.load_dataset()
   
    data_collator = DataCollatorWithPadding(data_tokenizer.get_tokenizer(),return_tensors="pt")
    train_dataloader = DataLoader(tokenized_datasets[constants.TRAIN], shuffle=True, batch_size=args.batch_size, collate_fn=data_collator)
    eval_dataloader = DataLoader(tokenized_datasets[constants.VALIDATION], batch_size=args.batch_size, collate_fn=data_collator)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = build_unbounded_model(args).to(device)
    
    optimizer = AdamW(model.parameters(), lr=args.lr)
    num_epochs = args.n_epochs
    num_training_steps = num_epochs * len(train_dataloader)
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    
    model.train()
    
    for epoch in range(num_epochs):
        train_losses=[]
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch[constants.INPUT_IDS],
                            attention_mask=batch[constants.ATTENTION_MASK],labels=batch[constants.LABELS])
            loss = outputs.loss
            train_losses.append(loss.detach().item())
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()


        val_losses=[]    
        model.eval()
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            val_losses.append(outputs.loss.item())
        print('After Epoch=',epoch, ' Train Loss=', torch.mean(torch.tensor(train_losses)).item()
              , ' Validation loss=',torch.mean(torch.tensor(val_losses)).item())
        
        train_losses.clear()
        val_losses.clear()

        model.train()
        
    
    checkpoint(args, model)
    return model

def inference(model, data_tokenizer):
    tokenizer = data_tokenizer.get_tokenizer()
    batch = tokenizer("I will generate a 2500 word long story. Once upon a time, there lived a king", return_tensors='pt').to('cuda')
    output_tokens = model.generate(**batch, max_length=2500)
    print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))
 
    


if __name__=='__main__':
    args= parse_args()
    data_tokenizer = DataTokenizer(args)
    model = train(args, data_tokenizer)
    inference(model, data_tokenizer)
    
    
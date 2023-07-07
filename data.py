import constants
from transformers import AutoTokenizer
from datasets import load_dataset
import datasets
import transformers
import platform

class DataTokenizer:
    
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.max_seq_length = args.max_seq_length
        self.text_col = constants.TEXT_COLUMN
        self.add_labels = args.add_labels
        self.dataset=None
        self.remove_columns=None
        self.num_procs=4
        if args.dataset == 'wikitext':
            self.dataset = load_dataset(args.dataset, args.dataset_config)
            self.remove_columns = constants.WIKITEXT_REMOVE_COLUMNS.split(",")
            self.num_procs = constants.WIKITEXT_NUM_PROCS
        else:
            self.dataset = load_dataset(args.dataset)
            self.remove_columns = constants.OPENASSISTANT_REMOVE_COLUMNS.split(",")
            self.num_procs = constants.OPENASSISTANT_NUM_PROCS
            
        
    def _tokenize_function(self, examples):
        result = self.tokenizer(
            examples[self.text_col],
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_length,
        )
        if self.add_labels:
            result["labels"] = result["input_ids"].copy()
        return result
    
    def get_tokenizer(self):
        return self.tokenizer;

    def load_dataset(self):

        tokenized_datasets = self.dataset.map(self._tokenize_function, batched=True, num_proc=self.num_procs, remove_columns=self.remove_columns)
        return tokenized_datasets
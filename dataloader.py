from torch.utils.data import DataLoader
from transformers import AutoTokenizer, RobertaTokenizer
from dataset import TextDatset
import torch

class TextDataLoader(DataLoader):
    def __init__(self, split, batch_size, model_use, shuffle):
        self.model_use = model_use
        if self.model_use == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
        self.dataset = TextDatset(split, model_use)
        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': batch_size,
            'shuffle': shuffle,
            'collate_fn': self.collate_fn,
        }
        super().__init__(**self.init_kwargs)
        
    def collate_fn(self, data):
        id, text, target = zip(*data)
        
        text_tokenized = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        text_ids, text_mask = text_tokenized.input_ids, text_tokenized.attention_mask
        
        
        if self.model_use == 't5':
            target_tokenized = self.tokenizer(target, padding=True, truncation=True, return_tensors='pt')
            labels, labels_mask = target_tokenized.input_ids, target_tokenized.attention_mask
        else:
            labels_mask = None
            labels = torch.tensor(target)

        return id, text_ids, text_mask, labels, labels_mask
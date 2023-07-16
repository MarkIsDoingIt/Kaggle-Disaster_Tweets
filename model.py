from transformers import T5ForConditionalGeneration, AutoTokenizer, RobertaTokenizer, RobertaModel
import torch
from peft import LoraConfig, get_peft_model, TaskType
import torch.nn as nn
import pdb

class LLMClassifier(nn.Module):
    def __init__(self, model_use):
        super().__init__()
        #Initialize model
        self.model_use = model_use
        if self.model_use == 'roberta':
            self.decoder = RobertaModel.from_pretrained('roberta-large', device_map={"":0})
            self.fc = nn.Linear(self.decoder.config.hidden_size, 2).to(torch.device('cuda'))
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        else:
            self.decoder = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", device_map={"":0})
            self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
            lora_config = LoraConfig(
                r=32,
                lora_alpha=16, 
                target_modules=["o", "wi_1", "v", "q", "wo", "wi_0"],
                lora_dropout=0.1, 
                bias="none", 
            )
            self.decoder = get_peft_model(self.decoder, lora_config)
       
        self.device = torch.device('cuda:0')
        
        #Initialize prefix prompt for prompt tuning
        self.prefix_mask = torch.load('/home/broca/kg/mask_roberta.pt').to(self.device)
        self.prefix_embedding = torch.load('/home/broca/kg/prefix_embedding_roberta.pt').to(self.device)
        


        
    def forward(self, text_ids, text_mask, labels = None, labels_mask = None):
        if self.model_use == 'roberta':
            text_embedding = self.decoder.embeddings(text_ids)
        else:
            text_embedding = self.decoder.shared(text_ids)
        
        prefix_embedding = self.prefix_embedding.expand(text_embedding.shape[0], -1, -1)
        prefix_mask = self.prefix_mask.expand(text_embedding.shape[0], -1)
        
        input_embedding = torch.cat((prefix_embedding, text_embedding), dim=1)
        input_mask = torch.cat((prefix_mask, text_mask), dim=1)
        
        # labels = labels.clone().detach()
        # labels[labels_mask == 0] = -100
        if self.model_use == 't5model':
            output = self.decoder(inputs_embeds=input_embedding, attention_mask=input_mask, labels=labels)
        else:
            output = self.decoder(inputs_embeds=input_embedding, attention_mask=input_mask).pooler_output
            output = self.fc(output)
            
        return output
    
    def classify(self, text_ids, text_mask):
        text_embedding = self.decoder.shared(text_ids)
        
        prefix_embedding = self.prefix_embedding.expand(text_embedding.shape[0], -1, -1)
        prefix_mask = self.prefix_mask.expand(text_embedding.shape[0], -1)
        
        input_embedding = torch.cat((prefix_embedding, text_embedding), dim=1)
        input_mask = torch.cat((prefix_mask, text_mask), dim=1)
        
        
        output = self.decoder.generate(inputs_embeds=input_embedding, attention_mask=input_mask)
        return output
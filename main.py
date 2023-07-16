import torch
import torch.nn as nn
from dataloader import TextDataLoader
from model import LLMClassifier
from tqdm import tqdm
from transformers import AutoTokenizer, RobertaTokenizer
import pandas as pd
import csv

epochs = 200
lr = 1e-6
batch_size = 32
test_batch_size = 32
model_use = 'roberta'
device = torch.device('cuda:0')
if model_use == 'roberta':
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
else:
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")

split = 'train'
train_loader = TextDataLoader(split, batch_size, model_use, shuffle=True)

split = 'valid'
valid_loader = TextDataLoader(split, test_batch_size, model_use, shuffle=False)

split = 'test'
test_loader = TextDataLoader(split, test_batch_size, model_use, shuffle=False)

model = LLMClassifier(model_use)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
check_step = 10
loss_accumulation = 0
loss_func = nn.CrossEntropyLoss()

do_training = True
if do_training:
    
    # valid_length, valid_matches = 0, 0       
    # for iteration, (id, text_ids, text_mask, labels, labels_mask) in enumerate(tqdm(valid_loader)):
    #     text_ids, text_mask, labels, labels_mask = text_ids.to(device), text_mask.to(device), labels.to(device), labels_mask.to(device)
    #     predicted_ids = model.classify(text_ids, text_mask)
    #     predicted_value = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)  
    #     ans_value = tokenizer.batch_decode(labels, skip_special_tokens=True)
    #     matches = [pre == ans for pre, ans in zip(predicted_value, ans_value)]
    #     valid_matches += sum(matches)
    #     valid_length += len(matches)

    # accuracy = valid_matches / valid_length
    # print('\nAccuracy before training is: ', accuracy, flush = True)  


    for epoch in range(epochs):
        print('Epoch ', epoch, flush=False)
        
        for iteration, (id, text_ids, text_mask, labels, labels_mask) in enumerate(tqdm(train_loader)):
            text_ids, text_mask = text_ids.to(device), text_mask.to(device)
            if model_use == 't5':
                labels, labels_mask = labels.to(device), labels_mask.to(device)
            output = model(text_ids, text_mask, labels, labels_mask)
            
            if model_use == 'roberta':
                output = output.to(torch.device('cpu'))
                loss = loss_func(output, labels)
            else:
                loss = output.loss
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  
            
            loss_accumulation += loss.item()
            if iteration % check_step == 0 and iteration > 0:
                print('\nLoss of ',iteration , 'iteration is: ', loss_accumulation / check_step, flush=True)
                loss_accumulation = 0
                # output = model.classify(text_ids, text_mask)
                # pred = tokenizer.batch_decode(output, skip_special_tokens=True)
                # ans = tokenizer.batch_decode(labels, skip_special_tokens=True)
                # print(pred, ans, flush=True)
            
        
        
        valid_length, valid_matches = 0, 0       
        for iteration, (id, text_ids, text_mask, labels, labels_mask) in enumerate(tqdm(valid_loader)):
            text_ids, text_mask = text_ids.to(device), text_mask.to(device)
            if model_use == 't5':
                labels, labels_mask = labels.to(device), labels_mask.to(device)
                
            if model_use == 'roberta':
                predicted = model(text_ids, text_mask)
                predicted_value = torch.argmax(predicted, dim=1)
                predicted_value = predicted_value.to(torch.device('cpu'))
                ans_value = labels
            else:
                predicted_ids = model.classify(text_ids, text_mask)
                predicted_value = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)  
                ans_value = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            matches = [pre == ans for pre, ans in zip(predicted_value, ans_value)]
            valid_matches += sum(matches)
            valid_length += len(matches)
        
        accuracy = valid_matches / valid_length
        print('\nAccuracy for ', epoch + 1, ' epoch is: ', accuracy, flush = True)        
        
        if epoch == 3:
            torch.save(model.state_dict(), '/home/broca/kg/LLMClassifier_4.pt')
        if epoch == 9:
            torch.save(model.state_dict(), '/home/broca/kg/LLMClassifier_10.pt')
        if epoch == 19:
            torch.save(model.state_dict(), '/home/broca/kg/LLMClassifier_20.pt')
        if epoch == 49:
            torch.save(model.state_dict(), '/home/broca/kg/LLMClassifier_50.pt')
        if epoch == 100:
            torch.save(model.state_dict(), '/home/broca/kg/LLMClassifier_99.pt')
        if epoch == 200:
            torch.save(model.state_dict(), '/home/broca/kg/LLMClassifier_199.pt')
            
            

            

state_dict = torch.load('/home/broca/kg/LLMClassifier_4.pt', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
    
result_list = []
model.eval()
with torch.no_grad():
    for iteration, (id, text_ids, text_mask, labels, labels_mask) in enumerate(tqdm(test_loader)):
        text_ids, text_mask = text_ids.to(device), text_mask.to(device)
        if model_use == 't5':
            labels, labels_mask = labels.to(device), labels_mask.to(device)

        
        if model_use == 'roberta':
            predicted = model(text_ids, text_mask)
            predicted_value = torch.argmax(predicted, dim=1)
            predicted_value = predicted_value.to(torch.device('cpu'))
        else:
            predicted_ids = model.classify(text_ids, text_mask)
            predicted_value = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)  
            
            
        for i in range(len(id)):
            result_list.append((id[i], int(predicted_value[i])))


with open('result.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'target'])
    for row in result_list:
        writer.writerow(row)



    

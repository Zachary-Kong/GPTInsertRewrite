from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import pipeline
from transformers import Trainer, TrainingArguments
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import argparse
import os


import torch
from csv_dataloader import csv_dataloader

#
def inference(model, tokenizer, device, save_path):
    model.eval()
    q = ['quit', 'q']
    while True:
        prompt = input("Input prompt for rewriting ('quit' or 'q' to exit): ")
        if prompt in q:
            break

        input_text = '<input> ' + prompt + ' <paraphrase>' #+ prompt[:prompt.find('@')] + '<p2>' #added prefix here
        input_text = input_text.replace('@', '<p1>', 1)
        input_text = input_text.replace('@', '<p1/>', 1)
        token = tokenizer(input_text, return_tensors='pt')
        x_text = token["input_ids"].to(device)
        x_mask = token["attention_mask"].to(device)

        output_token = model.generate(x_text, attention_mask=x_mask, max_new_tokens=64)
        output = tokenizer.decode(output_token[0])
        print(output)

        if save_path:
            with open(save_path + '/inference.txt', 'a', encoding="utf-8") as f:
                f.write("Input: " + input_text + '\n')
                f.write("Paraphrase: " + output + '\n\n')

def eval(dataset, tokenizer, model, device, save_path):
    model.eval()
    for text, _, num in DataLoader(dataset, shuffle=False):
        for i in range(num):
            token = tokenizer(text[i][0], return_tensors='pt')
            x_text = token["input_ids"].to(device)
            x_mask = token["attention_mask"].to(device)
            
            output_token = model.generate(x_text, attention_mask=x_mask, max_new_tokens=64)
            output = tokenizer.decode(output_token[0])

            if save_path:
                with open(save_path + '/eval.txt', 'a', encoding="utf-8") as f:
                    f.write("Input: " + text[i][0] + '\n')
                    f.write("Paraphrase: " + output + '\n\n')

def baseline(dataset, save_path):
    # Use a pipeline as a high-level helper
    pipe = pipeline("text-generation", model="SRM47/gpt2-paraphraser")
    for text, _, num in DataLoader(dataset, shuffle=False):
        for i in range(num):
            input_text = text[i][0]
            input_text = input_text.replace('<input>', '', 1)
            input_text = input_text.replace('<p1>', '<s>', 1)
            input_text = input_text.replace('<p1/>', '</s>', 1)
            input_text = input_text.replace(' <paraphrase>', '>>>><p>', 1)
            output = pipe(input_text)[0]['generated_text']
            end = output.rfind('</p>')

            if end > 0:
                output = output[:end+4]

            if save_path:
                with open(save_path + '/baseline.txt', 'a', encoding="utf-8") as f:
                    f.write("Input: " + input_text + '\n')
                    f.write("Paraphrase: " + output + '\n\n')



def train(dataset, model, batch, epochs, optimizer, device, save_step, save_path, token_path):
    model.train()
    num_training_ex = len(dataset)
    multiplier = 1
    lowest = 100

    for epoch in range(epochs):
        print("EPOCH ", epoch)
        avg_loss = 0
        for x_text, x_mask, num in DataLoader(dataset, batch_size=batch, shuffle=True):
            for i in range(num):
                multiplier = num

                optimizer.zero_grad()
                input_text, input_mask = x_text[i].to(device), x_mask[i].to(device)
                outputs = model(input_text, attention_mask=input_mask, labels=input_text)#, encoder_attention_mask=label_mask)
                loss = outputs.loss
                avg_loss += loss.item()
                loss.backward()
                optimizer.step()

        avg_loss /= (num_training_ex*multiplier)
        print("average loss: ", avg_loss)

        if epoch % save_step == 0:
            fname = '/' + str(epoch) + '.pt'
            torch.save(model.state_dict(), save_path + fname)
            dataset.get_tokenizer().save_pretrained(token_path)

            if avg_loss < lowest:
                lowest = avg_loss
                torch.save(model.state_dict(), save_path + '/best.pt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='path to data')
    parser.add_argument('--token-path', type=str, help='path to tokenizer')
    parser.add_argument('--weights', type=str, default='', help='description for option2')
    parser.add_argument('--device', type=str, default='cuda', help='description for option2')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--save-step', type=int, default=1, help='save weights every [--save-step] epoch')
    parser.add_argument('--save-path', type=str, help='path to directory where to save weight files')
    parser.add_argument('--save-token', type=str, help='path to directory where to save weight files')
    parser.add_argument('--tsv', action='store_true')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--baseline', action='store_true')
    opt = parser.parse_args()

    if opt.token_path:
        tokenizer = GPT2Tokenizer.from_pretrained(opt.token_path)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.add_special_tokens({"eos_token": tokenizer.eos_token})
        tokenizer.add_tokens(["<p>", "<p/>"])
        tokenizer.add_tokens(["<paraphrase>", "<input>"])
    #tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained('gpt2').to(opt.device)
    model.resize_token_embeddings(len(tokenizer))

    if opt.weights:
        model.load_state_dict(torch.load(opt.weights))
        model.to(opt.device)

    #Create folders
    if opt.save_path and not os.path.isdir(opt.save_path):
        os.mkdir(opt.save_path)

    if opt.save_token and not os.path.isdir(opt.save_token):
        os.mkdir(opt.save_token)
        
    
    optimizer = Adam(model.parameters(), lr=1e-5)
    
    if opt.tsv:
        sep = '\t'
    else:
        sep = ','

    if not opt.inference:
        if opt.eval or opt.baseline:
            mode = 'eval'
        else:
            mode = 'train'
        dataset = csv_dataloader(opt.path, sep, tokenizer, mode=mode)
    if opt.baseline:
        baseline(dataset, opt.save_path)
    elif opt.inference:
        inference(model, tokenizer, opt.device, opt.save_path)
    elif opt.eval:
        eval(dataset, tokenizer, model, opt.device, opt.save_path)
    else:
        train(dataset, model, opt.batch_size, opt.epochs, optimizer, opt.device, opt.save_step, opt.save_path, opt.save_token)
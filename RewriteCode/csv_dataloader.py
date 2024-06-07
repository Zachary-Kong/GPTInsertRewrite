from torch.utils.data import Dataset
import pandas as pd
import random

class csv_dataloader(Dataset):
    def __init__(self, path, separator, tokenizer, max_len=64, mode='train'):
        self.df = pd.read_csv(path, header=None, sep=separator, encoding='utf-8')
        self.shape = self.df.shape
        self.tokenizer = tokenizer
        self.mode = mode
        self.inputs = self.df[0]
        if self.shape == 2:
            self.labels = self.df[1]


    def __getitem__(self, idx):
        #Check if prefix and suffix are the same:
        s1 = str(self.inputs[idx])

        if self.shape[1] == 2:
            s2 = str(self.labels[idx])

            s1p = s1[:s1.find("@")]
            s2p = s2[:s2.find("@")]
            assert s1p.lower() == s2p.lower(), 'Training example ' + str(idx+1) + ' has different prefixes: ' + s1p + ' vs ' + s2p

            s1s = s1[s1.rfind("@")+1:]
            s2s = s2[s2.rfind("@")+1:]
            assert s1s.lower() == s2s.lower(), 'Training example ' + str(idx+1) + ' has different suffixes ' + s1s + ' vs ' + s2s

        if self.mode == 'train':
            flip = random.randint(0, 1)
            if flip:
                temp = s1
                s1 = s2
                s2 = temp

            s1 = s1.replace('@', '<p1>', 1)
            s1 = s1.replace('@', '<p1/>', 1)
            s2 = s2.replace('@', '<p2>', 1)
            s2 = s2.replace('@', '<p2/>', 1)

            input_text = []
            input_text.append('<input> ' + s1[:s1.find("<p1/>")+5] + ' <paraphrase> ' + s2[:s2.find("<p2/>")+5]) #prefix + middle   + ' '+  self.tokenizer.eos_token
            input_text.append(s1[s1.find("<p1>"):] + ' <paraphrase> ' + s2[s2.find("<p2>"):] + ' ' + self.tokenizer.eos_token) #middle + suffix    '<input> ' + 
            #input_text.append('<input> ' + s1 + ' <paraphrase> ' + s2 + ' ' + self.tokenizer.eos_token) # entire sentence# 

            token_input = []
            attention_mask = []
            for text in input_text:
                x = self.tokenizer(text, return_tensors='pt')
                token_input.append(x["input_ids"])
                attention_mask.append(x["attention_mask"])

        elif self.mode == 'eval':
            s1 = s1.replace('@', '<p1>', 1)
            s1 = s1.replace('@', '<p1/>', 1)

            input_text = []
            text1 = '<input> ' + s1 + ' <paraphrase>'

            input_text.append(text1)

            token_input = input_text
            attention_mask = [0]

        return token_input, attention_mask, len(token_input)

    
    def get_tokenizer(self):
        return self.tokenizer

    def __len__(self):
        return self.shape[0]
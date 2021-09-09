import os
import sys
import argparse
import pickle

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence as pad

import pytorch_lightning as pl

from transformers import BertJapaneseTokenizer
from transformers import AutoModel


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', help='the directory contains train.pkl, valid.pkl, test.pkl', type=str, default='./')
    parser.add_argument('--model_name', help='huggingface model name', type=str, default='cl-tohoku/bert-base-japanese-v2')
    parser.add_argument('--tokenizer_name', help='huggingface tokenizer name', type=str, default='cl-tohoku/bert-base-japanese-v2')
    parser.add_argument('--gpu_count', help='total count of gpus', type=int, default=1)
    parser.add_argument('--batch_size', help='batch size for train, valid, test', type=int, default=1)
    parser.add_argument('--num_workers', help='total count of threads for DataLoader', type=int, default=1)
    parser.add_argument('--learning_rate', help='learning rate for BERT training', type=float, default=1e-3)
    
    args = parser.parse_args()
    
    return args
    
class BERTDataset(torch.utils.data.Dataset):
    def __init__(self, source, target):
        self.source = source
        self.target = target
        self.lengths = torch.tensor([len(x) for x in source])
        self.size = len(source)
    
    def __len__(self):
        return self.size
            
    def __getitem__(self, index):
        return {
            'src':torch.tensor(self.source[index]),
            'tgt':torch.tensor(self.target[index]),
            'lens':self.lengths[index]}
    
    def collate(self, xs):
        src = pad([x['src'] for x in xs], batch_first=True)
        tgt = torch.stack([x['tgt'] for x in xs], dim=-1)
        lens = torch.stack([x['lens'] for x in xs], dim=-1)
        return {'src': src, 'tgt': tgt, 'lens': lens}
    
class BERTClassifier(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        self.loss = nn.CrossEntropyLoss().cuda()
        self.bert = AutoModel.from_pretrained(self.args.model_name)
        self.linear = nn.Linear(768, 2)
        
    def forward(self, x):
        x = self.bert(x)
        x = x['last_hidden_state'].permute(1,0,2)
        x = x[0]
        x = self.linear(x)
        return x
    
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x = batch['src']
        y = batch['tgt']
        
        x = self(x)
        
        loss = self.loss(x, y)
        
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch['src']
        y = batch['tgt']
        
        x = self(x)
        
        loss = self.loss(x, y)
        
        self.log("valid_loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        return {'optimizer': optimizer, 'frequency': 1}
    
def main():
    args = get_args()
    
    tokenizer = BertJapaneseTokenizer.from_pretrained(args.tokenizer_name)
    raw_data = {}
    all_data = []
    all_data = {'train':[], 'valid':[], 'test':[]}

    for datatype in ['train', 'valid', 'test']:
        with open(f'{args.data_dir}/{datatype}.pkl', 'rb') as f:
            raw_data[datatype] = pickle.load(f)
        for one_data in raw_data[datatype]:
            tokenized = tokenizer(one_data[0])
            all_data[datatype].append([tokenized['input_ids'], one_data[1]])
        
    dataset = {}
    data_loader = {}
    for datatype in ['train', 'valid', 'test']:
        dataset[datatype] = BERTDataset([x[0] for x in all_data[datatype]], [x[1] for x in all_data[datatype]])
        data_loader[datatype] = DataLoader(dataset[datatype], args.batch_size, num_workers=args.num_workers, collate_fn=dataset[datatype].collate)
    
    classifier = BERTClassifier(args)
    trainer = pl.Trainer(gpus=args.gpu_count, precision=32, accelerator="ddp")
    trainer.fit(classifier, data_loader['train'], data_loader['valid'])

if __name__ == '__main__':
    main()

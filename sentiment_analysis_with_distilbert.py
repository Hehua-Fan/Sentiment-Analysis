# Load Data and Configuration
texts = ['today is not that bad',
         'today is so bad',
         'so good tonight']
model_name = 'distilbert/distilbert-base-uncased-finetuned-sst-2-english'

# Instantiation
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Tokenizer
batch_input = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
'''
{'input_ids': tensor([[ 101, 2651, 2003, 2025, 2008, 2919,  102],
                      [ 101, 2651, 2003, 2061, 2919,  102,    0],
                      [ 101, 2061, 2204, 3892,  102,    0,    0]]), 
 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 0],
                           [1, 1, 1, 1, 1, 0, 0]])}
'''

# Model
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

with torch.no_grad():
    outputs = model(**batch_input) # SequenceClassifierOutput(loss=None, logits=tensor([[ 0.2347, -0.1015],[ 0.1364, -0.3081],[ 0.0071, -0.4359]], grad_fn=<AddmmBackward0>), hidden_states=None, attentions=None)
    logits = outputs.logits # tensor([[-3.4620,  3.6118],[ 4.7508, -3.7899],[-4.2113,  4.5724]])
    scores = F.softmax(logits, dim=-1) # tensor([[8.4632e-04, 9.9915e-01],[9.9980e-01, 1.9531e-04],[1.5318e-04, 9.9985e-01]])
    labels_ids = torch.argmax(scores, dim=-1) # tensor([1, 0, 1])
    labels = [model.config.id2label[id] for id in labels_ids.tolist()] # ['POSITIVE', 'NEGATIVE', 'POSITIVE']

# Save
target_cols = ['label']
submission = pd.DataFrame(labels, columns=target_cols)
submission.to_csv('submission.csv', index=False)
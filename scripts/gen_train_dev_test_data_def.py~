#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nltk
import random
import numpy as np
import re
import os

idataDir = '../data/textbooks_extracted_def'
odataDir = '../data/def_data'
books = ('open_stax_anatomy_physiology',
         'open_stax_astronomy',
         'open_stax_biology_2e',
         'open_stax_chemistry_2e',
         'open_stax_microbiology',
         'open_stax_university_physics_v3',
         'life_biology')

data = {}
for split in ['train', 'val', 'test']:
    data[split] = {}
    data[split]['sentences'] = []
    data[split]['labels'] = []

# load in and accumulate the data
for book in books:
    for type in ('def', 'nondef'):
        fname = "%s/%s_%s.txt" %(idataDir, book, type)
        print ("Reading file {}".format(fname))
        is_good = 0
        if type == 'def':
            is_good = 1
            
        if book == 'life_biology':
            split = 'test'
        else:
            split = 'train'

        with open(fname, 'r') as fin:
            text = fin.read()
            for sent in text.split("\n"):
                if re.search("^$", sent):
                    continue
                # remove small sentences
                tokens = nltk.word_tokenize(sent)
                if len(tokens) < 3:
                    continue
                data[split]['sentences'].append(" ".join(nltk.word_tokenize(sent)))
                data[split]['labels'].append(str(is_good))


## sanity check
for x in ("train", "test"):
    print (len(data[x]['sentences']))
    print (len(data[x]['labels']))
    assert(len(data[x]['sentences']) == len(data[x]['labels']))

## seed to fix randomization for repeatability
np.random.seed(10232)

## re-shuffle and split the train as 80-20%
length = len(data['train']['sentences'])
indices = np.arange(length)
np.random.shuffle(indices)
indices = list(indices)
dev_size = 0.2
train_size = int(length*(1 - dev_size))
small_size = 100

for x in ("full", "small"):
    for y in ("train", "val", "test"):
        if not os.path.exists(odataDir + "/" + x):
            os.makedirs(odataDir + "/" + x + "/" + y)
        elif not os.path.exists(odataDir + "/" + x + "/" + y):
            os.mkdir(odataDir + "/" + x + "/" + y)

# full
# train
with open(odataDir + "/full/train/sentences.txt", "w") as fout:
    fout.write("\n".join([data['train']['sentences'][x] for x in indices[0:train_size]]))
with open(odataDir + "/full/train/labels.txt", "w") as fout:
    fout.write("\n".join([data['train']['labels'][x] for x in indices[0:train_size]]))

# val
with open(odataDir + "/full/val/sentences.txt", "w") as fout:
    fout.write("\n".join([data['train']['sentences'][x] for x in indices[train_size:]]))
with open(odataDir + "/full/val/labels.txt", "w") as fout:
    fout.write("\n".join([data['train']['labels'][x] for x in indices[train_size:]]))

# small
# train
with open(odataDir + "/small/train/sentences.txt", "w") as fout:
    fout.write("\n".join([data['train']['sentences'][x] for x in indices[0:small_size]]))
with open(odataDir + "/small/train/labels.txt", "w") as fout:
    fout.write("\n".join([data['train']['labels'][x] for x in indices[0:small_size]]))

# val
with open(odataDir + "/small/val/sentences.txt", "w") as fout:
    fout.write("\n".join([data['train']['sentences'][x] for x in indices[small_size:small_size*2]]))
with open(odataDir + "/small/val/labels.txt", "w") as fout:
    fout.write("\n".join([data['train']['labels'][x] for x in indices[small_size:small_size*2]]))


# test data
length = len(data['test']['sentences'])
indices = np.arange(length)
np.random.shuffle(indices)
indices = list(indices)

# full
with open(odataDir + "/full/test/sentences.txt", "w") as fout:
    fout.write("\n".join([data['test']['sentences'][x] for x in indices]))
with open(odataDir + "/full/val/labels.txt", "w") as fout:
    fout.write("\n".join([data['test']['labels'][x] for x in indices]))

# small
with open(odataDir + "/small/test/sentences.txt", "w") as fout:
    fout.write("\n".join([data['test']['sentences'][x] for x in indices[0:small_size]]))
with open(odataDir + "/small/val/labels.txt", "w") as fout:
    fout.write("\n".join([data['test']['labels'][x] for x in indices[0:small_size]]))


#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nltk
import random
import numpy as np
import re
import os

dataDir = '../data/wcl_datasets_v1.2/wikipedia/'
dataFiles = ('wiki_good.txt',
             'wiki_bad.txt')


sentences = []
tags = []

for fname in dataFiles:
    is_good = 0
    if re.search("_good.txt$", fname):
        is_good = 1
    fname = dataDir + '/' + fname
    print ("Reading file {}".format(fname))
    with open(fname, 'r') as fin:
        data = fin.read()
        data = data.split("\n")
        for i in range(int(len(data)/2)):
            target = data[i*2+1].split(" ")[0].split(":")[0]
            if not is_good:
                target = target.replace("!", "")
            sent1 = data[i*2].replace("TARGET", target)
            tokens = nltk.word_tokenize(sent1)
            sentences.append(" ".join(tokens[1:]))
            tags.append(str(is_good))

print (len(sentences))
print (len(tags))

## sanity check
assert(len(sentences) == len(tags))

## re-shuffle and split 80-20%
indices = np.arange(len(sentences))
np.random.shuffle(indices)
indices = list(indices)

dev_size = 0.2
length = len(sentences)
train_size = int(length*(1 - dev_size))
debug_size = 10

for x in ("full", "small"):
    for y in ("train", "val", "test"):
        if not os.path.exists(dataDir + "/" + x):
            os.makedirs(dataDir + "/" + x + "/" + y)
        elif not os.path.exists(dataDir + "/" + x + "/" + y):
            os.mkdir(dataDir + "/" + x + "/" + y)

with open(dataDir + "/full/train/sentences.txt", "w") as fout:
    fout.write("\n".join([sentences[x] for x in indices[0:train_size]]))
with open(dataDir + "/full/train/labels.txt", "w") as fout:
    fout.write("\n".join([tags[x] for x in indices[0:train_size]]))

with open(dataDir + "/full/val/sentences.txt", "w") as fout:
    fout.write("\n".join([sentences[x] for x in indices[train_size:]]))
with open(dataDir + "/full/val/labels.txt", "w") as fout:
    fout.write("\n".join([tags[x] for x in indices[train_size:]]))

# small
with open(dataDir + "/small/train/sentences.txt", "w") as fout:
    fout.write("\n".join([sentences[x] for x in indices[0:debug_size]]))
with open(dataDir + "/small/train/labels.txt", "w") as fout:
    fout.write("\n".join([tags[x] for x in indices[0:debug_size]]))

with open(dataDir + "/small/val/sentences.txt", "w") as fout:
    fout.write("\n".join([sentences[x] for x in indices[debug_size:debug_size*2]]))
with open(dataDir + "/small/val/labels.txt", "w") as fout:
    fout.write("\n".join([tags[x] for x in indices[debug_size:debug_size*2]]))

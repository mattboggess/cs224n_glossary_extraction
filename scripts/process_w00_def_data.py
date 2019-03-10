#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nltk
import random
import numpy as np
import re
import os

dataDir = '../data/W00_dataset'
dataFiles = ('annotated.word',
             'annotated.tag')


sentences = []
tags = []

# Read all sentences
fname = dataDir + '/' + dataFiles[0]
with open(fname, 'r') as fin:
    data = fin.read()
    [sentences.append(x) for x in data.split("\n") if not re.search("^$", x)]

# Read tag data
fname = dataDir + '/' + dataFiles[1]
with open(fname, 'r') as fin:
    data = fin.read()
    for x in data.split("\n"):
        if (re.search("^$", x)):
            continue
        if re.search("TERM",x):
            tags.append(str(1))
        else:
            tags.append(str(0))


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

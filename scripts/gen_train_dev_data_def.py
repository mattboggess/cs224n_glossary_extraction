#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nltk
import random
import numpy as np
import re
import os

dataDir = '../data/textbooks_extracted_def'
books = ('open_stax_anatomy_physiology',
         'open_stax_astronomy',
         'open_stax_biology_2e',
         'open_stax_chemistry_2e',
         'open_stax_microbiology',
         'open_stax_university_physics_v1',
         'open_stax_university_physics_v2',
         'open_stax_university_physics_v3')

sentences = []
tags = []

for book in books:
    for type in ('def', 'nondef'):
        fname = "%s/%s_%s.txt" %(dataDir, book, type)
        is_good = 0
        if type == 'def':
            is_good = 1
        print ("Reading file {}".format(fname))
        with open(fname, 'r') as fin:
            data = fin.read().split("\n")
            [sentences.append(" ".join(nltk.word_tokenize(x))) for x in data if not re.search("^$", x)]
            [tags.append(str(is_good)) for x in data if not re.search("^$", x)]

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

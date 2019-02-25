#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nltk
import random
import numpy as np
import re
import os

dataDir = "../data"
textbooks = ("open_stax_anatomy_physiology",
             "open_stax_astronomy",
             "open_stax_biology_2e",
             "open_stax_chemistry_2e",
             "open_stax_microbiology",
             "open_stax_university_physics_v3")

sentences = []
tags = []

for book in textbooks:
    print ("Reading book %s" %(book))
    sfname = dataDir + "/textbooks_extracted/" + book + "_sentences.txt"
    tfname = dataDir + "/textbooks_extracted/" + book + "_sentence_tags.txt"
    with open(sfname, "r") as fin:
        data = fin.read()
        for sent in data.split("\n"):
            if re.search("^$", sent):
                continue
            tokens = nltk.word_tokenize(sent)
            sentences.append(" ".join(tokens))

    with open(tfname, "r") as fin:
        data = fin.read()
        i = 0
        for sent in data.split("\n"):
            if re.search("^$", sent):
                continue
            tags.append(sent)

print (len(sentences))
print (len(tags))

## sanity check
assert(len(sentences) == len(tags))
for sent, tag in zip(sentences, tags):
    sent = sent.split(" ")
    tag = tag.split(" ")
    assert(len(sent) == len(tag))
            
## re-shuffle and split 80-20%
indices = np.arange(len(sentences))
np.random.shuffle(indices)
indices = list(indices)

dev_size = 0.2
length = len(sentences)
train_size = int(length*dev_size)
debug_size = 10

for x in ("full", "small"):
    for y in ("train", "dev", "test"):
        if not os.path.exists(dataDir + "/" + x):
            os.makedirs(dataDir + "/" + x + "/" + y)
        elif not os.path.exists(dataDir + "/" + x + "/" + y):
            os.mkdir(dataDir + "/" + x + "/" + y)

with open(dataDir + "/full/train/sentences.txt", "w") as fout:
    fout.write("\n".join([sentences[x] for x in indices[0:train_size]]))
with open(dataDir + "/full/train/labels.txt", "w") as fout:
    fout.write("\n".join([tags[x] for x in indices[0:train_size]]))

with open(dataDir + "/full/dev/sentences.txt", "w") as fout:
    fout.write("\n".join([sentences[x] for x in indices[train_size:]]))
with open(dataDir + "/full/dev/labels.txt", "w") as fout:
    fout.write("\n".join([tags[x] for x in indices[train_size:]]))

# small
with open(dataDir + "/small/train/sentences.txt", "w") as fout:
    fout.write("\n".join([sentences[x] for x in indices[0:debug_size]]))
with open(dataDir + "/small/train/labels.txt", "w") as fout:
    fout.write("\n".join([tags[x] for x in indices[0:debug_size]]))

with open(dataDir + "/small/dev/sentences.txt", "w") as fout:
    fout.write("\n".join([sentences[x] for x in indices[debug_size:debug_size*2]]))
with open(dataDir + "/small/dev/labels.txt", "w") as fout:
    fout.write("\n".join([tags[x] for x in indices[debug_size:debug_size*2]]))





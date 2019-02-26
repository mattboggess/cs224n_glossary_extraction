import numpy as np
import re
import os
import json

dataDir = "../data"
textbooks = ("open_stax_anatomy_physiology",
             "open_stax_astronomy",
             "open_stax_biology_2e",
             "open_stax_chemistry_2e",
             "open_stax_microbiology",
             "open_stax_university_physics_v3",
             "life_biology")


data = {}
for split in ['train', 'val', 'test']:
    data[split] = {}
    data[split]['sentences'] = []
    data[split]['terms'] = []
    data[split]['labels'] = []

# load in and accumulate the data
for book in textbooks:
    print ("Reading book %s" %(book))
    sfname = dataDir + "/textbooks_extracted/" + book + "_sentences.txt"
    lfname = dataDir + "/textbooks_extracted/" + book + "_sentence_tags.txt"
    cfname = dataDir + "/textbooks_extracted/" + book + "_key_term_counts.json"
    tfname = dataDir + "/textbooks_extracted/" + book + "_key_terms.txt"

    if book == 'life_biology':
        split = 'test'
    elif book == 'open_stax_biology_2e':
        split = 'val'
    else:
        split = 'train'

    # load sentences
    with open(sfname, "r") as fin:
        text = fin.read()
        for sent in text.split("\n"):
            if re.search("^$", sent):
                continue
            data[split]['sentences'].append(sent)

    # load labels
    with open(lfname, "r") as fin:
        text = fin.read()
        for sent in text.split("\n"):
            if re.search("^$", sent):
                continue
            data[split]['labels'].append(sent)

    # load terms counts
    with open(cfname, 'r') as fin:
        counts = json.load(fin)

    # load terms screening ones that have no matches
    with open(tfname, 'r') as fin:
        text = fin.read()
        num_terms = 0
        num_zero_terms = 0
        for term in text.split("\n"):
            if re.search("^$", term):
                continue
            elif counts[term] > 0:
                num_terms += 1
                data[split]['terms'].append(term)
            else:
                num_zero_terms += 1
                num_terms += 1
        print('%d out of %d terms have 0 matches' % (num_zero_terms,
                                                     num_terms))


# write out the data
small_size = 20
for x in ("full", "small"):
    for y in ("train", "val", "test"):
        if not os.path.exists(dataDir + "/" + x):
            os.makedirs(dataDir + "/" + x + "/" + y)
        elif not os.path.exists(dataDir + "/" + x + "/" + y):
            os.mkdir(dataDir + "/" + x + "/" + y)

        data_path = '%s/%s/%s' % (dataDir, x, y)
        for data_type in ['sentences', 'labels', 'terms']:
            with open('%s/%s.txt' % (data_path, data_type), 'w') as fout:
                if x == 'full':
                    fout.write('\n'.join([s for s in data[y][data_type]]))
                else:
                    fout.write('\n'.join([s for s in data[y][data_type]][:small_size]))

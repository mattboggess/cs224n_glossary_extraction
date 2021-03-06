"""Build vocabularies of words and tags from datasets"""

import argparse
from collections import Counter
import json
import os
import gzip
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--min_count_word', default=1, help="Minimum count for words in the dataset", type=int)
parser.add_argument('--min_count_tag', default=1, help="Minimum count for tags in the dataset", type=int)
parser.add_argument('--data_dir', default='data/small', help="Directory containing the dataset")

# Hyper parameters for the vocab
PAD_WORD = '[PAD]'
PAD_TAG = 'O'
UNK_WORD = '[UNK]'


def save_vocab_to_txt_file(vocab, txt_path):
    """Writes one token per line, 0-based line id corresponds to the id of the token.

    Args:
        vocab: (iterable object) yields token
        txt_path: (stirng) path to vocab file
    """
    with open(txt_path, "w") as f:
        for token in vocab:
            f.write(token + '\n')


def save_dict_to_json(d, json_path):
    """Saves dict to json file

    Args:
        d: (dict)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)


def update_vocab(txt_path, vocab):
    """Update word and tag vocabulary from dataset

    Args:
        txt_path: (string) path to file, one sentence per line
        vocab: (dict or Counter) with update method

    Returns:
        dataset_size: (int) number of elements in the dataset
    """
    with open(txt_path) as f:
        for i, line in enumerate(f):
            vocab.update(line.strip().split(' '))

    return i + 1


if __name__ == '__main__':
    args = parser.parse_args()

    # Build word vocab with train and test datasets
    print("Building word vocabulary...")
    words = Counter()
    size_train_sentences = update_vocab(os.path.join(args.data_dir, 'train/sentences.txt'), words)
    size_dev_sentences = update_vocab(os.path.join(args.data_dir, 'val/sentences.txt'), words)
    size_test_sentences = update_vocab(os.path.join(args.data_dir, 'test/sentences.txt'), words)
    print("- done.")

    # Build stag vocab with train and test datasets
    stags = Counter()
    if os.path.exists(os.path.join(args.data_dir, 'train/slabels.txt')):
        print("Building stag vocabulary...")
        size_train_stags = update_vocab(os.path.join(args.data_dir, 'train/slabels.txt'), stags)
        size_dev_stags = update_vocab(os.path.join(args.data_dir, 'val/slabels.txt'), stags)
        size_test_stags = update_vocab(os.path.join(args.data_dir, 'test/slabels.txt'), stags)
        print("- done.")

        # Assert same number of examples in datasets
        assert size_train_sentences == size_train_stags
        assert size_dev_sentences == size_dev_stags
        assert size_test_sentences == size_test_stags
        stags = [tok for tok, count in stags.items() if count >= args.min_count_tag]
        if PAD_TAG not in stags: stags.append(PAD_TAG)

    # Build wtag vocab with train and test datasets
    wtags = Counter()
    if os.path.exists(os.path.join(args.data_dir, 'train/wlabels.txt')):
        print("Building wtag vocabulary...")
        size_train_wtags = update_vocab(os.path.join(args.data_dir, 'train/wlabels.txt'), wtags)
        size_dev_wtags = update_vocab(os.path.join(args.data_dir, 'val/wlabels.txt'), wtags)
        size_test_wtags = update_vocab(os.path.join(args.data_dir, 'test/wlabels.txt'), wtags)
        print("- done.")

        assert size_train_sentences == size_train_wtags
        assert size_dev_sentences == size_dev_wtags
        assert size_test_sentences == size_test_wtags
        wtags = [tok for tok, count in wtags.items() if count >= args.min_count_tag]
        if PAD_TAG not in stags: wtags.append(PAD_TAG)

    # Only keep most frequent tokens
    words = [tok for tok, count in words.items() if count >= args.min_count_word]

    # Build char vocab with train and test datasets
    print("Building character vocabulary...")
    chars = Counter()
    for word in words:
        for ch in word:
            chars.update(ch)
    chars = [tok for tok, count in chars.items() if count >= args.min_count_word]
    print("- done.")

    # Add pad tokens
    if PAD_WORD not in words: words.append(PAD_WORD)
    if PAD_WORD not in chars: chars.append(PAD_WORD)

    # add word for unknown words
    words.append(UNK_WORD)
    chars.append(UNK_WORD)

    # Save vocabularies to file
    print("Saving vocabularies to file...")
    save_vocab_to_txt_file(words, os.path.join(args.data_dir, 'words.txt'))
    save_vocab_to_txt_file(chars, os.path.join(args.data_dir, 'chars.txt'))
    save_vocab_to_txt_file(stags, os.path.join(args.data_dir, 'stags.txt'))
    save_vocab_to_txt_file(wtags, os.path.join(args.data_dir, 'wtags.txt'))
    print("- done.")

    # Load the glove word embeddings
    print("Processing GloVe Embeddings...")
    chunksize = 10 ** 4
    glove_embeddings = []
    glove_words = {}
    words = set(words)
    num_glove = 0
    glove_ix = 0
    average_embedding = np.zeros(300)
    with gzip.open('../data/embeddings/glove.840B.300d.txt.gz', 'rt') as fid:
        for line in fid:
            num_glove += 1
            values = line.split(' ')
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            average_embedding += embedding
            if word in words and word != UNK_WORD:
                glove_words[word] = glove_ix
                glove_embeddings.append(embedding)
                glove_ix += 1
    
    glove_words[UNK_WORD] = glove_ix
    glove_embeddings.append(average_embedding / num_glove)
    glove_words[PAD_WORD] = glove_ix + 1
    glove_embeddings.append(np.zeros(300))
    print("- done.")

    # Save glove embeddings to file
    print("Saving Reduced GloVe embeddings to file...")
    save_dict_to_json(glove_words, os.path.join(args.data_dir, 'glove_indices.json'))
    glove_path = os.path.join(args.data_dir, 'glove_embeddings.npz')
    np.savez_compressed(glove_path,
                        glove=np.array(glove_embeddings))
    print("- done.")

    # Save datasets properties in json file
    sizes = {
        'train_size': size_train_sentences,
        'dev_size': size_dev_sentences,
        'test_size': size_test_sentences,
        'vocab_size': len(words),
        'char_vocab_size': len(chars),
        'glove_vocab_size': len(glove_words.keys()),
        'glove_embedding_size': len(glove_embeddings[0]),
        'number_of_stags': len(stags),
        'number_of_wtags': len(wtags),
        'glove_path': glove_path,
        'pad_word': PAD_WORD,
        'pad_tag': PAD_TAG,
        'unk_word': UNK_WORD
    }
    save_dict_to_json(sizes, os.path.join(args.data_dir, 'dataset_params.json'))

    # Logging sizes
    to_print = "\n".join("- {}: {}".format(k, v) for k, v in sizes.items())
    print("Characteristics of the dataset:\n{}".format(to_print))

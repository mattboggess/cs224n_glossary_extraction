import random
import numpy as np
import os
import sys
import json

import torch
from torch.autograd import Variable

import utils

def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
    @param sents (list[list[int]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (int): padding token
    @returns sents_padded (list[list[int]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
        Output shape: (batch_size, max_sentence_length)
    """
    sents_padded = []

    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    for s in sents:
        padded = [pad_token] * max_len
        padded[:len(s)] = s
        sents_padded.append(padded)

    return sents_padded

def pad_sents_char(sents, char_pad_token):
    """ Pad list of sentences according to the longest sentence in the batch and max_word_length.
        @param sents (list[list[list[int]]]): list of sentences, result of `words2charindices()`
        from `vocab.py`
        @param char_pad_token (int): index of the character-padding token
        @returns sents_padded (list[list[list[int]]]): list of sentences where sentences/words shorter
            than the max length sentence/word are padded out with the appropriate pad token, such that
            each sentence in the batch now has same number of words and each word has an equal
            number of characters
            Output shape: (batch_size, max_sentence_length, max_word_length)
    """
    # Words longer than 21 characters should be truncated
    max_word_length = 21

    max_sent_len = max([len(sent) for sent in sents])
    pad_word = [char_pad_token] * max_word_length

    sents_padded = []
    for sent in sents:
        new_sent = []
        for word in sent:
            if len(word) > max_word_length:
                gew_sent.append(word[:max_word_length])
            else:
                num_word_pad = max_word_length - len(word)
                new_sent.append(word + [char_pad_token] * num_word_pad)

        num_sent_pad = max_sent_len - len(sent)
        sents_padded.append(new_sent + [pad_word] * num_sent_pad)

    return sents_padded

class DataLoader(object):
    """
    Handles all aspects of the data. Stores the dataset_params, vocabulary and tags with their mappings to indices.
    """
    def __init__(self, data_dir, params):
        """
        Loads dataset_params, vocabulary and tags. Ensure you have run `build_vocab.py` on data_dir before using this
        class.

        Args:
            data_dir: (string) directory containing the dataset
            params: (Params) hyperparameters of the training process. This function modifies params and appends
                    dataset_params (such as vocab size, num_of_tags etc.) to params.
        """

        # loading dataset_params
        json_path = os.path.join(data_dir, 'dataset_params.json')
        assert os.path.isfile(json_path), "No json file found at {}, run build_vocab.py".format(json_path)
        self.dataset_params = utils.Params(json_path)

        # loading vocab (we require this to map words to their indices)
        vocab_path = os.path.join(data_dir, 'words.txt')
        self.vocab2id = {}
        with open(vocab_path) as f:
            for i, l in enumerate(f.read().splitlines()):
                self.vocab2id[l] = i
        self.id2vocab = {v: k for k, v in self.vocab2id.items()}

        # setting the indices for UNKnown words and PADding symbols
        self.unk_ind = self.vocab2id[self.dataset_params.unk_word]
        self.pad_ind = self.vocab2id[self.dataset_params.pad_word]

        # loading glove mappings
        self.glove2id = {}
        glove_path = os.path.join(data_dir, 'glove_indices.json')
        with open(glove_path) as f:
            glove_ix = json.load(f)
        for word in self.vocab2id.keys():
            self.glove2id[word] = glove_ix.get(word, glove_ix[self.dataset_params.unk_word])
        self.id2glove = {v: k for k, v in self.glove2id.items()}

        # loading tags (we require this to map tags to their indices)
        tags_path = os.path.join(data_dir, 'tags.txt')
        self.tag2id = {}
        with open(tags_path) as f:
            for i, t in enumerate(f.read().splitlines()):
                self.tag2id[t] = i
        self.id2tag = {v: k for k, v in self.tag2id.items()}

        # adding character representation
        char_path = os.path.join(data_dir, 'words.txt')
        self.char2id = {}
        with open(char_path) as f:
            for i, l in enumerate(f.read().splitlines()):
                self.char2id[l] = i
        self.id2char = {v: k for k, v in self.char2id.items()}

        # adding dataset parameters to param (e.g. vocab size, )
        params.update(json_path)

    def load_sentences_labels(self, sentences_file, labels_file, terms_file, d):
        """
        Loads sentences, labels, and terms from their corresponding files.

        Args:
            sentences_file: (string) file with sentences with tokens space-separated
            labels_file: (string) file with NER tags for the sentences in labels_file
            terms_file: (string) file with key terms for the sentences in sentences_file
            d: (dict) a dictionary in which the loaded data is stored
        """

        sentences = []
        labels = []
        terms = []

        with open(sentences_file) as f:
            for sentence in f.read().splitlines():
                s = [token for token in sentence.split(' ')]
                sentences.append(s)

        with open(labels_file) as f:
            for sentence in f.read().splitlines():
                l = [label for label in sentence.split(' ')]
                labels.append(l)

        with open(terms_file) as f:
            for term in f.read().splitlines():
                terms.append(term)

        # checks to ensure there is a tag for each token
        assert len(labels) == len(sentences)
        for i in range(len(labels)):
            assert len(labels[i]) == len(sentences[i])

        # storing sentences and labels in dict d
        d['data'] = sentences
        d['labels'] = labels
        d['terms'] = terms
        d['size'] = len(sentences)

    def load_data(self, types, data_dir):
        """
        Loads the data for each type in types from data_dir.

        Args:
            types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
            data_dir: (string) directory containing the dataset

        Returns:
            data: (dict) contains the data with labels for each type in types

        """
        data = {}

        for split in ['train', 'val', 'test']:
            if split in types:
                sentences_file = os.path.join(data_dir, split, "sentences.txt")
                labels_file = os.path.join(data_dir, split, "labels.txt")
                terms_file = os.path.join(data_dir, split, "terms.txt")
                data[split] = {}
                self.load_sentences_labels(sentences_file, labels_file,
                                           terms_file, data[split])

        return data

    def words2charindices(self, sents):
        """ Convert list of sentences of words into list of list of list of character indices.
        @param sents (list[list[str]]): sentence(s) in words
        @return word_ids (list[list[list[int]]]): sentence(s) in indices
        """
        word_ids = []
        for sent in sents:
            sent_ids = []
            for word in sent:
                ch_ids = [self.char2id.get(ch, self.char_unk) for ch in word]
                sent_ids.append(ch_ids)
            word_ids.append(sent_ids)

        return word_ids

    def tags2indices(self, sents):
        """ Convert list of sentences of tags into list of list of indices.
           @param sents (list[list[str]]): sentence(s) in words
           @return word_ids (list[list[int]]): sentence(s) in indices
        """
        return [[self.tag2id[t] for t in s] for s in sents]

    def indices2tags(self, sents):
        """ Convert list of sentences of tags into list of list of indices.
           @param sents (list[list[str]]): sentence(s) in words
           @return word_ids (list[list[int]]): sentence(s) in indices
        """
        return [[self.id2tag[t] for t in s] for s in sents]

    def words2indices(self, sents, embed_type):
        """ Convert list of sentences of words into list of list of indices.
           @param sents (list[list[str]]): sentence(s) in words
           @return word_ids (list[list[int]]): sentence(s) in indices
        """
        if embed_type == 'word':
            return [[self.vocab2id.get(w, self.vocab2id[self.dataset_params.unk_word]) for w in s] for s in sents]
        elif embed_type == 'glove':
            return [[self.glove2id.get(w, self.glove2id[self.dataset_params.unk_word]) for w in s] for s in sents]

    def indices2words(self, word_ids):
        """ Convert list of indices into words.
            @param word_ids (list[int]): list of word ids
            @return sents (list[str]): list of words
        """
        return [self.id2vocab[w_id] for w_id in word_ids]

    def to_input_tensor(self, sents, embed_type):
        """ Convert list of sentences (words) into tensor with necessary padding for
           shorter sentences.

           @param sents (List[List[str]]): list of sentences (words)

           @returns sents_var: tensor of (max_sentence_length, batch_size)
        """
        if embed_type == 'char':
            return self.to_input_tensor_char(sents)
        elif embed_type == 'word':
            word_ids = self.words2indices(sents, 'word')
            sents_t = pad_sents(word_ids, self.vocab2id['<pad>'])
            sents_var = torch.tensor(sents_t, dtype=torch.long)
            return sents_var
        elif embed_type == 'tag':
            tag_ids = self.tags2indices(sents)
            sents_t = pad_sents(tag_ids, -1)
            sents_var = torch.tensor(sents_t, dtype=torch.long)
            return sents_var
        elif embed_type == 'glove':
            word_ids = self.words2indices(sents, 'glove')
            sents_t = pad_sents(word_ids, self.glove2id['<pad>'])
            sents_var = torch.tensor(sents_t, dtype=torch.long)
            return sents_var
        else:
            raise ValueError('Unsupported Embedding Type: %s' % embed_type)

    def to_input_tensor_char(self, sents):
        """ Convert list of sentences (words) into tensor with necessary padding for
            shorter sentences.

            @param sents (List[List[str]]): list of sentences (words)

            @returns sents_var: tensor of (max_sentence_length, batch_size, max_word_length)
        """
        char_ids = self.words2charindices(sents)
        sents_padded = pad_sents_char(char_ids, self.char2id['<pad>'])
        sents_t = torch.tensor(sents_padded, dtype=torch.long)
        return sents_t.permute(1, 0, 2)

    def data_iterator(self, data, params, shuffle=False):
        """
        Returns a generator that yields batches data with labels. Batch size is params.batch_size. Expires after one
        pass over the data.

        Args:
            data: (dict) contains data which has keys 'data', 'labels' and 'size'
            params: (Params) hyperparameters of the training process.
            shuffle: (bool) whether the data should be shuffled

        Yields:
            batch_data: (Variable) dimension batch_size x seq_len with the sentence data
            batch_labels: (Variable) dimension batch_size x seq_len with the corresponding labels

        """

        # make a list that decides the order in which we go over the data- this avoids explicit shuffling of data
        order = list(range(data['size']))
        if shuffle:
            random.seed(230)
            random.shuffle(order)

        # one pass over data
        for i in range((data['size']+1)//params.batch_size):

            batch = {}

            # fetch sentences and tags
            batch_sentences = [data['data'][idx] for idx in order[i*params.batch_size:(i+1)*params.batch_size]]
            batch_tags = [data['labels'][idx] for idx in order[i*params.batch_size:(i+1)*params.batch_size]]
            batch['sentences'] = batch_sentences

            # compute all desired embedding representation inputs
            for embed_type in params.embed_types:
                batch[embed_type] = self.to_input_tensor(batch_sentences, embed_type)
                if params.cuda:
                    batch[embed_type] = batch[embed_type].cuda()
                batch[embed_type] = Variable(batch[embed_type])

            # convert the tags to a label tensor
            batch['labels'] = self.to_input_tensor(batch_tags, 'tag')
            if params.cuda:
                batch['labels'] = batch['labels'].cuda()
            batch['labels'] = Variable(batch['labels'])

            ## prepare a numpy array with the data, initialising the data with pad_ind and all labels with -1
            ## initialising labels to -1 differentiates tokens with tags from PADding tokens
            #batch_data = self.pad_ind*np.ones((len(batch_sentences), batch_max_len))
            #batch_labels = -1*np.ones((len(batch_sentences), batch_max_len))

            ## copy the data to the numpy array
            #for j in range(len(batch_sentences)):
            #    cur_len = len(batch_sentences[j])
            #    batch_data[j][:cur_len] = batch_sentences[j]
            #    batch_labels[j][:cur_len] = batch_tags[j]

            ## since all data are indices, we convert them to torch LongTensors
            #batch_data, batch_labels = torch.LongTensor(batch_data), torch.LongTensor(batch_labels)


            # convert them to Variables to record operations in the computational graph
            # batch_data, batch_labels = Variable(batch_data), Variable(batch_labels)

            # yield batch_data, batch_labels
            yield batch

import random
import numpy as np
import os
import sys
import json

import torch
from torch.autograd import Variable
from pytorch_pretrained_bert import BertTokenizer

import utils


class DataLoader(object):
    """
    Handles all aspects of the data. Stores the dataset params, vocabulary and tags with their mappings to indices.
    Provides a generator that yields batches of data for batch training of different models.

    Currently supports the following representations of the input sentences:
        - Custom word embeddings: Creates a map from every word in the vocab to a custom trainable
        word embedding index.
        - GloVe word embeddings: Creates a map from every word in the vocab to a pre-trained GloVe
        word embedding vector index.
        - Char embeddings: Creates a map from every character in the character vocab to a custom
        trainable character embedding index.
        - BERT word piece embeddings: Uses the HuggingFace Pytorch BertTokenizer that tokenizes the
        sentences into word pieces and converts these word pieces to ids to be fed into the BERT
        models

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

        # load the dataset params
        json_path = os.path.join(data_dir, 'dataset_params.json')
        assert os.path.isfile(json_path), "No json file found at {}, run build_vocab.py".format(json_path)
        self.dataset_params = utils.Params(json_path)

        # load vocab and create vocab <--> id mapping
        vocab_path = os.path.join(data_dir, 'words.txt')
        self.vocab2id = {}
        with open(vocab_path) as f:
            for i, l in enumerate(f.read().splitlines()):
                self.vocab2id[l] = i
        self.id2vocab = {v: k for k, v in self.vocab2id.items()}

        # setting the indices for UNKnown words and PADding symbols
        self.unk_ind = self.vocab2id[self.dataset_params.unk_word]
        self.pad_ind = self.vocab2id[self.dataset_params.pad_word]

        # loading glove mappings and create vocab <--> glove id mapping
        glove_path = os.path.join(data_dir, 'glove_indices.json')
        self.glove2id = {}
        with open(glove_path) as f:
            glove_ix = json.load(f)
        # map words to their glove indices if existent, otherwise to the unknown glove vector
        for word in self.vocab2id.keys():
            self.glove2id[word] = glove_ix.get(word, glove_ix[self.dataset_params.unk_word])
        self.id2glove = {v: k for k, v in self.glove2id.items()}

        # load tags and create tag <--> id mapping
        # for NER model
        wtags_path = os.path.join(data_dir, 'wtags.txt')
        self.wtag2id = {}
        self.wid2tag = {}
        if os.path.exists(wtags_path):
            with open(wtags_path) as f:
                for i, t in enumerate(f.read().splitlines()):
                    self.wtag2id[t] = i
                self.wtag2id[self.dataset_params.pad_word] = -1
                self.wid2tag = {v: k for k, v in self.wtag2id.items()}

        # for DEF model
        stags_path = os.path.join(data_dir, 'stags.txt')
        self.stag2id = {}
        self.sid2tag = {}
        if os.path.exists(stags_path):
            with open(stags_path) as f:
                for i, t in enumerate(f.read().splitlines()):
                    self.stag2id[t] = i
                self.stag2id[self.dataset_params.pad_word] = -1
                self.sid2tag = {v: k for k, v in self.stag2id.items()}

        # adding character representation
        char_path = os.path.join(data_dir, 'chars.txt')
        self.char2id = {}
        with open(char_path) as f:
            for i, l in enumerate(f.read().splitlines()):
                self.char2id[l] = i
        self.id2char = {v: k for k, v in self.char2id.items()}

        # Bert mappings
        do_lower_case = params.bert_type != 'bert-base-cased'
        self.bert_tokenizer = BertTokenizer.from_pretrained(params.bert_type,
                                                            do_lower_case=do_lower_case)

        # adding dataset parameters to param (e.g. vocab size, )
        params.update(json_path)

        # store parameters
        self.params = params
        

    def load_sentences_labels(self, sentences_file, wlabels_file, slabels_file, terms_file, d):
        """
        Loads sentences, labels, and terms from their corresponding files.

        Args:
            sentences_file: (string) file with sentences with tokens space-separated generated
            using ntlk's word_tokenize function
            labels_file: (string) file with NER tags for the sentences in labels_file
            terms_file: (string) file with key terms for the entire textbook corpus represented in
            the sentences file. A separate key term on each line.
            d: (dict) a dictionary in which the loaded data is stored
        """

        sentences = []
        wlabels = []
        slabels = []
        terms = []

        with open(sentences_file) as f:
            for sentence in f.read().splitlines():
                s = [token for token in sentence.split(' ')]
                sentences.append(s)

        if os.path.exists(wlabels_file):
            with open(wlabels_file) as f:
                for sentence in f.read().splitlines():
                    l = [label for label in sentence.split(' ')]
                    wlabels.append(l)

            # checks to ensure there is a tag for each token
            assert len(wlabels) == len(sentences)
            for i in range(len(wlabels)):
                assert len(wlabels[i]) == len(sentences[i])

        if os.path.exists(slabels_file):
            with open(slabels_file) as f:
                for sentence in f.read().splitlines():
                    slabels.append(sentence)
                    
            # checks to ensure there is a tag for each sentence
            assert len(slabels) == len(sentences)

        if os.path.exists(terms_file):
            with open(terms_file) as f:
                for term in f.read().splitlines():
                    terms.append(term)

        # storing sentences and labels in dict d
        d['data'] = sentences
        d['wlabels'] = wlabels
        d['slabels'] = slabels
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
                wlabels_file = os.path.join(data_dir, split, "wlabels.txt")
                slabels_file = os.path.join(data_dir, split, "slabels.txt")
                terms_file = os.path.join(data_dir, split, "terms.txt")
                data[split] = {}
                self.load_sentences_labels(sentences_file, wlabels_file, slabels_file, terms_file, data[split])

        return data

    def words2charindices(self, sents):
        """ Convert list of sentences of words into list of list of list of character indices.

        Args:
            sents: list[list[str]] sentences(s) split into words

        Returns:
            char_ids: list[list[list[int]]] list of character indices for words across sentences
        """
        word_ids = []
        for sent in sents:
            sent_ids = []
            for word in sent:
                ch_ids = [self.char2id.get(ch, self.char2id[self.dataset_params.unk_word]) for ch in word]
                sent_ids.append(ch_ids)
            word_ids.append(sent_ids)

        return word_ids

    def tags2indices(self, sents):
        """ Convert list of sentences of tags into list of list of indices.

        Args:
            sents: list[list[str]] sentences(s) split into tags

        Returns:
            tag_ids: list[list[int]] list of tag ids across sentences
        """
        return [[self.wtag2id[t] for t in s] for s in sents]


    def words2indices(self, sents, embed_type):
        """ Convert list of sentences of words into list of list of indices.

        Args:
            sents: list[list[str]] sentences(s) split into words
            embed_type: str (glove|word) word embedding type to retrieve indices for

        Returns:
            word_ids: list[list[int]] list of word ids across sentences
        """
        if embed_type == 'word':
            return [[self.vocab2id.get(w, self.vocab2id[self.dataset_params.unk_word]) for w in s] for s in sents]
        elif embed_type == 'glove':
            return [[self.glove2id.get(w, self.glove2id[self.dataset_params.unk_word]) for w in s] for s in sents]
        else:
            raise ValueError('Unknown embedding type: %s' % embed_type)

    def words2bertindices(self, sents):
        """ Convert list of sentences of words into list of list of Bert wordpiece indices.

        Additionally creates a mask denoting which indices are wordpiece expansions of original
        words (denoted by -1). We will use this to re-map back to the original word token space
        when evaluating the tagging.

        Args:
            sents: list[list[str]] sentences(s) split into words

        Returns:
            bert_ids: list[list[int]] sentences expanded into wordpiece indices
            bert_masks: list[list[int] list of masks for each sentence that is -1 wherever a word
            got extended into additional word pieces
        """
        bert_sents = []
        bert_masks = []
        for sent in sents:
            bert_sent = []
            bert_mask = []
            sent = ['[CLS]'] + sent + ['[SEP]']
            for token in sent:
                bert_tokens = self.bert_tokenizer.tokenize(token)
                bert_tokenids = self.bert_tokenizer.convert_tokens_to_ids(bert_tokens)
                if len(bert_tokenids) > 0:
                    bert_sent += bert_tokenids
                    if token == '[CLS]' or token == '[SEP]':
                        bert_mask += [-1]
                    else:
                        bert_mask += ([1] + [-1] * (len(bert_tokenids) - 1))
            assert(len(bert_sent) == len(bert_mask)), sent
            bert_sents.append(bert_sent)
            bert_masks.append(bert_mask)

        return bert_sents, bert_masks

    def to_input_tensor(self, sents, embed_type):
        """ Convert list of sentences (words) into input tensor to be fed into models
            to retrieve appropriate embeddings.

        Args:
            sents: list[list[str]] sentences(s) split into words
            embed_type: str (glove|word|char|bert|label) embedding type to produce indices for

        Returns:
            id_tensor: tensor of (embedding_shape, batch_size)
            id_mask: tensor of (embedding_shape, batch_size)
        """
        if embed_type == 'char':
            char_ids = self.words2charindices(sents)
            pad_id = self.char2id[self.dataset_params.pad_word]
            sents_padded = pad_sents_char(char_ids, pad_id)
            sents_t = torch.tensor(sents_padded, dtype=torch.long)
            return sents_t.permute(1, 0, 2)
        elif embed_type == 'word':
            word_ids = self.words2indices(sents, 'word')
            pad_id = self.vocab2id[self.dataset_params.pad_word]
            sents_t = pad_sents(word_ids, pad_id)
            id_tensor = torch.tensor(sents_t, dtype=torch.long)
            return id_tensor
        elif embed_type == 'wtag':
            tag_ids = self.tags2indices(sents)
            pad_id = self.wtag2id[self.dataset_params.pad_word]
            sents_t = pad_sents(tag_ids, pad_id)
            id_tensor = torch.tensor(sents_t, dtype=torch.long)
            return id_tensor
        elif embed_type == 'stag':
            sents_t = [int(x) for x in sents]
            id_tensor = torch.tensor(sents_t, dtype=torch.float)
            return id_tensor
        elif embed_type == 'glove':
            word_ids = self.words2indices(sents, 'glove')
            pad_id = self.glove2id[self.dataset_params.pad_word]
            sents_t = pad_sents(word_ids, pad_id)
            sents_var = torch.tensor(sents_t, dtype=torch.long)
            return sents_var
        elif embed_type == 'bert':
            word_ids, word_masks = self.words2bertindices(sents)
            pad_id = self.bert_tokenizer.convert_tokens_to_ids([self.dataset_params.pad_word])[0]
            if self.params.fixed_sent_length:
                x = self.bert_tokenizer.convert_tokens_to_ids('[SEP]')[0]
                sents_t = pad_sents(word_ids, pad_id, self.params.fixed_sent_length, x)
                word_mask = pad_sents(word_masks, 0, self.params.fixed_sent_length, -1)
            else:
                sents_t = pad_sents(word_ids, pad_id)
                word_mask = pad_sents(word_masks, 0)
            sents_var = torch.tensor(sents_t, dtype=torch.long)
            berts_mask = torch.tensor(word_mask, dtype=torch.long)
            return sents_var, berts_mask
        else:
            raise ValueError('Unsupported Embedding Type: %s' % embed_type)

    def data_iterator(self, data, params, shuffle=False):
        """
        Returns a generator that yields batches of data with labels. Batch size is params.batch_size. Expires after one
        pass over the data.

        Args:
            data: (dict) contains data which has keys 'data', 'labels' and 'size'
            params: (Params) hyperparameters of the training process.
            shuffle: (bool) whether the data should be shuffled

        Yields:
            batch: (dictionary) dimension batch_size x seq_len with the sentence data

        """

        # make a list that decides the order in which we go over the data- this avoids explicit shuffling of data
        order = list(range(data['size']))
        if shuffle:
            random.seed(230)
            random.shuffle(order)

        # one pass over data
        for i in range((data['size']+1)//params.batch_size):

            batch = {}

            # fetch sentences and labels
            batch_sentences = [data['data'][idx] for idx in order[i*params.batch_size:(i+1)*params.batch_size]]
            batch_wtags = [data['wlabels'][idx] for idx in order[i*params.batch_size:(i+1)*params.batch_size] if idx < len(data['wlabels'])]
            batch_stags = [data['slabels'][idx] for idx in order[i*params.batch_size:(i+1)*params.batch_size] if idx < len(data['slabels'])]
            batch['sentences'] = batch_sentences
            batch['wtags'] = batch_wtags
            batch['stags'] = batch_stags

            # convert word tags to a label tensor
            if len(batch_wtags) > 0:
                batch['wlabels'] = self.to_input_tensor(batch_wtags, 'wtag')
                if params.cuda:
                    batch['wlabels'] = batch['wlabels'].cuda()
                    batch['wlabels'] = Variable(batch['wlabels'])

            # convert sentence tags to label tensor
            if len(batch_stags) > 0:
                batch['slabels'] = self.to_input_tensor(batch_stags, 'stag')
                if params.cuda:
                    batch['slabels'] = batch['slabels'].cuda()
                    batch['slabels'] = Variable(batch['slabels'])

            # compute all desired embedding representation inputs
            for embed_type in params.embed_types:
                if embed_type == 'bert':
                    sents, mask = self.to_input_tensor(batch_sentences, embed_type)
                    batch[embed_type] = sents
                    batch[embed_type + '_mask'] = mask
                    if params.cuda:
                        batch[embed_type] = batch[embed_type].cuda()
                        batch[embed_type + '_mask'] = batch[embed_type + '_mask'].cuda()
                    batch[embed_type] = Variable(batch[embed_type])
                else:
                    batch[embed_type] = self.to_input_tensor(batch_sentences, embed_type)
                    if params.cuda:
                        batch[embed_type] = batch[embed_type].cuda()
                    batch[embed_type] = Variable(batch[embed_type])

            yield batch

def pad_sents(sents, pad_token, fixed_length=None, sent_end_token=None):
    """ Pad list of sentences according to the longest sentence in the batch.
    @param sents (list[list[int]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (int): padding token
    @fixed_length (int): if non zero, every sent will be made of fixed length
    @returns sents_padded (list[list[int]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
        Output shape: (batch_size, max_sentence_length)
    """
    sents_padded = []

    if not fixed_length or fixed_length == 0:
        max_len = max(len(s) for s in sents)
    else:
        max_len = fixed_length
        
    for s in sents:
        padded = [pad_token] * max_len
        if len(s) < max_len:
            padded[:len(s)] = s
        else:
            padded[:max_len-1] = s[0:max_len-1]
            padded[max_len-1] = sent_end_token
        sents_padded.append(padded)

    return sents_padded

def pad_sents_char(sents, char_pad_token, max_word_length=21):
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
    max_sent_len = max([len(sent) for sent in sents])
    pad_word = [char_pad_token] * max_word_length

    sents_padded = []
    for sent in sents:
        new_sent = []
        for word in sent:
            if len(word) > max_word_length:
                new_sent.append(word[:max_word_length])
            else:
                num_word_pad = max_word_length - len(word)
                new_sent.append(word + [char_pad_token] * num_word_pad)

        num_sent_pad = max_sent_len - len(sent)
        sents_padded.append(new_sent + [pad_word] * num_sent_pad)

    return sents_padded

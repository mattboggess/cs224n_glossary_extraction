"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel

class HoveyMa(nn.Module):
    """
    Re-implementation of Hovey & Ma 2016 w/o the CRF layer.
    """

    def __init__(self, params):
        """
        Args:
            params: (Params) contains vocab_size, embedding_dim, lstm_hidden_dim
        """
        super(HoveyMa, self).__init__()

        # word embedding
        if 'glove' in params.embed_types:
            self.word_embed_type = 'glove'
            self.embed_size = params.glove_embedding_size
            self.word_embedding = nn.Embedding(params.glove_vocab_size,
                                               params.glove_embedding_size)

            # load in glove weights & fix
            glove_weights = np.load(params.glove_path)['glove']
            self.word_embedding.load_state_dict({'weight': torch.tensor(glove_weights)})
            self.word_embedding.weight.requires_grad = False
        else:
            self.word_embed_type = 'word'
            self.embed_size = params.word_embedding_size
            self.word_embedding = nn.Embedding(params.vocab_size, params.word_embedding_size)

        # char embedding
        self.char_embed = False
        if 'char' in params.embed_types:
            self.char_embed = True
            self.embed_size += params.cnn_num_filters
            self.char_embedding = nn.Embedding(params.char_vocab_size, params.char_embedding_size)
            self.conv = nn.Conv1d(params.char_embedding_size, params.cnn_num_filters,
                                  params.cnn_window_size, bias=True)

        self.dropout = nn.Dropout(params.dropout_rate)

        self.lstm = nn.LSTM(self.embed_size, params.lstm_hidden_size,
                            batch_first=True, bidirectional=True)

        self.fc = nn.Linear(2 * params.lstm_hidden_size, params.number_of_tags)

    def forward(self, batch):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of sentences, of dimension batch_size x seq_len, where seq_len is
               the length of the longest sentence in the batch. For sentences shorter than seq_len, the remaining
               tokens are PADding tokens. Each row is a sentence with each element corresponding to the index of
               the token in the vocab.

        Returns:
            out: (Variable) dimension batch_size*seq_len x num_tags with the log probabilities of tokens for each token
                 of each sentence.

        Note: the dimensions after each step are provided
        """

        # word embeddings
        word_embed = self.word_embedding(batch[self.word_embed_type])            # dim: batch_size x seq_len x embedding_dim

        # char embeddings
        if self.char_embed:
            char_embed = self.char_embedding(batch['char'])
            char_embed = self.dropout(char_embed)

            char_embed = char_embed.permute(0, 1, 3, 2)
            conv_shape = (char_embed.shape[0] * char_embed.shape[1],
                          char_embed.shape[2],
                          char_embed.shape[3])
            conv_input = char_embed.reshape(conv_shape)
            conv_output = self.conv(conv_input)
            conv_output = conv_output.max(dim=2)[0]
            conv_output = conv_output.reshape((char_embed.shape[0],
                                               char_embed.shape[1],
                                               conv_output.shape[1]))
            conv_output = conv_output.permute(1, 0, 2)
            embedding = torch.cat((word_embed, conv_output), -1)
        else:
            embedding = word_embed

        embedding = self.dropout(embedding)
        # run the LSTM along the sentences of length seq_len
        s, _ = self.lstm(embedding)              # dim: batch_size x seq_len x lstm_hidden_dim

        s = self.dropout(s)

        # make the Variable contiguous in memory (a PyTorch artefact)
        s = s.contiguous()

        # reshape the Variable so that each row contains one token
        s = s.view(-1, s.shape[2])       # dim: batch_size*seq_len x lstm_hidden_dim

        # apply the fully connected layer and obtain the output (before softmax) for each token
        s = self.fc(s)                   # dim: batch_size*seq_len x num_tags

        # apply log softmax on each token's output (this is recommended over applying softmax
        # since it is numerically more stable)
        return F.log_softmax(s, dim=1)   # dim: batch_size*seq_len x num_tags


class BertNER(nn.Module):

    def __init__(self, params):
        super(BertNER, self).__init__()
        self.bert = BertModel.from_pretrained(params.bert_type)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.fc = nn.Linear(self.bert.config.hidden_size, params.number_of_tags)

    def forward(self, batch):
        attention_mask = batch['bert_mask']
        attention_ix = attention_mask == -1
        attention_mask[attention_ix] = 1
        s, _ = self.bert(batch['bert'], attention_mask=attention_mask,
                         output_all_encoded_layers=False)
        attention_mask[attention_ix] = -1

        s = self.dropout(s)

        # make the Variable contiguous in memory (a PyTorch artefact)
        s = s.contiguous()

        # reshape the Variable so that each row contains one token
        s = s.view(-1, s.shape[2])       # dim: batch_size*seq_len x lstm_hidden_dim

        # apply the fully connected layer and obtain the output (before softmax) for each token
        s = self.fc(s)                   # dim: batch_size*seq_len x num_tags

        # apply log softmax on each token's output (this is recommended over applying softmax
        # since it is numerically more stable)
        return F.log_softmax(s, dim=1)   # dim: batch_size*seq_len x num_tags


def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs from the model and labels for all tokens. Assumes
    pad tokens have been screened out.

    Args:
        outputs: (Variable) dimension batch_size*seq_len x num_tags - log softmax output of the model
        labels: (Variable) dimension batch_size x seq_len where each element a label in [0, 1, ... num_tag-1],

    Returns:
        loss: (Variable) cross entropy loss for all tokens in the batch
    """


    num_tokens = len(labels)

    return -torch.sum(outputs[range(outputs.shape[0]), labels])/num_tokens


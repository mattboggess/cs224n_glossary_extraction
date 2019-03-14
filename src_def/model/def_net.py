"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    This is the standard way to define your own network in PyTorch. You typically choose the components
    (e.g. LSTMs, linear layers etc.) of your network in the __init__ function. You then apply these layers
    on the input step-by-step in the forward function. You can use torch.nn.functional to apply functions
    such as F.relu, F.sigmoid, F.softmax. Be careful to ensure your dimensions are correct after each step.

    You are encouraged to have a look at the network in pytorch/vision/model/net.py to get a better sense of how
    you can go about defining your own network.

    The documentation for all the various components available to you is here: http://pytorch.org/docs/master/nn.html
    """

    def __init__(self, params):
        """
        We define a definition model which predicts if a sentence is a definition setnence or not.
        - an embedding layer: maps input to word embeddings
        - cnn layer: convolves over words in sentences
        - max pool: pooling layer
        - bilstm: applying the Bi-LSTM on the sequential input returns an output for each token in the sentence
        - fc: a fully connected layer that converts the LSTM output to 1 or 0

        Args:
            params: (Params) contains 
               vocab_size          : size of vocabulary
               defm_embed_size     : word embedding size
               defm_cnn_num_filters: number of cnn filters
               defm_cnn_kernel     : cnn kernel
               defm_pool_kernel    : pooling kernel
               defm_pool_stride    : pooling stride
               defm_lstm_hidden_dim: bi-lstm hidden size
               defm_dcsn_threshold : decision threshold 
        """
        super(Net, self).__init__()

        self.threshold = params.defm_dcsn_threshold

        if 'glove' in params.embed_types:
            # the embedding takes as input the vocab_size and the embedding_dim
            self.word_embed_type = 'glove'
            self.defm_embed_size = params.glove_embedding_size
            self.embedding = nn.Embedding(params.glove_vocab_size, params.defm_embed_size)
            
            # load in glove weights & fix
            glove_weights = np.load(params.glove_path)['glove']
            self.embedding.load_state_dict({'weight': torch.tensor(glove_weights)})
            self.embedding.weight.requires_grad = False
        else:
            self.word_embed_type = 'word'
            self.defm_embed_size = params.defm_embed_size
            self.embedding = nn.Embedding(params.vocab_size, params.defm_embed_size)

        # cnn filters
        self.cnn = nn.Conv1d(params.defm_embed_size, 
                             params.defm_cnn_num_filters,
                             params.defm_cnn_kernel,
                             padding=int(params.defm_cnn_kernel/2),
                             bias=True)
        nn.init.xavier_uniform_(self.cnn.weight, gain=1)

        # max pool
        self.maxpool_layer = nn.MaxPool1d(params.defm_pool_kernel, stride=params.defm_pool_stride)
        
        # LSTM
        self.lstm = nn.LSTM(params.defm_cnn_num_filters,
                            params.defm_lstm_hidden_dim,
                            batch_first=True,
                            bidirectional=True)

        # dropout Layer
        self.dropout = nn.Dropout(params.defm_dropout_rate, inplace=True)

        # the fully connected layer transforms the output to give the final output layer
        self.fc = nn.Linear(params.defm_lstm_hidden_dim*2, 1, bias=True)

    def forward(self, s):
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
        #                                -> batch_size x seq_len
        # apply the embedding layer that maps each token to its embedding
        s = self.embedding(s)            # dim: batch_size x seq_len x defm_embed_size

        # run through cnn 
        s = s.permute(0,2,1)  # dim: batch_size x defm_embed_size x seq_len
        s = self.cnn(s)       # dim: batch_size x defm_cnn_num_filters x seq_len

        # run through pooling layer
        s = self.maxpool_layer(s) # dim: batch_size x defm_cnn_num_filters x seq_len/defm_pool_stride
        
        # apply dropout
        self.dropout(s)
        
        s = s.permute(0,2,1)  # dim: batch_size x seq_len/defm_pool_stride x defm_cnn_num_filters
        # run the LSTM along the sentences of length seq_len
        _, (s,__) = self.lstm(s)              # dim: 2 x batch_size x defm_lstm_hidden_dim

        # make the Variable contiguous in memory (a PyTorch artefact)
        s = s.contiguous()

        # reshape the Variable so that each row contains one token
        s = s.permute(1,0,2) # dim: batch_size x 2 x defm_lstm_hidden_dim
        s = s.contiguous()
        s = s.view(-1, s.shape[2]*2)       # dim: batch_size x 2*lstm_hidden_dim

        # apply dropout
        self.dropout(s)

        # apply the fully connected layer and obtain the output (before softmax) for each token
        s = self.fc(s)                   # dim: batch_size x 1

        # apply sigmoid function
        s = torch.sigmoid(s)
        
        return s


def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs from the model and labels for all tokens. Exclude loss terms
    for PADding tokens.

    Args:
        outputs: (Variable) dimension batch_size - sigmoid output of the model
        labels: (Variable) dimension batch_size  - 0|1 (no definition|definition)

    Returns:
        loss: (Variable) cross entropy loss for all tokens in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """

    labels = labels.squeeze(1)
    outputs = outputs.squeeze(1)
    loss = F.binary_cross_entropy(outputs, labels)
    # focal loss
    #CE(pt) = −log(pt)
    #FL(pt) = −(1 − pt)γ log(pt)
    #loss = -torch.sum((labels*torch.log(outputs) + (1-labels)*torch.log(1-outputs))*((1-outputs)**0.5))
    #loss = -torch.sum((labels*torch.log(outputs)*0.95 + (1-labels)*torch.log(1-outputs)*0.05))/len(labels)
    return loss


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all tokens. Exclude PADding terms.

    Args:
        outputs: (np.ndarray) dimension batch_size - sigmoid output of the model
        labels: (np.ndarray) dimension batch_size  - 0|1 (no definition|definition)

    Returns: (float) accuracy in [0,1]
    """
    # threshold the output
    #??fixme?? How to get threshold here
    outputs = outputs > 0.5
    labels = labels == 1

    # compare outputs with labels and divide by number of sentences
    return np.sum(outputs==labels)/float(labels.shape[0])

def f1metric(outputs, labels):
    """
    Compute precision, recall and f1 score.
    
    Args:
        outputs: (np.ndarray) dimension batch_size - sigmoid output of the model
        labels: (np.ndarray) dimension batch_size  - 0|1 (no definition|definition)

    Returns: float (prec, recall, f1)
    """

    outputs = outputs > 0.5
    labels = labels == 1
    tn = np.sum(np.logical_or(outputs, labels) == 0)
    tp = np.sum(np.logical_and(outputs, labels) == 1)
    x = np.logical_xor(outputs, labels)
    fn = np.sum(np.logical_and(x, labels))
    fp = np.sum(np.logical_and(x, outputs))
    p = r = f1 = 0.0
    if tp != 0:
        p = tp/(tp+fp)
        r = tp/(tp+fn)
        f1 = 2*(p*r)/(p+r)
   #print ("labels={}, outputs={}, tp={}, tn={}, fp={}, fn={}, p={}, r={}, f1={}".format(len(labels), len(outputs), tp, tn, fp, fn, p, r, f1))
    return (p,r,f1)
    

def f1score(outputs, labels):
    """
    Compute f1 score.
    
    Args:
        outputs: (np.ndarray) dimension batch_size - sigmoid output of the model
        labels: (np.ndarray) dimension batch_size  - 0|1 (no definition|definition)

    Returns: float
    """

    return f1metric(outputs, labels)[2]

def precision(outputs, labels):
    """
    Compute f1 score.
    
    Args:
        outputs: (np.ndarray) dimension batch_size - sigmoid output of the model
        labels: (np.ndarray) dimension batch_size  - 0|1 (no definition|definition)

    Returns: float
    """

    return f1metric(outputs, labels)[0]

def recall(outputs, labels):
    """
    Compute f1 score.
    
    Args:
        outputs: (np.ndarray) dimension batch_size - sigmoid output of the model
        labels: (np.ndarray) dimension batch_size  - 0|1 (no definition|definition)

    Returns: float
    """

    return f1metric(outputs, labels)[1]


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1score' : f1score,
    # could add more metrics such as accuracy for each token type
}

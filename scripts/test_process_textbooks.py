import numpy as np
from nltk import word_tokenize
from process_textbooks import tag_sentence


def test_tag_sentence():

    # test multiple terms per sentence, test len==2 term
    sentence = 'That is either a homogeneous mixture or a heterogeneous mixture.'
    term1 = 'homogeneous mixture'
    term2 = 'heterogeneous mixture'
    sentence = word_tokenize(sentence.lower())
    term1 = word_tokenize(term1)
    term2 = word_tokenize(term2)
    tags = ['O'] * len(sentence)

    tags1, counts = tag_sentence(sentence, term1, tags)
    assert tags1 == ['O', 'O', 'O', 'O', 'B', 'E', 'O', 'O', 'O', 'O', 'O']
    assert counts == 1
    tags2, counts = tag_sentence(sentence, term2, tags1)
    assert tags2 == ['O', 'O', 'O', 'O', 'B', 'E', 'O', 'O', 'B', 'E', 'O']
    assert counts == 1

    # test multiple same terms per sentence, test len==1 term
    sentence = 'Scientists need a hypothesis because a hypothesis helps.'
    term1 = 'hypothesis'
    sentence = word_tokenize(sentence.lower())
    term1 = word_tokenize(term1)
    tags = ['O'] * len(sentence)

    tags1, counts = tag_sentence(sentence, term1, tags)
    assert tags1 == ['O', 'O', 'O', 'S', 'O', 'O', 'S', 'O', 'O']
    assert counts == 2



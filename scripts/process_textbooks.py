from subprocess import call
from nltk.tokenize import sent_tokenize, word_tokenize
import json
import re
import os
from bs4 import BeautifulSoup
from collections import defaultdict

def match_pattern(elem, pattern):
    re_pattern = re.compile(pattern[1])
    if pattern[0] == 'text':
        return re_pattern.match(elem.text)
    else:
        return re_pattern.match(elem.attrs[pattern[0]])

def get_text_between_elements(spans, start_pattern, end_pattern, text_pattern):

    text = []
    between = False
    for span in spans:
        if between:
            if match_pattern(span, end_pattern):
                between = False
            else:
                if match_pattern(span, text_pattern):
                    text.append(span.text)
                else:
                    text.append('\n')
        else:
            if match_pattern(span, start_pattern):
                between = True

    return text


def extract_sentences(soup, pattern_info):

    # extract chapter text
    spans = soup.find_all('span')
    text = get_text_between_elements(spans,
                                     pattern_info['chapter_start_pattern'],
                                     pattern_info['chapter_end_pattern'],
                                     pattern_info['chapter_text_pattern'])
    text = ' '.join(text)
    text = text.replace('\n', ' ')

    # remove empty parens (missing figure references usually)
    text = re.sub('\(\s*\)', '', text)

    # split sentences, remove short < 3 words
    sentences = sent_tokenize(text)
    sentences = [sent for sent in sentences if len(word_tokenize(sent)) > 3]

    return sentences


def extract_key_terms(soup, pattern_info):

    spans = soup.find_all('span')
    terms = get_text_between_elements(spans,
                                      pattern_info['key_terms_start_pattern'],
                                      pattern_info['key_terms_end_pattern'],
                                      pattern_info['key_term_pattern'])
    terms = ''.join(terms).split('\n')

    # handle parens
    new_terms = []
    for term in terms:
        term = re.split('\(|\)', term)
        if len(term) <= 2:
            term = term[0].strip()
        else:
            term = term[0] + term[2] + '; ' + \
                   term[1] + term[2]

        new_terms.append(term)

    new_terms = [term for term in new_terms if len(term) > 1]

    return new_terms


def tag_sentence(sentence, term, tags):
    count = 0
    for ix in range(len(sentence) - len(term)):
        if sentence[ix:ix+len(term)] == term:
            count += 1
            if len(term) == 1:
                tags[ix] = 'S'
            else:
                for i in range(len(term)):
                    if i == 0:
                        tags[ix + i] = 'B'
                    elif i == len(term) - 1:
                        tags[ix + i] = 'E'
                    else:
                        tags[ix + i] = 'I'
    return tags, count


def tag_corpus(sentences, key_terms):
    term_counts = defaultdict(lambda: 0)
    corpus_tags = []

    for i, sentence in enumerate(sentences[1000:]):
        if i % 100 == 0: print(i)

        sentence = word_tokenize(sentence.lower())
        sentence_tags = ['O'] * len(sentence)

        for kt in key_terms:

            # iterate through all representations of a term
            terms = kt.lower().split(';')
            for term in terms:

                term = word_tokenize(term)

                sentence_tags, term_count = tag_sentence(sentence, term,
                                                         sentence_tags)

                term_counts[kt] += term_count

        corpus_tags.append(sentence_tags)

    return corpus_tags, term_counts


if __name__ == "__main__":

    # load textbook info
    with open('textbook_info.json') as f:
        textbook_info = json.load(f)
    textbooks = list(textbook_info.keys())


    # convert textbook pdfs to text for parsing
    for textbook in textbooks:
        print('Processing Textbook: %s' % textbook)

        # convert pdf to html using pdfminer
        print('Converting PDF to HTML')
        input_dir = '../data/textbooks_pdf'
        output_dir = '../data/textbooks_html'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists('%s/%s.html' % (output_dir, textbook)):
            call(['pdf2txt.py', '%s/%s.pdf' % (input_dir, textbook),
                  '-o', '%s/%s.html' % (output_dir, textbook),
                  '-t', 'html'])

        # load html for processing
        input_dir = '../data/textbooks_html'
        output_dir = '../data/textbooks_extracted'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open('%s/%s.html' % (input_dir, textbook)) as f:
            soup = BeautifulSoup(f, 'lxml')

        # extract and write chapter sentences
        print('Extracting Sentences')
        sentences = extract_sentences(soup, textbook_info[textbook])

        with open('%s/%s_sentences.txt' % (output_dir, textbook), 'w') as f:
            for sentence in sentences:
                f.write('%s\n' % sentence)

        # extract and write key terms
        print('Extracting Key Terms')
        key_terms = extract_key_terms(soup, textbook_info[textbook])

        with open('%s/%s_key_terms.txt' % (output_dir, textbook), 'w') as f:
            for term in key_terms:
                f.write('%s\n' % term)

        print('Creating Key Term Sentence Tags')
        # implement
        labels, counts = tag_corpus(sentences, key_terms)
        with open('%s/%s_sentence_tags.txt' % (output_dir, textbook), 'w') as f:
            for label in labels:
                f.write('%s\n' % ' '.join(label))
        print(counts)
        break


    # read in and process text
#    input_dir = 'data/textbooks_text'
#    output_dir = 'data/textbooks_processed'
#    for textbook in TEXTBOOKS:
#        with open('%s/%s.txt' % (input_dir, textbook)) as f:
#            text = f.readlines()
#        text = ' '.join(text)
#        text = text.replace('\n', '')
#
#        # extract sentences
#        sentences = sent_tokenize(text)
#        sentences = [sent for sent in sentences if len(sent.split(' ')) >= 5 and not
#                     sent[0].isdigit()]
#        with open('%s/%s_sentences.txt' % (output_dir, textbook), 'w') as f:
#            for sentence in sentences:
#                f.write('%s\n' % sentence)
#
#        print('Extracted %d sentences from %s' % (len(sentences), textbook))
#        break


        # extract key terms

    # create key term labels for each sentence


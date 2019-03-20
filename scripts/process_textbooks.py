from subprocess import call
from nltk.tokenize import sent_tokenize, word_tokenize
import json
import re
import os
import time
import pandas as pd
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

    # fix multi-line spanning words
    text = re.sub('-\s+', '', text)

    # split sentences, remove short < 3 words
    sentences = sent_tokenize(text)
    sentences = [sent for sent in sentences if len(word_tokenize(sent)) > 3]
    sentences = [sent for sent in sentences if len(word_tokenize(sent)) < 50]
    sentences = [' '.join(word_tokenize(sent)) for sent in sentences]

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
            if len((term[1] + term[2]).strip()) < 3:
                print(term[1] + term[2])
                term = term[0] + term[2]
                print(term)
            else:
                term = term[0] + term[2] + '; ' + \
                       term[1] + term[2]

        new_terms.append(term.strip())

    new_terms = [term for term in new_terms if len(term) > 1]

    return new_terms

def extract_key_terms_life(input_dir):
    terms = pd.read_excel('%s/life_biology_glossary.xlsx' % input_dir,
                          skiprows=4, header=None)

    new_terms = []
    for i in range(terms.shape[0]):
        term = terms.iloc[i, 0]
        acronym = terms.iloc[i, 2]
        if type(acronym) == str:
            if acronym.strip() != '' and len(acronym[1:-1].strip()) > 2:
                term = term + ';' + acronym[1:-1]
            else:
                print(acronym)
        new_terms.append(term)

    return new_terms


def tag_sentence(sentence, term, tags):
    count = 0
    for ix in range(len(sentence) - len(term)):
        if sentence[ix:ix+len(term)] == term:
            count += 1
            if len(term) == 1:
                # only put singleton if not subset of phrase
                if tags[ix] == 'O':
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

    for i, sentence in enumerate(sentences):

        sentence = sentence.lower().split(' ')
        sentence_tags = ['O'] * len(sentence)

        for kt in key_terms:

            # iterate through all representations of a term
            terms = kt.lower().split(';')
            for term in terms:
                if term != '':
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

    stats = []

    # convert textbook pdfs to text for parsing
    for textbook in textbooks:
        print('Processing Textbook: %s' % textbook)

        start_time = time.time()
        # convert pdf to html using pdfminer
        input_dir = '../data/textbooks_pdf'
        output_dir = '../data/textbooks_html'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists('%s/%s.html' % (output_dir, textbook)):
            print('Converting PDF to HTML')
            call(['pdf2txt.py', '%s/%s.pdf' % (input_dir, textbook),
                  '-o', '%s/%s.html' % (output_dir, textbook),
                  '-t', 'html'])

        # load html for processing
        input_dir = '../data/textbooks_html'
        output_dir = '../data/textbooks_extracted_copy'
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
        if textbook == 'life_biology':
            key_terms = extract_key_terms_life(input_dir)
        else:
            key_terms = extract_key_terms(soup, textbook_info[textbook])

        with open('%s/%s_key_terms.txt' % (output_dir, textbook), 'w') as f:
            for term in key_terms:
                f.write('%s\n' % term)

        if not os.path.exists('%s/%s_sentence_tags.txt' % (output_dir,
                                                           textbook)):
            print('Tagging Sentences')
            # implement
            labels, counts = tag_corpus(sentences, key_terms)

            # check the term counts
            num_terms = float(len(counts.keys()))
            num_zero = len([key for key in counts.keys() if counts[key] == 0])
            print('%d out of %d terms have 0 matches' % (num_zero, num_terms))

            with open('%s/%s_sentence_tags.txt' % (output_dir, textbook), 'w') as f:
                for label in labels:
                    f.write('%s\n' % ' '.join(label))

            with open('%s/%s_key_term_counts.json' % (output_dir, textbook), 'w') as f:
                json.dump(counts, f, indent=2)

        stats.append([textbook, len(sentences), len(key_terms)])

    stats.append(['Total', sum(s[1] for s in stats), sum(s[2] for s in stats)])
    stats = pd.DataFrame(stats, columns=['Textbook', '# Sentences', '# Key Terms'])
    stats.to_csv('%s/summary_statistics.csv' % output_dir, index=False)


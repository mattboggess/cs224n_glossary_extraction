from subprocess import call
from nltk.tokenize import sent_tokenize, word_tokenize
import json
import re
import os
from bs4 import BeautifulSoup

TEXTBOOKS = ('open_stax_anatomy_physiology',
             'open_stax_chemistry_2e',
             'open_stax_astronomy',
             'open_stax_biology_2e',
             'open_stax_concepts_of_biology',
             'open_stax_university_physics_v1',
             'open_stax_university_physics_v2',
             'open_stax_university_physics_v3')

TEXTBOOKS = ('open_stax_biology_2e',)

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
            if match_pattern(span, start_pattern):
                between = True

    return text


def extract_sentences(soup, pattern_info):


    spans = soup.find_all('span')
    #pattern_info = textbook_info[textbook]
    #chapter_pattern = re.compile('.*font-size:46px.*')
    #chapter_starts = soup.find_all('span', style=chapter_pattern)
    #key_term_starts = [p.parent for p in soup.find_all(text=)]

    #text_style = re.compile(".*'LiberationSans.*1[1-2]px.*")
    #text_pattern = ('style', text_style)
    #start_pattern = ('style', chapter_pattern)
    #end_pattern = ('text', re.compile('.*KEY TERMS.*'))
    text = get_text_between_elements(spans,
                                     pattern_info['chapter_start_pattern'],
                                     pattern_info['chapter_end_pattern'],
                                     pattern_info['chapter_text_pattern'])

    text = ' '.join(text)
    text = text.replace('\n', ' ')
    # remove empty parens, short sentences, ...
    sentences = sent_tokenize(text)
    return sentences


def extract_key_terms(soup, pattern_info):

    spans = soup.find_all('span')

    #text_style = re.compile(".*'LiberationSans-Bold.*12px.*")
    #text_pattern = ('style', text_style)
    #start_pattern = ('text', re.compile('.*KEY TERMS.*'))
    #end_pattern = ('text', re.compile('.*CHAPTER SUMMARY.*'))

    terms = get_text_between_elements(spans,
                                      pattern_info['key_terms_start_pattern'],
                                      pattern_info['key_terms_end_pattern'],
                                      pattern_info['key_term_pattern'])

    terms = [term.replace('\n', '') for term in terms]

    # handle abbreviations here
    return terms




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
        input_dir = 'data/textbooks_pdf'
        output_dir = 'data/textbooks_html'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists('%s/%s.html' % (output_dir, textbook)):
            call(['pdf2txt.py', '%s/%s.pdf' % (input_dir, textbook),
                  '-o', '%s/%s.html' % (output_dir, textbook),
                  '-t', 'html'])

        # load html for processing
        input_dir = 'data/textbooks_html'
        output_dir = 'data/textbooks_extracted'
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

        print('Creating Key Term Sentence Labels')
        # implement


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


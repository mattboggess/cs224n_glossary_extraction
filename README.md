# Automatically Extracting Glossaries from Textbooks Using Deep Learning

This repository hosts code for a class project completed for Stanford's Spring 2019 CS224N: Deep Learning for Natural Language Processing class by Matt Boggess (mattboggess) and Manish Singh (msingh9). The goal of the project was to build NLP deep learning models to automatically extract key terms and their definitions (glossaries) from textbooks as part of the larger [Stanford Inquire Research Project](http://web.stanford.edu/~vinayc/intelligent-life/).

A report giving an overview of the project and results can be found [here](http://web.stanford.edu/class/cs224n/reports/custom/15811430.pdf). "final report submitted commit" is the commit tag corresponding to the state of the code that produced the results for this report.

## Repo Overview

scripts: Contains Python scripts use to process the textbook dataset (www.openstax.com).
  - process_textbooks.py: Converts pdf representations of textbooks into text file list of chapter sentences, text file list of key terms, text file list of BIOES key term tags for each sentence, and json file of mapping from key terms to # of occurrences of key term in the text
  - gen_train_dev_test_data.py: Partitions textbooks into train, dev, and test splits for term identification task.
  - textbook_info.json: Contains regular expression patterns for extracting different parts of textbooks.
  - process_textbooks_for_def.py: Similarly converts pdf representations of textbooks into text file list of chapter sentences, text file lilist of definition sentence tags for the term definition extraction task.
  - gen_train_dev_test_data_def.py: Generates data partition for sentence identification task
  - process_w00_def_data.py & process_wiki_def_data.py: Processes ancillary definition datasets for the definition extraction task.

notebooks: Contains analysis notebook used to get statistics and generate figures for the report

src_ner: Code for training and evaluating term extraction models. The code for training and evaluating the deep learning models was adapted from an [example code base](https://github.com/cs230-stanford/cs230-code-examples/tree/master/pytorch/nlp) for Stanford's CS230 Deep Learning course with a few notable modifications for our project:
 - build_vocab.py: Builds the vocabulary from the dataset. Modified to support GloVe embeddings and character representations.
 - model/data_loader.py: Loads in batches of data and converts sentences to various embedding indices. Modified to support custom word embeddings, GloVe word embeddings, character embeddings, and BERT wordpiece embeddings.
 - model/net.py: Contains PyTorch models. Modified for models specifically used in our project.
 - evaluate.py: Contains evaluation code for the model. Custom evaluation scripts to assess term identification were added.
 - Other minor modifications to train.py as needed. 
One should refer to the documentation for the example code base for descriptions of how everything is run and organized.

src_def: Same as src_ner but adapted for the definition extraction task. 

## Limitations

This repo is primarily meant to serve as a reference for future groups on the project. It is recommended that one thoroughly test and clean up the code base if wanting to work with it directly (this project was completed under a very fast-paced deadline schedule so we did not have time to thoroughly test and organize everything). There are also major known limitations with the current formulation and data processing that are detailed in the report that should be taken into account. We have since provided a new dataset that processes the dataset in a cleaner way that should be preferred over the original dataset used for this report.


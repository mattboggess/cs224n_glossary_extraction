# Automatically Extracting Glossaries from Textbooks Using Deep Learning

Project completed for Stanford's CS224N: Deep Learning for Natural Language Processing class
by Matt Boggess (mattboggess) and Manish Singh (msingh9). Worked under supervision of
Dr. Vinay Chaudhri on the [Stanford Inquire Research Project](http://web.stanford.edu/~vinayc/intelligent-life/).

Final report located here: 
Git commit tag for final report version: "final report submitted commit"

scripts: Contains Python scripts use to process the textbook dataset.
notebooks: Contains analysis notebook used to get statistics and generate features
src: code for training and evaluating term extraction models 
src_def: code for training and evaluating definition extraction models

The code for training and evaluating the deep learning models was adapted from an [example code base](https://github.com/cs230-stanford/cs230-code-examples/tree/master/pytorch/nlp) for Stanford's CS230 Deep Learning course.

This code stills need to be cleaned up and organized. TODO:

	Data Processing:
		- Combine the definition and term extraction textbook processing scripts
		- Fully document the textbook processing scripts
		- Fix/handle the following known parsing issues:
			- plurals
			- morphological variations (depolarization vs. depolarized)
			- fix several key terms that got split onto multiple lines
			- broken words & other formatting issues
			- invalid bolded words for definition sentences
			- Section headings, figures, and other formatting sections to be screened
		- Potentially extract the glossary definitions directly in addition to candidate sentences from the chapter text
	Fully Clean up and Document Model Code
	Create an environment file to make code and environment reproducible.
	Add full documentation to this repo.

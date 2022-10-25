# this script creates a series of part of speech tags for a given list of german words and commits them to a text file
import spacy
import codecs
from tqdm import tqdm
from .helper_functions import print_word_list_to_text_file, replace_umlauts
import os

def pos_tag_word_list(input_text_paths, dataset_name, output_base_path, language='en'):
    print(f'producing part of speech tagged lists for input text files {input_text_paths}')
    output_noun_path = output_base_path + dataset_name + '_nouns.txt'
    output_verb_path = output_base_path + dataset_name + '_verbs.txt'
    output_adj_path = output_base_path + dataset_name + '_adjs.txt'
    output_other_path = output_base_path + dataset_name + '_others.txt'

    if language == 'en':
        nlp = spacy.load("en_core_web_sm")
    elif language =='de':
        nlp = spacy.load("de_core_news_sm")

    nouns = []
    verbs = []
    adjectives = []
    others = []

    input_text = []
    print('initiating spacy part of speech tagging on original data...')
    for input_path in input_text_paths:
        with codecs.open(input_path, 'r', encoding="utf8") as input_file:
            input_file_text = input_file.readlines()
            for line in input_file_text:
                input_text.append(line)

    for line in tqdm(input_text):
        # TODO scrub lines similar to BSG training process

        # TODO hauefigkeit miteinbeziehen

        line_tokens = nlp(line)
        for token in line_tokens:
            pos_tag = token.pos_
            token_text = token.text.lower()
            if language == 'de':
                token_text = replace_umlauts(token_text)

            if pos_tag == 'NOUN':
                if token_text not in nouns:
                    nouns.append(token_text)
            if pos_tag == 'VERB':
                if token_text not in verbs:
                    verbs.append(token_text)
            if pos_tag == 'ADJ':
                if token_text not in adjectives:
                    adjectives.append(token_text)
            else:
                if token_text not in others:
                    others.append(token_text)

    print('saving part of speech tagging results to files')
    print_word_list_to_text_file(nouns, output_noun_path)
    print_word_list_to_text_file(verbs, output_verb_path)
    print_word_list_to_text_file(adjectives, output_adj_path)
    print_word_list_to_text_file(others, output_other_path)


def check_pos_tagged_availability(input_base_path, output_base_path, reference_word_list_path):
    input_files = os.listdir(input_base_path)
    print(f"determining tagged word availability for entailment measurements")
    available_words = []
    print(f"grabbing reference word list from {reference_word_list_path}")
    with open(reference_word_list_path, 'r', encoding="utf8") as reference_file:
        reference_words = reference_file.readlines()
        for reference_word in reference_words:
            available_words.append(reference_word)
    print(f"checking lists: {input_files}")
    for input_filename in input_files:
        category = input_filename.split('.')[0].split('_')[-1]
        available_tagged_words = []
        with open((input_base_path+input_filename), 'r', encoding="utf8") as input_file:
            tagged_words = input_file.readlines()
            for tagged_word in tagged_words:
                if tagged_word in available_words:
                    available_tagged_words.append(tagged_word)
        print(f"determined {len(available_tagged_words)} words of {len(tagged_words)} from  {input_filename} to be available for entailment testing")
        output_file_path = output_base_path + input_filename
        print_word_list_to_text_file(available_tagged_words, output_file_path)


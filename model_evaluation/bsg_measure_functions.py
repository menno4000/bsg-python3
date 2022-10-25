# this file holds functions that produce entailment measurements from the BSG inference API
import requests
import os
from tqdm import tqdm
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.cistem import Cistem

entailment_url = "http://127.0.0.1:5000/entailment"
# english language lemmatizer
lemmatizer = WordNetLemmatizer()
# german language lemmatizer
stemmer = Cistem()


def lemmatize_word(word, language='de'):
    if language == 'de':
        stemmed_segments = stemmer.segment(word)
        return stemmed_segments[0]
    if language == 'en':
        return lemmatizer.lemmatize(word)


def produce_entailment_measurements(input_word, input_word_list, input_word_list_name, output_path, language='de'):
    print(f'producing entailment results for {input_word} from reference list {input_word_list_name}')
    results = {}
    for word_2 in tqdm(input_word_list):
        if input_word != word_2:
            # lemmatize to avoid words of the same tree
            input_word_lemma = lemmatize_word(input_word, language)
            word2_lemma = lemmatize_word(word_2)
            if input_word_lemma != word2_lemma:
                key = input_word + '_' + word_2
                if key not in results.keys() and (word_2 + '_' + input_word) not in results.keys():
                    params = {"word1": input_word, "word2": word_2}
                    r = requests.post(entailment_url, params=params)
                    if r.status_code != 200:
                        continue
                    else:
                        response_json = r.json()
                        entailment_results = response_json["entailment prediction results"][0]["entailment"]

                        cos = entailment_results["cosine similarity score"]
                        kl1 = float(entailment_results["word1 -> word2 (KL)"])
                        kl2 = float(entailment_results["word2 -> word1 (KL)"])
                        kl_mean = (kl1 + kl2) / 2

                        results[key] = [cos, kl_mean, kl1, kl2]

    print(f'successfully collected {len(results.keys())} entailment measurements, saving to {output_path}')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_filename = input_word + '_entailment_results.csv'
    output_file = output_path + output_filename

    with open(output_file, 'w', encoding="utf8") as of:

        of.write("Word 1,Word2,Cosine Sim,KL 1->2,KL 2->1,KL Mean\n")
        for result in results.keys():
            res = results[result]
            if res is not None:
                key_parts = result.split('_')
                word_2 = key_parts[1].strip()
                of.write(f"{input_word},{word_2},{res[0]},{res[2]},{res[3]},{res[1]}\n")


def read_kl_from_entailment_results_file(input_path):
    data = []
    with open(input_path, 'r', encoding="utf8") as input_file:
        results = input_file.readlines()
        for result in results[1:]:
            result_data = result.split(',')
            if len(result_data) > 1:
                result_word = result_data[1].strip()
                result_value = float(result_data[5])
                data.append([result_word, result_value])

    return data


def read_cos_from_entailment_results_file(input_path):
    data = []
    with open(input_path, 'r', encoding="utf8") as input_file:
        results = input_file.readlines()
        for result in results[1:]:
            result_data = result.split(',')
            if len(result_data) > 1:
                result_word = result_data[1].strip()
                result_value = float(result_data[2])
                data.append([result_word, result_value])

    return data


# calculates a set of entailment results from part-of-speech tagged word lists
# from each part-of-speech word list, a set of a given size is taken and compared to all other words from all lists
def produce_pos_tagged_entailment_measurements(tagged_list_input_path, output_base_path, size=10, language='en'):
    tagged_lists = os.listdir(tagged_list_input_path)
    print(f"producing entailment results for pos tagged lists {tagged_lists}")
    tagged_word_data = []
    print("gathering tagged words")
    for tagged_list in tagged_lists:
        tagged_words = []
        with open((tagged_list_input_path+tagged_list), 'r') as tagged_file:
            for line in tagged_file.readlines():
                tagged_words.append(line.strip())
        tagged_word_data.append([tagged_list, tagged_words])

    for target_word_data in tagged_word_data:
        word_list_name = target_word_data[0]
        target_pos_category = word_list_name.split('.')[0].split('_')[-1]
        print(f"beginning entailment measurements for target part of speech category {target_pos_category}")
        word_list = target_word_data[1]
        if size > 0:
            word_list_subset = word_list[:size]
        else:
            word_list_subset = word_list[size:] # set size to negative to produce a set of different size
        for target_word in word_list_subset:
            for reference_word_data in tagged_word_data:
                reference_data_name = reference_word_data[0]
                reference_pos_category = reference_data_name.split('.')[0].split('_')[-1]
                output_directory = output_base_path + target_pos_category + '_to_' + reference_pos_category + '/'
                produce_entailment_measurements(target_word, reference_word_data[1], reference_data_name, output_directory, language)


# calculate part-of-speech-tagged entailment results for a given tuple list
def produce_pos_tagged_entailment_measurements_from_tuple_list(target_tuples, tagged_list_input_path, output_base_path, size=10, language='en'):
    tagged_lists = os.listdir(tagged_list_input_path)
    print(f"producing entailment results for pos tagged lists {tagged_lists}")
    tagged_word_data = []
    print("gathering tagged words")
    for tagged_list in tagged_lists:
        tagged_words = []
        with open((tagged_list_input_path+tagged_list), 'r') as tagged_file:
            for line in tagged_file.readlines():
                tagged_words.append(line.strip())
        tagged_word_data.append([tagged_list, tagged_words])

    for target_tuple in target_tuples:
        target_word = target_tuple[0]
        target_pos_category = target_tuple[1]
        for reference_word_data in tagged_word_data:
            reference_data_name = reference_word_data[0]
            reference_pos_category = reference_data_name.split('.')[0].split('_')[-1]
            output_directory = output_base_path + target_pos_category + '_to_' + reference_pos_category + '/'
            produce_entailment_measurements(target_word, reference_word_data[1], reference_data_name, output_directory, language)


def read_word_count_dict_from_file(input_path):
    word_counts = {}
    with open(input_path, 'r') as tclf:
        for word_line in tclf.readlines():
            word_count_data = word_line.split(',')
            word = word_count_data[0].strip()
            count = int(word_count_data[1].strip())
            word_counts[word] = count
    return word_counts
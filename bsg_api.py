import os
import logging
import argparse
import numpy as np
from flask import Flask, jsonify, make_response, abort, request, json
from timeit import default_timer as timer

from prep.train.bsg.interfaces.interface_configurator import InterfaceConfigurator
from prep.train.bsg.libraries.tools.vocabulary import Vocabulary
from prep.train.bsg.libraries.evaluation.entailment.support import read_vectors_to_dict, read_vectors_to_arrays
from prep.train.bsg.libraries.simulators.support import cosine_sim

train_data_path = './path/to/docs' # change the path! must point to directory containing .txt input files
output_base_path = "./output/my-dataset/" # change the path (optional)

mus_x = []
mus_and_sigmas = {}
mu_hash_dict = {}
spcial_char_map = {ord('ä'):'a', ord('ü'):'u', ord('ö'):'o', ord('ß'):'ss'}

app = Flask(__name__)

logger = logging.getLogger('werkzeug')
logger.setLevel(logging.ERROR)
endpoints = [
    {
        'title': 'nearest neighbour prediction for text queries',
        'description': 'computes five nearest neighbours of each known word of a text query',
        'url': '<host>/nearest_neighbours',
        'query parameters': {
            'query': 'The Full Text Query for which five Nearest Neighbours are to be predicted'
        }
    }
]


@app.route('/', methods=['GET'])
def get_endpoints():
    return jsonify({'endpoints': endpoints})


@app.route('/nearest_neighbours', methods=['POST'])
def get_nearest_neighbours_for_query():
    query = request.args.get('query')
    if query is None:
        abort(400, description='Parameter \'query\' not found. Please provide a text query to infer recommendations on.')
    words = [clean_target_word(w) for w in query.split(' ')]
    recommendations = {}
    time_elapsed_arr = []

    # for each word of the query
    for word in words:
        # given that the word is in the model vocabulary
        if word in mus_and_sigmas:
            nearest_neighbours, time_elapsed = get_nearest_neighbours(word)
            time_elapsed_arr.append(time_elapsed)

        # print(f"recommendations for word {word}: {nearest_neighbours} calculated in {time_elapsed} seconds")
            recommendations[word] = nearest_neighbours
        # else:
            # print(f"word {word} not in dictionary")
    if recommendations:
        if len(words) == 1:
            result = {
                "recommendations": recommendations[query],
                "time_elapsed": np.average(time_elapsed_arr)
            }
        else:
            result = {
                "recommendations": recommendations,
                "time_elapsed": np.average(time_elapsed_arr)
            }
    else:
        result = {
            "recommendations": []
        }
    return jsonify(result)



def get_nearest_neighbours(target_word):
    start_time = timer()
    mu_vector = mus_and_sigmas.get(target_word)[0]
    # nearest neighbour implementation borrowed from BSG evaluation
    dists_exact = np.linalg.norm(mus_x - mu_vector, axis=1) ** 2
    dists_exact_sorted_indices = dists_exact.argsort()
    other_exact_sorted = mus_x[dists_exact_sorted_indices][1:]
    nearest_neighbours_exact_mus = other_exact_sorted[:5]
    nearest_neighbours = [mu_hash_dict[hash(str(m))] for m in nearest_neighbours_exact_mus]
    stop_time = timer()
    time_elapsed = stop_time - start_time
    return nearest_neighbours, time_elapsed


def clean_target_word(target_word):
    return target_word.translate(spcial_char_map)


def parse_args():
    parser = argparse.ArgumentParser(description="get model path")
    parser.add_argument('--model_index', help="directory of the word embedding model to be loaded", default="0")
    return parser.parse_args()


def load_model(model_index):
    output_folder_path = output_base_path + model_index + '/'
    vocab_file_path = output_folder_path + 'vocab.txt'

    # load the model
    # i_model = InterfaceConfigurator.get_interface(train_data_path,
    #                                               vocab_file_path,
    #                                               output_folder_path,
    #                                               model_file_path=output_folder_path+"model.pkl")
    vocab = Vocabulary()
    vocab.load(vocab_file_path=vocab_file_path)
    mu_vecs = os.path.join(output_folder_path+"mu.vectors")
    sigma_vecs = os.path.join(output_folder_path+"sigma.vectors")

    globals()['mus_and_sigmas'] = read_vectors_to_dict(mu_vecs, sigma_vecs, log_sigmas=True)

    globals()['mus_x'] = np.asarray([x[0].tolist() for x in mus_and_sigmas.values()], dtype='float32')

    words = mus_and_sigmas.keys()
    mus_hashes = [hash(str(mus)) for mus in mus_x]
    globals()['mu_hash_dict'] = dict(zip(mus_hashes, words))
    print(f"model {model_index} loaded")


def main():
    args = parse_args()
    model_index = args.model_index
    load_model(model_index)
    app.run(debug=False)


if __name__ == '__main__':
    main()

# this file contains an example on how to run the bayesian skip-gram model
import os
import argparse
from interfaces.interface_configurator import InterfaceConfigurator

train_data_path = './path/to/docs' # change the path! must point to directory containing .txt input files
vocab_file_path = './output/invoice/invoice.txt' # if the file does not exist - it will be created
output_folder_path = "./output/my-dataset/"  # change the path(optional)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default='2', help='number of epochs for training')
    parser.add_argument('--max_vocab_size', type=int, default='100000', help='size of the model vocabulary that will be created')
    parser.add_argument('--alpha', type=float, default='0.0015', help='learning rate for training')
    parser.add_argument('--embedding_size', type=int, default='200', help='size of the models embedding vectors')
    parser.add_argument('--batch_size', type=int, default='500', help="batch size for context window creation")
    parser.add_argument('--nr_neg_samples', type=int, default='10', help="number of negative samples for skip gram algorithm")
    return parser.parse_args()


def train(args):
    # obtain the interface to interact with the model. If one wants to change hyper-param the manual modification of the below class's method will be necessary!
    i_model = InterfaceConfigurator.get_interface(train_data_path, vocab_file_path, output_folder_path='invoice',
                                                  epochs=args.epochs, alpha=args.alpha,
                                                  max_vocab_size=args.max_vocab_size,
                                                  batch_size=args.batch_size,
                                                  nr_neg_samples=args.nr_neg_samples,
                                                  embedding_size=args.embedding_size)

    i_model.train_workflow()

    # store the temporary vocab, because it can be different from the original one(e.g. smaller number of words)
    vocab = i_model.vocab
    temp_vocab_file_path = os.path.join(i_model.output_path, "vocab.txt")
    vocab.write(temp_vocab_file_path)

    mu_vecs = [os.path.join(i_model.output_path, "mu.vectors")]
    sigma_vecs = [os.path.join(i_model.output_path, "sigma.vectors")]

    # a complex of word embedding evaluations(word similarity, entailment, directional entailment)
    #evaluate(mu_vectors_files=mu_vecs, sigma_vectors_files=sigma_vecs, vocab_file=temp_vocab_file_path, log_sigmas=False,
    #         full_sim=True, vocab=vocab)

    # run additionally lexical substitution evaluation
    #run_lexsub(input_folder=i_model.output_path, output_path=i_model.output_path)


# needed on windows, otherwise spawning get's problems
if __name__ == '__main__':
    train(parse_args())

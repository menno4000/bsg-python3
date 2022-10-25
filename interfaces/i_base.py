from libraries.utils.paths_and_files import get_subdir_number
import sys, os
from interfaces.support import load, save, metrics_to_str
from libraries.tools.log import Log
from libraries.utils.other import merge_ordered_dicts
from interfaces.support import infer_attributes_to_log, format_experimental_setup
from libraries.tools.ordered_attrs import OrderedAttrs

# a dirty hack from:
# http://stackoverflow.com/questions/24171725/scikit-learn-multicore-attributeerror-stdin-instance-has-no-attribute-close
# to avoid iPython in pycharm crushing
if not hasattr(sys.stdin, 'close'):
    def dummy_close():
        pass
    sys.stdin.close = dummy_close


class IBase(OrderedAttrs):
    """
    Base interface class that contains methods that must be implemented and the ones that can be used directly in children classes.

    """
    def __init__(self, model_class, vocab, epochs=5, learning_rate=0.001, max_vocab_size=50000, batch_size=100, nr_neg_samples=5, embedding_size=100,
                 train_data_path=None, val_data_path=None, test_data_path=None,
                 output_dir=None):
        OrderedAttrs.__init__(self)

        # will be assigned later on in the child class
        self.model = None
        self.init_iterator = None

        self.model_class = model_class
        self.vocab = vocab
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.test_data_path = test_data_path
        self.epochs = epochs

        output_dir = os.path.join(os.getcwd(), output_dir) if output_dir else os.path.join(os.getcwd(), 'output')
        print(f'output_dir: {output_dir}')
        output_number = get_subdir_number(output_dir)
        learning_rate_for_dir = str(learning_rate)[2:]
        vocab_size_for_dir = f'{str(max_vocab_size)[:-3]}k'
        output_index = f'{output_number}_lr{learning_rate_for_dir}_e{str(epochs)}_v{vocab_size_for_dir}_bs{batch_size}_nns{nr_neg_samples}_es{embedding_size}'
        self.output_path = os.path.join(output_dir, output_index)
        self.log = Log(self.output_path)  # will write log to a current w. dir. if not provide

    def init_model(self, **kwargs):
        """
        Initializes the actual model.

        """
        self.model = self.model_class(**kwargs)
        self.record_experimental_setup()

    def train_workflow(self, evaluate=True, save_model=True):
        """
        Runs a workflow of steps such as training and evaluation. One could modify it in order to create other procedures.
        :param evaluate: if True will evaluate otherwise not.

        """
        assert self.train_data_path

        for epoch in range(1, self.epochs+1):

            self.log.write('epoch %d' % epoch)
            self.train(data_path=self.train_data_path)

            # evaluate training and validation accuracy and loss
            if evaluate:
                # FIXME: at the moment the training evaluation is disabled, as it's too expensive to perform evaluation over the whole large dataset.
                # metrics = self._measure_performance(data_path=self.train_data_path)
                # if metrics:
                #     self.log.write(metrics_to_str(metrics, "training"))

                if self.val_data_path:
                    metrics = self._measure_performance(data_path=self.val_data_path)
                    self.log.write(metrics_to_str(metrics, "validation"))

        if evaluate and self.test_data_path:
            metrics = self._measure_performance(data_path=self.test_data_path)
            self.log.write(self.log.write(metrics_to_str(metrics, "test")))

        # save the actual model
        if save_model:
            self.save_model(os.path.join(self.output_path, 'model.pkl'))
            self.log.write("model is saved to: %s" % self.output_path)

        # run post training functions
        self._post_training_logic()

    def train(self, data_path):
        """
        A user accessible train function that wraps the model's train function.
        :type data_path: str

        """
        iterator = self.init_iterator(data_path)
        for counter, batch in enumerate(iterator, 1):
            metrics = self._train(batch=batch)
            if counter % 10 == 0:
                self.log.write(metrics_to_str(metrics, prefix="chunk's # %d" % counter))

    def load_model(self, model_file_path):
        """
        :param model_file_path: pre-saved pkl file with a model.

        """
        self.model = load(model_file_path)
        self.record_experimental_setup()
        self.log.write("loaded the model from: %s" % model_file_path)

    def save_model(self, file_path):
        assert self.model  # the model should be initialized
        save(self.model, file_path)

    def load_params(self, params_dump_file_path=None, exclude_params=[]):
        """
        Loads parameters from dumps or/and embeddings from a file.
        :param exclude_params: an array of parameter names that should NOT be initialized.

        """
        assert self.model  # the model should be initialized

        # general parameters loading
        if params_dump_file_path:
            init_params = self.model.load_params(file_path=params_dump_file_path, exclude_params=exclude_params)
            self.log.write("loaded parameters from: %s" % params_dump_file_path)
            self.log.write("initialized the following parameters: %s" % (", ".join(init_params)))

    def record_experimental_setup(self):
        """
        Records the experimental setup: basic and the model specific to a log file.

        """
        setup = merge_ordered_dicts(infer_attributes_to_log(self.model), infer_attributes_to_log(self))
        self.log.write(format_experimental_setup(setup), include_timestamp=False)

    # the following functions will be implemented in children classes
    # TODO: write a basic documentation for those functions

    def _train(self, **kwargs):
        """
        A specific wrapper over the model's training function.

        """
        raise NotImplementedError

    def _measure_performance(self, **kwargs):
        """
        Computes the performance of the model and returns a dictionary with names and values.

        """
        raise NotImplementedError

    def _post_training_logic(self, **kwargs):
        """
        Logic that is desired to be executed in the train_workflow after the model has finished training.

        """
        pass

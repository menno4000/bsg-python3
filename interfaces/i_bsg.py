from models.bsg import BSG
from interfaces.i_base import IBase
from collections import OrderedDict
from libraries.batch_iterators.window_batch_iterator import WindowBatchIterator as BatchIterator
from interfaces.support import compute_loss


class IBSG(IBase):
    """
    Interface class that builds on top of the BSG model. Specifically, it wraps the model's methods to easy user access.

    """
    def __init__(self, data_iterator, vocab, half_window_size=5, nr_neg_samples=5, batch_size=5,
                 epochs=5, max_vocab_size=50000, learning_rate=0.0001, embedding_size=100,
                 subsampling_threshold=None, **kwargs):
        # init the parent object
        IBase.__init__(self, vocab=vocab, model_class=BSG,
                       epochs=epochs,
                       max_vocab_size=max_vocab_size,
                       learning_rate=learning_rate,
                       batch_size=batch_size,
                       nr_neg_samples=nr_neg_samples,
                       embedding_size=embedding_size, **kwargs)

        # general attributes
        self.batch_size = batch_size
        self.half_window_size = half_window_size
        self.nr_neg_samples = nr_neg_samples
        self.subsampling_threshold = subsampling_threshold

        self.init_iterator = lambda data_path: BatchIterator(vocab, data_path, data_iterator, half_window_size=half_window_size, nr_neg_samples=nr_neg_samples,
                                                             subsampling_threshold=subsampling_threshold, batch_size=batch_size)

    def _measure_performance(self, data_path):
        return {"loss": compute_loss(self.init_iterator(data_path), loss_func=self.loss_func)}

    def _train(self, batch):
        mean_margin, mean_kl, avg_log_det = self.model.train(batch.pos_context_words, batch.neg_context_words,
                                                             batch.center_words, batch.mask)
        return OrderedDict((("margin", mean_margin), ("kl", mean_kl), ("log_det", avg_log_det)))

    def loss_func(self, batch):
        return self.model.loss(batch.pos_context_words, batch.neg_context_words, batch.center_words, batch.mask)

    def _post_training_logic(self):
        # will execute this code after the training workflow is finished, can contain custom functions, e.g. saving
        # of word embeddings
        self.model.save_word_vectors(self.vocab, vectors_folder=self.output_path)
        self.log.write("Word vectors are saved to: %s" % self.output_path)
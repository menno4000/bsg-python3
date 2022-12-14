from nltk import word_tokenize as default_tokenizer
from libraries.data_iterators.support import deal_with_accents
from libraries.utils.paths_and_files import get_file_paths
import codecs #Python 3

class OpenTextDataIterator():

    def __init__(self, tokenizer=None):
        """
        Text data iterator for open text, that returns tokenized sentences. Assumes that each sentence is separated
        by a new line.

        """
        self.tokenizer = tokenizer if tokenizer else default_tokenizer
        self.data_path = None

    def set_data_path(self, data_path):
        self.data_path = data_path

    def __iter__(self):
        if not self.data_path:
            raise ValueError("please specify the data_path first by calling set_data_path()")
        for filename in get_file_paths(self.data_path):
            with codecs.open(filename,'r',encoding='utf-8') as f: #Python 3
                for line in f:
                    # tokens = self.tokenizer(deal_with_accents(line.strip().lower().decode('utf-8', 'ignore')))
                    tokens = self.tokenizer(deal_with_accents(line.strip().lower())) # python 3
                    yield tokens,

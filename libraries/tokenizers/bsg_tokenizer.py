from libraries.tools.word_processor import WordProcessor
from nltk import word_tokenize as default_tokenizer
try:
    import re2 as re
except ImportError:
    import re

TOKENS = {"URL_TOKEN": "<URL>", "FLOAT_TOKEN": "<FLOAT>"}


class BSGTokenizer:
    """
    A more advanced tokenizer that both tokenizes and cleans textual data. It's specifically tailored for BSG models.

    """
    def __init__(self, word_processor_type='none', use_external_tokenizer=True):
        """
        :param use_external_tokenizer: whether to use NLTK tokenizer or rely on the simple splitter(x.split()).
        """
        if use_external_tokenizer:
            self.tokenizer = default_tokenizer
        else:
            self.tokenizer = split  # assuming that data was already tokenized
        self.word_processor = WordProcessor(word_processor_type=word_processor_type)

    def __call__(self, sentence):
        """
        :param sentence: a string of words
        :return: a list of clean tokens

        """
        words = self.tokenizer(sentence)
        tokens = []
        for word in words:
            # check if the word matches some known token
            token = self.__match_to_known_token(word)
            if not token:
                # clean the word otherwise
                token = self.word_processor.process(word)
                if token == "":
                    continue
            tokens.append(token)

        return tokens

    @staticmethod
    def __match_to_known_token(word):
        # URL
        if re.match(r"^(https?:\/\/(?:www\.|(?!www))[^\s\.]+\.[^\s]{2,}|www\.[^\s]+\.[^\s]{2,})$", word):
            return TOKENS["URL_TOKEN"]
        # FLOAT
        if re.match(r'^([0-9]+\.)[0-9]+$', word):
            return TOKENS["FLOAT_TOKEN"]
            
# Wordprocessor and lambda is used in multiprocessing processes. Hence it need to be pickled.
# But pickled objects can't include method definitions. Hence they need to be made explicite
# see https://laszukdawid.com/2017/12/13/multiprocessing-in-python-all-about-pickling/
 
def split(x):
    return x.split()

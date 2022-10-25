# word processors that use regular expressions to clean words and tokenization
try:
    import re2 as re
except ImportError:
    import re

# Wordprocessor is used in multiprocessing processes. Hence, it will be pickled.
# The original definition gave the well known error "Python multiprocessing PicklingError: 
# Can't pickle <type 'function'>". By creating ean explicite process method, we can get
# around this error.
    
class WordProcessor:
    def __init__(self, word_processor_type='default'):
        self.__allowed_types = ['none', 'default', 'open_text']
        # sanity checks for input
        assert word_processor_type in self.__allowed_types
        # assigning processing function
        self.word_processor_type = word_processor_type
        
    def process(self,x):
        if self.word_processor_type == 'none':
            return self._id(x)
        if self.word_processor_type == 'default':
            return self._default(x)
        if self.word_processor_type == 'open_text':
            return self._open_text_cleaner(x)

    @staticmethod
    def _open_text_cleaner(word):
        """
        Direct copy from the original BSG setup. The tokens matching logic was moved to bsg_tokenizer.py

        """
        word = re.sub(r'[^\w\'\-]|[\'\-\_]{2,}', "", word)
        if len(word) == 1:
            word = re.sub(r'[^\daiu]', '', word)
        return word
        
    @staticmethod
    def _id(x):
        return x
        
    @staticmethod
    def _default(word):
        return re.sub(r'[^\w_,.?@!$#\':\/\-()]|[,\'?@$#]{2,}', "", word)
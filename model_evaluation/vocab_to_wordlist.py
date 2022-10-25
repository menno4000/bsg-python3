import re
import codecs

def vocab_to_word_list(input_vocab_path, output_word_list_path):
    print(f'creating word list from vocab file {input_vocab_path}')
    words = []

    with codecs.open(input_vocab_path, 'r') as vocab_file:
        vocab_text = vocab_file.readlines()
        for line in vocab_text:
            word = line.split(' ')[0]
            word = re.sub('[^a-zA-Z]+', '', word)
            if len(word) > 1:
                words.append(word)

    print(f"saving {len(words)} words to {output_word_list_path}")
    with open(output_word_list_path, 'w', encoding="utf8") as word_list_file:
        for word in words:
            word_list_file.write("%s\n" % word)


def vocab_to_word_list_counted(input_vocab_path, output_word_list_path):
    print(f'creating word list from vocab file {input_vocab_path}')
    words = []

    with codecs.open(input_vocab_path, 'r') as vocab_file:
        vocab_text = vocab_file.readlines()
        for line in vocab_text:
            line_data = line.split(' ')
            word = line_data[0]
            count = line_data[1]
            word = re.sub('[^a-zA-Z]+', '', word)
            if len(word) > 1:
                words.append((word, count))

    print(f"saving {len(words)} words to {output_word_list_path}")
    with open(output_word_list_path, 'w', encoding="utf8") as word_list_file:
        for word in words:
            word_list_file.write(f"{word[0]},{word[1]}")

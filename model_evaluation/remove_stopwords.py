from stop_words import get_stop_words
from .helper_functions import replace_umlauts


def remove_stopwords(input_word_list_path, output_word_list_path, language='en'):
    print(f"creating stop word filtered version of input word list {input_word_list_path}")
    stop_words = []
    if language == 'de':
        stop_words = get_stop_words('german')
        stop_words_mod = []
        for stop_word in stop_words:
            stop_words_mod.append(replace_umlauts(stop_word))
        stop_words = stop_words_mod
    elif language == 'en':
        stop_words = get_stop_words('english')

    print(f"gathered {len(stop_words)} from the provided library")
    nonstop_words = []

    with open(input_word_list_path, 'r', encoding="utf8") as words_file:
        words = words_file.readlines()
        print(f"filtering {len(words)} words")
        for word in words:
            if word.strip() not in stop_words:
                nonstop_words.append(word)

    print(f"saving {len(nonstop_words)} non-stop-words to {output_word_list_path}")
    with open(output_word_list_path, 'w', encoding="utf8") as nonstop_words_file:
        for ns_word in nonstop_words:
            nonstop_words_file.write("%s" % ns_word)


def remove_stopwords_counted(input_word_list_path, output_word_list_path, language='en'):
    print(f"creating stop word filtered version of input word list {input_word_list_path}")
    stop_words = []
    if language == 'de':
        stop_words = get_stop_words('german')
        stop_words_mod = []
        for stop_word in stop_words:
            stop_words_mod.append(replace_umlauts(stop_word))
        stop_words = stop_words_mod
    elif language == 'en':
        stop_words = get_stop_words('english')

    print(f"gathered {len(stop_words)} from the provided library")
    nonstop_words = []

    with open(input_word_list_path, 'r', encoding="utf8") as words_file:
        word_count_lines = words_file.readlines()
        print(f"filtering {len(word_count_lines)} words")
        for word_count_line in word_count_lines:
            word_count_data = word_count_line.split(',')
            word = word_count_data[0]
            if word not in stop_words:
                count = int(word_count_data[1])
                nonstop_words.append((word, count))

    print(f"saving {len(nonstop_words)} non-stop-words to {output_word_list_path}")
    with open(output_word_list_path, 'w', encoding="utf8") as nonstop_words_file:
        for ns_word in nonstop_words:
            nonstop_words_file.write(f"{ns_word[0]}, {ns_word[1]}\n")

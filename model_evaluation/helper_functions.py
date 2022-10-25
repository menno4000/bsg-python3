# this script includes helper functions used throughout the BSG evaluation project.


def replace_umlauts(umlaut_word):
    u = 'ü'.encode()
    U = 'Ü'.encode()
    a = 'ä'.encode()
    A = 'Ä'.encode()
    o = 'ö'.encode()
    O = 'Ö'.encode()
    ss = 'ß'.encode()

    umlaut_word = umlaut_word.encode()
    umlaut_word = umlaut_word.replace(u, b'u')
    umlaut_word = umlaut_word.replace(U, b'U')
    umlaut_word = umlaut_word.replace(a, b'a')
    umlaut_word = umlaut_word.replace(A, b'A')
    umlaut_word = umlaut_word.replace(o, b'o')
    umlaut_word = umlaut_word.replace(O, b'O')
    umlaut_word = umlaut_word.replace(ss, b'')

    umlaut_word = umlaut_word.decode('utf-8')
    return umlaut_word


def print_word_list_to_text_file(input_list, path):
    with open(path, 'w', encoding="utf8") as text_file:
        for entry in input_list:
            word = entry.strip()
            text_file.write("%s\n" % word)


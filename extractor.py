import glob
import os
import re
import string
from collections import Counter

import nltk
import spacy
from pdfminer.high_level import extract_text
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from spacy.lang.en.stop_words import STOP_WORDS


def read_file(fpath):
    content = extract_text(fpath)
    return content


def data_pre_process(data):
    data = data.lower()
    data = data.replace('\n\n', ' ')  # Replacing extra new lines.
    data = data.replace('\n', ' ')
    data = data.replace('•', ',')
    data = data.replace('', ',')

    return data


def ext_sent(lines, output_fname):
    fw = open('./output/' + output_fname, 'w')  # Writing results operations

    lines = data_pre_process(lines.strip())

    stop_words = set(stopwords.words('english'))
    punctuations = string.punctuation
    new_stp = ["____________________________"]

    res = sent_tokenize(lines)

    fw.write(lines)

    fw.write('\n\n ***** NLTK ***** ')
    fw.write(f'\n\n --- sentences ----\n\n {res}')
    fw.write(f'\n\n total sentences --> {len(res)}')
    fw.write('\n\n --- Sentence Display --- \n\n')

    for sentence in res:
        fw.write(f'\n\n sent --> {sentence}')
        sentence = sentence.strip()
        remove_sw = " ".join([word for word in sentence.split() if word not in stop_words])
        fw.write(f'\n\n filtered sentence [stopwords] --> {remove_sw}')
        tokens = word_tokenize(remove_sw)
        fw.write(f'\n\n Tokens --> {tokens}')
        fw.write(f'\n\n Total Tokens --> {len(tokens)}')
        remove_new_stp = " ".join([tk for tk in tokens if tk not in new_stp])
        remove_punch = " ".join([tk for tk in tokens if tk not in punctuations])
        fw.write(f'\n\n Filtered Tokens [New Stop Words] --> {remove_new_stp}')
        fw.write(f'\n\n Filtered Tokens [punctuations] --> {remove_punch}')
        fw.write(f'\n\n {"------"*10}')

    # --- spacy ----
    # nlp = spacy.load("en_core_web_sm")
    nlp = spacy.load("./trained_models/")
    nlp.add_pipe('sentencizer')
    doc = nlp(lines)
    fw.write('\n\n ***** SPACY ***** ')
    fw.write(f'\n\n --- doc --- \n {doc}')
    fw.write('\n\n ------ Spacy Sentences ------ \n\n ')

    filtered_sentence = []

    for sent in doc.sents:
        fw.write(f'\n\n sent --> {sent}')
        for sent_tk in sent:
            if sent_tk.text.strip() not in punctuations:
                lexeme = nlp.vocab[sent_tk.text]
                if lexeme.is_stop is False:
                    filtered_sentence.append(sent_tk)
        for y in sent.ents:
            # print('\n\n sentence --> ', sent.text.strip())
            # print('\n\n ents --> ', sent.ents)
            # print('\n\n --- NER ---\n\n')
            # print(f'\n\n {y.text} -> {y.label_}')
            fw.write(f'\n\n sentence entities --> {sent.ents}')
            fw.write(f'\n\n {y.text} -> {y.label_}')
            # print('\n\n ---------------------------')

    fw.write(f'\n\n filtered sentence --> {filtered_sentence}')


if __name__ == '__main__':
    i_path = './input/'
    o_path = './output/'

    i_path_exist = os.path.exists(i_path)  # Check the user input path
    o_path_exist = os.path.exists(o_path)  # Check the user output path

    if i_path_exist and o_path_exist:
        for file in glob.glob(i_path + '*'):
            fname = os.path.basename(file)  # Get file name from path
            filename, ext = os.path.splitext(fname)  # Extract filename & extension
            output_fname = filename + '_output.txt'

            if fname.endswith('.pdf') and ext == '.pdf':
                file_data = read_file(file)
                ext_sent(file_data, output_fname)
            else:
                print('\n\n Invalid File Type !')
    else:
        print("\n \n Given paths does not exists")


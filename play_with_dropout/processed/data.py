""" A neural chatbot using sequence to sequence model with
attentional decoder. 
"""
from __future__ import print_function

import os
import random
import re

import numpy as np
import csv
import config_tf as config


##############




stop_word = ['?', '.']
def clearing_word(word):
    word = re.sub('\x8e', 'é', word)
    word = re.sub('\x88', 'à', word)
    word = re.sub('\x9d', 'ù', word)
    word = re.sub('\x8f', 'è', word)
    word = re.sub('\x9e', 'û', word)
    word = re.sub('\x90', 'ê', word)
    word = re.sub('\x99', 'ô', word)
    word = re.sub('\x94', 'î', word)
    word = re.sub('\x8f', 'è', word)
    return word
    
def clearing(pharse):
    """
    Arg: Data is a list of questions or answers
    Return: clean questions et answers 
    """
    #clean_data = []
    pharses =[]
    for word in pharse.split(' '):
       # line= data[i].lower().split(' ')
        
        #for word in line:
        word = clearing_word(word)
        if word not in stop_word and len(word) !=0:
                    
            pharses.append(word)
        
    return ' '.join(pharses)





stop_word = ['?', '.']
def clearing_word(word):
    word = re.sub('\x8e', 'é', word)
    word = re.sub('\x88', 'à', word)
    word = re.sub('\x9d', 'ù', word)
    word = re.sub('\x8f', 'è', word)
    word = re.sub('\x9e', 'û', word)
    word = re.sub('\x90', 'ê', word)
    word = re.sub('\x99', 'ô', word)
    word = re.sub('\x94', 'î', word)
   # word = re.sub('\x8f', 'è', word)
    word = re.sub('\x8d', 'ç', word)
    word = re.sub('õ', '', word)
    word = re.sub('Ê', '', word)
    word = re.sub('[?,.,!, \,,  %]', '', word)
    if word == 'û' or word == 'v' or word == 'é':
        word = ''
    if word =="2017êles":
        word = "2017"
    if "ênox" in word:
        word ="nox"
    return word
    
def clearing(pharse):
    """
    Arg: Data is a list of questions or answers
    Return: clean questions et answers 
    """
    #clean_data = []
    pharses =[]
    for word in pharse.strip().lower().split(' '):
       # line= data[i].lower().split(' ')
        
        #for word in line:
        word = clearing_word(word)
        if word not in stop_word and len(word) !=0:       
            pharses.append(word)
        
    return ' '.join(pharses)



def get_all_convos():
    convos = []
    file1 = 'chatbot_tout_corpus.txt'
    file2 = 'convos_120_nettoye.txt'
    liste_file = [file1, file2]
    for file in liste_file:
        with open(file) as f:
            i=0
            for line in f:
                if i%2==0:
                    question = clearing(line)
                    if '++++' in question:
                         question = question[9:]
                else:
                    answer = clearing(line)
                    if '++++' in answer:
                        answer = answer[9:]
                    convos.append([question, answer])
                i+=1
        f.close()
    return convos

def train_test():
    convos = get_all_convos()
    trains, tests = [], []
    rang = len(convos)
    index_of_train = random.sample(range(rang), int(0.8*rang))
    for i in range(rang):
        if i in index_of_train:
            trains.append(convos[i])
        else:
            tests.append(convos[i])
    save_train_test_in_file(trains, False)
    save_train_test_in_file(tests, True)
    
def save_train_test_in_file(CONVOS, answer=True):
    if not answer:
        out_path1 = os.path.join(config.PROCESSED_PATH, 'train_question.txt')
        out_path2 = os.path.join(config.PROCESSED_PATH, 'train_answer.txt')
    else: 
        out_path1 = os.path.join(config.PROCESSED_PATH, 'test_question.txt')
        out_path2 = os.path.join(config.PROCESSED_PATH, 'test_answer.txt')
    question, answer = [], []
    for i in range(len(CONVOS)):
        question.append(CONVOS[i][0])
        answer.append(CONVOS[i][1])
    with open(out_path1, 'w') as f1:
        for i in range(len(question)):
            f1.write(question[i]+'\n')
    f1.close()
    with open(out_path2, 'w') as f2:
        for i in range(len(answer)):
            f2.write(answer[i]+'\n')
    f2.close()

def question_answers():
    """ Divide the dataset into two sets: questions and answers. """
    convos = get_all_convos()
    questions_train, answers_train = [], []
    questions_test, answers_test = [], []
    index_of_train = random.sample(range(len(convos)), int(0.8*len(convos)))
    for i in range(len(convos)):
        convo = convos[i]
        question = [];answer = []
        for word in convo[0].split():
            if word not in config.STOPWORDS and word not in "([.,!?\"'-<>:;)(])":
                question.append(word)
        for word in convo[-1].split():
            if word not in config.STOPWORDS and word not in "([.,!?\"'-<>:;)(])":
                answer.append(word)
        question = ' '.join(word for word in question)
        answer = ' '.join(word for word in answer)
        if i in index_of_train:
            questions_train.append(question)
            answers_train.append(answer)
        else:
            questions_test.append(question)
            answers_test.append(answer)
    return questions_train, answers_train, questions_test, answers_test


        
def prepare_dataset():
    make_dir(config.PROCESSED_PATH)
    path1= os.path.join(config.PROCESSED_PATH, 'question_train.txt')
    path2 = os.path.join(config.PROCESSED_PATH, 'answer_train.txt')
    try: 
        os.remove(path1);   os.remove(path2)
    except FileNotFoundError:
        pass
    with open(path1, 'w') as f:
        for i in range(len(questions_train)):
            f.write(questions_train[i]+'\n')
    f.close()
    with open(path2, 'w') as f:
        for i in range(len(answers_train)):
            f.write(answers_train[i]+'\n')
    f.close()
    path3= os.path.join(config.PROCESSED_PATH, 'question_test.txt')
    path4 = os.path.join(config.PROCESSED_PATH, 'answer_test.txt')
    try: 
        os.remove(path3);   os.remove(path4)
    except FileNotFoundError:
        pass
    with open(path3, 'w') as f:
        for i in range(len(questions_test)):
            f.write(questions_test[i]+'\n')
    f.close()
    with open(path4, 'w') as f:
        for i in range(len(answers_test)):
            f.write(answers_test[i]+'\n')
    f.close()

def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass


def basic_tokenizer(lines, normalize_digits=True):
    """ A basic tokenizer to tokenize text into tokens.
    Feel free to change this to suit your need. """
    words = []
    if type(lines) == str:
        for word in lines.strip().lower().split(' '):
                if not word:
                      continue
                words.append(word)
    else:
        for line in lines:
            for word in line.strip().lower().split(' '):
                    if not word:
                        continue
                    words.append(word)
    return words


def build_vocab():
         
    convos = get_all_convos()
    out_path_dec = os.path.join(config.PROCESSED_PATH, 'decoder_vocab.txt')
    out_path_enc = os.path.join(config.PROCESSED_PATH, 'encoder_vocab.txt')
    try: 
        os.remove(out_path_dec) and os.remove(out_path_enc)
    except FileNotFoundError:
        pass
    questions = []
    answers = []
    for i in range(len(convos)):
        convo = convos[i]
        for word in convo[0].strip().split():
            if word not in config.STOPWORDS and word not in "([.,!?\"'-<>:;)(])":
                if word not in questions:
                    questions.append(word)
        for word in convo[-1].split():
            if word not in config.STOPWORDS and word not in "([.,!?\"'-<>:;)(])":
                if word not in answers:
                    answers.append(word)
    with open(out_path_enc, 'w') as f:
            f.write('<pad>' + '\n')
            f.write('<unk>' + '\n')
            f.write('<s>' + '\n')
            f.write('<\s>' + '\n') 
            for word in questions:
                f.write(word + '\n')
    f.close()
    with open(out_path_dec, 'w') as f:
            f.write('<pad>' + '\n')
            f.write('<unk>' + '\n')
            f.write('<s>' + '\n')
            f.write('<\s>' + '\n') 
            for word in answers:
                f.write(word + '\n')
    f.close()
    enc, dec = len(questions), len(answers)
    with open('config.py', 'a') as cf:
                cf.write('DEC_VOCAB = '+str(dec)+'\n')
                cf.write('NUM_SAMPLES = '+str(dec-1)+'\n')
                cf.write('ENC_VOCAB = '+str(enc)+'\n')
    cf.close()

def load_vocab(vocab_path):
    with open(vocab_path, 'r') as f:
        words = f.read().splitlines()
    return words, {words[i]: i for i in range(len(words))}

def sentence2id(vocab, line):
    line = basic_tokenizer(line)
    words = []
    for word in line:
        if word in vocab:
            words.append(word)
    return [vocab.get(token) for token in words]

def token2id(data, answer):
    """ Convert all the tokens in the data into their corresponding
    index in the vocabulary. """
    if answer: 
         vocab_path = 'decoder_vocab.txt'
    else:
        vocab_path = 'encoder_vocab.txt'
    out_path = data[:-4]+'2id.txt'
    _, vocab = load_vocab(os.path.join(config.PROCESSED_PATH, vocab_path))
    in_file = open(os.path.join(config.PROCESSED_PATH, data), 'r')
    out_file = open(os.path.join(config.PROCESSED_PATH, out_path), 'w')
    
    lines = in_file.read().splitlines()
    for line in lines:
        if data == "answer_test.txt" or data == "answer_train.txt":
            ids = [vocab['<s>']]
        else:
            ids = []
        ids.extend(sentence2id(vocab, line))#vocab est un dict
        if data == "answer_test.txt" or data == "answer_train.txt":
            ids.append(vocab['<\s>'])
        out_file.write(' '.join(str(id_) for id_ in ids) + '\n')

def prepare_raw_data():
    print('Preparing raw data into train set ...')
    questions_train, answers_train, questions_test, answers_test = question_answers()
    prepare_dataset()
    #build_vocab(questions, False)
    #build_vocab(answers, True)

def process_data():
    print('Preparing data to be model-ready ...')
    token2id('question_train.txt', False)
    token2id('answer_train.txt', True)
    token2id('question_test.txt', False)
    token2id('answer_test.txt', True)
    print('Done!')

def load_data(enc_filename, dec_filename, max_training_size=None):
    """
    Arg: enc_filename is questions2id.txt file and dec_filename is answers2id.txt file
    """
    encode_file = open(os.path.join(config.PROCESSED_PATH, enc_filename), 'r')
    decode_file = open(os.path.join(config.PROCESSED_PATH, dec_filename), 'r')
    encode, decode = encode_file.readline(), decode_file.readline()
    data_buckets = [[] for _ in config.BUCKETS]
    i = 0
    while encode and decode:
        if (i + 1) % 100 == 0:
            print("Bucketing conversation number", i+1)
        encode_ids = [int(id_) for id_ in encode.split()]
        decode_ids = [int(id_) for id_ in decode.split()]
        for bucket_id, (encode_max_size, decode_max_size) in enumerate(config.BUCKETS):
            if len(encode_ids) <= encode_max_size and len(decode_ids) <= decode_max_size:
                data_buckets[bucket_id].append([encode_ids, decode_ids])
                break
        encode, decode = encode_file.readline(), decode_file.readline()
        i += 1
    return data_buckets
#BIEN: Make input_ to have size=size by add some zero element to the end of input_
def _pad_input(input_, size):
    return input_ + [config.PAD_ID] * (size - len(input_))
#BIEN: Following function will change the shape of inputs, example,
#from 3x5 to 5x3, but not by an operation like matrix transpose 
# it takes all first element to make in an array, and second, and third...
def _reshape_batch(inputs, size, batch_size):
    """ Create batch-major inputs. Batch inputs are just re-indexed inputs
    """
    batch_inputs = []
    for length_id in range(size):
        batch_inputs.append(np.array([inputs[batch_id][length_id]
                                    for batch_id in range(batch_size)], dtype=np.int32))
    return batch_inputs


def get_batch(data_buckets, bucket_id, batch_size=1):
    """ Return one batch to feed into the model """
    # only pad to the max length of the bucket
    encoder_size, decoder_size = config.BUCKETS[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    for _ in range(batch_size):
        #BIEN: Choose randomly a couple of question-answer in the data_buckets
        encoder_input, decoder_input = random.choice(data_buckets)
        # pad both encoder and decoder, reverse the encoder
        # So, encoder start by 0, 0, ...
        encoder_inputs.append(list(reversed(_pad_input(encoder_input, encoder_size))))
        decoder_inputs.append(_pad_input(decoder_input, decoder_size))

    # now we create batch-major vectors from the data selected above.
    batch_encoder_inputs = _reshape_batch(encoder_inputs, encoder_size, batch_size)
    batch_decoder_inputs = _reshape_batch(decoder_inputs, decoder_size, batch_size)

    # create decoder_masks to be 0 for decoders that are padding.
    batch_masks = []
    for length_id in range(decoder_size):
        batch_mask = np.ones(batch_size, dtype=np.float32)
        for batch_id in range(batch_size):
            # we set mask to 0 if the corresponding target is a PAD symbol.
            # the corresponding decoder is decoder_input shifted by 1 forward.
            if length_id < decoder_size - 1:
                target = decoder_inputs[batch_id][length_id + 1]
            if length_id == decoder_size - 1 or target == config.PAD_ID:
                batch_mask[batch_id] = 0.0
        batch_masks.append(batch_mask)
    return batch_encoder_inputs, batch_decoder_inputs, batch_masks

if __name__ == '__main__':
    questions_train, answers_train, questions_test, answers_test = question_answers()
    prepare_raw_data()
    build_vocab()
    process_data()
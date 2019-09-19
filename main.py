import os
import numpy as np
import ast
import csv
from tqdm import tqdm
from mtranslate import translate


# Read training data
file_path = os.path.dirname(os.path.abspath(__file__))
data_path = file_path + '/data/description_collected.csv'
index_description = 1
low_fre_thres = 0

char_chklst = ['<br>', '\n', '<b>', '</b>', '<br />', '<B>', '<\/B>', '[/Liste]', '[/liste]', '<\/b>', '<BR>', '<td>',
               '<tr>', '</td>', '</tr>', '<p>', '</p>', '<strong>', '</strong>', '<ul>', '</ul>', '<li>', '</li>',
               '<table>', '<tbody>', '<h1>', '</h1>', '<h2>', '</h2>', '<h3>', '</h3>', '<h4>', '</h4>', '<label>',
               '</label>', '<div>', '</div>', '<u>', '</u>', '</span>', '</table>', '</iframe>', '<hr />', '<sub>',
               '</sub>', '||', '[Liste]', '[liste]', '/liste]', '/Liste]', '[/Liste', ]

char_chkpair = [['[Liste titel="', '"]'], ['[liste titel="', '"]'], ['[liste Titel="', '"]'], ['[Liste Titel="', '"]'],
                ['[Liste titel=', '"]'], ['Liste titel="', '"]'], ['<p class="', '">'], ['<span style="', '">'],
                ['<br style="', '" />'], ['<ul class="', '">'], ['<h2 class="', '">'], ['<h3 class="', '">'],
                ['<div id="', '">'], ['<iframe class="', '">'], ['<td class="', '">'], ['<img class="', '" />'],
                ['<p style="', '">'], ['<h4 class="', '">'], ['<div class="', '">'], ['<li id="', '">'],
                ['<span class="', '">'], [' <div style="', '">'], ['<table border', '>'], ['<td colspan', '>']]

char_rep = ['&nbsp;', '', '', '', '', '', '', '', '', '', '', '']

with open('data/vocab_types_de_non_verb.json') as f:
    vocab = f.readlines()
    vocab = ast.literal_eval(vocab[0])


def remove_sub_sentence(input_word):
    num_word = len(input_word.split())

    # Replace htlm charac
    for check_id in range(len(char_rep)):
        input_word = input_word.replace(char_rep[check_id], ' ')

    # Remove notation
    for check_id in range(len(char_chklst)):
        input_word = input_word.replace(char_chklst[check_id], ' . ')

    # Remove sub-sentence
    for check_id in range(np.shape(char_chkpair)[0]):
        for word in range(num_word):
            if (char_chkpair[check_id][0] in input_word) and (char_chkpair[check_id][1] in input_word):
                start_idx = input_word.find(char_chkpair[check_id][0])
                end_idx = start_idx + input_word[start_idx:].find(char_chkpair[check_id][1])
                if end_idx > start_idx:
                    input_word = input_word[:start_idx] + ' . ' + input_word[end_idx + len(char_chkpair[check_id][1]):]

    return input_word


def extract_keywords(editor_note):
    keyword_pairs = []
    for i, c in enumerate(editor_note.split()):
        if c.lower() in vocab.keys():
            keyword_pairs.append((vocab[c.lower()], c.lower()))

    return keyword_pairs


# Read resource for training and testing
des_count = 0
res = []
with open('./data/description_collected.csv', 'r') as df:
    reader = csv.reader(df)
    for i, line in enumerate(reader):
        if i != 0:  # ignore header row
            des = line[index_description]
            if des != '' and des != 'NULL':
                des = remove_sub_sentence(des)

                des_count += 1
                des_ls = []
                for wid in range(len(des.split())):
                    word = des.split()[wid]
                    if word != '.':
                        des_ls.append(word)
                    else:
                        if wid + 1 < len(des.split()):
                            if des.split()[wid+1] != '.':
                                des_ls.append(word)
                if len(des_ls) > 10:
                    if des_ls[-1] != '.':
                        des_ls.append('.')
                    if des_ls[0] == '.':
                        del des_ls[0]
                    des = str(' '.join(des_ls))
                    print(des)
                    res.append(des)

                    # if des_count == 1000:
                    #     break

# check_duplicate
res_dup = []
count_dup = 0
for des in res:
    if des not in res_dup:
        res_dup.append(des)
    else:
        count_dup += 1
res = res_dup
print('Remove %d duplicated', count_dup)

# Translate to english description using goslate
print('Start to translate document')
res_trans = []
count_trans_error = 0
for des_id in tqdm(range(len(res))):
    try:
        des_trans = translate(res[des_id])
        res_trans.append(des_trans)
    except:
        print('Translate error')
        count_trans_error += 1
print('Number of translate error: ', count_trans_error)
res = res_trans

# with open('./data/description_collected_english.csv', 'w+') as writeFile:
#     writer = csv.writer(writeFile)
#     for item in res:
#         writeFile.write(str(''.join(item)) + '\n')
#     writeFile.close()
#
# with open('./data/description_collected_english.txt', 'w+') as writeFile:
#     for r in res:
#         writeFile.write(str(r) + '\n')



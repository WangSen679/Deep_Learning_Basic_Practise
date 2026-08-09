# file: prepro.py
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import random
import itertools
import codecs
from collections import Counter
from hyperparams import Hyperparams as hp

def build_dataset():
    animals = {"duck": "鸭子", "cat": "猫", "dog": "狗", "cow": "牛", "bird": "鸟", "horse": "马", "bear": "熊", "lion": "狮子"}
    colors = {"quiet": "安静", "red": "红色", "black": "黑色", "white": "白色", "blue": "蓝色", "green": "绿色", "yellow": "黄色", "brown": "棕色"}
    locations = {"next to": "旁边", "behind": "后面", "in front of": "前面", "on": "上面", "under": "下面", "near": "附近"}
    objects = {"door": "门", "tree": "树", "table": "桌子", "chair": "椅子", "box": "盒子", "window": "窗户", "car": "汽车", "apple": "苹果"}
    names = {"alice": "爱丽丝", "bob": "鲍勃", "charlie": "查理", "david": "大卫", "emma": "艾玛", "fiona": "菲奥娜", "george": "乔治", "henry": "亨利"}
    numbers = {1: ("a", "one", "一"), 2: ("two", "two", "两"), 3: ("three", "three", "三"), 4: ("four", "four", "四"), 5: ("five", "five", "五"), 6: ("six", "six", "六")}

    pattern1_all = []
    for num, color, animal, loc, obj in itertools.product(numbers.keys(), colors.keys(), animals.keys(), locations.keys(), objects.keys()):
        en_num = numbers[num][0]
        zh_num = numbers[num][2]
        if num == 1:
            en = f"there is {en_num} {color} {animal} {loc} the {obj}"
        else:
            en = f"there are {en_num} {color} {animal}s {loc} the {obj}"
        zh = f"{objects[obj]} {locations[loc]} 有 {zh_num} 只 {colors[color]} {animals[animal]}"
        pattern1_all.append((en, zh))

    pattern2_all = []
    for name, num, color, obj in itertools.product(names.keys(), numbers.keys(), colors.keys(), objects.keys()):
        en_num = numbers[num][1]
        zh_num = numbers[num][2]
        if num == 1:
            en = f"{name} has {en_num} {color} {obj}"
        else:
            en = f"{name} has {en_num} {color} {obj}s"
        zh = f"{names[name]} 有 {zh_num} 个 {colors[color]} {objects[obj]}"
        pattern2_all.append((en, zh))

    random.seed(42)
    random.shuffle(pattern1_all)
    random.shuffle(pattern2_all)
    selected = pattern1_all[:1920] + pattern2_all[:1920]
    random.shuffle(selected)
    return selected

def make_vocab(sents, fname):
    '''Constructs vocabulary. Writes vocabulary line by line to `preprocessed/fname`'''
    words = []
    for sent in sents:
        words.extend(sent.split())
    word2cnt = Counter(words)
    
    if not os.path.exists('preprocessed'): 
        os.mkdir('preprocessed')
    with codecs.open('preprocessed/{}'.format(fname), 'w', 'utf-8') as fout:
        fout.write("{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format("<PAD>", "<UNK>", "<s>", "</s>"))
        for word, cnt in word2cnt.most_common():
            fout.write("{}\t{}\n".format(word, cnt))

if __name__ == '__main__':
    raw_data = build_dataset()
    total_size = len(raw_data) # 3840
    train_size = int(total_size * 0.7) # 2688
    val_size = int(total_size * 0.2)   # 768
    test_size = total_size - train_size - val_size # 384

    train_data = raw_data[:train_size]
    val_data = raw_data[train_size:train_size + val_size]
    test_data = raw_data[train_size + val_size:]

    if not os.path.exists('preprocessed'):
        os.mkdir('preprocessed')

    def save_split(data_list, prefix):
        with codecs.open(f'preprocessed/{prefix}.en', 'w', 'utf-8') as f_en, \
             codecs.open(f'preprocessed/{prefix}.zh', 'w', 'utf-8') as f_zh:
            for en, zh in data_list:
                f_en.write(en + '\n')
                f_zh.write(zh + '\n')

    save_split(train_data, 'train')
    save_split(val_data, 'val')
    save_split(test_data, 'test')

    train_en_sents = [pair[0] for pair in train_data]
    train_zh_sents = [pair[1] for pair in train_data]

    make_vocab(train_en_sents, "en.vocab.tsv")
    make_vocab(train_zh_sents, "zh.vocab.tsv")

    print(f"Preprocess Done! Data Split -> Train: {len(train_data)} (70%), Val: {len(val_data)} (20%), Test: {len(test_data)} (10%)")
    print("Vocabularies and datasets saved in preprocessed/ directory.")

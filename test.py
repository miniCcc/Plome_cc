import torch
from utils import read_train_ds, collate_fn, MyData#, convert_example
from utils import get_zi_py_matrix, convert_single_example
import tokenization
from pinyin_tool import PinyinTool
import opencc
from pypinyin import pinyin, lazy_pinyin, Style


def getAllHanzi():

    err_list = []

    with open("./pinyin_data/zi_py.txt", 'r', encoding='UTF-8') as f:
        index = 1
        with open("./datas/hanzi.txt", 'w', encoding='UTF-8') as f2:
            for line in f:
                res = ""
                res_y = ""
                if index > 2:
                    line = line.strip('\n')
                    els = line.split("\t")
                    res, res_y = opt(els[0])
                    # print(type(res), type(res_y))
                    str = els[0] + "\t" + els[1] + "\t" + res + "\t" + res_y + "\n"
                    print(str)
                    f2.writelines(str)
                    if els[1] != res + res_y:
                        err_list.append(str)
                index += 1
    return err_list




if __name__ == '__main__':
    # train_list = list(read_train_ds('datas/train.txt'))  # list[dict: key1 = source, key2 = target]
    # tokenizer = tokenization.FullTokenizer(vocab_file="datas/pretrained_plome/vocab.txt", do_lower_case=False)
    # pytool = PinyinTool(py_dict_path='pinyin_data/zi_py.txt', py_vocab_path='pinyin_data/py_vocab.txt', py_or_sk='py')
    # sktool = PinyinTool(py_dict_path='stroke_data/zi_sk.txt', py_vocab_path='stroke_data/sk_vocab.txt', py_or_sk='sk')
    #
    # train_ids = [convert_single_example(example, 180, tokenizer, pytool, sktool) for example in train_list]

    err_list = getAllHanzi()
    # 把错误的写入文件
    with open("./datas/err_list.txt", "w", encoding='UTF-8') as f:
        for i, val in enumerate(err_list):
            f.writelines(val)
    print(len(err_list))


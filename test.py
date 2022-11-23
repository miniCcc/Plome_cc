import torch
from utils import read_train_ds, collate_fn, MyData#, convert_example
from utils import get_zi_py_matrix, convert_single_example
import tokenization
from pinyin_tool import PinyinTool



if __name__ == '__main__':
    train_list = list(read_train_ds('datas/train.txt'))  # list[dict: key1 = source, key2 = target]
    tokenizer = tokenization.FullTokenizer(vocab_file="datas/pretrained_plome/vocab.txt", do_lower_case=False)
    pytool = PinyinTool(py_dict_path='pinyin_data/zi_py.txt', py_vocab_path='pinyin_data/py_vocab.txt', py_or_sk='py')
    sktool = PinyinTool(py_dict_path='stroke_data/zi_sk.txt', py_vocab_path='stroke_data/sk_vocab.txt', py_or_sk='sk')

    train_ids = [convert_single_example(example, 180, tokenizer, pytool, sktool) for example in train_list]


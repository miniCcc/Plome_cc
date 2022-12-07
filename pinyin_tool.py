#encoding:utf8
import sys
import numpy as np
import re

from utils import get_zi_py_matrix, convert_single_example
import tokenization

# py 字典重新定义
py_dict = {}


# sk 字典重新定义

# ----------------- 自己写U的关于声母韵母匹配工具 -----------------
Shengmu2ID = {'b':1, 'p':2, 'm':3, 'f':4, 'd':5, 't':6, 'n':7, 'l':8, 'g':10, 'k':11, 'h':12, 'j':13, 'q':14, 'x': 15, 'zh': 16,
              'ch':17, 'sh':17, 'r':19, 'z':16, 'c':17, 's':18, 'y':19, 'w':20}
Yunmu2ID = {'a':1, 'o':2, 'e':3, 'i':4, 'u':5, 'v':6, 'ai':7, 'ei':8, 'ui':9, 'ao':10, 'ou':11, 'iu':12, 'ie':13, 've':14, 'er':15,
            'an':16, 'ang':16, 'en':17, 'eng':17, 'in':18, 'ing':18, 'un': 19, 'vn':20, 'ong':21}

# 做yinpin的dict映射
def split_pystr():
    with open('pinyin_data/py_split.txt', 'w', encoding='utf-8') as f:
      for line in open('pinyin_data/py_vocab2.txt', encoding='utf-8'):
            line = line.strip()
            ans = []
            els = re.split(r"[ ]+", line)
            print(els)
            if len(els) > 1:
                if els[1] == '[N]':
                    ans.append(0)
                else:
                    ans.append(Shengmu2ID[els[1]])
                if len(els) >= 3:
                    ans.append(Yunmu2ID[els[2]])
                else:
                    ans.append(0)
                if len(els) >= 4:
                    ans.append(Yunmu2ID[els[3]])
                else:
                    ans.append(0)
            str2 = ",".join(str(i) for i in ans)
            str2 += '\n'
            f.writelines(els[0] + '    ' + str2)


class PinyinTool:
    def __init__(self, py_dict_path, py_vocab_path, py_split_path, py_or_sk='py'):
        self.zi_pinyin = self._load_pydict(py_dict_path)
        self.vocab = self._load_pyvocab(py_vocab_path) # key = py or sk, value = index(py/sk _vocab.txt)
        self.vocab_py_split_dict = self._load_pySplitDict(py_split_path)
        if 'py' in py_or_sk:
            self.ZM2ID = {':':1, 'a':2, 'c':3, 'b':4, 'e':5, 'd':6, 'g':7, 'f':8, 'i':9, 'h':10, 'k':11, 'j':12, 'm':13, 'l':14, 'o':15, 'n':16, 'q':17, 'p':18, 's':19, 'r':20, 'u':21, 't':22, 'w':23, 'v':24, 'y':25, 'x':26, 'z':27}
            self.PYLEN = 3
        else:
            self.ZM2ID = {'1': 1, '2':2, '3':3, '4':4, '5':5}
            self.PYLEN = 10

    def _load_pydict(self, fpath):
        ans = {}
        for line in open(fpath, encoding='utf-8'):
            line = line.strip()#.decode('utf8')
            tmps = line.split('\t')
            if len(tmps) != 2: continue
            ans[tmps[0]] = tmps[1]
        return ans

    def _load_pySplitDict(self, fpath):
        def convert(nums):
            res = []
            for _, value in enumerate(nums):
                res.append(int(value))
            return res

        ans = {}
        for line in open(fpath, encoding='utf-8'):
            line = line.strip()
            els = re.split(r"[ ]+", line)
            if len(els) > 1:
                ans[els[0]] = convert(els[1].split(","))
        return ans


    def _load_pyvocab(self, fpath):
        ans = {'PAD': 0, 'UNK': 1}
        idx = 2
        for line in open(fpath, encoding='utf-8'):
            line = line.strip()#.decode('utf8')
            if len(line) < 1: continue
            ans[line] = idx
            idx += 1
        return ans

    """
        Function: 获取这个汉字拼音在 vocab 的 index
    """
    def get_pinyin_id(self, zi_unicode):
        py = self.zi_pinyin.get(zi_unicode, None)
        if py is None:
            return self.vocab['UNK']
        return self.vocab.get(py, self.vocab['UNK'])

    """
        ans = [
            [0,0,0,0]   每一行都是一个字的映射
            [0,0,0,0]
            [....]
        ]
        不是所有汉字的，而是所有拼音的，也就是 py_vocab 对应的
    """
    def get_pyid2seq_matrix(self):
        ans = [[0] * self.PYLEN, [0] * self.PYLEN] #PAD, UNK
        rpyvocab = {v: k  for k, v in self.vocab.items()}  # key = index  value = py / sk
        for k in range(2, len(rpyvocab), 1):
            pystr = rpyvocab[k]
            seq = []
            seq.append(self.vocab_py_split_dict[pystr])
            # for c in pystr:
                # seq.append(self.ZM2ID[c])
            seq = [0] * self.PYLEN + seq
            seq = seq[-self.PYLEN:]
            ans.append(seq)
        return np.asarray(ans, dtype=np.int32)

if __name__ == '__main__':
    pytool = PinyinTool(py_dict_path='pinyin_data/zi_py.txt', py_vocab_path='pinyin_data/py_vocab.txt',
                        py_split_path='pinyin_data/py_split.txt',
                        py_or_sk='py')
    # pytool.get_pyid2seq_matrix()
    print(111)
    # split_pystr()
    # tokenizer = tokenization.FullTokenizer(vocab_file="datas/pretrained_plome/vocab.txt", do_lower_case=False)
    # zi_py_matrix = get_zi_py_matrix(pytool, tokenizer)


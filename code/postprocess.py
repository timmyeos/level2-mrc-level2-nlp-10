import pickle

filePath = "./stop_list.txt"
with open(filePath, "rb") as lf:
    stoplist = pickle.load(lf)

from konlpy.tag import Mecab

mecab = Mecab()


def postprocess(x):
    word_tokens = mecab.morphs(x)
    result = [word for word in word_tokens if not word in stoplist]
    return " ".join(result)

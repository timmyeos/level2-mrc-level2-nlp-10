# import emoji
from soynlp.normalizer import repeat_normalize
import re
import glob
from nltk.tokenize import word_tokenize
import unicodedata
import json
from soynlp.word import WordExtractor
import pickle
from tqdm.auto import tqdm

stopword_dir = "./stopword/*"
stopword_list = glob.glob(stopword_dir)

stopword_set = set()
for file in stopword_list:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            if file == "./stopword/한국어불용어100.txt":
                data = line.split()
                stopword_set.add(data[0].replace("\n", ""))
            elif file == "./stopword/1.txt":
                for data in line:
                    if data not in [" ", '"', ",", "[", "]"]:
                        stopword_set.add(data)
            else:
                stopword_set.add(line.replace("\n", ""))

# emojis = "".join(emoji.UNICODE_EMOJI.keys())
pattern = re.compile(f"[^ .,?!/@$%~％·∼()\x00-\x7Fa-zA-Z0-9가-힇ㄱ-ㅎㅏ-ㅣぁ-ゔァ-ヴー々〆〤一-龥]")
url_pattern = re.compile(
    r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
)


def clean(x):
    x = pattern.sub(" ", x)
    x = url_pattern.sub("", x)
    x = x.strip()
    x = repeat_normalize(x, num_repeats=2)
    return x


def stopword(text):
    tmp_txt = []
    tokens = word_tokenize(text)
    for word in tokens:
        if word not in stopword_set:
            tmp_txt.append(word)
    tmp_str = " ".join(tmp_txt)
    return tmp_str


def preprocess(text):
    x = unicodedata.normalize("NFKC", clean(text))
    return x


# files = "../data/wikipedia_documents.json"
# with open(files, "r", encoding="utf-8") as f:
#     wiki = json.load(f)
# contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))

# for idx in tqdm(range(len(contexts))):
#     contexts[idx] = preprocess(contexts[idx])


# word_extractor = WordExtractor()
# word_extractor.train(contexts)
# word_score_table = word_extractor.extract()


# with open("soynlp_dict.pkl", "wb") as f:
#     pickle.dump(word_score_table, f)

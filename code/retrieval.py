import json
import os
import pickle
import time
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union
import torch
import faiss
import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm
from preprocess import preprocess
from transformers import AutoTokenizer
from dpr import dpr, p_emd, q_emd
from BM25 import BM25Okapi


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class SparseRetrieval:
    def __init__(
        self,
        tokenize_fn,
        datasets,
        data_args,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> NoReturn:

        """
        Arguments:
            tokenize_fn:
                기본 text를 tokenize해주는 함수입니다.
                아래와 같은 함수들을 사용할 수 있습니다.
                - lambda x: x.split(' ')
                - Huggingface Tokenizer
                - konlpy.tag의 Mecab

            data_path:
                데이터가 보관되어 있는 경로입니다.

            context_path:
                Passage들이 묶여있는 파일명입니다.

            data_path/context_path가 존재해야합니다.

        Summary:
            Passage 파일을 불러오고 TfidfVectorizer를 선언하는 기능을 합니다.
        """
        self.data_args = data_args
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(contexts)}")
        self.ids = list(range(len(contexts)))
        self.tokenize_fn = tokenize_fn

        self.p_embedding_dpr = None
        self.p_embedding_sparse = None
        self.p_embedding_bm = None
        # get_sparse_embedding()로 생성합니다
        # dpr, tf, bm 중 선택해서 생성합니다.

        self.ngram = 2  # bm25 용 ngram parameter
        self.indexer = None  # build_faiss()로 생성합니다.
        self.datasets = datasets
        self.datasets_train = self.datasets["train"]
        self.datasets_valid = self.datasets["validation"]
        if self.data_args.use_preprocess is False:
            self.contexts = contexts
        if self.data_args.use_preprocess:
            if self.data_args.use_parasplit:
                para_num_limit = 5  # 문단 별 최소 문장 개수
                para_len_limit = 15  # 문단 별 최소 길이
                contexts_2 = []
                tmp_para = ""

                for text in tqdm(contexts):
                    tmp = text.split("\n")
                    tmp_para = ""
                    cnt = 0
                    for data in tmp:
                        if data != "":
                            cnt += 1
                            if preprocess(data) + " " != " ":
                                tmp_para += preprocess(data) + " "
                            if cnt % para_num_limit == 0:
                                if len(tmp_para) > para_len_limit:
                                    contexts_2.append(tmp_para)
                                    tmp_para = ""
                    if tmp_para != "" and len(tmp_para) > para_len_limit:
                        contexts_2.append(tmp_para)
                self.contexts = contexts_2
                for data in self.contexts:
                    if data == "":
                        print("공백이들어감.")
                print("문장분리 후 위키데이터 길이: ", len(self.contexts))
            else:
                self.contexts = datasets
                for idx in tqdm(range(len(self.contexts))):
                    self.contexts[idx] = preprocess(self.contexts[idx])
                print("전처리 후 위키데이터길이: ", len(self.contexts))
            for idx in tqdm(range(len(self.datasets_train))):
                self.datasets_train[idx]["context"] = preprocess(
                    self.datasets_train[idx]["context"]
                )
            for idx in tqdm(range(len(self.datasets_valid))):
                self.datasets_valid[idx]["context"] = preprocess(
                    self.datasets_valid[idx]["context"]
                )

            print("훈련/검증/위키데이터 전처리 완료")

    def get_sparse_embedding(self) -> NoReturn:

        """
        Summary:
            DPR, Sparse+TFIDF, BM25 Embedding을 만들고
            TFIDF와 각 Embedding을 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """

        # Pickle을 저장합니다.
        dpr_name = f"dense_embedding_inbatch(bert 40epoch).bin"
        sparse_name = f"sparse.bin"
        tfidfv_name = f"tfidv.bin"
        bm25_name = f"bm25.bin"

        dpr_path = os.path.join(self.data_path, dpr_name)
        sparse_path = os.path.join(self.data_path, sparse_name)
        tfidfv_path = os.path.join(self.data_path, tfidfv_name)
        bm25_path = os.path.join(self.data_path, bm25_name)

        # dpr pickle load
        # if os.path.isfile(dpr_path):
        #     with open(dpr_path, "rb") as file:
        #         self.p_embedding_dpr = pickle.load(file)
        #         self.q_encoder = torch.load("./pretrained_dpr/q_encoder_inbatch39.pt")
        #     print("dpr_Embedding pickle load.")

        # # sparse+tfidfv pickle load
        # if os.path.isfile(tfidfv_path) and os.path.isfile(sparse_path):
        #     with open(sparse_path, "rb") as file:
        #         self.p_embedding_sparse = pickle.load(file)
        #     with open(tfidfv_path, "rb") as file:
        #         self.tfidfv = pickle.load(file)
        #     print("sparse_Embedding & tfidfv pickle load.")

        # bm25 pickle load
        if os.path.isfile(bm25_path):
            with open(bm25_path, "rb") as file:
                self.p_embedding_bm = pickle.load(file)
            print("bm25_Embedding pickle load.")

        # # dpr pickle save
        # if os.path.isfile(dpr_path) is False:
        #     print("Build DPR")
        #     self.p_encoder, self.q_encoder = dpr(self.datasets_train, self.contexts)
        #     self.p_embedding_dpr = p_emd(self.p_encoder, self.contexts)
        #     with open(dpr_path, "wb") as file:
        #         pickle.dump(self.p_embedding_dpr, file)
        #     print("DPR Embedding pickle saved.")

        # # sparse+tfidfv pickle save
        # if os.path.isfile(sparse_path) is False or os.path.isfile(tfidfv_path) is False:
        #     print("Build Sparse + tfidf")
        #     # Transform by vectorizer
        #     self.tfidfv = TfidfVectorizer(
        #         tokenizer=self.tokenize_fn,
        #         ngram_range=(1, 2),
        #         max_features=50000,
        #     )
        #     self.p_embedding_sparse = self.tfidfv.fit_transform(self.contexts)

        #     with open(sparse_path, "wb") as file:
        #         pickle.dump(self.p_embedding_sparse, file)
        #     print("Sparse Embedding pickle saved.")

        #     with open(tfidfv_path, "wb") as file:
        #         pickle.dump(self.tfidfv, file)
        #     print("tfidfv pickle saved.")

        # bm25 pickle save
        if os.path.isfile(bm25_path) is False:
            print("Build passage embedding")
            tokenized_contexts = []
            for text in tqdm(self.contexts):
                tmp = []
                # print(self.tokenize_fn(text))
                tmp.extend(self.tokenize_fn(text))
                assert self.ngram <= len(
                    tmp
                ), f"tokenized 된 길이가 ngram({self.ngram}) 보다 작습니다."
                tmp.extend(
                    [
                        "".join(tmp[i : i + self.ngram])
                        for i in range(len(tmp) - self.ngram + 1)
                    ]
                )
                tokenized_contexts.append(tmp)

            self.p_embedding_bm = BM25Okapi(tokenized_contexts)
            with open(bm25_path, "wb") as file:
                pickle.dump(self.p_embedding_bm, file)
            print("BM25 Embedding pickle saved.")

    def build_faiss(self, num_clusters=64) -> NoReturn:
        # 현재 사용 X
        """
        Summary:
            속성으로 저장되어 있는 Passage Embedding을
            Faiss indexer에 fitting 시켜놓습니다.
            이렇게 저장된 indexer는 `get_relevant_doc`에서 유사도를 계산하는데 사용됩니다.

        Note:
            Faiss는 Build하는데 시간이 오래 걸리기 때문에,
            매번 새롭게 build하는 것은 비효율적입니다.
            그렇기 때문에 build된 index 파일을 저정하고 다음에 사용할 때 불러옵니다.
            다만 이 index 파일은 용량이 1.4Gb+ 이기 때문에 여러 num_clusters로 시험해보고
            제일 적절한 것을 제외하고 모두 삭제하는 것을 권장합니다.
        """

        indexer_name = f"faiss_clusters{num_clusters}.index"
        indexer_path = os.path.join(self.data_path, indexer_name)
        if os.path.isfile(indexer_path):
            print("Load Saved Faiss Indexer.")
            self.indexer = faiss.read_index(indexer_path)

        else:
            # p_emb = self.p_embedding.astype(np.float32).toarray()
            p_emb = self.p_embedding.cpu().detach().numpy()
            emb_dim = p_emb.shape[-1]

            num_clusters = num_clusters
            quantizer = faiss.IndexFlatL2(emb_dim)

            self.indexer = faiss.IndexIVFScalarQuantizer(
                quantizer, quantizer.d, num_clusters, faiss.METRIC_L2
            )
            self.indexer.train(p_emb)
            self.indexer.add(p_emb)
            faiss.write_index(self.indexer, indexer_path)
            print("Faiss Indexer Saved.")

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
        """

        # assert (
        #     self.p_embedding_sparse is not None
        # ), "get_sparse_embedding() 메소드를 먼저 수행해줘야합니다."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context_id": doc_indices[idx],
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        with timer("transform"):
            query_vec = self.tfidfv.transform([query])
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        with timer("query ex search"):
            result = query_vec * self.p_embedding_sparse.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()

        sorted_result = np.argsort(result.squeeze())[::-1]
        doc_score = result.squeeze()[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]
        return doc_score, doc_indices

    def _softmax(self, x):
        max = np.max(
            x, axis=1, keepdims=True
        )  # returns max of each row and keeps same dims
        e_x = np.exp(x - max)  # subtracts each row with its max value
        sum = np.sum(
            e_x, axis=1, keepdims=True
        )  # returns sum of each row and keeps same dims
        f_x = e_x / sum
        return f_x

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """
        # TFIDF로 점수 측정
        # query_vec = self.tfidfv.transform(queries)
        # assert (
        #     np.sum(query_vec) != 0
        # ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        # DPR로 점수 측정
        # query_vecs_dpr = q_emd(self.q_encoder, queries)
        # print(query_vecs_dpr.size())
        # print(self.p_embedding_dpr.size())
        # result = query_vecs_dpr.matmul(self.p_embedding_dpr.T).numpy()
        # if not isinstance(result, np.ndarray):
        #     result = result.toarray()
        # doc_scores_dpr = []
        # doc_indices_dpr = []
        # print(result.shape[0])
        # print(result.shape[1])
        # for i in range(result.shape[0]):
        #     sorted_result = np.argsort(result[i, :])[::-1]
        #     doc_scores_dpr.append(result[i, :][sorted_result].tolist()[:k])
        #     doc_indices_dpr.append(sorted_result.tolist()[:k])

        # BM으로 점수 측정
        doc_scores_bm = []
        doc_indices_bm = []
        for query in tqdm(queries, desc=f"Top-k({k}) retrieval: "):
            tmp = []
            tokenized_query = self.tokenize_fn(query)
            tmp.extend(tokenized_query)
            assert self.ngram <= len(
                tmp
            ), f"tokenized 된 길이가 ngram({self.ngram}) 보다 작습니다."
            tmp.extend(
                [
                    "".join(tmp[i : i + self.ngram])
                    for i in range(len(tmp) - self.ngram + 1)
                ]
            )

            scores, indices = self.p_embedding_bm.get_top_n(tmp, self.contexts, n=k)
            doc_scores_bm.append(scores)
            doc_indices_bm.append(indices)

        # # 점수 가중치
        # weight_dpr = self.data_args.weight_dpr
        # weight_bm = 1 - weight_dpr

        # # 점수 정규화
        # for idx, item in enumerate(doc_scores_dpr):
        #     doc_scores_dpr[idx] = doc_scores_dpr[idx] / np.sum(doc_scores_dpr[idx])

        # for idx, item in enumerate(doc_scores_bm):
        #     doc_scores_bm[idx] = doc_scores_bm[idx] / np.sum(doc_scores_bm[idx])
        # doc_scores_dpr = self._softmax(doc_scores_dpr)
        # doc_scores_bm = self._softmax(doc_scores_bm)

        # 보간된 점수 및 인덱스
        # doc_scores_inter = np.zeros((len(doc_scores_dpr), k))
        # doc_indices_inter = np.zeros((len(doc_scores_dpr), k), dtype=np.int32)
        # for i in range(len(doc_scores_dpr)):
        #     tmp_dict = dict()
        #     for idx, score in zip(doc_indices_dpr[i], doc_scores_dpr[i]):
        #         tmp_dict[idx] = weight_dpr * score
        #     for idx, score in zip(doc_indices_bm[i], doc_scores_bm[i]):
        #         if idx in tmp_dict.keys():
        #             tmp_dict[idx] += weight_bm * score
        #         else:
        #             tmp_dict[idx] = weight_bm * score
        #     sorted_data = sorted(
        #         [(score, idx) for idx, score in tmp_dict.items()], reverse=True
        #     )[:k]
        #     for j in range(k):
        #         doc_scores_inter[i][j] = sorted_data[j][0]
        #         doc_indices_inter[i][j] = sorted_data[j][1]
        # doc_scores = torch.from_numpy(doc_scores_inter)
        # doc_indices = torch.from_numpy(doc_indices_inter)
        return doc_scores_bm, doc_indices_bm

    def retrieve_faiss(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
            retrieve와 같은 기능을 하지만 faiss.indexer를 사용합니다.
        """

        assert self.indexer is not None, "build_faiss()를 먼저 수행해주세요."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc_faiss(
                query_or_dataset, k=topk
            )
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            queries = query_or_dataset["question"]
            total = []

            with timer("query faiss search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk_faiss(
                    queries, k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context_id": doc_indices[idx],
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            return pd.DataFrame(total)

    def get_relevant_doc_faiss(
        self, query: str, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        query_vec = self.tfidfv.transform([query])
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        q_emb = query_vec.toarray().astype(np.float32)
        with timer("query faiss search"):
            D, I = self.indexer.search(q_emb, k)

        return D.tolist()[0], I.tolist()[0]

    def get_relevant_doc_bulk_faiss(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        print("bulk_faiss 시작")
        if self.data_args.use_faiss:
            query_vecs = q_emd(self.q_encoder, queries)
        else:
            query_vecs = self.tfidfv.transform(queries)
        assert (
            np.sum(query_vecs) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."
        q_embs = query_vecs.numpy().astype(np.float32)
        # q_embs = query_vecs.toarray().astype(np.float32)
        D, I = self.indexer.search(q_embs, k)
        print(D)
        print(I)
        return D.tolist(), I.tolist()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--dataset_name", metavar="./data/train_dataset", type=str, help=""
    )
    parser.add_argument(
        "--model_name_or_path",
        metavar="bert-base-multilingual-cased",
        type=str,
        help="",
    )
    parser.add_argument("--data_path", metavar="./data", type=str, help="")
    parser.add_argument(
        "--context_path", metavar="wikipedia_documents", type=str, help=""
    )
    parser.add_argument("--use_faiss", metavar=False, type=bool, help="")

    args = parser.parse_args()

    # Test sparse
    org_dataset = load_from_disk(args.dataset_name)
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
    )

    retriever = SparseRetrieval(
        tokenize_fn=tokenizer.tokenize,
        data_path=args.data_path,
        context_path=args.context_path,
        datasets=datasets,
    )

    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

    if args.use_faiss:

        # test single query
        with timer("single query by faiss"):
            scores, indices = retriever.retrieve_faiss(query)

        # test bulk
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve_faiss(full_ds)
            df["correct"] = df["original_context"] == df["context"]

            print("correct retrieval result by faiss", df["correct"].sum() / len(df))

    else:
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve(full_ds)
            df["correct"] = df["original_context"] == df["context"]
            print(
                "correct retrieval result by exhaustive search",
                df["correct"].sum() / len(df),
            )

        with timer("single query by exhaustive search"):
            scores, indices = retriever.retrieve(query)

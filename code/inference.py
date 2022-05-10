"""
Open-Domain Question Answering 을 수행하는 inference 코드 입니다.

대부분의 로직은 train.py 와 비슷하나 retrieval, predict 부분이 추가되어 있습니다.
"""
# 지민님 eval 명령어
# python inference.py --output_dir ./outputs/val_dataset/ 
#                     --dataset_name ../data/train_dataset/ 
#                     --model_name_or_path ./models/train_dataset/ 
#                     --do_eval

# 추론 명령어
# python inference.py --output_dir ./outputs/test_dataset/ 
#                     --dataset_name ../data/test_dataset/ 
#                     --model_name_or_path ./models/train_dataset/ 
#                     --do_predict

# dense_retrieval eval 명령어
# python inference.py --output_dir ./outputs/val_Dataset_dense/ --dataset_name ../data/train_dataset/ --model_name_or_path ./models/train_dataset/ --do_eval --overwrite_output_dir

import logging
import sys
from typing import Callable, Dict, List, NoReturn, Tuple

import numpy as np
from arguments import DataTrainingArguments, ModelArguments
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
    load_from_disk,
    load_metric,
)
from retrieval import SparseRetrieval
from BM25 import BM25SparseRetrieval
from dense_retrieval import DenseRetrival, BertEncoder
# import torch
from trainer_qa import QuestionAnsweringTrainer
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from utils_qa import check_no_error, postprocess_qa_predictions

logger = logging.getLogger(__name__)


def main():
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.
    # torch.cuda.empty_cache()
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.do_train = True

    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("Training/evaluation parameters %s", training_args)

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(training_args.seed)

    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)
    '''
    < datasets["validation"][0] 예시 >
    {'__index_level_0__': 2146,
    'answers': {'answer_start': [284], 'text': ['한보철강']},
    'context': '순천여자고등학교 졸업, 1973년 이화여자대학교를 졸업하고 1975년 제17회 사법시험에 합격하여 판사로 임용되었고 '
                '대법원 재판연구관, 수원지법 부장판사, 사법연수원 교수, 특허법원 부장판사 등을 거쳐 능력을 인정받았다. 2003년 '
                '최종영 대법원장의 지명으로 헌법재판소 재판관을 역임하였다.\\n\\n경제민주화위원회(위원장 장하성이 소액주주들을 '
                '대표해 한보철강 부실대출에 책임이 있는 이철수 전 제일은행장 등 임원 4명을 상대로 제기한 손해배상청구소송에서 '
                '서울지방법원 민사합의17부는 1998년 7월 24일에 "한보철강에 부실 대출하여 은행에 막대한 손해를 끼친 점이 '
                '인정된다"며 "원고가 배상을 청구한 400억원 전액을 은행에 배상하라"고 하면서 부실 경영인에 대한 최초의 배상 '
                '판결을 했다. \\n\\n2004년 10월 신행정수도의건설을위한특별조치법 위헌 확인 소송에서 9인의 재판관 중 '
                '유일하게 각하 견해를 내었다. 소수의견에서 전효숙 재판관은 다수견해의 문제점을 지적하면서 관습헌법 법리를 부정하였다. '
                '전효숙 재판관은 서울대학교 근대법학교육 백주년 기념관에서 열린 강연에서, 국회가 고도의 정치적인 사안을 정치로 '
                '풀기보다는 헌법재판소에 무조건 맡겨서 해결하려는 자세는 헌법재판소에게 부담스럽다며 소회를 밝힌 바 있다.',
    'document_id': 9027,
    'id': 'mrc-0-003264',
    'question': '처음으로 부실 경영인에 대한 보상 선고를 받은 회사는?',
    'title': '전효숙'}
    '''

    # AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
    # argument로 원하는 모델 이름을 설정하면 옵션을 바꿀 수 있습니다.
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        use_fast=True,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )

    # True일 경우 : run passage retrieval
    if data_args.eval_retrieval:
        print('eval_retrieval 실행..')
        # datasets(do_predict) :
        # {'validation':{"context(하나로 합쳐진 Top-k 문장)","id(dataset 내의 id'mrc-..')","question"}}
        # datasets(do_eval) :
        # {'validation':{"answers","context(하나로 합쳐진 Top-k 문장)","id(dataset 내의 id'mrc-..')","question"}}
        # datasets = run_sparse_retrieval(
        #     tokenizer.tokenize, datasets, training_args, data_args,
        # )
        datasets = run_dense_retrieval(
            datasets, training_args, data_args,
        )

    # eval or predict mrc model
    if training_args.do_eval or training_args.do_predict:
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)


def run_sparse_retrieval(
    tokenize_fn: Callable[[str], List[str]],
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    data_path: str = "../data",
    context_path: str = "wikipedia_documents.json",
) -> DatasetDict:
    # datasets['validation'] :
    # {'answers','context','document_id','id(dataset 내의 id'mrc-..')','question','title'}

    # Query에 맞는 Passage들을 Retrieval 합니다.
    # retriever = SparseRetrieval(
    #     tokenize_fn=tokenize_fn, data_path=data_path, context_path=context_path
    # )
    retriever = BM25SparseRetrieval(
        tokenize_fn=tokenize_fn, data_path=data_path, context_path=context_path
    )
    retriever.get_sparse_embedding()

    if data_args.use_faiss: # default(False)
        retriever.build_faiss(num_clusters=data_args.num_clusters)
        df = retriever.retrieve_faiss(
            datasets["validation"], topk=data_args.top_k_retrieval
        )
    else: # default(Tfidf, Top-k)
        # df(do_predic) (test/validation 경우) :
        # -> DataFrame{"question","id(dataset 내의 id'mrc-..')","context_id(추출context의 인덱스들)","context(하나로 합쳐진 Top-k 문장)"}
        # df(do_eval) (train/validation 경우 정답추가) : 
        # -> 위에서 추가 {"original_context(원래 dataset에 포함된 문장)", "answers"}
        df = retriever.retrieve(datasets["validation"], topk=data_args.top_k_retrieval)

    # DataSet 자료형으로 변환하기 위해 Features 자료형 생성
    # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    if training_args.do_predict:
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )

    # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
    elif training_args.do_eval:
        f = Features(
            {
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
    # 반환값(do_predict) :
    # {'validation':{"context(하나로 합쳐진 Top-k 문장)","id(dataset 내의 id'mrc-..')","question"}}
    # 반환값(do_eval) :
    # {'validation':{"answers","context(하나로 합쳐진 Top-k 문장)","id(dataset 내의 id'mrc-..')","question"}}
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets

def run_dense_retrieval(
    # tokenize_fn: Callable[[str], List[str]],
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    data_path: str = "../data",
    # context_path: str = "wikipedia_documents.json",
) -> DatasetDict:
    # datasets['validation'] :
    # {'answers','context','document_id','id(dataset 내의 id'mrc-..')','question','title'}

    # Query에 맞는 Passage들을 Retrieval 합니다.
    retriever = DenseRetrival(
        data_path=data_path, 
        encoder_model_path = "./encoder_model"
    )
    retriever.get_dense_embedding()

    if data_args.use_faiss: # default(False)
        assert True, 'faiss 기능 추후 구현예정..'
        # retriever.build_faiss(num_clusters=data_args.num_clusters)
        # df = retriever.retrieve_faiss(
        #     datasets["validation"], topk=data_args.top_k_retrieval
        # )
    else: # default(Tfidf, Top-k)
        # df(do_predic) (test/validation 경우) :
        # -> DataFrame{"question","id(dataset 내의 id'mrc-..')","context_id(추출context의 인덱스들)","context(하나로 합쳐진 Top-k 문장)"}
        # df(do_eval) (train/validation 경우 정답추가) : 
        # -> 위에서 추가 {"original_context(원래 dataset에 포함된 문장)", "answers"}
        df = retriever.retrieve(datasets["validation"], topk=data_args.top_k_retrieval)

    # DataSet 자료형으로 변환하기 위해 Features 자료형 생성
    # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    if training_args.do_predict:
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )

    # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
    elif training_args.do_eval:
        f = Features(
            {
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
    # 반환값(do_predict) :
    # {'validation':{"context(하나로 합쳐진 Top-k 문장)","id(dataset 내의 id'mrc-..')","question"}}
    # 반환값(do_eval) :
    # {'validation':{"answers","context(하나로 합쳐진 Top-k 문장)","id(dataset 내의 id'mrc-..')","question"}}
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets

def run_mrc(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
) -> NoReturn:

    # eval 혹은 prediction에서만 사용함
    column_names = datasets["validation"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding에 대한 옵션을 설정합니다.
    # (question|context) 혹은 (context|question)로 세팅 가능합니다.
    pad_on_right = tokenizer.padding_side == "right"

    # 오류가 있는지 확인합니다.
    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )

    # Validation preprocessing / 전처리를 진행합니다.
    def prepare_validation_features(examples):
        # examples(do_predict) :
        # {"context(하나로 합쳐진 Top-k 문장)","id(dataset 내의 id'mrc-..')","question"}
        # examples(do_eval) :
        # {"answers","context(하나로 합쳐진 Top-k 문장)","id(dataset 내의 id'mrc-..')","question"}

        # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
        # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=False, # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # 길이가 긴 context가 등장할 경우 truncate를 진행해야하므로, 해당 데이터셋을 찾을 수 있도록 mapping 가능한 값이 필요합니다.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # evaluation을 위해, prediction을 context의 substring으로 변환해야합니다.
        # corresponding example_id를 유지하고 offset mappings을 저장해야합니다.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # sequence id를 설정합니다 (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # 하나의 example이 여러개의 span을 가질 수 있습니다.
            sample_index = sample_mapping[i]
            # example_id는 dataset 내에서의 순서를 인덱스로 한것
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # context의 일부가 아닌 offset_mapping을 None으로 설정하여 토큰 위치가 컨텍스트의 일부인지 여부를 쉽게 판별할 수 있습니다.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        print("총 tokenized_examples:",len(tokenized_examples["input_ids"]))
        # 반환값(tokenized_examples) :
        # {'input_ids','token_type_ids','attention_mask','offset_mapping'<-changed,'example_id(dataset 내에서의 순서)'<-added}
        return tokenized_examples

    # eval_datasets(do_predict) :
    # {"context(하나로 합쳐진 Top-k 문장)","id(dataset 내의 id'mrc-..')","question"}
    # eval_datasets(do_eval) :
    # {"answers","context(하나로 합쳐진 Top-k 문장)","id(dataset 내의 id'mrc-..')","question"}
    eval_dataset = datasets["validation"]

    # Validation Feature 생성
    eval_dataset = eval_dataset.map(
        prepare_validation_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    # Data collator
    # flag가 True이면 이미 max length로 padding된 상태입니다.
    # 그렇지 않다면 data collator에서 padding을 진행해야합니다.
    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )

    # Post-processing:
    def post_processing_function(
        examples,
        features,
        predictions: Tuple[np.ndarray, np.ndarray],
        training_args: TrainingArguments,
    ) -> EvalPrediction:
        # examples :
        # {"context(하나로 합쳐진 Top-k 문장)","id(dataset 내의 id'mrc-..')","question"}
        # features :
        # {'input_ids', 'token_type_ids', 'attention_mask', 'offset_mapping', 'example_id(dataset 내에서의 순서)'}

        # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.
        # predictions : {id(dataset 내의 id'mrc-..'): 모델의 예상 정답 }
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=data_args.max_answer_length,
            output_dir=training_args.output_dir,
        )
        # Metric을 구할 수 있도록 Format을 맞춰줍니다.
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]

        if training_args.do_predict:
            return formatted_predictions
        elif training_args.do_eval:
            references = [
                {"id": ex["id"], "answers": ex[answer_column_name]}
                for ex in datasets["validation"]
            ]

            return EvalPrediction(
                predictions=formatted_predictions, label_ids=references
            )

    metric = load_metric("squad")

    def compute_metrics(p: EvalPrediction) -> Dict:
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    print("init trainer...")
    # Trainer 초기화
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=eval_dataset,
        eval_examples=datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    logger.info("*** Evaluate ***")

    #### eval dataset & eval example - predictions.json 생성됨
    if training_args.do_predict:
        # predictions : {'id(dataset 내의 id'mrc-..')','prediction_text'}
        predictions = trainer.predict(
            test_dataset=eval_dataset, test_examples=datasets["validation"]
        )

        # predictions.json 은 postprocess_qa_predictions() 호출시 이미 저장됩니다.
        print(
            "No metric can be presented because there is no correct answer given. Job done!"
        )

    if training_args.do_eval:
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)


if __name__ == "__main__":
    main()

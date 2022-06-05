import torch
import streamlit as st
# from model import MyEfficientNet
import yaml
from typing import Callable, List, Tuple, Dict
import numpy as np

from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    # HfArgumentParser,
    TrainingArguments,
    # set_seed,
)
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Value,
    load_from_disk,
    load_metric,
)
import tokenizers
from retrieval import SparseRetrieval
from trainer_qa import QuestionAnsweringTrainer
from utils_qa import check_no_error, postprocess_qa_predictions

def run_sparse_retrieval(
    tokenize_fn: Callable[[str], List[str]],
    query: str,
    data_path: str = "../data",
    context_path: str = "wikipedia_documents.json",
) -> DatasetDict:

    # Query에 맞는 Passage들을 Retrieval 합니다.
    retriever = SparseRetrieval(
        tokenize_fn=tokenize_fn, data_path=data_path, context_path=context_path
    )
    retriever.get_sparse_embedding()

    df40, df = retriever.retrieve(query, topk=40)
    f = Features(
        {
            "context": Value(dtype="string", id=None),
            "id": Value(dtype="string", id=None),
            "question": Value(dtype="string", id=None),
        }
    )
    # st.write(df)
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets, df40

def run_mrc(
    # training_args: TrainingArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
):

    # eval 혹은 prediction에서만 사용함
    column_names = datasets["validation"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding에 대한 옵션을 설정합니다.
    # (question|context) 혹은 (context|question)로 세팅 가능합니다.
    pad_on_right = tokenizer.padding_side == "right"

    # # 오류가 있는지 확인합니다.
    # last_checkpoint, max_seq_length = check_no_error(
    #     data_args, training_args, datasets, tokenizer
    # )

    # Validation preprocessing / 전처리를 진행합니다.
    def prepare_validation_features(examples):
        # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
        # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=512,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=False, # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
            padding="max_length",
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
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # context의 일부가 아닌 offset_mapping을 None으로 설정하여 토큰 위치가 컨텍스트의 일부인지 여부를 쉽게 판별할 수 있습니다.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        return tokenized_examples

    eval_dataset = datasets["validation"]
    # Validation Feature 생성
    eval_dataset = eval_dataset.map(
        prepare_validation_features,
        batched=True,
        num_proc=None,
        remove_columns=column_names,
        load_from_cache_file=True,
    )
    # Data collator
    # flag가 True이면 이미 max length로 padding된 상태입니다.
    # 그렇지 않다면 data collator에서 padding을 진행해야합니다.
    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 # if training_args.fp16 else None
    )

    # Post-processing:
    def post_processing_function(
        examples,
        features,
        predictions: Tuple[np.ndarray, np.ndarray],
        training_args: TrainingArguments,
    ) -> EvalPrediction:
        # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.
        global all_nbest_json
        predictions, all_nbest_json = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=512,
            output_dir='./outputs/test',
        )
        # Metric을 구할 수 있도록 Format을 맞춰줍니다.
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]
        # st.write(all_nbest_json)
        return formatted_predictions

    metric = load_metric("squad")

    def compute_metrics(p: EvalPrediction) -> Dict:
        return metric.compute(predictions=p.predictions, references=p.label_ids)
    
    training_args = TrainingArguments(output_dir='./outputs/test')
    # training_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


    #### eval dataset & eval example - predictions.json 생성됨
    predictions = trainer.predict(
        test_dataset=eval_dataset, test_examples=datasets["validation"]
    )
    return predictions


@st.cache(hash_funcs={tokenizers.Tokenizer: lambda _: None})
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = AutoConfig.from_pretrained(
        '/opt/ml/level2-mrc-level2-nlp-10/code/models/train_dataset/'
    )
    tokenizer = AutoTokenizer.from_pretrained(
        '/opt/ml/level2-mrc-level2-nlp-10/code/models/train_dataset/',
        use_fast=True,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        '/opt/ml/level2-mrc-level2-nlp-10/code/models/train_dataset/',
        from_tf=bool(".ckpt" in '/opt/ml/level2-mrc-level2-nlp-10/code/models/train_dataset/'),
        config=config,
    ).to(device)
    # model = MyEfficientNet(num_classes=18).to(device)
    # model.load_state_dict(torch.load(config['model_path'], map_location=device))
    
    return model, tokenizer


def get_prediction(model, tokenizer, query: str) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with st.spinner("Wikipedia에서 지문 추출 중..."):
        datasets, df40 = run_sparse_retrieval(
            tokenizer.tokenize, query,
        )
        with st.expander("질문과 관련된 TOP-40 Wikipedia 지문"):
            st.table(df40[['context_id', 'context']])
    with st.spinner("추출된 지문에서 답변 찾는 중..."):
        answer = run_mrc(datasets, tokenizer, model)
        with st.expander("추출된 Wikipedia 지문에서 찾은 TOP-20 답변"):
            # st.table(df40[['context_id', 'context']])
            st.table(all_nbest_json['0'])
    return answer

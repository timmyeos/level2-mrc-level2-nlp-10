import logging
import os
import sys
from typing import NoReturn

from arguments import DataTrainingArguments, ModelArguments
from datasets import DatasetDict, load_from_disk, load_metric, Dataset
import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd
from trainer_qa import QuestionAnsweringTrainer
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed
)
from utils_qa import check_no_error, postprocess_qa_predictions
from retrieval import SparseRetrieval
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel
import json
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class Model(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.conv1 = nn.Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size*2,
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=config.hidden_size*2,
            out_channels=config.hidden_size,
            kernel_size=1,
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = sequence_output.permute(0,2,1)

        for _ in range(5):
            out = self.conv1(sequence_output)
            out = self.conv2(out)
            out = sequence_output + torch.relu(out)
            sequence_output = self.layer_norm(out)

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

   
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def main():
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.num_train_epochs = 5
    training_args.learning_rate = 3e-6
    training_args.fp16 = True

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("Training/evaluation parameters %s", training_args)

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(training_args.seed)

    datasets = load_from_disk(data_args.dataset_name)

    # with open("augmentation.json", "r") as file:
    #     trans = json.load(file)

    # train_ = pd.DataFrame({"__index_level_0__": (trans['__index_level_0__'] + datasets['train']['__index_level_0__']), 
    #                        "answers": (trans['answers'] + datasets['train']['answers']),
    #                        "context": trans['context'] + datasets['train']['context'],
    #                        "document_id": (trans['document_id'] + datasets['train']['document_id']),
    #                        "id": (trans['id'] + datasets['train']['id']),
    #                        "question": (trans['question'] + datasets['train']['question']),
    #                        "title": (trans['title'] + datasets['train']['title'])          
    #                        })
    # validation_ = pd.DataFrame({"__index_level_0__": datasets['validation']['__index_level_0__'], 
    #                        "answers": datasets['validation']['answers'],
    #                        "context": datasets['validation']['context'],
    #                        "document_id": datasets['validation']['document_id'],
    #                        "id": datasets['validation']['id'],
    #                        "question": datasets['validation']['question'],
    #                        "title": datasets['validation']['title']                 
    #                        })
    # datasets = DatasetDict({'train':Dataset(pa.Table.from_pandas(train_)), 'validation':Dataset(pa.Table.from_pandas(validation_))})


    # AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
    # argument로 원하는 모델 이름을 설정하면 옵션을 바꿀 수 있습니다.

    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name is not None
        else model_args.model_name_or_path,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name,
        # if model_args.tokenizer_name is not None
        # else model_args.model_name_or_path,
        # # 'use_fast' argument를 True로 설정할 경우 rust로 구현된 tokenizer를 사용할 수 있습니다.
        # # False로 설정할 경우 python으로 구현된 tokenizer를 사용할 수 있으며,
        # # rust version이 비교적 속도가 빠릅니다.
        use_fast=True,
    )

    model = Model(config=config)
    # model = AutoModelForQuestionAnswering.from_pretrained(
    #     model_args.model_name_or_path,
    #     from_tf=bool(".ckpt" in model_args.model_name_or_path),
    #     config=config,
    # )

    # do_train mrc model 혹은 do_eval mrc model
    if training_args.do_train or training_args.do_eval:
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)


def run_mrc(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
) -> NoReturn:

    # dataset을 전처리합니다.
    # training과 evaluation에서 사용되는 전처리는 아주 조금 다른 형태를 가집니다.
    if training_args.do_train:
        column_names = datasets["train"].column_names # ['title', 'context', 'question', 'id', 'answers', 'document_id', '__index_level_0__']
    else:
        column_names = datasets["validation"].column_names # ['title', 'context', 'question', 'id', 'answers', 'document_id', '__index_level_0__']
    
    question_column_name = "question" if "question" in column_names else column_names[0] # 'question'
    context_column_name = "context" if "context" in column_names else column_names[1] # 'context'
    answer_column_name = "answers" if "answers" in column_names else column_names[2] # 'answer'
    
    # Padding에 대한 옵션을 설정합니다.
    # (question|context) 혹은 (context|question)로 세팅 가능합니다.
    pad_on_right = tokenizer.padding_side == "right" # True
    
    # 오류가 있는지 확인합니다.
    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )
    # None
    # max_seq_length

    # Train preprocessing / 전처리를 진행합니다.
    def prepare_train_features(examples):
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
        # token의 캐릭터 단위 position를 찾을 수 있도록 offset mapping을 사용합니다.
        # start_positions과 end_positions을 찾는데 도움을 줄 수 있습니다.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # 데이터셋에 "start position", "enc position" label을 부여합니다.
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)  # cls index

            # sequence id를 설정합니다 (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # 하나의 example이 여러개의 span을 가질 수 있습니다.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]

            # answer가 없을 경우 cls_index를 answer로 설정합니다(== example에서 정답이 없는 경우 존재할 수 있음).
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # text에서 정답의 Start/end character index
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # text에서 current span의 Start token index
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # text에서 current span의 End token index
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # 정답이 span을 벗어났는지 확인합니다(정답이 없는 경우 CLS index로 label되어있음).
                if not (
                    offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char
                ):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # token_start_index 및 token_end_index를 answer의 끝으로 이동합니다.
                    # Note: answer가 마지막 단어인 경우 last offset을 따라갈 수 있습니다(edge case).
                    while (
                        token_start_index < len(offsets)
                        and offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)
        '''
        print(tokenizer.decode(tokenized_examples['input_ids']))
        print(tokenized_examples['attention_mask'])
        print(tokenized_examples['token_type_ids'])
        print(tokenized_examples['start_positions'])
        print(tokenized_examples['end_positions'])
        '''
        return tokenized_examples
    
    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]

        # dataset에서 train feature를 생성합니다.
        train_dataset = train_dataset.map(
            prepare_train_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Validation preprocessing
    def prepare_validation_features(examples):
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
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping을 None으로 설정해서 token position이 context의 일부인지 쉽게 판별 할 수 있습니다.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        '''
        print(tokenizer.decode(tokenized_examples['input_ids']))
        print(tokenized_examples['attention_mask'])
        print(tokenized_examples['token_type_ids'])
        print(tokenized_examples['example_id'])
        print(tokenized_examples['offset_mapping'])
        '''
        return tokenized_examples

    if training_args.do_eval:
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
    '''
    DataCollatorWithPadding(tokenizer=PreTrainedTokenizerFast(name_or_path='klue/bert-base', 
                                                              vocab_size=32000, 
                                                              model_max_len=512, 
                                                              is_fast=True, 
                                                              padding_side='right', 
                                                              special_tokens={'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'}), 
                                                              padding=True, 
                                                              max_length=None, 
                                                              pad_to_multiple_of=None)
    '''
    
    # Post-processing:
    def post_processing_function(examples, features, predictions, training_args):
        '''
        <<EXCUTION ONLY WHEN do_eval or do_predict>>

        Dataset({
            features: ['__index_level_0__', 'answers', 'context', 'document_id', 'id', 'question', 'title'],
            num_rows: 240
        })
        Dataset({
            features: ['attention_mask', 'example_id', 'input_ids', 'offset_mapping', 'token_type_ids'],
            num_rows: 474
        })
        (array, array, dtype) -> array.shape : (features_num_rows, max_seq_length)
        '''
       
        # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.
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
            '''
            formatted_predictions : [{'id': 'mrc-0-003264', 'prediction_text': '각하 견해를 내었다. 소수의'}, ...]
            '''
            
            return formatted_predictions

        elif training_args.do_eval:
            references = [
                {"id": ex["id"], "answers": ex[answer_column_name]}
                for ex in datasets["validation"]
            ]
            
            '''
            formatted_predictions : [{'id': 'mrc-0-003264', 'prediction_text': '각하 견해를 내었다. 소수의'}, ...]
            references : [{'id': 'mrc-0-003264', 'answers': {'answer_start': [284], 'text': ['한보철강']}}, ...]
            
            '''
        
                
            return EvalPrediction(
                predictions=formatted_predictions, label_ids=references
            )

    metric = load_metric("squad") # Exact Match, F1

    def compute_metrics(p: EvalPrediction):
        '''
        p.predictions = formatted_predictions
        p.label_ids = references
        '''
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    # Trainer 초기화
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
 
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")

        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        # State 저장
        trainer.state.save_to_json(
            os.path.join(training_args.output_dir, "trainer_state.json")
        )

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()

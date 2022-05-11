import os
import json
import numpy as np
import pickle
import pandas as pd
import random
from tqdm import tqdm, trange
from pprint import pprint
from typing import List, NoReturn, Optional, Tuple, Union

import faiss
import time
from contextlib import contextmanager


import torch
import torch.nn.functional as F
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel
from transformers import (
    BertModel, 
    BertPreTrainedModel, 
    AutoTokenizer,
    AdamW, 
    TrainingArguments, 
    get_linear_schedule_with_warmup
)
from torch.utils.data import (
    DataLoader, 
    RandomSampler, 
    TensorDataset, 
    SequentialSampler
)
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
    load_from_disk,
    load_metric,
    load_dataset
)

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

class DenseRetrival:
    def __init__(
        self,
        # data_path: Optional[str] = "../data/",
        # encoder_model_path: Optional[str] = "./encoder_model"
        data_path = "../data/",
        encoder_model_path = "./encoder_model"
    ):

        assert os.path.exists(data_path), "Data_Path not exists..!"
        assert os.path.exists(data_path), "Encoder_Model_Path not exists..!"

        self.data_path = data_path
        self.contexts = None
        self.context_idx = None

        with open(os.path.join(encoder_model_path,"dpr_tokenizer.bin"), "rb") as file:
            self.tokenizer = pickle.load(file)
        self.p_encoder = torch.load(os.path.join(encoder_model_path, "p_encoder.pt"))
        self.q_encoder = torch.load(os.path.join(encoder_model_path, "q_encoder.pt"))
        self.p_embeddings = None

    def get_dense_embedding(
        self,
        eval_batch_size = 8,
        doc_stride = 128
    ):

        # wiki : {'text',document_id,'title',..}
        with open(os.path.join(self.data_path,"wikipedia_documents.json"), "r", encoding="utf-8") as f:
            wiki = json.load(f)
        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        print(f"Lengths of unique wiki_contexts : {len(self.contexts)}")

        # Construt dataloader
        valid_p_seqs = self.tokenizer(
            self.contexts, 
            padding="max_length", 
            truncation="only_first",
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_tensors='pt'
        )

        self.context_idx = np.array(valid_p_seqs['overflow_to_sample_mapping'])
        
        valid_dataset = TensorDataset(valid_p_seqs['input_ids'], valid_p_seqs['attention_mask'], valid_p_seqs['token_type_ids'])
        valid_sampler = SequentialSampler(valid_dataset)
        valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=eval_batch_size)

        # Inference using the passage encoder to get dense embeddeings
        p_embs = []
        print('DPR passage embedding generation..')
        with torch.no_grad():

            epoch_iterator = tqdm(valid_dataloader, desc="Iteration", position=0, leave=True)
            self.p_encoder.eval()

            for _, batch in enumerate(epoch_iterator):
                batch = tuple(t.cuda() for t in batch)

                p_inputs = {'input_ids': batch[0],
                            'attention_mask': batch[1],
                            'token_type_ids': batch[2]
                            }
                    
                outputs = self.p_encoder(**p_inputs).to('cpu').numpy()
                p_embs.extend(outputs)

        # (num_passage, emb_dim)
        self.p_embeddings = np.array(p_embs)

    def retrieve(
        self, 
        # query_or_dataset: Union[str, Dataset], 
        # topk: Optional[int] = 1
        query_or_dataset, 
        topk = 1,
        bm25_use = False
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        # query_or_dataset :
        # {'answers','context','document_id','id(mrc-..)','question','title'}

        if isinstance(query_or_dataset, str):
            assert isinstance(query_or_dataset, str), "단일 Question(str 타입)은 아직 지원하지 않습니다."
        
        elif isinstance(query_or_dataset, Dataset):
            total = []
            doc_scores = None
            doc_indices = None

            with timer("query exhaustive search"):
                # top_k score,인덱스 정보를 가져옴
                # doc_scores : (num_query,top_k)
                # doc_indices : (num_query,top_k)
                if bm25_use:
                    doc_scores, doc_indices = self.get_relevant_doc_bulk(
                        query_or_dataset["question"], k=2000
                    )
                else:
                    doc_scores, doc_indices = self.get_relevant_doc_bulk(
                        query_or_dataset["question"], k=topk
                    )

            if bm25_use:
                retriever = BM25SparseRetrieval(
                    tokenize_fn=tokenize_fn, data_path=data_path, context_path=context_path
                )
                retriever.get_sparse_embedding()
                bm_scores, bm_indices = retriever.retrieve(query_or_dataset, topk=topk, get_scores=True)
                
                pass

            # dataframe으로 변환
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Dense retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"], # 예)'mrc-0-003264'
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context_id": doc_indices[idx], # doc_indices[0] = [10,153,15,..]
                    "context": " ".join( # Top-k 문장을 하나로 합침 ['순천여자고..','소수의견에서..',..]
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                # train/validation 데이터의 경우 자료형 추가
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            # 반환값(test/validation 경우) :
            # -> DataFrame{"question","id(dataset 내의 id'mrc-..')","context_id(추출context의 인덱스들)","context(하나로 합쳐진 Top-k 문장)"}
            # 반환값(train/validation 경우 정답추가) : 
            # -> 위에서 추가 {"original_context(query에 포함된 문장)", "answers"}
            return cqas
    
    def get_relevant_doc_bulk(
        # self, queries: List, k: Optional[int] = 1
        self, queries, k = 1
    ) -> Tuple[List, List]:

        valid_q_seqs = self.tokenizer(queries, padding="max_length", truncation=True, return_tensors='pt').to('cuda')

        with torch.no_grad():
            self.q_encoder.eval()
            q_embs = self.q_encoder(**valid_q_seqs).to('cpu').numpy()

        torch.cuda.empty_cache()

        if torch.cuda.is_available():
            p_embs_cuda = torch.Tensor(self.p_embeddings).to('cuda')
            q_embs_cuda = torch.Tensor(q_embs).to('cuda')

        print('Scores Calculation..')
        dot_prod_scores = torch.matmul(q_embs_cuda, torch.transpose(p_embs_cuda, 0, 1))
        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()

        doc_scores = []
        doc_indices = []

        for i in tqdm(range(len(queries)), desc=f"Top-k({k}) retrieval: "):
            tmp = []
            # tokenized_query = self.tokenize_fn(query)
            # tmp.extend(tokenized_query)
            # assert self.ngram <= len(tmp), f"tokenized 된 길이가 ngram({ngram}) 보다 작습니다."
            # tmp.extend([''.join(tmp[i:i+self.ngram]) for i in range(len(tmp)-self.ngram+1)])
            
            scores = [ dot_prod_scores[i][rank[i][0]] ] # rank[i][]: passage_index
            indices = [ self.context_idx[rank[i][0]] ]
            
            for idx in range(1,len(rank[i])):
                if len(scores) >= k:
                    break
                if self.context_idx[rank[i][idx]] != self.context_idx[rank[i][idx-1]]:
                    scores.append(dot_prod_scores[i][rank[i][idx]])
                    indices.append(self.context_idx[rank[i][idx]])
            # scores,indices = self.bm25.get_top_n(tmp, self.contexts, n=k)
            doc_scores.append(scores)
            doc_indices.append(indices)
        
        return doc_scores, doc_indices


class BertEncoder(BertPreTrainedModel):
    def __init__(
        self, 
        config
        ) -> NoReturn:

        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()

    def forward(
        self, 
        input_ids, 
        attention_mask=None, 
        token_type_ids=None): 

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
            )

        pooled_output = outputs[1]

        return pooled_output

class RobertaEncoder(RobertaPreTrainedModel):
    def __init__(self, config):
        super(RobertaEncoder, self).__init__(config)
        self.roberta = RobertaModel(config)
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
        return_dict=None
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
        
        pooled_output = outputs[1]
   
        return pooled_output


def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Configure 설정
    data_path = "../data"
    output_dir = "./encoder_model"
    # model_checkpoint = "bert-base-multilingual-cased"
    model_checkpoint = 'klue/bert-base'
    max_token_length = 512
    eval_batch_size = 8
    doc_stride = 128
        
    # 데이터 경로 이상유무 검증
    # assert os.path.exists(data_path), "Wiki_Data_path not exists..!"
    assert os.path.exists(data_path), "Data_path not exists..!"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print('Dataset Load..')
    # datasets : {'train','validation'}
    #   -> datasets['train'] / datasets['validation'] : 
    #       {'id','document_id','title','context','question','answers'}
    datasets = load_from_disk(os.path.join(data_path,"train_dataset"))
    # train_dataset = datasets["train"]
    # eval_dataset = datasets["validation"]

    print('Tokenizer Load..')
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    training_dataset = datasets['train']
    print('Data length:',len(datasets['train']))

    #######################################
    # def prepare_train_features(examples):
    #     tokenized_examples = dict()

    #     modified_train_context = []

    #     for i in tqdm(range(len(examples['context']))):
    #         answer_len = len(examples['answers'][i]['text'][0])
    #         answer_start_offset = examples['answers'][i]['answer_start'][0]
    #         answer_end_offset = answer_start_offset + answer_len - 1
    #         context_start_offset = max((answer_start_offset+answer_end_offset)//2-max_token_length//2,0)
    #         modified_train_context.append(examples['context'][i][context_start_offset:])

    #     p_seqs = tokenizer(
    #         modified_train_context,
    #         padding="max_length", 
    #         truncation=True,
    #         return_tensors='pt'
    #     )
    #     # tokenized_examples['p_seqs'] = [p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids']]
    #     tokenized_examples['p_seqs_input_ids'] = [p_seqs['input_ids']]
    #     tokenized_examples['p_seqs_attention_mask'] = [p_seqs['attention_mask']]
    #     tokenized_examples['p_seqs_token_type_ids'] = [p_seqs['token_type_ids']]

    #     q_seqs = tokenizer(
    #         examples['question'],
    #         padding="max_length", 
    #         truncation=True,
    #         return_tensors='pt'
    #     )
    #     # tokenized_examples['q_seqs'] = [q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids']]
    #     tokenized_examples['q_seqs_input_ids'] = [q_seqs['input_ids']]
    #     tokenized_examples['q_seqs_attention_mask'] = [q_seqs['attention_mask']]
    #     tokenized_examples['q_seqs_token_type_ids'] = [q_seqs['token_type_ids']]

    #     return tokenized_examples
    
    # train_dataset = datasets['train'].map(
    #     prepare_train_features,
    #     batched=True,
    #     num_proc=4,
    #     remove_columns=datasets["train"].column_names,
    #     load_from_cache_file=True,
    # )
    ########################################
    print("Modifing Data..")
    modified_train_context = []

    for i in tqdm(range(len(training_dataset['context']))):
        answer_len = len(training_dataset['answers'][i]['text'][0])
        answer_start_offset = training_dataset['answers'][i]['answer_start'][0]
        answer_end_offset = answer_start_offset + answer_len - 1
        context_start_offset = max((answer_start_offset+answer_end_offset)//2-max_token_length//2,0)
        modified_train_context.append(training_dataset['context'][i][context_start_offset:])
    print('Modified Data length:',len(modified_train_context))

    q_seqs = tokenizer(training_dataset['question'], 
        padding="max_length", 
        truncation=True,
        return_tensors='pt'
    )
    p_seqs = tokenizer(modified_train_context, 
        padding="max_length", 
        truncation=True, 
        return_tensors='pt'
    )

    train_dataset = TensorDataset(p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'], 
                                q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids'])
    ##############################


    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=20,
        weight_decay=0.01
    )
    # load pre-trained model on cuda (if available)
    # 
    p_encoder = BertEncoder.from_pretrained(model_checkpoint)
    q_encoder = BertEncoder.from_pretrained(model_checkpoint)
    # p_encoder = RobertaEncoder.from_pretrained(model_checkpoint)
    # q_encoder = RobertaEncoder.from_pretrained(model_checkpoint)

    if torch.cuda.is_available():
        p_encoder.cuda()
        q_encoder.cuda()
    
    p_encoder, q_encoder = train(args, train_dataset, p_encoder, q_encoder)

    with open(os.path.join(args.output_dir,"dpr_tokenizer.bin"), "wb") as file:
        pickle.dump(tokenizer, file)
    torch.save(p_encoder, os.path.join(args.output_dir, "p_encoder.pt"))
    torch.save(q_encoder, os.path.join(args.output_dir, "q_encoder.pt"))


def train(
    args, 
    dataset, 
    p_model, 
    q_model
):

    # Dataloader
    train_sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=args.per_device_train_batch_size)

    # Optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in p_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in p_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in q_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in q_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # Start training!
    global_step = 0

    p_model.zero_grad()
    q_model.zero_grad()
    torch.cuda.empty_cache()

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")

    for _ in train_iterator:
        # epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")

        for step, batch in enumerate(epoch_iterator):
            q_model.train()
            p_model.train()
            
            if torch.cuda.is_available():
                batch = tuple(t.cuda() for t in batch)

            ######################################
            # p_inputs = {'input_ids': batch['p_seqs_input_ids'][0].cuda(),
            #             'attention_mask': batch['p_seqs_attention_mask'][0].cuda(),
            #             'token_type_ids': batch['p_seqs_token_type_ids'][0].cuda()
            #             }

            # q_inputs = {'input_ids': batch['q_seqs_input_ids'][0].cuda(),
            #             'attention_mask': batch['q_seqs_attention_mask'][0].cuda(),
            #             'token_type_ids': batch['q_seqs_token_type_ids'][0].cuda()
            #             }
            #######################################
            p_inputs = {'input_ids': batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2]
                        }

            q_inputs = {'input_ids': batch[3],
                        'attention_mask': batch[4],
                        'token_type_ids': batch[5]}
            ########################################

            p_outputs = p_model(**p_inputs)  # (batch_size, emb_dim)
            q_outputs = q_model(**q_inputs)  # (batch_size, emb_dim)


            # Calculate similarity score & loss
            sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))  # (batch_size, emb_dim) x (emb_dim, batch_size) = (batch_size, batch_size)

            # target: position of positive samples = diagonal element 
            targets = torch.arange(0, args.per_device_train_batch_size).long()
            if torch.cuda.is_available():
                targets = targets.to('cuda')

            sim_scores = F.log_softmax(sim_scores, dim=1)

            loss = F.nll_loss(sim_scores, targets)
            epoch_iterator.set_description("Iteration|Loss->%.4f" % loss)
            # print(loss)

            loss.backward()
            optimizer.step()
            scheduler.step()
            q_model.zero_grad()
            p_model.zero_grad()
            global_step += 1

            torch.cuda.empty_cache()

    return p_model, q_model


if __name__ == "__main__":
    main()
import logging
import os
import sys
from typing import NoReturn
import re
import json
import numpy as np
from pprint import pprint
from arguments import DataTrainingArguments, ModelArguments,RetrieverArguments
from datasets import DatasetDict, load_from_disk, load_metric
from trainer_qa import QuestionAnsweringTrainer
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,BertModel, 
    BertPreTrainedModel,
    get_linear_schedule_with_warmup,
    AdamW
)
from utils_qa import check_no_error, postprocess_qa_predictions
from typing import Optional
import torch      
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from tqdm.auto import tqdm, trange
import torch.nn.functional as F


# parsing
def dense_args(retriever_args:RetrieverArguments):# : RetrieverArguments):
    '''
    inference TrainingArguments
    '''
    args = TrainingArguments(
            output_dir="dense_retrieval",
            evaluation_strategy="epoch",
            learning_rate=retriever_args.dpr_learning_rate,
            per_device_train_batch_size=retriever_args.dpr_train_batch,
            per_device_eval_batch_size=retriever_args.dpr_eval_batch,
            num_train_epochs=retriever_args.dpr_epochs,
            weight_decay=retriever_args.dpr_weight_decay,
            overwrite_output_dir = True,
            eval_steps = retriever_args.dpr_eval_steps
            )

    retriever_dir = retriever_args.retriever_dir
    p,q = 'p_encoder','q_encoder'

    if (os.path.isdir(os.path.join(retriever_dir,p)) and os.path.isdir(os.path.join(retriever_dir,q))):
        print('Fine-tuned DPR exists... check directory again if using model_checkpoints...')
        config_p =  AutoConfig.from_pretrained(os.path.join(retriever_dir, p))
        config_q =  AutoConfig.from_pretrained(os.path.join(retriever_dir, p))
        p_encoder  = BertEncoder.from_pretrained(os.path.join(retriever_dir, p), config = config_p)
        q_encoder = BertEncoder.from_pretrained(os.path.join(retriever_dir, q), config = config_q)

    else:
        p_encoder  = BertEncoder.from_pretrained(retriever_args.dpr_model)
        q_encoder = BertEncoder.from_pretrained(retriever_args.dpr_model)
        print('No fine-tuned DPR exists ... newly train Dense Passage Retriever...')
    
    tokenizer = AutoTokenizer.from_pretrained(retriever_args.dpr_model)
    
    return args, tokenizer, p_encoder, q_encoder


# encoder
class BertEncoder(BertPreTrainedModel):
    """
    BertEncoder : special mission 4 참고
    """
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)
        self.bert = BertModel(config)
        self.init_weights()

    def forward(self,input_ids, attention_mask=None,token_type_ids=None): 

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        pooled_output = outputs[1]
        return pooled_output


# Answer
class DenseRetrieval:

    def __init__(self, args, dataset, num_neg, tokenizer, p_encoder, q_encoder):

        '''
        special mission 4 참고
        '''
        self.args = args
        self.dataset = dataset
        self.num_neg = num_neg

        self.tokenizer = tokenizer
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder

        self.prepare_in_batch_negative(num_neg=num_neg)

    def prepare_in_batch_negative(self, dataset=None, num_neg=2, tokenizer=None):

        if dataset is None:
            dataset = self.dataset

        if tokenizer is None:
            tokenizer = self.tokenizer

        # 1. In-Batch-Negative 만들기
        # CORPUS를 np.array로 변환해줍니다.        
        corpus = np.array(list(set([example for example in dataset['context']])))
        p_with_neg = []

        for c in dataset['context']:
            
            while True:
                neg_idxs = np.random.randint(len(corpus), size=num_neg)

                if not c in corpus[neg_idxs]:
                    p_neg = corpus[neg_idxs]

                    p_with_neg.append(c)
                    p_with_neg.extend(p_neg)
                    break

        # 2. (Question, Passage) 데이터셋 만들어주기
        q_seqs = tokenizer(dataset['question'], padding="max_length", truncation=True, return_tensors='pt')
        p_seqs = tokenizer(p_with_neg, padding="max_length", truncation=True, return_tensors='pt')

        max_len = p_seqs['input_ids'].size(-1)
        p_seqs['input_ids'] = p_seqs['input_ids'].view(-1, num_neg+1, max_len)
        p_seqs['attention_mask'] = p_seqs['attention_mask'].view(-1, num_neg+1, max_len)
        p_seqs['token_type_ids'] = p_seqs['token_type_ids'].view(-1, num_neg+1, max_len)

        train_dataset = TensorDataset(
            p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'], 
            q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids']
        )

        self.train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.args.per_device_train_batch_size)

        valid_seqs = tokenizer(dataset['context'], padding="max_length", truncation=True, return_tensors='pt')
        passage_dataset = TensorDataset(
            valid_seqs['input_ids'], valid_seqs['attention_mask'], valid_seqs['token_type_ids']
        )
        self.passage_dataloader = DataLoader(passage_dataset, batch_size=self.args.per_device_train_batch_size)


    def train(self, args=None):

        if args is None:
            args = self.args
        batch_size = args.per_device_train_batch_size

        # Optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.p_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.q_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        t_total = len(self.train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

        # Start training!
        global_step = 0

        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = tqdm(range(int(args.num_train_epochs)), desc="Epoch")
        # for _ in range(int(args.num_train_epochs)):
        for _ in train_iterator:

            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:

                    p_encoder.train()
                    q_encoder.train()
            
                    targets = torch.zeros(batch_size).long() # positive example은 전부 첫 번째에 위치하므로
                    targets = targets.to(args.device)

                    p_inputs = {
                        'input_ids': batch[0].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                        'attention_mask': batch[1].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                        'token_type_ids': batch[2].view(batch_size * (self.num_neg + 1), -1).to(args.device)
                    }
            
                    q_inputs = {
                        'input_ids': batch[3].to(args.device),
                        'attention_mask': batch[4].to(args.device),
                        'token_type_ids': batch[5].to(args.device)
                    }
            
                    p_outputs = self.p_encoder(**p_inputs)  # (batch_size*(num_neg+1), emb_dim)
                    q_outputs = self.q_encoder(**q_inputs)  # (batch_size*, emb_dim)

                    # Calculate similarity score & loss
                    p_outputs = p_outputs.view(batch_size, self.num_neg + 1, -1)
                    q_outputs = q_outputs.view(batch_size, 1, -1)

                    sim_scores = torch.bmm(q_outputs, torch.transpose(p_outputs, 1, 2)).squeeze()  #(batch_size, num_neg + 1)
                    sim_scores = sim_scores.view(batch_size, -1)
                    sim_scores = F.log_softmax(sim_scores, dim=1)

                    loss = F.nll_loss(sim_scores, targets)
                    tepoch.set_postfix(loss=f'{str(loss.item())}')

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    self.p_encoder.zero_grad()
                    self.q_encoder.zero_grad()

                    global_step += 1

                    torch.cuda.empty_cache()

                    del p_inputs, q_inputs


    def get_relevant_doc(self, query, k=1, args=None, p_encoder=None, q_encoder=None):

        if args is None:
            args = self.args

        if p_encoder is None:
            p_encoder = self.p_encoder

        if q_encoder is None:
            q_encoder = self.q_encoder

        with torch.no_grad():
            p_encoder.eval()
            q_encoder.eval()

            q_seqs_val = self.tokenizer([query], padding="max_length", truncation=True, return_tensors='pt').to(args.device)
            q_emb = q_encoder(**q_seqs_val).to('cpu')  # (num_query=1, emb_dim)

            p_embs = []
            for batch in self.passage_dataloader:

                batch = tuple(t.to(args.device) for t in batch)
                p_inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2]
                }
                p_emb = p_encoder(**p_inputs).to('cpu')
                p_embs.append(p_emb)

        p_embs = torch.stack(p_embs, dim=0).view(len(self.passage_dataloader.dataset), -1)  # (num_passage, emb_dim)

        dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
        return rank[:k]

def main():
    logger = logging.getLogger(__name__)

    parser = HfArgumentParser((DataTrainingArguments, RetrieverArguments))
    data_args, retriever_args= parser.parse_args_into_dataclasses()

    print(f"data is from {data_args.dataset_path}")

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    # 일단 train만 학습
    dataset = load_from_disk(os.path.join(data_args.dataset_path , 'train_dataset'))
    args, tokenizer, p_encoder, q_encoder = dense_args(retriever_args)
    p_encoder = p_encoder.to(args.device)
    q_encoder = q_encoder.to(args.device)
    train_dataset = dataset['train']

#     args = TrainingArguments(
#     output_dir="dense_retireval",
#     evaluation_strategy="epoch",
#     learning_rate=3e-6,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     num_train_epochs=5,
#     weight_decay=0.01
# )  
    model_checkpoint = 'klue/bert-base'
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    p_encoder = BertEncoder.from_pretrained(model_checkpoint).to(args.device)
    q_encoder = BertEncoder.from_pretrained(model_checkpoint).to(args.device)
    retriever = DenseRetrieval(args=args, dataset=train_dataset, num_neg=5, tokenizer=tokenizer, p_encoder=p_encoder, q_encoder=q_encoder)

    model_dir = './models/retriever'

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    p_encoder.save_pretrained(os.path.join(model_dir,'p_encoder'))
    q_encoder.save_pretrained(os.path.join(model_dir,'q_encoder'))

    print(f'passage & question encoders successfully saved at {model_dir}')

    # Retriever 사용 예시
    query = '제주도 시청의 주소는 뭐야?'
    results = retriever.get_relevant_doc(query=query, k=5)

    print(f"[Search Query] {query}\n")

    indices = results.tolist()
    for i, idx in enumerate(indices):
        print(f"Top-{i + 1}th Passage (Index {idx})")
        pprint(retriever.dataset['context'][idx])

if __name__ == '__main__':
    main()
    
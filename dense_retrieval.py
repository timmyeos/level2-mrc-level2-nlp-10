import os
import numpy as np
from tqdm import tqdm, trange
import random
import torch
import torch.nn.functional as F
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


class DenseRetrival:
    def __init__(
        self,
        tokenize_fn
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
        encoder_model_path,
        top_k
        ) -> NoReturn:

        # 위키피디아 파일을 불러와서 text를 추출하고 순서대로 인덱스를 부여
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        self.p_encoder = None
        self.q_encoder = None
        self.top_k = top_k

    def get_dense_embedding():
        if torch.cuda.is_available():
            p_embs_cuda = torch.Tensor(p_embs).to('cuda')
            q_embs_cuda = torch.Tensor(q_embs).to('cuda')

        dot_prod_scores = torch.matmul(q_embs_cuda, torch.transpose(p_embs_cuda, 0, 1))
        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()

        for i, q in enumerate(query[:self.top_k]):
            r = rank[i]
            for j in range(k):
                print("Top-%d passage with score %.4f" % (j+1, dot_prod_scores[i][r[j]]))
                print(search_corpus[r[j]])

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


def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    data_path = "../data/train_dataset"
    output_dir = "./encoder_model"

    model_checkpoint = "bert-base-multilingual-cased"
        
    assert os.path.exists(data_path), "Data_path not exists..!"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    datasets = load_from_disk(data_path)
    # train_dataset = datasets["train"]
    # eval_dataset = datasets["validation"]

    # if os.path.isfile(emd_path)

    corpus = [example['context'] for example in datasets['train']]
    print('* 총 데이터 개수: ',len(corpus))

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    training_dataset = datasets['train']

    q_seqs = tokenizer(training_dataset['question'], 
        padding="max_length", 
        truncation=True,
        return_tensors='pt'
    )
    p_seqs = tokenizer(training_dataset['context'], 
        padding="max_length", 
        truncation=True, 
        return_tensors='pt'
    )

    train_dataset = TensorDataset(p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'], 
                                q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids'])

    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=2,
        weight_decay=0.01
    )
    # load pre-trained model on cuda (if available)
    p_encoder = BertEncoder.from_pretrained(model_checkpoint)
    q_encoder = BertEncoder.from_pretrained(model_checkpoint)

    if torch.cuda.is_available():
        p_encoder.cuda()
        q_encoder.cuda()
    
    p_encoder, q_encoder = train(args, train_dataset, p_encoder, q_encoder)


    ##########################################
    search_corpus = list(set([example['context'] for example in dataset['validation']]))


    eval_batch_size = 8

    # Construt dataloader
    valid_p_seqs = tokenizer(search_corpus, padding="max_length", truncation=True, return_tensors='pt')
    valid_dataset = TensorDataset(valid_p_seqs['input_ids'], valid_p_seqs['attention_mask'], valid_p_seqs['token_type_ids'])
    valid_sampler = SequentialSampler(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=eval_batch_size)

    # Inference using the passage encoder to get dense embeddeings
    p_embs = []

    with torch.no_grad():

    epoch_iterator = tqdm(valid_dataloader, desc="Iteration", position=0, leave=True)
    p_encoder.eval()

    for _, batch in enumerate(epoch_iterator):
        batch = tuple(t.cuda() for t in batch)

        p_inputs = {'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2]
                    }
            
        outputs = p_encoder(**p_inputs).to('cpu').numpy()
        p_embs.extend(outputs)

    p_embs = np.array(p_embs)
    p_embs.shape  # (num_passage, emb_dim)    


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
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")

        for step, batch in enumerate(epoch_iterator):
            q_encoder.train()
            p_encoder.train()

            if torch.cuda.is_available():
                batch = tuple(t.cuda() for t in batch)

            p_inputs = {'input_ids': batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2]
                        }

            q_inputs = {'input_ids': batch[3],
                        'attention_mask': batch[4],
                        'token_type_ids': batch[5]}

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
            print(loss)

            loss.backward()
            optimizer.step()
            scheduler.step()
            q_model.zero_grad()
            p_model.zero_grad()
            global_step += 1

            torch.cuda.empty_cache()

    return p_model, q_model

def get_dense_embedding():
    pass

def train_encoder(
    tokenizer,
    model,
) -> NoReturn:
    

if __name__ == "__main__":
    main()
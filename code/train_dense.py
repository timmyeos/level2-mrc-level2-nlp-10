from transformers import (
    AutoTokenizer,
    BertModel,
    BertPreTrainedModel,
    AdamW,
    get_linear_schedule_with_warmup,
    TrainingArguments,
)
from transformers.models.roberta.modeling_roberta import (
    RobertaModel,
    RobertaPreTrainedModel,
)
from datasets import Dataset, concatenate_datasets, load_from_disk
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
import torch
from tqdm import tqdm, trange
import torch.nn.functional as F
import pickle
import os
import numpy as np
import json


class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):

        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        pooled_output = outputs[1]  # embedding을 뽑아내는 것

        return pooled_output


class RobertaEncoder(RobertaPreTrainedModel):
    def __init__(self, config):
        super(RobertaEncoder, self).__init__(config)

        self.roberta = RobertaModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):

        outputs = self.roberta(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        pooled_output = outputs[1]  # embedding을 뽑아내는 것

        return pooled_output


def prepare_dataset(retriever, tokenizer, inbatch):

    dataset = load_from_disk("../data/train_dataset")

    # train_dataset(train, validation 모두 합쳐 4192개)으로 encoder 훈련
    # full_dataset = concatenate_datasets(
    #     [
    #         dataset["train"].flatten_indices(),
    #         dataset["validation"].flatten_indices(),
    #     ]
    # )
    # print(full_dataset)
    dataset = dataset["train"]

    if inbatch == False:
        # sparse embedding -> df : 각 question에 대해 topk passage의 결과를 담은 dataframe
        retriever.get_sparse_embedding()

        num_topk = 6  ### 수정가능
        df = retriever.retrieve(dataset, topk=num_topk)

        # negative sampling : context(passage_list;TF-IDF의 값이 높은 passage)에서 정답을 포함하지 않는 passage를 구하여 context값으로 지정
        num_p_with_negs = 16  ### 수정 가능
        p_with_negs = []
        corpus = np.array(list(set([ex for ex in dataset["context"]])))

        for idx in range(len(df)):
            p_with_neg = []
            p_with_neg.append(df.loc[idx]["original_context"])  # ground truth
            for context in df.loc[idx]["context"]:  # topk passages
                if not df.loc[idx]["answers"]["text"][0] in context:
                    p_with_neg.append(context)
                if len(p_with_neg) == num_p_with_negs:
                    break
            if len(p_with_neg) < num_p_with_negs:
                # select random neagative sample
                while True:
                    neg_idxs = np.random.randint(
                        len(corpus), size=num_p_with_negs - len(p_with_neg)
                    )
                    if not df.loc[idx]["original_context"] in corpus[neg_idxs]:
                        p_neg = corpus[neg_idxs]
                        p_with_neg.extend(p_neg)
                        break
            p_with_negs.extend(p_with_neg)
        print(
            f"prepare negative samples (n_context:{len(df)} * num_p_with_negs:{num_p_with_negs}) = {len(p_with_negs)}"
        )

        # 1. (Question, Passage) 데이터셋 만들어주기
        q_seqs = tokenizer(
            list(df["question"]),
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        p_seqs = tokenizer(
            p_with_negs, padding="max_length", truncation=True, return_tensors="pt"
        )

        max_len = p_seqs["input_ids"].size(-1)
        p_seqs["input_ids"] = p_seqs["input_ids"].view(-1, num_p_with_negs, max_len)
        p_seqs["attention_mask"] = p_seqs["attention_mask"].view(
            -1, num_p_with_negs, max_len
        )
        p_seqs["token_type_ids"] = p_seqs["token_type_ids"].view(
            -1, num_p_with_negs, max_len
        )

        print("q_seqs size:", q_seqs["input_ids"].size())
        print("p_seqs size:", p_seqs["input_ids"].size())

        # 2. Tensor dataset
        train_dataset = TensorDataset(
            p_seqs["input_ids"],
            p_seqs["attention_mask"],
            p_seqs["token_type_ids"],
            q_seqs["input_ids"],
            q_seqs["attention_mask"],
            q_seqs["token_type_ids"],
        )

        return train_dataset

    else:  # inbatch
        # 1. (Question, Passage) 데이터셋 만들어주기
        q_seqs = tokenizer(
            dataset["question"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        p_seqs = tokenizer(
            dataset["context"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # 2. Tensor dataset
        train_dataset = TensorDataset(
            p_seqs["input_ids"],
            p_seqs["attention_mask"],
            p_seqs["token_type_ids"],
            q_seqs["input_ids"],
            q_seqs["attention_mask"],
            q_seqs["token_type_ids"],
        )

        return train_dataset


def train(args, train_dataset, p_model, q_model, num_p_with_negs):

    # Dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.per_device_train_batch_size,
    )

    # Optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in p_model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in p_model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
        {
            "params": [
                p
                for n, p in q_model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in q_model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    t_total = (
        len(train_dataloader)
        // args.gradient_accumulation_steps
        * args.num_train_epochs
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Start training!
    global_step = 0

    p_model.zero_grad()
    q_model.zero_grad()
    torch.cuda.empty_cache()

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")

    # config={"epochs": args.num_train_epochs, "batch_size": args.per_device_train_batch_size, "learning_rate" : args.learning_rate}
    # wandb.init(project="MRCProject", config=config, name="train_encoder_8b_5e")
    for num_epochs in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        print(num_epochs)
        loss_value = 0
        matches = 0
        for step, batch in enumerate(epoch_iterator):
            p_model.train()
            p_model.train()

            targets = torch.zeros(args.per_device_train_batch_size).long()
            if torch.cuda.is_available():
                batch = tuple(t.cuda() for t in batch)
                targets = targets.cuda()

            p_inputs = {
                "input_ids": batch[0].view(
                    args.per_device_train_batch_size * (num_p_with_negs), -1
                ),
                "attention_mask": batch[1].view(
                    args.per_device_train_batch_size * (num_p_with_negs), -1
                ),
                "token_type_ids": batch[2].view(
                    args.per_device_train_batch_size * (num_p_with_negs), -1
                ),
            }

            q_inputs = {
                "input_ids": batch[3],
                "attention_mask": batch[4],
                "token_type_ids": batch[5],
            }

            p_outputs = p_model(**p_inputs)  # (batch_size*(num_p_with_negs), emb_dim)
            q_outputs = q_model(**q_inputs)  # (batch_size*, emb_dim)

            # Calculate similarity score & loss
            p_outputs = torch.transpose(
                p_outputs.view(args.per_device_train_batch_size, num_p_with_negs, -1),
                1,
                2,
            )
            q_outputs = q_outputs.view(args.per_device_train_batch_size, 1, -1)

            sim_scores = torch.bmm(
                q_outputs, p_outputs
            ).squeeze()  # (batch_size, num_p_with_negs)
            sim_scores = sim_scores.view(args.per_device_train_batch_size, -1)
            sim_scores = F.log_softmax(sim_scores, dim=1)
            preds = torch.argmax(sim_scores, dim=-1)

            loss = F.nll_loss(sim_scores, targets)
            loss_value += loss
            matches += (preds == targets).sum()

            loss.backward()
            optimizer.step()
            scheduler.step()
            q_model.zero_grad()
            p_model.zero_grad()
            global_step += 1

            torch.cuda.empty_cache()

        # 학습된 모델 저장하기
        MODEL_PATH = "./models"
        torch.save(p_model, os.path.join(MODEL_PATH, f"p_encoder{num_epochs}.pt"))
        torch.save(q_model, os.path.join(MODEL_PATH, f"q_encoder{num_epochs}.pt"))
        print("model_saved")
        train_loss = loss_value / len(epoch_iterator)
        train_acc = matches / len(train_dataset)
        print(
            f"Epoch {num_epochs} || training loss {train_loss:4.4} || training accuracy {train_acc:4.2%}"
        )
        # wandb.log({'epoch' : num_epochs, 'training accuracy':  train_acc, 'training loss': train_loss})
        # valid_epoch(q_model, p_model, valid_dataset, args.per_device_eval_batch_size, num_p_with_negs, num_epochs)

    return p_model, q_model


def train_inbatch(args, train_dataset, p_model, q_model):

    # Dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.per_device_train_batch_size,
    )

    # Optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in p_model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in p_model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
        {
            "params": [
                p
                for n, p in q_model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in q_model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    t_total = (
        len(train_dataloader)
        // args.gradient_accumulation_steps
        * args.num_train_epochs
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Start training!
    global_step = 0

    p_model.zero_grad()
    q_model.zero_grad()
    torch.cuda.empty_cache()

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")

    for num_epochs in train_iterator:

        with tqdm(train_dataloader, unit="batch") as tepoch:
            for idx, batch in enumerate(tepoch):
                if torch.cuda.is_available():
                    batch = tuple(t.cuda() for t in batch)
                p_model.train()
                q_model.train()

                p_inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }

                q_inputs = {
                    "input_ids": batch[3],
                    "attention_mask": batch[4],
                    "token_type_ids": batch[5],
                }

                p_outputs = p_model(**p_inputs)  # (batch_size, emb_dim)
                q_outputs = q_model(**q_inputs)  # (batch_size, emb_dim)

                # target position : diagonal
                targets = torch.arange(0, args.per_device_train_batch_size).long()
                if torch.cuda.is_available():
                    targets = targets.to("cuda")

                # Calculate similarity score & loss
                sim_scores = torch.matmul(
                    q_outputs, torch.transpose(p_outputs, 0, 1)
                )  # (batch_size, batch_size)
                sim_scores = F.log_softmax(sim_scores, dim=1)
                loss = F.nll_loss(sim_scores, targets)
                if idx % 1900 == 0:
                    print(f"training loss : {loss:.4f}")

                loss.backward()
                # optimizer.zero_grad()
                optimizer.step()
                scheduler.step()
                p_model.zero_grad()
                q_model.zero_grad()
                global_step += 1
                torch.cuda.empty_cache()

                del p_inputs, q_inputs

        MODEL_PATH = "./models"
        torch.save(
            p_model, os.path.join(MODEL_PATH, f"p_encoder_inbatch{num_epochs}.pt")
        )
        torch.save(
            q_model, os.path.join(MODEL_PATH, f"q_encoder_inbatch{num_epochs}.pt")
        )

    return p_model, q_model


def run_dpr(retriever, context, tokenizer, inbatch):
    # dense embedding만 새로 생성하고자 하는 경우
    q_encoder_name = f"q_encoder2.pt"  ### 수정 가능 - 저장된 encoder 중 원하는 모델로
    p_encoder_name = f"p_encoder2.pt"  ### 수정 가능 - 저장된 encoder 중 원하는 모델로
    q_model_path = os.path.join("./models", q_encoder_name)
    p_model_path = os.path.join("./models", p_encoder_name)

    if os.path.isfile(q_model_path):
        q_encoder = torch.load(q_model_path)
        p_encoder = torch.load(p_model_path)
    else:
        # load pre-trained model on cuda (if available)
        model_name = "klue/roberta-large"  ### 수정 가능 - model.args 받아와서 model_args.model_name_or_path 해도 됨
        p_encoder = BertEncoder.from_pretrained(model_name).cuda()
        q_encoder = BertEncoder.from_pretrained(model_name).cuda()
        # tokenizer = AutoTokenizer.from_pretrained(model_name) ### reader와 다른 모델 사용 시 주석 제거

        # model_dict = torch.load("./dense_encoder/encoder.pth")  # 모델 파라미터들은 다운받았던 거 활용
        # p_encoder.load_state_dict(model_dict['p_encoder'])
        # q_encoder.load_state_dict(model_dict['q_encoder'])

        # negative sampling한 dataset
        train_dataset = prepare_dataset(retriever, tokenizer, inbatch)

        args = TrainingArguments(
            output_dir="dense_retireval",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,  ###
            per_device_eval_batch_size=16,  ###
            num_train_epochs=20,  ### 수정 가능
            weight_decay=0.01,
            fp16=True,
        )

        # 학습
        # 현재 epoch 마다 encoder 모델이 저장됩니다. 마지막 encoder만 저장하고 싶으면 train 함수의 마지막을 수정해주세요.
        if inbatch == False:
            p_encoder, q_encoder = train(
                args, train_dataset, p_encoder, q_encoder, num_p_with_negs=16
            )
        else:
            p_encoder, q_encoder = train_inbatch(
                args, train_dataset, p_encoder, q_encoder
            )

    # dense embedding 결과
    print("make_dense_embedding")
    with torch.no_grad():
        p_encoder.eval()
        q_encoder.eval()

        p_embs = []
        for p in tqdm(context):
            p = tokenizer(
                p, padding="max_length", truncation=True, return_tensors="pt"
            ).to("cuda")
            p_emb = p_encoder(**p).to("cpu").numpy()
            p_embs.append(p_emb)
    p_embs = torch.Tensor(p_embs).squeeze()  # (num_passage, emb_dim)

    # dense embedding 결과 저장
    data_path = "../data/"
    if inbatch == False:
        pickle_name = f"dense_embedding.bin"
    else:
        pickle_name = f"dense_embedding_inbatch.bin"

    emd_path = os.path.join(data_path, pickle_name)
    with open(emd_path, "wb") as file:
        pickle.dump(p_embs, file)
    print("Dense Embedding pickle saved.")

    return q_encoder, p_embs

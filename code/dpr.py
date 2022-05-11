from tqdm import tqdm, trange
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AdamW,
    TrainingArguments,
    AutoModel,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from transformers.models.roberta.modeling_roberta import (
    RobertaModel,
    RobertaPreTrainedModel,
)


# def neg_sampling(dataset, wikiset, num_neg):
#     # dataset = np.array(dataset)
#     wikiset = np.array(wikiset)
#     p_with_neg = []

#     for c in tqdm(dataset):
#         # print(c)
#         while True:
#             neg_idxs = np.random.randint(len(wikiset), size=num_neg)
#             # print(wikiset[neg_idxs])
#             if c not in wikiset[neg_idxs]:
#                 p_neg = wikiset[neg_idxs]

#                 p_with_neg.append(c)
#                 p_with_neg.extend(p_neg)
#                 break
#     print("neg_sampling done")
#     return p_with_neg


def dataset_func(tokenizer, dataset, wikiset, num_neg):
    q_seqs = tokenizer(
        dataset["question"],
        padding="max_length",
        truncation=True,
        return_token_type_ids=False,
        return_tensors="pt",
    )
    # p_with_neg = neg_sampling(dataset["context"], wikiset, num_neg)
    p_seqs = tokenizer(
        dataset["context"],
        padding="max_length",
        truncation=True,
        return_token_type_ids=False,
        return_tensors="pt",
    )

    # max_len = p_seqs["input_ids"].size(-1)
    # p_seqs["input_ids"] = p_seqs["input_ids"].view(-1, num_neg + 1, max_len)
    # p_seqs["attention_mask"] = p_seqs["attention_mask"].view(-1, num_neg + 1, max_len)
    # p_seqs["token_type_ids"] = p_seqs["token_type_ids"].view(-1, num_neg + 1, max_len)

    print("DPR p_seqs size: ", p_seqs["input_ids"].size())

    train_dataset = TensorDataset(
        p_seqs["input_ids"],
        p_seqs["attention_mask"],
        # p_seqs["token_type_ids"],
        q_seqs["input_ids"],
        q_seqs["attention_mask"],
        # q_seqs["token_type_ids"],
    )

    return train_dataset


class RobertaEncoder(RobertaPreTrainedModel):
    def __init__(self, config):
        super(RobertaEncoder, self).__init__(config)

        self.bert = RobertaModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids
        )

        pooled_output = outputs[1]

        return pooled_output


args = TrainingArguments(
    output_dir="dense_retireval",
    evaluation_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    weight_decay=0.01,
    # fp16=True,
)


def train(args, num_neg, dataset, model_checkpoint):
    # p,q model
    p_model = RobertaEncoder.from_pretrained(model_checkpoint)
    q_model = RobertaEncoder.from_pretrained(model_checkpoint)

    if torch.cuda.is_available():
        p_model.cuda()
        q_model.cuda()

    # Dataloader
    train_sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(
        dataset, sampler=train_sampler, batch_size=args.per_device_train_batch_size
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

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")

        for step, batch in enumerate(epoch_iterator):
            q_model.train()
            p_model.train()

            if torch.cuda.is_available():
                batch = tuple(t.cuda() for t in batch)

            p_inputs = {"input_ids": batch[0], "attention_mask": batch[1]}

            q_inputs = {
                "input_ids": batch[2],
                "attention_mask": batch[3],
            }

            p_outputs = p_model(**p_inputs)  # (batch_size*(num_neg+1), emb_dim)
            q_outputs = q_model(**q_inputs)  # (batch_size*, emb_dim)

            # Calculate similarity score & loss
            sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))
            # (batch_size, emb_dim) x (emb_dim, batch_size) = (batch_size, batch_size)

            # target: position of positive samples = diagonal element
            targets = torch.arange(0, args.per_device_train_batch_size).long()
            if torch.cuda.is_available():
                targets = targets.to("cuda")

            sim_scores = F.log_softmax(sim_scores, dim=1)
            loss = F.nll_loss(sim_scores, targets)

            if global_step % 500 == 0:
                print(loss)

            loss.backward()
            optimizer.step()
            scheduler.step()
            q_model.zero_grad()
            p_model.zero_grad()
            global_step += 1

            torch.cuda.empty_cache()

    return p_model, q_model


def dpr(dataset_train, wikiset):
    model_checkpoint = "klue/roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint,
        # 'use_fast' argument를 True로 설정할 경우 rust로 구현된 tokenizer를 사용할 수 있습니다.
        # False로 설정할 경우 python으로 구현된 tokenizer를 사용할 수 있으며,
        # rust version이 비교적 속도가 빠릅니다.
        use_fast=True,
    )
    num_neg = 15

    train_dataset = dataset_func(tokenizer, dataset_train, wikiset, num_neg)
    p_encoder, q_encoder = train(args, num_neg, train_dataset, model_checkpoint)
    p_encoder.save_pretrained("./p_encoder")
    q_encoder.save_pretrained("./q_encoder")
    return p_encoder, q_encoder


def p_emd(p_encoder, wikiset):
    model_checkpoint = "klue/roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint,
        # 'use_fast' argument를 True로 설정할 경우 rust로 구현된 tokenizer를 사용할 수 있습니다.
        # False로 설정할 경우 python으로 구현된 tokenizer를 사용할 수 있으며,
        # rust version이 비교적 속도가 빠릅니다.
        use_fast=True,
    )
    with torch.no_grad():
        p_encoder.eval()

        p_embs = []
        for p in wikiset:
            p = tokenizer(
                p,
                padding="max_length",
                truncation=True,
                return_token_type_ids=False,
                return_tensors="pt",
            ).to("cuda")
            p_emb = p_encoder(**p).to("cpu").numpy()
            p_embs.append(p_emb)

    p_embs = torch.Tensor(p_embs).squeeze()  # (num_passage, emb_dim)
    print("p_embs size: ", p_embs.size())
    return p_embs


def q_emd(q_encoder, queries):
    model_checkpoint = "klue/roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint,
        # 'use_fast' argument를 True로 설정할 경우 rust로 구현된 tokenizer를 사용할 수 있습니다.
        # False로 설정할 경우 python으로 구현된 tokenizer를 사용할 수 있으며,
        # rust version이 비교적 속도가 빠릅니다.
        use_fast=True,
    )
    with torch.no_grad():
        q_encoder.eval()

        q_seqs_val = tokenizer(
            queries,
            padding="max_length",
            truncation=True,
            return_token_type_ids=False,
            return_tensors="pt",
        ).to("cuda")
        q_embs = q_encoder(**q_seqs_val).to("cpu")  # (num_query, emb_dim)
        print("q_embs size: ", q_embs.size())
    return q_embs

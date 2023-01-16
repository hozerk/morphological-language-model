from pathlib import Path

from tqdm import tqdm
from transformers import RobertaTokenizer
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
import torch
from transformers import AdamW
from datasets import load_dataset
import warnings
import psutil
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForMaskedLM
from transformers import TrainingArguments
from transformers import Trainer




data_path_log = 'BertLM'
#PATH = data_path_log+'/bertMorphtrain'+str(1)+'loss.pt'
PATH = '/bertMorphtrain'+str(1)+'loss.pt'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

''' 
paths = [str(x) for x in Path(data_path_log).glob('**/*.txt')]

from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(files=paths[:5],vocab_size=30_522,  min_frequency=2,
                special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'])
tokenizer.save_model(data_path_log)'''

def main():
    config = RobertaConfig(
        vocab_size=16000,  # we align this to the tokenizer vocab_size
        max_position_embeddings=514,
        hidden_size=512,
        num_attention_heads=8,
        num_hidden_layers=6,
        type_vocab_size=1
    )
    model = RobertaForMaskedLM(config)
    model.to(device)
    tokenizer = RobertaTokenizer.from_pretrained(data_path_log, max_len=512)
    dataset = load_dataset("csv",
                           data_files="LMModelDenemeBigTrainALLcsv8.csv")  # ,streaming=True
    dataset_eval = load_dataset("csv",
                           data_files="LMModelDenemeBigTrainALLcsv8test.csv")  # ,streaming=True

    tokenized_dataset = dataset.map(
        lambda x: tokenizer(x["updated"], max_length=512, padding='max_length', truncation=True))
    tokenized_dataset_eval = dataset_eval.map(
        lambda x: tokenizer(x["updated"], max_length=512, padding='max_length', truncation=True))
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_eval_batch_size=32,
        evaluation_strategy="steps",
        eval_steps=5_000,
        learning_rate=2e-5,
        num_train_epochs=40,
        weight_decay=0.01,
        per_device_train_batch_size=32, 
        gradient_accumulation_steps=4,
        ignore_data_skip=True

    )
    data = tokenized_dataset.with_format("torch")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset_eval["train"],
        # eval_dataset=lm_dataset["test"],
        data_collator=data_collator
    )
    trainer.train("./results/checkpoint-198000")


if __name__ == "__main__":
    
    main()

'''print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
print(f"Number of files in dataset : {dataset['train'].dataset_size}")
size_gb = dataset['train'].dataset_size / (1024**3)
print(f"Dataset size (cache file) : {size_gb:.2f} GB")'''






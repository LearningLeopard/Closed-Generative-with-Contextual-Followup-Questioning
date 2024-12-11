# %%
from datasets import load_from_disk
from transformers import T5ForConditionalGeneration, T5Tokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
import matplotlib.pyplot as plt

# %%
CHECKPOINT = "google/flan-t5-base"
DATASET_PATH = "../Datasets/Visa_QA_V3/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
## Tokenization code
checkpoint_tokenizer = T5Tokenizer.from_pretrained(CHECKPOINT, return_tensors='pt')
def tokenize_and_create_prompt(sample):
    question_string = sample['question']
    tags_meta_data = sample['meta_tags'].split(",")
    prompt = f"You are an expert in dealing with questions of immigration and international travel. Answer the following question with proper rationale and use the keywords to get some documents from memory about the answer: Keywords: {', '.join(tags_meta_data)} Question: {question_string}"
    tokenized_question = checkpoint_tokenizer(prompt, max_length=512, truncation=True, padding="max_length")
    tokenized_answer = checkpoint_tokenizer(sample['answer'], max_length=512, truncation=True, padding="max_length")
    return {
        "input_ids": tokenized_question.input_ids,
        "attention_mask": tokenized_question.attention_mask,
        'labels': tokenized_answer.input_ids
    }

# %%
visa_qa_dataset = load_from_disk(DATASET_PATH)
preprocessed_visa_questions = visa_qa_dataset.map(tokenize_and_create_prompt)


# %%
preprocessed_visa_questions

# %%
preprocessed_visa_questions.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
preprocessed_visa_questions= preprocessed_visa_questions.select_columns(['input_ids', 'attention_mask', 'labels'])
preprocessed_visa_questions

# %%
qa_model = T5ForConditionalGeneration.from_pretrained(CHECKPOINT).to(device)


# %%
ques_data_collator = DataCollatorForSeq2Seq(tokenizer = checkpoint_tokenizer, model=qa_model, label_pad_token_id=-100)
train_args = Seq2SeqTrainingArguments(
    output_dir="../Model_Checkpoints/closed-generative-qa-FLAN-T5",
    num_train_epochs=30,
    warmup_steps=500,
    weight_decay=0.01,
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    gradient_accumulation_steps=16,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    logging_dir="../.logs"
)

trainer = Seq2SeqTrainer(
    model=qa_model,
    args=train_args,
    data_collator=ques_data_collator,
    train_dataset=preprocessed_visa_questions['train'],
    eval_dataset=preprocessed_visa_questions['validation'],
    tokenizer=checkpoint_tokenizer,
)

trainer.train()

train_losses = []
eval_losses = []

for log in trainer.state.log_history:
    if 'loss' in log:
        train_losses.append(log['loss'])
    if 'eval_loss' in log:
        eval_losses.append(log['eval_loss'])

# Plotting the Training and Validation Loss
plt.figure(figsize=[6, 6])
plt.plot(train_losses)
plt.plot(eval_losses, "--")
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend(['train', 'val'])
plt.title('Fine Tuning Training and Validation Losses for FLAN T5 base')
plt.show()
plt.savefig('../Graph_outputs/QA_FLAN_T5_pretrained.png', bbox_inches='tight')
plt.close()

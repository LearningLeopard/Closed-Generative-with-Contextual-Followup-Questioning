# %%
import torch
from transformers import AutoModelWithLMHead, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration
from datasets import load_from_disk
import evaluate

# %%
# CHECKPOINT = "../Model_Checkpoints/closed-generative-qa/checkpoint-156" # Don't forget to replace this with the best checkpoint
CHECKPOINT = "../Model_Checkpoints/closed-generative-qa-FLAN-T5/checkpoint-390"
# CHECKPOINT = "google/flan-t5-large"

# %%
visa_qa_testing = load_from_disk("../Datasets/Visa_QA_V4/")["test"]

# %%
tokenizer = T5Tokenizer.from_pretrained(CHECKPOINT)
model = T5ForConditionalGeneration.from_pretrained(CHECKPOINT)
def generate_output(data_point):
    question = data_point['question']
    prompt = f"You are an expert in dealing with questions of immigration and international travel. Answer the following question with proper reasoning and use the keywords to get some hints about the answer: Keywords: {', '.join(data_point['meta_tags'].split(','))} Question: {question}"
    encoded_input = tokenizer(prompt, return_tensors='pt')
    output = model.generate(**encoded_input)
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"Predicted Answer": "".join(output)}

# %%
prediction = visa_qa_testing.map(generate_output)

# %%
prediction

# %%
rogue = evaluate.load("rouge")
bleu = evaluate.load("bleu")
predictions = prediction['Predicted Answer']
ground_truth = prediction['answer']
print("====== ROUGE SCORE ======")
print(rogue.compute(predictions=predictions, references = ground_truth))
print("====== BLEU SCORE ======")
print(bleu.compute(predictions=predictions, references=ground_truth))

prediction.save_to_disk("../Datasets/Visa_prediction_FineTuned_FLANT5")

# %%
import torch
from transformers import AutoModelWithLMHead, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration
from datasets import load_from_disk
import evaluate
from ques_gen_chatgpt import generate_follow_up_question
from user_agent_simulation_chatgpt import simulate_user_response
import torch.nn as nn
from typing import Type
from torch import Tensor



# %%
class CorrectnessModuleLLM(nn.Module):
    def __init__(self: Type["CorrectnessModuleLLM"],
                 checkpoint: str) -> None:
        super(CorrectnessModuleLLM, self).__init__()
        bert_config = AutoConfig.from_pretrained(checkpoint, output_attention=True, output_hidden_states=False)
        self.embedding_body = AutoModel.from_pretrained(checkpoint, 
                                                        config=bert_config)
        self.dense_hidden_1 = nn.Linear(in_features=bert_config.hidden_size, 
                                        out_features=512,
                                        bias=True)
        self.dense_hidden_activation_1 = nn.Tanh()
        self.dense_hidden_2 = nn.Linear(in_features=512, 
                                        out_features=64,
                                        bias=True)
        self.dense_hidden_activation_2 = nn.ReLU()
        self.logit_transform = nn.Linear(in_features=64, 
                                         out_features=1,
                                         bias=True)
        return
    
    def forward(self: Type["CorrectnessModuleLLM"],
                input_ids: Tensor,
                attention_mask: Tensor,
                token_type_ids) -> Tensor:
        llm_embeddings = self.embedding_body(input_ids=input_ids,
                                             attention_mask=attention_mask,
                                             token_type_ids=token_type_ids)
        cls_token_output = llm_embeddings.last_hidden_state[:, 0, :]
        dense_intermediate_1 = self.dense_hidden_1(cls_token_output)
        dense_intermediate_activated_1 = self.dense_hidden_activation_1(dense_intermediate_1)
        dense_intermediate_2 = self.dense_hidden_2(dense_intermediate_activated_1)
        dense_intermediate_activated_2 = self.dense_hidden_activation_2(dense_intermediate_2)
        logits = self.logit_transform(dense_intermediate_activated_2)
        return logits

    def save_pretrained(self, save_directory: str) -> None:
        """
        Save the model's base transformer and custom layers.
        """
        os.makedirs(save_directory, exist_ok=True)
        # Save the transformer part
        self.embedding_body.save_pretrained(save_directory)
        # Save the custom layers
        torch.save(self.dense_hidden_1.state_dict(), f"{save_directory}/dense_hidden_1.pt")
        torch.save(self.dense_hidden_2.state_dict(), f"{save_directory}/dense_hidden_2.pt")
        torch.save(self.logit_transform.state_dict(), f"{save_directory}/logit_transform.pt")
        # Save the activations (optional, if needed for any configuration metadata)
        # Save any additional model configuration (if applicable)
        config = {"hidden_activation_1": "tanh", "hidden_activation_1": "ReLU", "output_activation": "relu"}  
        with open(f"{save_directory}/custom_config.json", "w") as f:
            json.dump(config, f)

    @classmethod
    def from_pretrained(cls, save_directory: str, *model_args, **kwargs) -> Type["CorrectnessModuleLLM"]:
        """
        Load the model's base transformer and custom layers.
        """
        # Load transformer configuration
        config = AutoConfig.from_pretrained(save_directory)
        # Initialize the model
        model = cls(checkpoint=save_directory)
        # Load transformer weights
        model.embedding_body = AutoModel.from_pretrained(save_directory, config=config)
        # Load custom layer weights
        model.dense_hidden_1.load_state_dict(torch.load(f"{save_directory}/dense_hidden_1.pt"))
        model.dense_hidden_2.load_state_dict(torch.load(f"{save_directory}/dense_hidden_2.pt"))
        model.logit_transform.load_state_dict(torch.load(f"{save_directory}/logit_transform.pt"))
        return model


# %%
# CHECKPOINT = "../Model_Checkpoints/closed-generative-qa/checkpoint-156" # Don't forget to replace this with the best checkpoint
QA_CHECKPOINT = "../Model_Checkpoints/closed-generative-qa-FLAN-T5/checkpoint-390"
CORRECTNESS_MODULE_CHECKPOINT = "../Model_Checkpoints/correctness-module-checkpoints-bert-uncased/bert-correctness-batch_32_lr_cosine_5e6_20"

# %%
visa_qa_testing = load_from_disk("../Datasets/Visa_QA_V4/")["test"].select(range(3))

# %%
qa_model_tokenizer = T5Tokenizer.from_pretrained(QA_CHECKPOINT, return_tensors='pt')
correctness_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased-whole-word-masking-finetuned-squad", return_tensors='pt')

# %%
qa_model = T5ForConditionalGeneration.from_pretrained(QA_CHECKPOINT)
correctness_model = CorrectnessModuleLLM(checkpoint=CORRECTNESS_MODULE_CHECKPOINT).from_pretrained(CORRECTNESS_MODULE_CHECKPOINT)

# %%
def generate_qa_model_answer(question, meta_tags, context=""):
    base_prompt = "You are an expert in dealing with questions of immigration and international travel. Answer the following question with proper reasoning"
    if len(context) == 0:
        prompt = f"{base_prompt}. Use the keywords to get some hints about the answer: Keywords: {', '.join(meta_tags.split(','))} Question: {question}"
    else:
        prompt = f"{base_prompt}. Take hints from the provided context and give a complete personalized answer: Context: {context}, Question: {question}"
    encoded_input = qa_model_tokenizer(prompt, return_tensors='pt')
    output = qa_model.generate(**encoded_input)
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    return "".join(output)

# %%
def get_confidence_level(question, answer):
    question_string = sample['question']
    answer_string = sample["answer"]
    final_input_string = f"{question_string} [SEP] {answer_string}"
    tokenized_inputs = correctness_tokenizer(final_input_string, add_special_tokens=True)
    output_score = correctness_model.forward(**tokenized_inputs).item()
    return output_score

# %%
MAX_STEPS = 5
CONFIDENCE_THRESHOLD = 25
def generate_answer(row):
    question = row['question']
    tags = row['meta_tags']
    init_ans = generate_qa_model_answer(question, tags)
    answers_history = [init_ans]
    confidence = get_confidence_level(question, init_ans)
    steps = 0
    while steps < MAX_STEPS and confidence < CONFIDENCE_THRESHOLD:
        new_question = generate_follow_up_question(question, answers_history[-1])
        followup_ans = simulate_user_response(new_question, question)
        new_ans = generate_qa_model_answer(question, tags, answers_history[-1])
        confidence = get_confidence_level(question, new_ans)

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



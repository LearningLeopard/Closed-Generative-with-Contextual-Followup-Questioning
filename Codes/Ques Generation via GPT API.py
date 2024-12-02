import openai
import os

# Set the API key (either through environment variable or directly)
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure that the API key is set in your environment or set directly in the script.

# Function to generate a follow-up question based on the provided question and answer
def generate_follow_up_question(question, answer):
    # Polite, empathetic prompt to gather user information for more accurate answers
    prompt = f"Question: {question}\nAnswer: {answer}\nCould you please ask a follow-up question that helps gather more information about the user's situation? The goal is to understand the user's circumstances better to provide a more personalized and accurate response."
    
    try:
        # Use the OpenAI API to generate the follow-up question
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # You can use "gpt-4" if you have access
            messages=[{"role": "system", "content": "You are a helpful assistant who asks polite, open-ended questions to gather more details about the user's situation."},
                      {"role": "user", "content": prompt}],
            max_tokens=100
        )
        
        # Extract and return the generated follow-up question
        follow_up_question = response['choices'][0]['message']['content'].strip()
        return follow_up_question
    
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

# Example usage
question = "What is the process for DACA?"
answer = "The process for DACA involves filing Form I-821D for deferred action."
follow_up_question = generate_follow_up_question(question, answer)

print("Generated Follow-Up Question:", follow_up_question)


### Prompts Version 2
'''
{"role": "system", "content": "You are a helpful assistant who is well versed with the policies and regulations related to immigration and international travel. You ask polite, open-ended follow-up questions to gather more details about the user's situation. Generate a follow-up question for the user, given the user query and a incomplete answer. The goal is to understand the user's circumstances better to provide a more personalized and accurate response."},
 {"role": "user", "content": prompt}
 prompt = f"Question: {question}\nAnswer: {answer}"
'''

'''
import openai
import os

# Set the API key (either through environment variable or directly)
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure that the API key is set in your environment or set directly in the script.

# Function to generate a follow-up question based on the provided question and answer
def generate_follow_up_question(question, answer):
    prompt = f"Question: {question}\nAnswer: {answer}\nGenerate a follow-up question related to the answer."
    
    try:
        # Use the new method for OpenAI API call
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # You can use "gpt-4" if you have access
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}],
            max_tokens=100
        )
        
        # Extract and return the generated follow-up question
        follow_up_question = response['choices'][0]['message']['content'].strip()
        return follow_up_question
    
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

# Example usage
question = "What is the process for DACA?"
answer = "The process for DACA involves filing Form I-821D for deferred action."
follow_up_question = generate_follow_up_question(question, answer)

print("Generated Follow-Up Question:", follow_up_question)

#pip install openai

import openai

# Set up OpenAI API key (replace YOUR_API_KEY with your actual API key)
openai.api_key = "sk-p"

# Function to generate follow-up question using GPT-3
def generate_follow_up_question(question, answer):
    # Prepare the prompt to feed into GPT-3
    prompt = f"""
    Here is a question-answer pair:
    Question: {question}
    Answer: {answer}
    Please generate a relevant follow-up question to ask the user for more details or clarification.
    """
    
    # Make the API request to OpenAI's GPT-3 model
    response = openai.Completion.create(
        engine="text-davinci-003",  # You can also use "gpt-4" if you have access
        prompt=prompt,
        max_tokens=100,  # Maximum number of tokens for the follow-up question
        n=1,  # Generate one completion
        stop=None,  # Let the model decide where to stop
        temperature=0.7  # Controls randomness in the output (0.7 is a balanced value)
    )
    
    # Extract and return the generated follow-up question
    follow_up_question = response.choices[0].text.strip()
    return follow_up_question

# Example usage
question = "What is the process for DACA?"
answer = "The process for DACA involves filing Form I-821D for deferred action."
follow_up_question = generate_follow_up_question(question, answer)

print("Generated Follow-Up Question:", follow_up_question)'''


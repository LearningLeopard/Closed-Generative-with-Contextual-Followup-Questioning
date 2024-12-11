import openai
import os

# Set the API key (either through environment variable or directly)
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure that the API key is set in your environment or set directly in the script.
openai.api_key = "<OPENAI-API-KEY>"

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
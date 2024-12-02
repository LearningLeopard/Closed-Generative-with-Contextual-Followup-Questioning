import openai
import os
import random

# Set the API key 
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure that the API key is set in your environment or set directly in the script.

# List of random countries with states 
countries_with_states = {
    "United States": ["California", "Texas", "New York", "Florida", "Illinois", "Pennsylvania", "Ohio"],
    "Canada": ["Ontario", "Quebec", "British Columbia", "Alberta", "Manitoba", "Saskatchewan"],
    "Mexico": ["CDMX", "Jalisco", "Nuevo Leon", "Chihuahua", "Yucatan", "Guanajuato"],
    "India": ["Uttar Pradesh", "Maharashtra", "Bengaluru", "Delhi", "Tamil Nadu", "Kerala"],
    "Germany": ["Bavaria", "Berlin", "North Rhine-Westphalia", "Hamburg", "Hesse", "Saxony"],
    "Brazil": ["Sao Paulo", "Rio de Janeiro", "Bahia", "Minas Gerais", "Parana"],
    "Australia": ["New South Wales", "Victoria", "Queensland", "Western Australia", "South Australia"],
    "France": ["Île-de-France", "Provence-Alpes-Côte d'Azur", "Nouvelle-Aquitaine", "Hauts-de-France"],
    "United Kingdom": ["England", "Scotland", "Wales", "Northern Ireland"],
    "Japan": ["Tokyo", "Osaka", "Hokkaido", "Kyoto", "Aichi"]
}

# List of random countries that don't have states (random)
countries_without_states = ["Argentina", "South Africa", "Russia", "Egypt", "Singapore"]

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
      
# Function to simulate a conversation with GPT as the agent (user responds directly to the follow-up question)
def simulate_user_response(follow_up_question, initial_question, answer):
    # Step 1: Adjust the follow-up question when the user is required to provide a country or location
    if "country" in follow_up_question.lower() or "state" in follow_up_question.lower():
        country = random.choice(list(countries_with_states.keys()) + countries_without_states)
        
        if country in countries_with_states:
            state = random.choice(countries_with_states[country])
            simulated_user_response = f"I am currently a citizen of {country}, residing in {state}. I have been living here for 3 years."
        else:
            simulated_user_response = f"I am currently a citizen of {country} and have been living here for 3 years."
    else:
        # Step 2: If the follow-up question is not about location, generate a response based on the context
        simulated_user_response = ""

        if "immigration status" in follow_up_question.lower():
            simulated_user_response = "I am currently a lawful permanent resident of the United States and have been residing here for 5 years."

        elif "challenges" in follow_up_question.lower():
            simulated_user_response = "I am facing challenges with the documentation required for my DACA application and need some clarification."

        else:
            simulated_user_response = "I have submitted my DACA application and am currently waiting for a response."

    # Ensure the response is concise and realistic
    if simulated_user_response.lower().startswith("sorry"):
        simulated_user_response = "I am currently residing in the United States and have been living here for 2 years."  # Default concise response for missing data
    
    print(f"Simulated User Response: {simulated_user_response}")

# Example usage with dynamic question-answer pairs
question = "What is the process for DACA?"
answer = "The process for DACA involves filing Form I-821D for deferred action."

# Step 1: Generate the follow-up question based on the provided question-answer pair
follow_up_question = generate_follow_up_question(question, answer)
print(f"Generated Follow-Up Question: {follow_up_question}")

# Step 2: Simulate the user response based on the generated follow-up question
simulate_user_response(follow_up_question, question, answer)

question2 = "How do I apply for citizenship?"
answer2 = "You need to fill out Form N-400 and submit it to USCIS."

# Step 1: Generate the follow-up question based on the provided question-answer pair
follow_up_question2 = generate_follow_up_question(question2, answer2)
print(f"Generated Follow-Up Question: {follow_up_question2}")

# Step 2: Simulate the user response based on the generated follow-up question
simulate_user_response(follow_up_question2, question2, answer2)

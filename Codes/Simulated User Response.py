import openai
import os
import random

# Set the API key 
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
      
# Function to simulate a conversation with GPT as the agent (user responds directly to the follow-up question)
def simulate_user_response(follow_up_question, initial_question, answer):
    # Step 1: Adjust the follow-up question when the user is required to provide a country or location
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

    #if "country" in follow_up_question.lower() or "state" in follow_up_question.lower():
       # country = random.choice(list(countries_with_states.keys()) + countries_without_states)

    # Other user scenarios (work, education, financial status)
    work_status = ["Employed full-time", "Part-time worker", "Self-employed", "Looking for work", "Retired"]
    education_level = ["High school graduate", "Bachelor's degree", "Master's degree", "Doctorate", "Some college", "No formal education"]
    income_status = ["Less than $30,000", "$30,000 - $60,000", "$60,000 - $100,000", "Over $100,000"]
    
    # Randomly select a country and state
    country = random.choice(list(countries_with_states.keys()) + countries_without_states)
    state = random.choice(countries_with_states.get(country, ["Unknown"]))
    
    # Randomly choose other details for the user
    user_work_status = random.choice(work_status)
    user_education = random.choice(education_level)
    user_income = random.choice(income_status)

    prompt = f"Simulate a user's response to the following follow-up question: '{follow_up_question}'. The user is currently living in {country}, specifically in {state}. Their work status is: {user_work_status}. They have the following education level: {user_education}. Their income is approximately: {user_income}."
    
    try:
        # Use the OpenAI API to generate the simulated user response
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # You can use "gpt-4" if you have access
            messages=[{"role": "system", "content": "You are a simulated user providing a concise and relevant response to a follow-up question."},
                      {"role": "user", "content": prompt}],
            max_tokens=100
        )
        
        # Extract and return the generated simulated user response
        user_response = response['choices'][0]['message']['content'].strip()
        print(user_response)
        return user_response

    except Exception as e:
        print(f"Error occurred: {e}")
        return None
        
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

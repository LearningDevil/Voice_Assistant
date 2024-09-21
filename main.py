import speech_recognition as sr
import webbrowser
import pyttsx3
import pywhatkit as kit
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

r = sr.Recognizer()
engine = pyttsx3.init()
# newsapi = "API from nwesapi website"
# Load the Hugging Face conversational model (DialoGPT)
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

def speak(text):
    engine.say(text)
    engine.runAndWait()

# def chat_with_ai(user_input):
#     # Tokenize user input
#     new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

#     # Generate a response from the model
#     bot_input_ids = new_input_ids
#     chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

#     # Decode the response
#     response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
#     return response

def chat_with_ai(user_input):
    # Tokenize user input
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Create an attention mask
    attention_mask = torch.ones(new_input_ids.shape, dtype=torch.long)

    # Generate a response from the model
    chat_history_ids = model.generate(new_input_ids, max_length=1000, attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id)

    # Decode the response
    response = tokenizer.decode(chat_history_ids[:, new_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response


def processCommand(c):
    print(c)
    if c.lower().startswith("open"):
        website = c.lower().split(" ")[1]
        webbrowser.open(f"https://{website}.com")
    elif c.lower().startswith("play"):
        song = " ".join(c.split(" ")[1:])
        kit.playonyt(song)
    # elif "news" in c.lower():
    #     r = requests.get(f"https://newsapi.org/v2/top-headlines?country=us&apiKey={newsapi}")
    #     if r.status_code == 200:
    #         # Parse the JSON response
    #         data = r.json()
    #         # Extract the headlines
    #         articles = data.get('articles', [])
    #         # Print out the top headlines
    #         for i, article in enumerate(articles, start=1):
    #             print(f"{i}. {article['title']}")
    #             speak(article['title'])
    #     else:
    #         print(f"Failed to fetch news. Status code: {r.status_code}")

    else:
         # AI Assistant for general conversation
        ai_response = chat_with_ai(c)
        print(f"AI: {ai_response}")
        speak(ai_response)

if __name__ == "__main__":
    speak("Initializing jarvis...")
    while True:
        print("Recognizing....")
        try:
            with sr.Microphone() as source:
                print("Listening....")
                r.adjust_for_ambient_noise(source, duration=0.5)  # Fine-tune for background noise
                audio = r.listen(source, timeout=3, phrase_time_limit=1)
            word = r.recognize_google(audio)
            print(word)
            if(word.lower() == "jarvis"):
                speak("How may i help you")
                # Listen for command
                with sr.Microphone() as source:
                    print("Waiting for your command.....")
                    r.adjust_for_ambient_noise(source, duration=0.5)
                    audio = r.listen(source, timeout=5, phrase_time_limit=5)
                    command = r.recognize_google(audio)

                    processCommand(command)
            elif word.lower() == "activate":
                ai_active = True
                speak("AI activated. You can talk to me.")
                while ai_active:
                    print("Listening for your command.....")
                    with sr.Microphone() as source:
                        r.adjust_for_ambient_noise(source, duration=0.5)
                        audio = r.listen(source, timeout=5, phrase_time_limit=5)
                        ai_command = r.recognize_google(audio)
                        if ai_command.lower() == "exit":
                            ai_active = False
                            speak("AI deactivated.")
                        else:
                            processCommand(ai_command)

        except Exception as e:
            print(f"Error: {e}")

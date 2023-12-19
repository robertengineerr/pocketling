from vosk import Model, KaldiRecognizer 
import pyaudio
import json

# Imports google translate
from googletrans import Translator

# Try gTTS again
from gtts import gTTS
import os

# Variables that I will need
fromLanguage = "en"
targetLanguge = "es"
integerForFile = 1

# Pass the absolute path by adding 'r'
model = Model(r"/Volumes/OTHER_OS/PocketLing/vosk_files/vosk-model-small-en-us-0.15")

# Recognize
recognizer = KaldiRecognizer(model, 16000)

# Use pyaudio to record
mic = pyaudio.PyAudio()

try:
    stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
    stream.start_stream()

except Exception as e:
    print(f"An error occurred while setting up the microphone: {e}")

# Make a loop to record
while True:
    try:
        # Listen for user input
        data = stream.read(4096)
        if recognizer.AcceptWaveform(data):
            # You have to use json here or it will read "text" in the output
            userInput = json.loads(recognizer.Result())["text"]
            print(userInput)
            
            # Declare a function
            tl = Translator()

            # Translate input
            translated_text = tl.translate(userInput, dest=targetLanguge)

            # Print to terminal
            print(translated_text.text)
            

            # Text-to-speech
            integerForFile = 1
            file = "file" + str(integerForFile) + ".mp3"
            tts = gTTS(translated_text.text, lang=fromLanguage)  # Set the language to the language of the original speaker
            tts.save(file)
            os.system("mpg123 " + file)
            integerForFile += 1
    
    #error handling:

    except ValueError as e:
        print(f"A ValueError occurred while processing your input: {e}")

    except KeyError as e:
        print(f"A KeyError occurred while processing your input: {e}")

    except Exception as e:
        print(f"An error occurred while processing your input: {e}")

    finally:
        # Clean up resources
        stream.stop_stream()
        stream.close()
        mic.terminate()

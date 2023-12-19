# Import the necessary libraries
from vosk import Model, KaldiRecognizer
import pyaudio
import json
import torch
import os
from gtts import gTTS

# Import the PyTorch modules for machine translation
import torchtext
from torchtext import TranslationDataset, Multi30k
from torchtext import Field, BucketIterator

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

# Set up the PyTorch fields for the machine translation model
SRC = Field(tokenize="spacy",
            tokenizer_language="de",
            init_token="<sos>",
            eos_token="<eos>",
            lower=True)

TRG = Field(tokenize="spacy",
            tokenizer_language="en",
            init_token="<sos>",
            eos_token="<eos>",
            lower=True)

# Load the data and build the vocabulary
train_data, valid_data, test_data = Multi30k.splits(exts=(".de", ".en"), fields=(SRC, TRG))
SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

# Set up the device and batch size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128

# Set up the iterator
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device)

# Define a function for translating a sentence using the PyTorch model
def translate_sentence(sentence, src_field, trg_field, model, device, max_length=50):
    model.eval()
    
    # Tokenize the input sentence
    tokens = src_field.tokenize(sentence)
    
    # Add the start of sentence and end of sentence tokens
    tokens = ["<sos>"] + tokens + ["<eos>"]
    
    # Convert the tokens to a numerical tensor
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.tensor(src_indexes).unsqueeze(1).to(device)
    
    # Initialize the translation
    trg_indexes = [trg_field.vocab.stoi["<sos>"]]
    
    # Create a mask over the input tensor to prevent attention over padding
    src_mask = src_tensor.eq(0).squeeze(1)
    
    # Iterate over the input sentence
    for i in range(max_length):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(1).to(device)
        with torch.no_grad():
            # Predict the next token
            output = model(src_tensor, src_mask, trg_tensor, 1)
            next_token = output.argmax(2)[-1, :]
        
        # Add the predicted token to the translation
        trg_indexes.append(next_token.item())
        
        # Stop translation if the end of sentence token is predicted
        if next_token == trg_field.vocab.stoi["<eos>"]:
            break
    
    # Convert the translation to a string of text
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    return " ".join(trg_tokens[1:])

# Make a loop to record
while True:
    try:
        # Listen for user input
        data = stream.read(4096)
        if recognizer.AcceptWaveform(data):
            # You have to use json here or it will read "text" in the output
            userInput = json.loads(recognizer.Result())["text"]
            print(userInput)
            
            # Use the PyTorch model to translate the input
            translated_text = translate_sentence(userInput, SRC, TRG, model, device, max_length=50)

            # Print to terminal
            print(translated_text)
            
            # Text-to-speech
            integerForFile = 1
            file = "file" + str(integerForFile) + ".mp3"
            tts = gTTS(translated_text, lang=fromLanguage)  # Set the language to the language of the original speaker
            tts.save(file)
            os.system("play " + file)  # Use the "play" program to play the audio file
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

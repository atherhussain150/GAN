import os
import librosa
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt



# Function to extract audio features (e.g., spectrograms)
def extract_audio_features(audio_file, sample_rate=22050):
    audio, _ = librosa.load(audio_file, sr=sample_rate)
    
    # Extract spectrogram using librosa
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Intensity (dB)')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title('Spectrogram')
    plt.show()
    return spectrogram

# Function to tokenize text transcripts
def tokenize_text(transcript):
    # Tokenize text using NLTK
    tokens = word_tokenize(transcript.lower())
    return tokens

# Path to the directory containing the audio files
audio_dir = "D://audio ai//data//LJSpeech-1.1//wavs"

# Path to the preprocessed metadata CSV file containing the transcript data
metadata_file = "D://audio ai//data//LJSpeech-1.1//metadata_preprocessed.csv"  # Replace with the actual filename

# Load the preprocessed metadata CSV file into a pandas DataFrame
df = pd.read_csv(metadata_file)

# List to store spectrograms and tokens
spectrograms = []
tokens_list = []

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    audio_file = os.path.join(audio_dir, row["ID"].strip()) + ".wav"
    
    # Extract audio features
    spectrogram = extract_audio_features(audio_file)
    spectrograms.append(spectrogram)
    
    # Tokenize text transcript
    tokens = tokenize_text(row["Transcript"])
    tokens_list.append(tokens)

print(f"spectogram: {spectrograms.sh}")
print(f"tokens: {tokens_list}")

# Convert spectrograms and tokens to numpy arrays
spectrograms = np.array(spectrograms)
tokens_list = np.array(tokens_list)

# Split the dataset into training and validation sets (e.g., 80% train, 20% validation)
train_size = int(0.8 * len(spectrograms))
train_spectrograms, val_spectrograms = spectrograms[:train_size], spectrograms[train_size:]
train_tokens, val_tokens = tokens_list[:train_size], tokens_list[train_size:]


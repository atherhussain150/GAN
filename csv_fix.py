import pandas as pd

import pandas as pd

def preprocess_transcript_csv(csv_file):
    transcripts = []

    with open(csv_file, 'r', encoding='utf-8') as file:  # Specify UTF-8 encoding
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace
            if "|" in line:  # If the line contains the separator "|"
                id_, *text_cells = line.split("|")
                # Start building the transcript for the current ID
                transcript = " ".join(cell.strip() for cell in text_cells if cell.strip())  # Concatenate text cells with text
                transcripts.append((id_, transcript))
            else:
                print(f"Invalid format in line: {line}")

    # Convert the list of transcripts to a DataFrame
    df = pd.DataFrame(transcripts, columns=["ID", "Transcript"])

    # Write the DataFrame to a new CSV file
    output_csv = csv_file.replace(".csv", "_preprocessed.csv")
    df.to_csv(output_csv, index=False)

    print(f"Preprocessed transcript data saved to {output_csv}")

# Path to the original CSV file containing the transcript data
csv_file = "D://audio ai//data//LJSpeech-1.1//metadata.csv"  # Replace with the actual path

# Preprocess the CSV file
preprocess_transcript_csv(csv_file)


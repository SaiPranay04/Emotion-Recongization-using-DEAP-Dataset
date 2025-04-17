import json
import random

def prepare_prompt_completion_pairs(verbal_reps, emotion_labels):
    """
    Pair each verbalized prompt with its corresponding emotion label.
    
    Parameters:
      verbal_reps (list): List of strings containing the verbal descriptions.
      emotion_labels (list): List of strings containing the emotion labels.
    
    Returns:
      pairs (list): List of dictionaries with keys "prompt" and "completion".
    """
    pairs = []
    for prompt, label in zip(verbal_reps, emotion_labels):
        pairs.append({"prompt": prompt, "completion": label})
    return pairs

if __name__ == "__main__":
    # Load verbal representations from file (one per line)
    verbal_reps_path = "D:/EEG/src/data/verbal_representations.txt"
    with open(verbal_reps_path, "r") as f:
        verbal_reps = [line.strip() for line in f if line.strip()]
    
    # Attempt to load emotion labels from a file; if not found, assign randomly for demonstration.
    emotion_labels_path = "D:/EEG/src/data/emotion_labels.txt"
    try:
        with open(emotion_labels_path, "r") as f:
            emotion_labels = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("emotion_labels.txt not found. Assigning random emotion classes for demonstration.")
        possible_labels = [
            "High Valence – High Arousal",
            "High Valence – Low Arousal",
            "Low Valence – High Arousal",
            "Low Valence – Low Arousal"
        ]
        emotion_labels = [random.choice(possible_labels) for _ in range(len(verbal_reps))]
    
    # Ensure the number of prompts and labels match.
    if len(verbal_reps) != len(emotion_labels):
        raise ValueError("Mismatch between number of verbal representations and emotion labels!")
    
    # Prepare the prompt-completion pairs.
    pairs = prepare_prompt_completion_pairs(verbal_reps, emotion_labels)
    
    # Save the pairs to a JSONL file (one JSON object per line)
    output_file = "D:/EEG/src/data/prompt_completion_pairs.jsonl"
    with open(output_file, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")
    
    print(f"Prepared {len(pairs)} prompt-completion pairs.")

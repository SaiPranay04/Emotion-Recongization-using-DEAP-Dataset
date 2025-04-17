
# 🧠 Emotion Recognition using DEAP Dataset

This project focuses on classifying human emotions using EEG data from the [DEAP dataset](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/). We implement both **traditional ML/DL models** and **LLM-based pipelines** for emotion recognition across four emotion dimensions:

- **Valence**
- **Arousal**
- **Dominance**
- **Liking**

---

## 📂 Dataset

Download the DEAP preprocessed `.dat` files from the link below and place them in:

```
data/data_preprocessed_python/
```

🔗 [Download DEAP Dataset](https://drive.google.com/drive/folders/REPLACE_WITH_YOUR_LINK)


---

## 🗂️ Folder Structure

```
.
├── extract_data.py                     # Combines .dat files into .npy arrays
├── Arousal/
│   ├── Arousal_SVM.py
│   ├── Arousal_CNN.py
│   ├── Arousal_LSTM.py
│   ├── Arousal_Attention.py
│   └── Arousal_Ensemble.py
├── Valence/                            # Same structure as Arousal
├── Dominance/                          # Same structure
├── Liking/                             # Same structure
└── LLM/
    ├── gpt.py                          # Segment EEG data into epochs
    ├── feature.py                      # Extract features from epochs
    ├── NLD.py                          # Convert features into verbal form
    ├── prompt.py                       # Generate prompt-completion JSONL
    ├── tuning.py                       # Fine-tune a causal language model
    └── eval.py                         # Evaluate LLM performance
```

---

## 🛠️ Installation

Install the required dependencies using pip:

```bash
pip install numpy scipy scikit-learn tensorflow torch transformers datasets
```

- TensorFlow: for CNN/LSTM models  
- PyTorch & Transformers: for LLM pipeline  
- Scikit-learn: for SVM and evaluation metrics

---

## 🚀 Getting Started

### 🔹 Step 1: Clone the Repo

```bash
git clone https://github.com/SaiPranay04/Emotion-Recongization-using-DEAP-Dataset.git
cd Emotion-Recongization-using-DEAP-Dataset
```

### 🔹 Step 2: Data Extraction

Run the script to extract and generate combined `.npy` files:

```bash
python extract_data.py
```

Generated files:

- `combined_data.npy` — EEG signal data  
- `combined_labels.npy` — Labels for [Valence, Arousal, Dominance, Liking]

---

## 🤖 Traditional Models

Each of the emotion folders contains 5 types of models:

- `*_SVM.py` — Support Vector Machine
- `*_CNN.py` — Convolutional Neural Network
- `*_LSTM.py` — Long Short-Term Memory
- `*_Attention.py` — Attention-based model
- `*_Ensemble.py` — Combines predictions from multiple models

### ▶️ Example Run

```bash
python Arousal/Arousal_CNN.py
```

> 📌 Make sure to update the `np.load(...)` paths inside each script to point to the correct location of your `.npy` files.

---

## 💬 LLM-based Classification Pipeline

This optional pipeline uses language models (e.g., GPT-style) by converting EEG features into descriptive text.

### 🔸 Step 1: Segment EEG Signals

```bash
python LLM/gpt.py
```

### 🔸 Step 2: Feature Extraction

```bash
python LLM/feature.py
```

### 🔸 Step 3: Convert to Verbal Form

```bash
python LLM/NLD.py
```

### 🔸 Step 4: Create Prompt–Label Pairs

```bash
python LLM/prompt.py
```

### 🔸 Step 5: Fine-Tune the Language Model

```bash
python LLM/tuning.py
```

### 🔸 Step 6: Evaluate Model

```bash
python LLM/eval.py
```

---

## ⚙️ Customization Tips

- 🛤️ **Paths**: Update `"D:/EEG/src/..."` paths inside scripts to your local structure.
- 🧪 **Experiment** with different hyperparameters like epochs, batch size, etc.
- 📦 Create a `requirements.txt` file using:

  ```bash
  pip freeze > requirements.txt
  ```

- 📊 Add TensorBoard, MLflow, or wandb to monitor training metrics.

---

## 📌 Future Ideas

- Add GUI or Streamlit app for demo.
- Implement real-time EEG input integration.
- Improve LLM summaries using RLHF techniques.

---

## 📬 Contact

Developed with ❤️ by **Sai Pranay** and **Sai Ganesh Reddy**

📧 Feel free to [open an issue](https://github.com/SaiPranay04/Emotion-Recongization-using-DEAP-Dataset/issues) for any bugs or suggestions.

---

## 🪪 License

This project is licensed under the **MIT License**.  
You’re free to use, modify, and distribute with attribution.

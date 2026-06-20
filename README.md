# SMS and Email Spam Classifier using PyTorch and GRU

A simple and effective Deep Learning project that classifies text messages (SMS) and emails as either **Ham (Normal)** or **Spam (Fraud/Advertisement)**. 

Instead of using traditional Machine Learning, this project uses a **Recurrent Neural Network (RNN)** variant called **GRU (Gated Recurrent Unit)** implemented in PyTorch to understand the sequence of words in a message.

---

## 🚀 Features
* **Custom Text Preprocessing:** Cleans, tokenizes, and pads text sequences to a fixed length of 50 words.
* **Vocabulary Generator:** Automatically builds a vocabulary mapping from scratch based on word frequency.
* **Deep Learning Pipeline:** Built using **PyTorch**, featuring an Embedding layer, a GRU layer, and a Linear Classifier.
* **Custom Inference:** Includes a function to test your own custom messages in real-time.

---

## 📊 Project Workflow

1. **Data Cleaning:** Lowercasing text and splitting messages into individual words.
2. **Vocabulary Building:** Creating a dictionary where each unique word gets a specific number ID (`<PAD>` is 0, `<UNK>` is 1).
3. **Padding & Truncating:** Making sure every single message is exactly 50 words long so the neural network can process them in batches.
4. **Train-Test Split:** Dividing the dataset into 80% for training the model and 20% for testing its accuracy.
5. **Model Training:** Training the GRU model using `DataLoader` batches, tracking Loss and Accuracy across 15 epochs.
6. **Inference:** Testing live custom text inputs to see if the model predicts them correctly.

---

## 🛠️ Model Architecture

The neural network is built with the following layers:
* **Embedding Layer:** Converts word token IDs into dense vectors of size 64.
* **GRU Layer:** Processes the sequence of word vectors and remembers the context of the sentence (Hidden dimension = 128).
* **Linear Layer:** A fully connected layer that maps the final hidden state to the 2 output classes (Ham vs Spam).

---

## 💻 Tech Stack Used
* **Language:** Python
* **Deep Learning Framework:** PyTorch
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning Utilities:** Scikit-Learn (for train-test split)

---

## 📈 Results
* **Training Framework:** Batch Size = 32 | Optimizer = Adam | Loss = CrossEntropyLoss
* The model successfully trains over 15 epochs, significantly reducing loss and achieving high accuracy on unseen test data.

---

## 🔮 How to Test Custom Messages
Once the model is trained, you can test any custom message using the built-in function:

```python
# Testing a normal message
predict_spam_or_ham("Hey friend, are we still meeting for lunch today?")
# Output: HAM (Normal) ✅

# Testing a spam message
predict_spam_or_ham("WINNER! You have won a free cash prize of 1000 dollars. Call now to claim!")
# Output: SPAM 🚨
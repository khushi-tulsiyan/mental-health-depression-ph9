import pandas as pd
import shap
import torch
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import joblib
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
from huggingface_hub import login
import warnings
from dotenv import load_dotenv


load_dotenv()
warnings.filterwarnings("ignore", category=FutureWarning)


token = os.getenv("Huggingface")

login(token)

bnb_config = BitsAndBytesConfig(load_in_8bit=True)

try:
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", model_max_length=512)
    llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map="auto", quantization_config = bnb_config)
except Exception as e:
    print("Llama 2 model unavailable. Falling back to Mistral-7B.")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", model_max_length=512)
    llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map="auto", quantization_config = bnb_config)


X_train = pd.read_csv("./data/processed/X_train.csv")
X_test = pd.read_csv("./data/processed/X_test.csv")
y_train = pd.read_csv("./data/processed/y_train.csv").values.ravel()
y_test = pd.read_csv("./data/processed/y_test.csv").values.ravel()

X_test = X_test[X_train.columns]

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', max_depth=6, random_state=42),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, alpha=0.01, random_state=42)
}


best_model = None
best_score = 0
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
    
    results[name] = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-score": f1, "ROC-AUC": roc_auc}
    
    print(f"{name} Performance:")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, ROC-AUC: {roc_auc:.4f}\n")
    
    if accuracy > best_score:
        best_model = model
        best_score = accuracy

joblib.dump(best_model, "best_model.pkl")
print(f"Best Model Saved: {best_model}")


if isinstance(best_model, (RandomForestClassifier, XGBClassifier)):
    explainer = shap.TreeExplainer(best_model)
else:
    explainer = shap.KernelExplainer(best_model.predict, shap.sample(X_train, 100))

shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)


def generate_llama_explanation(prediction):
    prompt = f"A user has been classified as having {prediction} depression severity. Provide a brief explanation and coping mechanisms."
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output = llama_model.generate(input_ids, max_length=150)
    return tokenizer.decode(output[0], skip_special_tokens=True)


# Example usage
example_prediction = "Moderate"
explanation = generate_llama_explanation(example_prediction)
print(f"Llama 2 Explanation for {example_prediction} Depression:\n{explanation}")

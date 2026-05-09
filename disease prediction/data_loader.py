import os
import requests
import pandas as pd

def download_file(url, local_path):
    if not os.path.exists(local_path):
        print(f"Downloading {local_path} from {url}...")
        response = requests.get(url)
        response.raise_for_status()  # Check that the request was successful
        with open(local_path, "wb") as f:
            f.write(response.content)
        print(f"Saved to {local_path}")
    else:
        print(f"{local_path} already exists. Skipping download.")

def load_data():
    dataset_dir = "dataset"
    os.makedirs(dataset_dir, exist_ok=True)
    
    training_url = "https://raw.githubusercontent.com/sohamvsonar/Disease-Prediction-and-Medical-Recommendation-System/main/dataset/Training.csv"
    training_path = os.path.join(dataset_dir, "Training.csv")
    
    download_file(training_url, training_path)
    
    df_train = pd.read_csv(training_path)
    
    return df_train

if __name__ == "__main__":
    df_train = load_data()
    print("Data shape:", df_train.shape)

import os
import torch
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, AdamW, AutoConfig
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Add the load_text_files and create_dataset functions here

def load_text_files(folder, event_int):
    """
    Load text files from a folder and return a list of dictionaries with the event and disclosure.
    """
    file_list = os.listdir(folder)
    text_data = []

    for file in file_list:
        with open(os.path.join(folder, file), 'r', encoding='utf-8') as f:
            content = f.read()
            text_data.append({"event": event_int, "disclosure": content})
    
    return text_data

def create_dataset(folders):
    """
    Create a dataset from a list of folders.
    """
    all_data = []
    # create a sorted list of folder names to create consistent label mappings
    folder_names = sorted(os.path.basename(folder) for folder in folders)
    # create the event mapping
    event_mapping = {name: i for i, name in enumerate(folder_names)}

    for folder in folders:
        event_int = event_mapping[os.path.basename(folder)]
        text_data = load_text_files(folder, event_int)
        all_data.extend(text_data)
    
    dataset = pd.DataFrame(all_data)

    # Separate features (X) and labels (y)
    X = dataset['disclosure']
    y = dataset['event']

    return X, y, event_mapping

class EventAnalysisDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx].values
        text = str(row[0])  # Ensure text is a string
        y = row[1]

        inputs = self.tokenizer(
            text, 
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding="max_length",
            add_special_tokens=True
            )

        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]

        return input_ids, attention_mask, y

def main():
    main_dataset_dir = './train_dataset'

    # get all the folders
    folders = [os.path.join(main_dataset_dir, folder_name) for folder_name in os.listdir(main_dataset_dir)]

    # Use the `create_dataset` function from the first script to create the dataset
    X, y, event_mapping = create_dataset(folders)

    num_classes = len(folders)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create test data as DataFrame
    test_data = pd.concat([X_test, y_test], axis=1)

    # Create the EventAnalysisDataset object for the test dataset
    test_dataset = EventAnalysisDataset(test_data)

    # Load the saved model with the weights
    config = AutoConfig.from_pretrained("monologg/koelectra-base-v3-discriminator", num_labels=num_classes)
    model = AutoModelForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator", config=config)

    # Load the weights from the 'weights' folder
    save_model_path = Path('./weights2')
    model.load_state_dict(torch.load(save_model_path / "koelectra-base-finetuned-sentiment-analysis_best_table_parse.bin"))
    #model.load_state_dict(torch.load(save_model_path / "koelectra-base-finetuned-sentiment-analysis3.bin"))
    model = model.to(device)

    # Create a DataLoader for the test dataset
    batch_size = 16
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Get the predictions
    y_pred = []
    y_true = []
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, 1)
            y_pred.extend(preds.tolist())
            y_true.extend(labels.tolist())

    # Generate classification report
    print(classification_report(y_true, y_pred, target_names=event_mapping.keys()))

if __name__ == "__main__":
    main()
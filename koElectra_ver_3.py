import os
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import logging

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, AutoConfig
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn.functional as F
import logging.handlers
import json

# Configure logging
log_filename = 'log_history.log'
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(),
                              logging.handlers.RotatingFileHandler(log_filename, maxBytes=1000000, backupCount=5)])

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
torch.set_num_threads(1)

def load_text_files(folder, event_int):
    """
    Load text files from a specific folder.

    :param folder: the path to the directory containing the text files
    :param event_int: the integer value representing the event (label)
    :return: list of dictionaries, each representing one text file with the event and its content
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

    :param folders: the list of folders paths
    :return: a tuple containing the feature matrix X, target vector y, and the event mapping dictionary
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
        """
        Initialization of EventAnalysisDataset class.

        :param dataset: a DataFrame containing the dataset
        """
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

    def __len__(self):
        """
        :return: the number of samples in the dataset
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        :param idx: the index of the sample to fetch
        :return: the input_ids, attention_mask, and the label of the sample
        """
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

def train(model, train_loader, optimizer, device):
    """
    Train a model for one epoch.

    :param model: the model to be trained
    :param train_loader: DataLoader for the training data
    :param optimizer: the optimizer to use
    :param device: the device to use for training ('cuda' or 'cpu')
    :return: average loss and accuracy for this epoch
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    batches = 0

    for input_ids_batch, attention_masks_batch, y_batch in tqdm(train_loader):
        optimizer.zero_grad()
        y_batch = y_batch.to(device)
        y_pred = model(input_ids_batch.to(device), attention_mask=attention_masks_batch.to(device))[0]
        loss = F.cross_entropy(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(y_pred, 1)
        correct += (predicted == y_batch).sum()
        total += len(y_batch)

        batches += 1
        if batches % 100 == 0:
            avg_loss = total_loss / batches
            print(f"Batch: {batches}, Average Loss: {avg_loss}, Accuracy: {correct.float() / total}")
            logger.info(f"Batch {batches} - Average Loss: {avg_loss}, Accuracy: {correct.float() / total}")
    
    return total_loss / batches, correct.float() / total

def evaluate(model, test_loader, device):
    """
    Evaluate the model's performance on a validation/test set.

    :param model: the model to be evaluated
    :param test_loader: DataLoader for the test data
    :param device: the device to use for evaluation ('cuda' or 'cpu')
    :return: accuracy on the test set
    """
    model.eval()
    test_correct = 0
    test_total = 0

    for input_ids_batch, attention_masks_batch, y_batch in test_loader:
        y_batch = y_batch.to(device)
        y_pred = model(input_ids_batch.to(device), attention_mask=attention_masks_batch.to(device))[0]
        _, predicted = torch.max(y_pred, 1)
        test_correct += (predicted == y_batch).sum()
        test_total += len(y_batch)

    return test_correct.float() / test_total

def main():
    """
    The main function that orchestrates the creation of the dataset, training, evaluation, and saving the model.
    - The function first collects all the directories in the main dataset directory.
    - It then creates the dataset, and splits it into training and testing datasets.
    - Next, it creates DataFrames for both training and testing data.
    - It defines the path where the model should be saved and creates the directory if it doesn't exist.
    - It then prepares the model for training, including setting the tokenizer, loading the pre-trained model, and setting up the optimizer.
    - The training loop is executed over a specified number of epochs, and the model is evaluated at the end of each epoch.
    - If the model's performance improves, it is saved.
    - Finally, the function logs the model's performance over the epochs and evaluates it on the testing set. The model is then saved one last time.
    """

    main_dataset_dir = './train_dataset'

    # get all the folders
    folders = [os.path.join(main_dataset_dir, folder_name) for folder_name in os.listdir(main_dataset_dir)]

    # Use the `create_dataset` function from the first script to create the dataset
    X, y, event_mapping = create_dataset(folders)

    with open('event_mapping2.json', 'w') as f:
        json.dump(event_mapping, f)

    # 400 classes
    num_classes = len(folders)
    print(f"Number of classes: {num_classes}")

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create train and test data as DataFrames
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    # Create the EventAnalysisDataset objects
    train_dataset = EventAnalysisDataset(train_data)
    test_dataset = EventAnalysisDataset(test_data)

    # Create the DataLoaders for our training and testing datasets
    save_model_path = Path('./weights2')
    save_model_path.mkdir(parents=True, exist_ok=True)

    # Set device
    tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Load Pre-trained model and update the configuration to handle `num_classes`
    config = AutoConfig.from_pretrained("monologg/koelectra-base-v3-discriminator", num_labels=num_classes)
    model = AutoModelForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator", config=config).to(device)
    
    # Train
    epochs = 20
    batch_size = 16
    optimizer = AdamW(model.parameters(), lr=1e-5)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    losses = []
    accuracies = []

    best_val_accuracy = 0.0

    for i in range(epochs):
        epoch_loss, epoch_accuracy = train(model, train_loader, optimizer, device)
        val_accuracy = evaluate(model, test_loader, device)
    
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), save_model_path / "koelectra-base-finetuned-sentiment-analysis_best_table_parse.bin")

        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)
        print(f"Train Epoch {i + 1} Loss: {epoch_loss}, Accuracy: {epoch_accuracy}")
        logger.info(f"Train Epoch {i + 1} Loss: {epoch_loss}, Accuracy: {epoch_accuracy}")

    logger.info(f"Losses: {losses}, Accuracies: {accuracies}")

    # Evaluate
    test_accuracy = evaluate(model, test_loader, device)
    logger.info(f"Test Accuracy: {test_accuracy}")

    # Save model
    torch.save(model.state_dict(), save_model_path / "koelectra-base-finetuned-sentiment-analysis_table_parse.bin")

if __name__ == "__main__":
    main()

    
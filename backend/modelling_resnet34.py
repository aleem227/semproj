import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoFeatureExtractor, ResNetModel
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
from tqdm import tqdm
import torch

print('CUDA available:', torch.cuda.is_available())
print('CUDA device count:', torch.cuda.device_count())
print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class UTKFaceDataset(Dataset):
    def __init__(self, dataframe, feature_extractor, transform=None):
        self.dataframe = dataframe
        self.feature_extractor = feature_extractor
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        # Get image path and labels
        img_path = self.dataframe.iloc[idx]['image_path']
        age = self.dataframe.iloc[idx]['age']
        gender = self.dataframe.iloc[idx]['gender']
        
        # Read and preprocess image
        image = Image.open(img_path).convert('RGB')
        
        # Apply additional transforms if specified (data augmentation)
        if self.transform:
            image = self.transform(image)
        
        # Apply feature extractor
        # This returns a dict with 'pixel_values', etc.
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        # We squeeze out the batch dimension for each item so shapes line up
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        # Return the feature-extractor dict plus the age/gender
        return inputs, torch.tensor(age, dtype=torch.float32), torch.tensor(gender, dtype=torch.float32)

class UTKFaceModel(nn.Module):
    """
    Uses ResNetModel as a raw backbone, then manually applies global average pooling
    to get a [batch_size, 512] feature vector, feeding it into custom heads.
    """
    def __init__(self, pretrained_model_name="microsoft/resnet-34"):
        super(UTKFaceModel, self).__init__()
        
        # Load the raw ResNet backbone (no classification head)
        self.resnet = ResNetModel.from_pretrained(pretrained_model_name)
        
        # For microsoft/resnet-34, the last_hidden_state has 512 channels
        hidden_size = 512
        
        # Define custom heads for age and gender
        self.age_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
        self.gender_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, pixel_values):
        # If we get a dict, extract pixel_values
        if isinstance(pixel_values, dict) and 'pixel_values' in pixel_values:
            outputs = self.resnet(pixel_values=pixel_values['pixel_values'])
        else:
            outputs = self.resnet(pixel_values=pixel_values)
        
        # outputs.last_hidden_state is [batch_size, 512, H, W]
        # We apply global average pooling over H and W:
        hidden = outputs.last_hidden_state
        features = hidden.mean(dim=[2, 3])  # => [batch_size, 512]
        
        # Pass through our heads
        age_output = self.age_head(features)
        gender_output = self.gender_head(features)
        
        return age_output, gender_output

class UTKFaceTrainer:
    def __init__(self, batch_size=32, device=None):
        # Force CUDA if available, else CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        
        self.batch_size = batch_size
        self.model = None
        self.feature_extractor = None
        
        # For plotting/tracking
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_gender_acc": [],
            "val_gender_acc": []
        }
        
    def prepare_data(self, df):
        """
        Creates coarser bins for age, merges with gender to form 'strata',
        tries a stratified split, else random if not feasible.
        """
        temp_age_group = pd.cut(
            df['age'],
            bins=[0, 12, 30, 100],  # Coarser bins
            labels=['child', 'young_adult', 'adult']
        )

        # Combine with gender into a single strata column
        df['strata'] = temp_age_group.astype(str) + '_' + df['gender'].astype(str)

        # Count
        strata_counts = df['strata'].value_counts()

        # Filter out strata with < 2 samples
        valid_strata = strata_counts[strata_counts >= 2].index.tolist()
        
        # Mark valid vs. invalid
        df['valid_strata'] = df['strata'].apply(lambda x: x if x in valid_strata else None)

        # If any stratum has fewer than 2 samples, fallback to random split
        min_count = strata_counts.min()
        if min_count < 2:
            print("Warning: Some stratum has < 2 samples. Using random split.")
            stratify_col = None
        else:
            stratify_col = df['valid_strata']

        # Split: Train vs (Val+Test)
        train_df, temp_df = train_test_split(
            df,
            test_size=0.3,
            stratify=stratify_col,
            random_state=42
        )

        # For the second split (Val vs Test)
        if stratify_col is not None:
            temp_strata_counts = temp_df['valid_strata'].value_counts()
            if temp_strata_counts.min() < 2:
                print("Warning: Some stratum in temp split < 2 samples. Using random split.")
                stratify_col_temp = None
            else:
                stratify_col_temp = temp_df['valid_strata']
        else:
            stratify_col_temp = None

        valid_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            stratify=stratify_col_temp,
            random_state=42
        )
        
        # Drop helper columns
        for d in (train_df, valid_df, test_df):
            d.drop(['strata', 'valid_strata'], axis=1, inplace=True, errors='ignore')
        
        print("Dataset splits:")
        print(f"Train set: {len(train_df)} samples")
        print(f"Validation set: {len(valid_df)} samples")
        print(f"Test set: {len(test_df)} samples")
        
        return train_df, valid_df, test_df
    
    def create_data_loaders(self, train_df, valid_df, test_df):
        """Creates DataLoaders for train/val/test."""
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-34")
        
        # Data augmentation transforms (training only)
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))
        ])
        
        train_dataset = UTKFaceDataset(train_df, self.feature_extractor, transform=train_transform)
        valid_dataset = UTKFaceDataset(valid_df, self.feature_extractor)
        test_dataset = UTKFaceDataset(test_df, self.feature_extractor)
        
        num_workers = 0
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        
        return train_loader, valid_loader, test_loader
    
    def build_model(self):
        """Instantiates the model and moves it to self.device."""
        self.model = UTKFaceModel()
        self.model.to(self.device)  # move parameters/buffers to GPU if available
        return self.model
    
    def train_model(self, train_loader, valid_loader, epochs=50, learning_rate=0.001, weight_decay=1e-5):
        """Runs the training loop on self.device (GPU if available)."""
        # Loss functions
        age_criterion = nn.MSELoss()
        gender_criterion = nn.BCELoss()
        
        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # LR scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, patience=5, min_lr=1e-6
        )
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # ===================== TRAIN =====================
            self.model.train()
            train_loss = 0.0
            correct_gender = 0
            total_gender = 0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for inputs, age_targets, gender_targets in train_pbar:
                # Move all data to self.device
                if isinstance(inputs, dict) and 'pixel_values' in inputs:
                    pixel_values = inputs['pixel_values'].to(self.device)
                else:
                    pixel_values = inputs.to(self.device)
                
                age_targets = age_targets.to(self.device)
                gender_targets = gender_targets.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward
                age_outputs, gender_outputs = self.model(pixel_values)
                
                # Compute losses
                age_loss = age_criterion(age_outputs.squeeze(), age_targets)
                gender_loss = gender_criterion(gender_outputs.squeeze(), gender_targets)
                
                # Weighted combination
                loss = 0.5 * age_loss + 1.0 * gender_loss
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * pixel_values.size(0)
                
                # Gender accuracy
                predicted_gender = (gender_outputs.squeeze() > 0.5).float()
                correct_gender += (predicted_gender == gender_targets).sum().item()
                total_gender += gender_targets.size(0)
                
                train_pbar.set_postfix({
                    'loss': loss.item(),
                    'gender_acc': correct_gender / total_gender if total_gender > 0 else 0
                })
            
            # Average over the entire dataset
            train_loss /= len(train_loader.dataset)
            train_gender_acc = correct_gender / total_gender if total_gender > 0 else 0
            
            # ===================== VALIDATION =====================
            self.model.eval()
            val_loss = 0.0
            correct_gender = 0
            total_gender = 0
            
            val_pbar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{epochs} [Valid]")
            with torch.no_grad():
                for inputs, age_targets, gender_targets in val_pbar:
                    if isinstance(inputs, dict) and 'pixel_values' in inputs:
                        pixel_values = inputs['pixel_values'].to(self.device)
                    else:
                        pixel_values = inputs.to(self.device)
                    age_targets = age_targets.to(self.device)
                    gender_targets = gender_targets.to(self.device)
                    
                    age_outputs, gender_outputs = self.model(pixel_values)
                    
                    age_loss = age_criterion(age_outputs.squeeze(), age_targets)
                    gender_loss = gender_criterion(gender_outputs.squeeze(), gender_targets)
                    loss = 0.5 * age_loss + 1.0 * gender_loss
                    
                    val_loss += loss.item() * pixel_values.size(0)
                    
                    # Gender accuracy
                    predicted_gender = (gender_outputs.squeeze() > 0.5).float()
                    correct_gender += (predicted_gender == gender_targets).sum().item()
                    total_gender += gender_targets.size(0)
                    
                    val_pbar.set_postfix({
                        'loss': loss.item(),
                        'gender_acc': correct_gender / total_gender if total_gender > 0 else 0
                    })
            
            val_loss /= len(valid_loader.dataset)
            val_gender_acc = correct_gender / total_gender if total_gender > 0 else 0
            
            # Update LR scheduler
            scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Gender Acc: {train_gender_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Gender Acc: {val_gender_acc:.4f}")
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_gender_acc'].append(train_gender_acc)
            self.history['val_gender_acc'].append(val_gender_acc)
            
            # Best model tracking
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                print(f"New best model with validation loss: {best_val_loss:.4f}")
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
                
        # Load best model parameters
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            
        return self.history
    
    def plot_training_history(self):
        """Plot training curves for loss and gender accuracy."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        ax1.plot(self.history['train_loss'], label='Training Loss')
        ax1.plot(self.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Gender accuracy
        ax2.plot(self.history['train_gender_acc'], label='Training Accuracy')
        ax2.plot(self.history['val_gender_acc'], label='Validation Accuracy')
        ax2.set_title('Gender Classification Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def evaluate_model(self, test_loader):
        """Evaluate model performance on the test set."""
        self.model.eval()
        
        age_preds = []
        gender_preds = []
        true_ages = []
        true_genders = []
        
        with torch.no_grad():
            for inputs, age_targets, gender_targets in tqdm(test_loader, desc="Evaluating"):
                # Move data to GPU/CPU
                if isinstance(inputs, dict) and 'pixel_values' in inputs:
                    pixel_values = inputs['pixel_values'].to(self.device)
                else:
                    pixel_values = inputs.to(self.device)
                age_targets = age_targets.to(self.device)
                gender_targets = gender_targets.to(self.device)
                
                # Forward pass
                age_outputs, gender_outputs = self.model(pixel_values)
                
                # Store predictions
                age_preds.extend(age_outputs.squeeze().cpu().numpy())
                gender_preds.extend((gender_outputs.squeeze() > 0.5).float().cpu().numpy())
                true_ages.extend(age_targets.cpu().numpy())
                true_genders.extend(gender_targets.cpu().numpy())
        
        # Convert to numpy arrays
        age_preds = np.array(age_preds)
        gender_preds = np.array(gender_preds)
        true_ages = np.array(true_ages)
        true_genders = np.array(true_genders)
        
        # Evaluate age
        age_mae = np.mean(np.abs(age_preds - true_ages))
        print("\nAge Prediction MAE:", age_mae)
        
        # Evaluate gender
        print("\nGender Classification Report:")
        print(classification_report(true_genders, gender_preds))
        
        # Confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(true_genders, gender_preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - Gender Classification')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks([0.5, 1.5], ['Male (0)', 'Female (1)'])
        plt.yticks([0.5, 1.5], ['Male (0)', 'Female (1)'])
        plt.show()
        
        # Plot age error distribution
        plt.figure(figsize=(12, 6))
        age_errors = age_preds - true_ages
        plt.hist(age_errors, bins=30)
        plt.title('Age Prediction Error Distribution')
        plt.xlabel('Prediction Error (years)')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Age MAE by gender
        male_indices = (true_genders == 0)
        female_indices = (true_genders == 1)
        
        male_mae = np.mean(np.abs(age_preds[male_indices] - true_ages[male_indices]))
        female_mae = np.mean(np.abs(age_preds[female_indices] - true_ages[female_indices]))
        
        plt.figure(figsize=(8, 6))
        plt.bar(['Male', 'Female'], [male_mae, female_mae])
        plt.title('Age Prediction MAE by Gender')
        plt.ylabel('Mean Absolute Error (years)')
        plt.grid(True, alpha=0.3)
        plt.text(0, male_mae, f'{male_mae:.2f}', ha='center', va='bottom')
        plt.text(1, female_mae, f'{female_mae:.2f}', ha='center', va='bottom')
        plt.show()
    
    def save_model(self, filepath):
        """Save the model (weights + feature_extractor name)."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'feature_extractor_name': "microsoft/resnet-34"
        }, filepath)
        print(f"Model saved to {filepath}")

def main():
    # Load the dataset (CSV must have ['image_path', 'age', 'gender'])
    df = pd.read_csv('age_gender_dataset.csv')
    
    # Initialize trainer
    trainer = UTKFaceTrainer(batch_size=32)
    
    # Prepare data (split into train/valid/test)
    train_df, valid_df, test_df = trainer.prepare_data(df)
    
    # Create DataLoaders
    train_loader, valid_loader, test_loader = trainer.create_data_loaders(train_df, valid_df, test_df)
    
    # Build the model and move to GPU if available
    trainer.build_model()
    
    # Train
    trainer.train_model(train_loader, valid_loader, epochs=50, learning_rate=1e-3)
    
    # Plot training history
    trainer.plot_training_history()
    
    # Evaluate on the test set
    trainer.evaluate_model(test_loader)
    
    # Save final model
    trainer.save_model('utk_face_model_resnet34.pt')

if __name__ == "__main__":
    main()

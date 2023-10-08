import random

import os
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
from torchvision import transforms

dataset_root = "./data/"
batch_size = 32
test_percentage = 0.10

input_size = 224 * 224 * 3
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

num_folds = 5
num_epochs = 5

lr = 1e-3
l2_lambda = 1e-4

class SoftmaxLoss(nn.Module):
    def __init__(self):
        super(SoftmaxLoss, self).__init__()

    def forward(self, outputs, labels):    
        max_values, _ = torch.max(outputs, dim=1, keepdim=True)
        normalized_outputs = outputs - max_values

        exp_shifted_outputs = torch.exp(normalized_outputs)
        sum_exp_shifted_outputs = torch.sum(exp_shifted_outputs, dim=1, keepdim=True)
        log_softmax = normalized_outputs - torch.log(sum_exp_shifted_outputs)
        
        log_probabilities = log_softmax.gather(1, labels.view(-1, 1))

        loss = -torch.sum(log_probabilities)

        return loss.mean()

class SVMLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(SVMLoss, self).__init__()
        self.margin = margin

    def forward(self, outputs, labels):
        correct_scores = outputs.gather(1, labels.view(-1, 1))

        margins = outputs - correct_scores + self.margin
        margins.scatter_(1, labels.view(-1, 1), 0)

        loss = torch.sum(torch.clamp(margins, min=0))

        return loss

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, data=None):
        self.transform = transform
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = self._make_dataset()
    
    def _make_dataset(self):
        images = []
        for target_class in self.classes:
            class_dir = os.path.join(self.root_dir, target_class)
            for filename in os.listdir(class_dir):
                img_path = os.path.join(class_dir, filename)
                item = (img_path, self.class_to_idx[target_class])
                images.append(item)
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):        
        img_path, target = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, target

class Model(nn.Module):
    def __init__(self, input_size, num_classes, loss_function):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)
        self.loss_function = loss_function

    def regularization_loss(self):
        regularization_loss = torch.tensor(0.0)
        for param in self.parameters():
            regularization_loss += torch.norm(param, p=2)
        return regularization_loss

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def train_model(self, train_loader, optimizer):
        self.train()

        total_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.loss_function(outputs, labels) + self.regularization_loss()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(train_loader)
    
    def evaluate_model(self, dataloader):
        self.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                outputs = self(inputs)
                loss = self.loss_function(outputs, labels) + self.regularization_loss()
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        return total_loss / len(dataloader), correct / total

def main():
    class_dirs = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]

    train_data = []
    validation_data = []
    test_data = []

    # Split the images into train, validation, and test sets
    for class_dir in class_dirs:
        class_path = os.path.join(dataset_root, class_dir)
        class_images = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.endswith(".jpg")]

        random.shuffle(class_images)

        num_images = len(class_images)
        num_test = int(num_images * test_percentage)
        num_train_validation = num_images - num_test
        num_train = int(0.9 * num_train_validation)
        num_validation = num_train_validation - num_train

        train_data.extend(class_images[:num_train])
        validation_data.extend(class_images[num_train:num_train + num_validation])
        test_data.extend(class_images[-num_test:])

    # Create datasets
    train_dataset = CustomImageDataset(root_dir=dataset_root, transform=transform, data=train_data)
    validation_dataset = CustomImageDataset(root_dir=dataset_root, transform=transform, data=validation_data)
    test_dataset = CustomImageDataset(root_dir=dataset_root, transform=transform, data=test_data)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    num_samples = len(train_dataset)
    fold_size = num_samples // num_folds

    for fold in range(num_folds):
        print(f"Fold {fold + 1}/{num_folds}")

        validation_indices = list(range(fold * fold_size, (fold + 1) * fold_size))
        train_indices = [i for i in range(num_samples) if i not in validation_indices]

        train_sampler = SubsetRandomSampler(train_indices)
        validation_sampler = SubsetRandomSampler(validation_indices)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, sampler=validation_sampler)

        softmax_model = Model(
            input_size, 
            len(class_dirs), 
            SoftmaxLoss()
        )
        softmax_optimizer = torch.optim.Adam(softmax_model.parameters(), lr=lr, weight_decay=l2_lambda)

        print("\nSoftmax training")

        for epoch in range(num_epochs):
            train_loss = softmax_model.train_model(train_loader, softmax_optimizer)
            validation_loss, validation_accuracy = softmax_model.evaluate_model(validation_loader)
            print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {train_loss:.4f}, Validation Loss: {validation_loss:.4f}, Validation Accuracy: {validation_accuracy:.2%}")

        svm_model = Model(
            input_size, 
            len(class_dirs),
            SVMLoss(margin=1.0)
        )
        svm_optimizer = torch.optim.Adam(svm_model.parameters(), lr=lr, weight_decay=l2_lambda)

        print("\nSVM training")

        for epoch in range(num_epochs):
            train_loss = svm_model.train_model(train_loader, svm_optimizer)
            validation_loss, validation_accuracy = svm_model.evaluate_model(validation_loader)
            print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {train_loss:.4f}, Validation Loss: {validation_loss:.4f}, Validation Accuracy: {validation_accuracy:.2%}")

        test_loss_softmax, test_accuracy_softmax = softmax_model.evaluate_model(test_dataloader)
        test_loss_svm, test_accuracy_svm = svm_model.evaluate_model(test_dataloader)

        print(f"\nTest Results for Fold {fold + 1}:")
        print(f"Softmax Model - Test Loss: {test_loss_softmax:.4f}, Test Accuracy: {test_accuracy_softmax:.2%}")
        print(f"SVM Model - Test Loss: {test_loss_svm:.4f}, Test Accuracy: {test_accuracy_svm:.2%}")

if __name__ == "__main__":
    main()
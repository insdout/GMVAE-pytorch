import torch
from utils import get_model
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import os


def flatten_mnist(tensor):
    return tensor.reshape(-1, 28*28)


class Trainer:
    def __init__(self, model, optimizer, criterion, train_loader, test_loader, device, transform_fn=flatten_mnist):
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.history = defaultdict(list)
        self.current_epoch = 0
        self.device = torch.device("cuda" if (torch.cuda.is_available() and device == "cuda") else "cpu")
        self.transform_fn = transform_fn
        self.model = model.to(self.device)

    def train(self, epochs):
        for epoch in range(epochs):
            self.train_epoch()
            self.evaluate()
            train_loss = self.history["train_loss"][-1]
            test_loss = self.history["test_loss"][-1]
            test_acc = self.history["test_accuracy"][-1]
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f} Test Acc: {test_acc:.4f}")

    def get_accuracy(self, y_true, y_pred):
        """
        Get accuracy.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        corrected_pred = -np.ones_like(y_pred)

        for cls in np.unique(y_pred):
            indx = y_pred == cls
            true_cls = np.bincount(y_true[indx]).argmax()
            corrected_pred[indx] = true_cls
        acc = np.mean(y_true == corrected_pred)
        return acc

    def train_epoch(self):
        """
        Training loop
        """
        model = self.model.train()
        optimizer = self.optimizer
        criterion = self.criterion
        dataloader = self.train_loader
        device = self.device

        running_loss = 0.0
        running_entropy = 0.0
        pred_labels = []
        true_labels = []

        for data, labels in dataloader:
            data = data.to(device)
            if self.transform_fn:
                data = self.transform_fn(data)
            labels = labels.to(device)

            optimizer.zero_grad()
            out_train, out_infer = model(data)
            loss = criterion(data, out_train)
            loss['total_loss'].backward()
            optimizer.step()

            running_loss += loss['total_loss'].item()
            running_entropy += loss["cond_entropy"].item()

            pred_labels.extend(out_infer["y"].detach().cpu().numpy())
            true_labels.extend(labels.detach().cpu().numpy())

        train_loss = running_loss / len(dataloader.dataset)
        cond_entropy = running_entropy / len(dataloader.dataset)
        self.history["train_loss"].append(train_loss)
        self.history["train_cond_entropy"].append(loss["cond_entropy"])
        self.history["train_accuracy"].append(self.get_accuracy(true_labels, pred_labels))
        self.history["train_cond_entropy"].append(cond_entropy)
        self.current_epoch += 1
        return train_loss, out_infer

    def evaluate(self):
        """
        Training loop
        """
        model = self.model.eval()
        criterion = self.criterion
        dataloader = self.test_loader
        device = self.device

        running_loss = 0.0
        running_entropy = 0.0
        pred_labels = []
        true_labels = []

        with torch.no_grad():
            for data, labels in dataloader:
                data = data.to(device)
                if self.transform_fn:
                    data = self.transform_fn(data)
                labels = labels.to(device)

                out_train, out_infer = model(data)
                loss = criterion(data, out_train)

                running_loss += loss['total_loss'].item()
                running_entropy += loss["cond_entropy"].item()

                pred_labels.extend(out_infer["y"].detach().cpu().numpy())
                true_labels.extend(labels.detach().cpu().numpy())

        test_loss = running_loss / len(dataloader.dataset)
        cond_entropy = running_entropy / len(dataloader.dataset)
        self.history["test_loss"].append(test_loss)
        self.history["test_cond_entropy"].append(loss["cond_entropy"])
        self.history["test_accuracy"].append(self.get_accuracy(true_labels, pred_labels))
        self.history["testn_cond_entropy"].append(cond_entropy)
        return test_loss, out_infer

    def plot_images(self, imgs, lbls, save_folder):
        num_rows = 4
        num_cols = 10

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 8))
        fig.tight_layout()

        for i in range(num_rows):
            for j in range(num_cols):
                index = i * num_cols + j
                axes[i][j].imshow(imgs[index], cmap='gray')
                axes[i][j].axis('off')
                axes[i][j].set_title(lbls[index])

        # Create the save folder if it doesn't exist
        os.makedirs(save_folder, exist_ok=True)

        # Save the plot as images
        for i in range(num_rows):
            for j in range(num_cols):
                index = i * num_cols + j
                save_path = os.path.join(save_folder, f'image_{index}.png')
                plt.savefig(save_path)

        plt.close(fig)

if __name__ == "__main__":
    k = 10
    encoder_type = "FC"
    input_size = 28*28
    hidden_size = 128
    latent_dim = 32

    model, criterion = get_model(k, encoder_type, input_size, hidden_size, latent_dim,
                                recon_loss_type="BCE", return_probs=True, eps=1e-8,
                                encoder_kwargs={}, decoder_kwargs={})


    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Set up data loaders
    # Define the transformation
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='data', train=False, transform=transform)


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2048, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2048, shuffle=False)

    # Move model to device
    model.to(device)

    # Define loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    trainer = Trainer(model, optimizer, criterion, train_loader, test_loader, device, transform_fn=flatten_mnist)
    trainer.train(10git)
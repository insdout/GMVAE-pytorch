import torch
from utils import get_model
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from loss2 import loss_function

k = 10
encoder_type = "FC"
input_size = 28*28
hidden_size = 128
latent_dim = 32

model, loss_cls = get_model(k, encoder_type, input_size, hidden_size, latent_dim,
                             recon_loss_type="MSE", return_probs=False, eps=1e-8,
                             encoder_kwargs={}, decoder_kwargs={})

criterion = loss_function

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

# Training loop
def train(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader):
        images = images.to(device)
        images = images.reshape(-1, 28*28)
        labels = labels.to(device)

        optimizer.zero_grad()
        out_train, out_infer = model(images)
        loss = criterion(images, out_train)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        #print(f"loss batch {loss.item()}")


    train_loss = running_loss / len(dataloader)
    return train_loss, out_infer


# Evaluation loop
def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            images = images.reshape(-1, 28*28)
            labels = labels.to(device)

            out_train, out_infer = model(images)
            loss = criterion(images, out_train)

            running_loss += loss.item()
            
    
    test_loss = running_loss / len(dataloader)
    return test_loss, out_infer

# Train the model
num_epochs = 20
test_loss = 0
for epoch in range(num_epochs):
    train_loss, out_infer= train(model, train_loader, criterion, optimizer)
    #test_loss, out_infer = evaluate(model, test_loader, criterion)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

ims = out_infer["x_hat"].detach().cpu().numpy()
for i in range(10):
    im = ims[i]
    plt.imshow(im.reshape(1, 28, 28).squeeze(0), cmap='gray')
    plt.show()

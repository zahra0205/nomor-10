import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ambil path absolut berdasarkan lokasi file ini
base_dir = os.path.dirname(__file__)
train_dir = os.path.join(base_dir, "dataset", "train")
val_dir = os.path.join(base_dir, "dataset", "validation")
test_image_path = os.path.join(base_dir, "test_image", "test1.jpg")

# Cek isi folder untuk memastikan terbaca
print("Train douglas_fir:", os.listdir(os.path.join(train_dir, "douglas_fir")))
print("Train white_pine :", os.listdir(os.path.join(train_dir, "white_pine")))

# Transformasi untuk dataset
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor()
])

# Load dataset
train_data = datasets.ImageFolder(root=train_dir, transform=transform)
val_data = datasets.ImageFolder(root=val_dir, transform=transform)

train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
val_loader = DataLoader(val_data, batch_size=2)

# Model CNN
class PineNet(nn.Module):
    def __init__(self):
        super(PineNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 37 * 37, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

model = PineNet().to(device)
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
print("Mulai training...")
for epoch in range(5):
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss {total_loss:.4f}")

# Simpan model
torch.save(model.state_dict(), "pinetree_model.pt")

# Prediksi Gambar Baru
def predict_image(img_path):
    model.eval()
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        pred = output.item()
        if pred > 0.5:
            print(f"Gambar diprediksi: White Pine ({pred:.2%})")
        else:
            print(f"Gambar diprediksi: Douglas Fir ({1 - pred:.2%})")

predict_image(test_image_path)

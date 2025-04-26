import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.optim import Adam
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np



# splits the input image into patches and creates linear embeddings for the model
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

# allows model to learn positions of image patches

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        pos_encoder = torch.zeros(max_seq_length, d_model)

        for pos in range(max_seq_length):
            for i in range(d_model):
                if i % 2 == 0:
                    pos_encoder[pos][i] = np.sin(pos / (10000 ** (i / d_model)))
                else:
                    pos_encoder[pos][i] = np.cos(pos / (10000 ** ((i - 1) / d_model)))

        self.register_buffer('pos_encoder', pos_encoder.unsqueeze(0))

    def forward(self, x):
        batch_size = x.size(0)

        # expand class token for each batch
        tokens_batch = self.cls_token.expand(batch_size, -1, -1)

        # add class tokens
        x = torch.cat((tokens_batch, x), dim=1)

        # add positional encodings
        positional_encodings = self.pos_encoder
        x = x + positional_encodings
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embedding dim % num heads != 0"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.shape

        # batch, seq_length, 3 * embed_dim-  query, key, value
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)

        # separate q,k,v
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # calculate attention scores
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_probs = attn_scores.softmax(dim=-1)

        # attention prob matmul values
        attn_output = attn_probs @ v

        # add attention heads
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_length, embed_dim)
        output = self.out_proj(attn_output)

        return output

class TransformerEncoder(nn.Module):
  def __init__(self, d_model, n_heads, r_mlp=4):
    super().__init__()
    self.d_model = d_model
    self.n_heads = n_heads

    # normalized multi head attention layer
    self.ln1 = nn.LayerNorm(d_model)
    self.mha = MultiHeadSelfAttention(d_model, n_heads)

    # normalized multilayer perceptron layer
    self.ln2 = nn.LayerNorm(d_model)
    self.mlp = nn.Sequential(nn.Linear(d_model, d_model*r_mlp), nn.GELU(), nn.Linear(d_model*r_mlp, d_model))

  def forward(self, x):
    # add normalized attention
    out = x + self.mha(self.ln1(x))

    # add normalized perceptron
    out = out + self.mlp(self.ln2(out))

    return out

class VisionTransformer(nn.Module):
  def __init__(self, d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers):
    super().__init__()
    assert img_size % patch_size == 0, "img_size dimensions must be divisible by patch_size dimensions"
    assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

    self.d_model = d_model # Dimensionality of model
    self.n_classes = n_classes # Number of classes
    self.img_size = img_size # Image size
    self.patch_size = patch_size # Patch size
    self.n_channels = n_channels # Number of channels
    self.n_heads = n_heads # Number of attention heads

    self.n_patches = (self.img_size // self.patch_size)**2
    self.max_seq_length = self.n_patches + 1

    self.patch_embedding = PatchEmbedding(img_size=self.img_size, patch_size=self.patch_size, in_channels=self.n_channels, embed_dim=self.d_model)
    self.positional_encoding = PositionalEncoding(self.d_model, self.max_seq_length)
    self.transformer_encoder = nn.Sequential(*[TransformerEncoder(self.d_model, self.n_heads) for _ in range(n_layers)])

    # classification perceptron
    self.classifier = nn.Sequential(nn.Linear(self.d_model, self.n_classes))

  def forward(self, images):
    x = self.patch_embedding(images)

    x = self.positional_encoding(x)

    x = self.transformer_encoder(x)

    x = self.classifier(x[:,0])

    return x

class testDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_filepath).convert('RGB')

        if self.transform:
            image = self.transform(image)
        return image, label


# vision transformer on our datasets
transform = transforms.Compose([transforms.ToTensor()])

data_set_paths = 'datasets/combined-resized-denoised-normalized'
for dataset in data_set_paths:
      dataset_root = dataset
      classes = sorted(os.listdir(dataset_root))
      class_to_idx = {cls: idx for idx, cls in enumerate(classes)}# map class labels

      image_paths = []
      labels = []

      for cls in classes:
          class_folder = os.path.join(dataset_root, cls)
          for filename in os.listdir(class_folder):
              image_paths.append(os.path.join(class_folder, filename))
              labels.append(class_to_idx[cls])  # assign class labels

      # training
      train_paths, test_paths, train_labels, test_labels = train_test_split(image_paths, labels, test_size=0.1, random_state=1)
      train_paths, val_paths, train_labels, val_labels = train_test_split(train_paths, train_labels, test_size=0.1, random_state=1)
      print(f"Train: {len(train_paths)}, Validation: {len(val_paths)}, Test: {len(test_paths)}")


      d_model = 256
      img_size = 256
      patch_size = 16
      n_channels = 3
      n_heads = 4
      n_layers = 3
      batch_size = 64
      epochs = 40
      alpha = 0.005
      n_classes = len(classes)

      transform = T.Compose([T.ToTensor()])

      train_set = testDataset(train_paths, train_labels, transform=transform)
      val_set = testDataset(val_paths, val_labels, transform=transform)
      test_set = testDataset(test_paths, test_labels, transform=transform)

      train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
      test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size)
      val_loader = DataLoader(val_set, shuffle=False, batch_size=batch_size)


      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

      transformer = VisionTransformer(d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers).to(device)

      optimizer = Adam(transformer.parameters(), lr=alpha)
      criterion = nn.CrossEntropyLoss()


      for epoch in range(epochs):
          transformer.train()
          training_loss = 0.0
          for i, data in enumerate(train_loader, 0):
              inputs, labels = data
              inputs, labels = inputs.to(device), labels.to(device)

              optimizer.zero_grad()

              outputs = transformer(inputs)
              loss = criterion(outputs, labels)
              loss.backward()
              optimizer.step()

              training_loss += loss.item()

          avg_train_loss = training_loss / len(train_loader)
          print(f'Epoch {epoch + 1}/{epochs} - Training loss: {avg_train_loss:.3f}')

          # evaluation
          transformer.eval()
          val_loss = 0.0
          correct = 0
          total = 0
          all_preds = []
          all_labels = []

          with torch.no_grad():
              for inputs, labels in val_loader:
                  inputs, labels = inputs.to(device), labels.to(device)
                  outputs = transformer(inputs)
                  loss = criterion(outputs, labels)
                  val_loss += loss.item()

                  _, predicted = torch.max(outputs, 1)
                  total += labels.size(0)
                  correct += (predicted == labels).sum().item()

                  all_preds.extend(predicted.cpu().numpy())
                  all_labels.extend(labels.cpu().numpy())

          avg_val_loss = val_loss / len(val_loader)
          val_accuracy = 100 * correct / total
          print(f'Validation loss: {avg_val_loss:.3f}, Accuracy: {val_accuracy:.2f}%')

          # confusion matrix if last epoch
          if epoch == epochs - 1:
            cm = confusion_matrix(all_labels, all_preds)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
            disp.plot(xticks_rotation=45, cmap=plt.cm.Blues)
            plt.title(f'Confusion Matrix (Epoch {epoch + 1})')
            plt.tight_layout()
            plt.show()



      correct = 0
      total = 0

      # testing
      with torch.no_grad():
        for data in test_loader:
          images, labels = data
          images, labels = images.to(device), labels.to(device)

          outputs = transformer(images)

          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
        print(f'\nModel Accuracy: {100 * correct // total} %')

import torch
from torchvision import models, transforms
from PIL import Image

def compute_embedding(image_path):
    """
    Computes the embedding of an image using a pre-trained ResNet50 model.

    Args:
        image_path (str): Path to the input image.

    Returns:
        torch.Tensor: Normalized embedding vector of the image.
    """
    # Load the image
    img = Image.open(image_path).convert("RGB")

    # Preprocess the image
    input_tensor = preprocess(img).unsqueeze(0)

    # Get the embedding
    with torch.no_grad():
        embedding = model(input_tensor).squeeze()
        embedding = embedding / embedding.norm()  # Normalize the embedding

    # Preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Model
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()

    # Load image
    img = Image.open("image.jpg").convert("RGB")
    input_tensor = preprocess(img).unsqueeze(0)

    # Get embedding
    with torch.no_grad():
        embedding = model(input_tensor).squeeze()
        embedding = embedding / embedding.norm()

    # Compare to other embeddings via dot product
    return embedding

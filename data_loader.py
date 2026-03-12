from utils import *

# Custom PyTorch Dataset for multimodal (image + text) garbage classification.
# It reads images organized in class-labeled folders, automatically assigns
# numeric labels, and extracts text descriptions from image filenames.
# The text is cleaned (removing digits and underscores), tokenized using a
# provided tokenizer (e.g., DistilBERT), and padded/truncated to max_len.
# Each sample returns:
#   - transformed image tensor
#   - tokenized text (input_ids, attention_mask)
#   - original cleaned text
#   - numeric class label
# This dataset is designed to be used with a DataLoader for training or validation.

class GarbageImageTextDataset(Dataset):
    """
    PyTorch Dataset for multimodal garbage classification (image + text).

    This dataset assumes the following directory structure:
        CVPR_2024_dataset_Train/
            Black/
                image1.jpg
                image2.png
            Blue/
                image3.jpg
                ...

    Each subfolder represents a class label. The class name is mapped
    automatically to a numeric index.

    For each image:
        - The filename (without extension) is used as a text description.
        - Underscores are replaced with spaces.
        - Digits are removed.
        - Text is converted to lowercase.
        - The cleaned text is tokenized using the provided tokenizer.

    Args:
        base_dir (str): Root directory containing class subfolders.
        transform (callable, optional): Image transformations to apply.
        tokenizer (transformers tokenizer, optional): Tokenizer used to
            encode text descriptions.
        max_len (int, optional): Maximum token length for text encoding.
            Default is 25.

    Returns:
        dict:
            {
                'text' (str): Cleaned text description,
                'input_ids' (Tensor): Tokenized text ids (flattened),
                'attention_mask' (Tensor): Attention mask (flattened),
                'image' (Tensor): Transformed image tensor,
                'label' (Tensor): Class index (long)
            }
    """
    def __init__(self, base_dir, transform=None, tokenizer=None, max_len=25):

        self.base_dir = base_dir          # Root directory containing class folders
        self.transform = transform        # Image transformation pipeline

        self.tokenizer = tokenizer        # Text tokenizer (e.g., DistilBERT)
        self.max_len = max_len            # Maximum token length for text encoding

        self.samples = []                 # List to store (image_path, text, label)
        self.class_to_idx = {}            # Mapping from class name to numeric index

        # Iterate over sorted class folders to ensure consistent label indexing
        for idx, label_folder in enumerate(sorted(os.listdir(base_dir))):
            folder_path = os.path.join(base_dir, label_folder)
            if not os.path.isdir(folder_path):
                continue  # Skip non-directory files

            self.class_to_idx[label_folder] = idx  # Assign numeric label

            # Iterate over image files inside each class folder
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                    image_path = os.path.join(folder_path, filename)

                    # Extract and clean text from filename
                    text = os.path.splitext(filename)[0]  # Remove extension
                    text = text.replace('_', ' ')         # Replace underscores
                    text = re.sub(r'\d+', '', text)       # Remove digits
                    text = text.strip().lower()           # Clean spaces & lowercase

                    # Store sample as (image path, cleaned text, label index)
                    self.samples.append((image_path, text, idx))

    def __len__(self):
        return len(self.samples)  # Total number of samples

    def __getitem__(self, idx):
        image_path, text_description, label = self.samples[idx]

        # Tokenize and encode text description
        encoding = self.tokenizer.encode_plus(
            text_description,
            add_special_tokens=True,      # Add [CLS], [SEP]
            max_length=self.max_len,
            return_token_type_ids=False,  # Not needed for single-sentence input
            padding='max_length',         # Pad to fixed length
            truncation=True,              # Truncate if longer than max_len
            return_attention_mask=True,
            return_tensors='pt'           # Return PyTorch tensors
        )        

        # Load and convert image to RGB
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)  # Apply augmentations / normalization

        # Return multimodal sample as dictionary
        return {
            'text': text_description,                              # Cleaned text
            'input_ids': encoding['input_ids'].flatten(),          # Token IDs
            'attention_mask': encoding['attention_mask'].flatten(),# Attention mask
            'image': image,                                        # Image tensor
            'label': torch.tensor(label, dtype=torch.long)         # Class label
        }
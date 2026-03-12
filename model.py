from utils import *

class ImageModel(nn.Module):
    """
    Image feature extractor with configurable backbone (ResNet18, ViT-B_16, DenseNet121).

    Args:
        backbone_name (str): Name of the backbone model.
        dropout_rate (float): Dropout applied after feature projection.
        feature_dim (int): Output feature dimension of the model.
        trainable (bool): Whether to fine-tune the backbone weights.
    """
    def __init__(self, backbone_name='resnet18', dropout_rate=0.5,
                 feature_dim=512, trainable=False):
        super(ImageModel, self).__init__()

        self.backbone_name = backbone_name.lower()
        self.feature_dim = feature_dim
        self.trainable = trainable

        # Load backbone model and remove original classifier head
        if self.backbone_name == 'resnet18':
            self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            backbone_out_dim = self.model.fc.in_features  # Input features for classifier
            self.model.fc = nn.Identity()  # Remove final fc layer

        elif self.backbone_name == 'vit_b_16':
            self.model = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
            backbone_out_dim = self.model.heads.head.in_features
            self.model.heads.head = nn.Identity()  # Remove classification head

        elif self.backbone_name == 'densenet121':
            self.model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
            backbone_out_dim = self.model.classifier.in_features
            self.model.classifier = nn.Identity()  # Remove classifier
            
        elif self.backbone_name == 'mobilenet_v3_large':
            self.model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
            backbone_out_dim = self.model.classifier[0].in_features
            self.model.classifier = nn.Identity()  # Remove classifier

        else:
            raise ValueError(f"Backbone {backbone_name} not supported.")

        # Linear projection layer to obtain fixed-size feature vector
        self.feature_extractor = nn.Sequential(
            nn.Linear(backbone_out_dim, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate))

        # Set backbone trainability
        print("ImageModel weight is trainable:", self.trainable)
        self.set_trainable(self.trainable)

    def set_trainable(self, trainable):
        """Freeze or unfreeze backbone parameters based on `trainable`."""
        for param in self.model.parameters():
            param.requires_grad = trainable

    def forward(self, x):
        """Forward pass: backbone -> feature projection"""
        x = self.model(x)
        
        # Replace the classifier head

        x = self.feature_extractor(x)
        return x


class TextModel(nn.Module):
    """
    Text feature extractor using HuggingFace transformers (BERT / DistilBERT).

    Args:
        model_name (str): Transformer model name (e.g., 'distilbert-base-uncased').
        feature_dim (int): Output feature dimension of text embeddings.
        dropout_rate (float): Dropout applied after pooling.
        trainable (bool): Whether to fine-tune transformer weights.
    """
    def __init__(self, model_name: str = "distilbert-base-uncased",
                 feature_dim: int = 128, dropout_rate: float = 0.3,
                 trainable: bool = False):
        super().__init__()

        # Load pretrained transformer model
        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_size = self.transformer.config.hidden_size  # Hidden size of transformer
        self.drop = nn.Dropout(dropout_rate)
        self.trainable = trainable
        self.feature_dim = feature_dim

        # Linear projection to desired feature dimension
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        # Freeze transformer if not trainable
        print("TextModel weight is trainable:", self.trainable)
        self.set_trainable(self.trainable)

    def set_trainable(self, trainable):
        """Freeze or unfreeze transformer parameters."""
        for param in self.transformer.parameters():
            param.requires_grad = trainable

    def forward(self, input_ids, attention_mask):
        """
        Forward pass: transformer -> mean pooling -> projection.

        Args:
            input_ids (Tensor): Token IDs of input text.
            attention_mask (Tensor): Attention mask for padding tokens.

        Returns:
            Tensor: Projected text feature vector.
        """
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)

        # Mean pooling over valid tokens
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * input_mask_expanded, dim=1)
        sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        # Project to desired feature dimension
        return self.projection(mean_pooled)


class MultiModalGarbageClassifier(nn.Module):
    """
    Multimodal classifier combining image and text features.

    Args:
        num_classes (int): Number of target classes.
        dropout_rate (float): Dropout applied in fusion layers.
        input_image_feature (int): Dimension of image features.
        input_text_feature (int): Dimension of text features.
    """
    def __init__(self, num_classes=4, dropout_rate=0.5,
                 input_image_feature=128, input_text_feature=128):
        super(MultiModalGarbageClassifier, self).__init__()

        self.input_image_feature = input_image_feature
        self.input_text_feature = input_text_feature
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes

        # Fully connected layers for feature fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(self.input_image_feature + self.input_text_feature, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),

            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),

            nn.Linear(256, self.num_classes)
        )

    def forward(self, image_feature, text_feature):
        """
        Forward pass: concatenate image & text features -> fusion layers.

        Args:
            image_feature (Tensor): Extracted image features. # of size torch.Size([bs, 512])
            text_feature (Tensor): Extracted text features.   # of size torch.Size([bs, 512])

        Returns:
            Tensor: Predicted class logits.
        """
        # Concatenate image and text features along feature dimension
        combined_features = torch.cat((image_feature, text_feature), dim=1)

        # Pass through fusion layers to get final logits
        combined_output = self.fusion(combined_features)
        return combined_output


class AttentionFusionGarbageClassifier(nn.Module):

    """
    Attention-based fusion of image and text features for multimodal classification.

    - Takes image and text feature vectors as input (each [batch_size, feature_dim]).
    - Stacks them as a 2-token sequence and applies scaled dot-product attention
      to model cross-modal interactions.
    - Produces a fused feature vector by averaging the attended features.
    - Passes the fused vector through a small feed-forward classifier to output
      logits for the target classes.

    Advantages:
    - Explicitly captures the relationship between modalities.
    - Dynamically weights image vs text contributions per sample.
    - More expressive than simple concatenation-based fusion.
    """
    def __init__(self, num_classes=4, dropout_rate=0.5, input_image_feature=128, input_text_feature=128):
        super(AttentionFusionGarbageClassifier, self).__init__()
        self.input_image_feature = input_image_feature
        self.input_text_feature = input_text_feature
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Query, Key, Value for attention
        self.query = nn.Linear(self.input_text_feature, self.input_text_feature)
        self.key = nn.Linear(self.input_text_feature, self.input_text_feature)
        self.value = nn.Linear(self.input_text_feature, self.input_text_feature)

        # Final classifier after attention-based fusion
        self.classifier = nn.Sequential(
            nn.Linear(self.input_text_feature, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, image_feature, text_feature):
        # Stack features into a "sequence" of 2 tokens
        # shape: [batch_size, 2, feature_dim]
        features = torch.stack([image_feature, text_feature], dim=1)

        # Compute Q, K, V
        Q = self.query(features)  # [bs, 2, 512]
        K = self.key(features)    # [bs, 2, 512]
        V = self.value(features)  # [bs, 2, 512]

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.input_text_feature ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)  # [bs, 2, 2]

        # Weighted sum of V
        fused_features = torch.matmul(attn_weights, V)  # [bs, 2, 512]

        # Option 1: take mean across the 2 tokens (fused vector)
        fused_vector = fused_features.mean(dim=1)  # [bs, 512]

        # Classifier
        logits = self.classifier(fused_vector)
        return logits
    


class SimpleWeightedFusionClassifier(nn.Module):
    """
    Simple weighted fusion of image and text features for multimodal classification.

    - Learns a single weight parameter to balance image vs. text contribution.
    - Fuses image and text features using a learnable sigmoid-scaled weight.
    - Passes the fused feature vector through a feed-forward classifier.

    Args:
        feature_dim (int): Dimension of input image and text features.
        num_classes (int): Number of target classes.
    """
    def __init__(self, num_classes=4, dropout_rate=0.5, input_image_feature=128, input_text_feature=128):
        super(SimpleWeightedFusionClassifier, self).__init__()
        self.input_image_feature = input_image_feature
        self.input_text_feature = input_text_feature
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Learnable weight to balance image vs text features
        self.alpha = nn.Parameter(torch.randn(1))

        # Feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(input_image_feature, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, image_feature, text_feature):
        """
        Forward pass: compute weighted fusion and classify.

        Args:
            image_feature (Tensor): Image feature vector [batch_size, feature_dim]
            text_feature (Tensor): Text feature vector [batch_size, feature_dim]

        Returns:
            Tensor: Class logits [batch_size, num_classes]
        """
        # Sigmoid to constrain alpha between 0 and 1
        weight = torch.sigmoid(self.alpha)

        # Weighted fusion of image and text features
        fused = weight * image_feature + (1 - weight) * text_feature

        # Classifier output
        logits = self.classifier(fused)
        return logits
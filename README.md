Multi-Modal Garbage Classification - Group 2
A PyTorch-based deep learning framework for classifying garbage images using both image and text features. This project combines CNN models with DistilBERT-based language models for multimodal fusion

Features
Multimodal Input: Leverages both image and text data from file names.

Flexible Image Backbones: Supports ResNet18, ViT-B_16, DenseNet121and mobilenet_v3_large.

Transformer-based Text Encoding: Supports BERT and DistilBERT (AutoModel from HuggingFace).

Stage-wise Training:

Stage 1 (s1): Freeze backbones, train fusion layers.
Stage 2 (s2): Fine-tune entire network by enabling all weights trainable for a few epochs.
Early Stopping & LR Scheduler: Prevent overfitting and improve convergence.

Label Smoothing: Reduces overconfidence in classification.

Visualization: Automatically plots training curves and confusion matrices.

Evaluation: Computes accuracy, class-wise accuracy, confusion matrix, ROC-AUC per class.

📂 Repository Structure
├── train.py                 # Main training script
├── test.py                  # Evaluation & testing script
├── train_test_wandb.py      # Main training and testing scripts integrated with Weights & Biases (W&B) for experiment tracking
├── model.py                 # Model definitions (ImageModel, TextModel, Fusion)
├── data_loader.py           # Custom PyTorch Dataset & DataLoader
├── utils.py                 # Training utilities, metrics, plotting
├── predictions_of_the_model.ipynb  # Include metrics, figures of the incorrect classifications, etc.)
├── garbage_data/            # Dataset folder (images organized by class)
├── requirements.txt/        # required package
├── saved_model_{backbone}/...      # Model checkpoints and train/val curves per models
├── test_result_{backbone}/...      # Evaluation results for each model, including ROC curves, confusion matrices, and a detailed .txt file containing additional performance metrics.
├── requirements.txt/        # required package
└── README.md
🔧 Installation
1. Clone repository:

git clone https://github.com/nedatghd/garbage_data_classification.git
cd garbage_data_classification
2. Create Python environment (recommended):

conda create -n garbage_classification python=3.12 -y
conda activate garbage_classification
3. Install dependencies:

pip install -r requirements.txt
Requirements include: torch, torchvision, transformers, scikit-learn, numpy, pandas, matplotlib, seaborn, tqdm, wandb, PIL, wandb.

🗂 Dataset
Dataset organization is as follows:

garbage_data/
├── CVPR_2024_dataset_Train/
│   ├── Black/
│   ├── Blue/
│   ├── Green/
│   └── TTR/
├── CVPR_2024_dataset_Val/
│   ├── Black/
│   ├── Blue/
│   ├── Green/
│   └── TTR/
├── CVPR_2024_dataset_Test/
│   ├── Black/
│   ├── Blue/
│   ├── Green/
│   └── TTR/
Each folder contains images named as textual descriptions (e.g., plastic_bottle.png) for multimodal training.

⚙ Training
Stage 1: Train Classifier Head (Backbone Frozen)

In Stage 1, the pretrained image backbone is frozen and only the classifier head is trained.

python train.py --traindata_dir "garbage_data/CVPR_2024_dataset_Train" --valdata_dir "garbage_data/CVPR_2024_dataset_Val" --epochs 20 --batch_size 16 --lr 1e-3 --weight_decay 2e-5 --label_smoothing 2e-5 --tokenizer_name "distilbert-base-uncased" --image_backbone "resnet18" --stage "s1"
Stage 2: Fine-tune all weights

python train.py --traindata_dir "garbage_data/CVPR_2024_dataset_Train" --valdata_dir "garbage_data/CVPR_2024_dataset_Val" --epochs 5 --batch_size 16 --lr 2e-5 --weight_decay 2e-6 --tokenizer_name "distilbert-base-uncased" --image_backbone "resnet18" --stage "s2"
⚙ Experiment Tracking with Weights & Biases (Optional)
To enable experiment tracking, use the train_test_wandb.py script.

Stage 1: Stage 1 with W&B

python train_test_wandb.py --epochs 20 --batch_size 16 --lr 1e-3 --weight_decay 2e-5 --label_smoothing 0.05 --tokenizer_name distilbert-base-uncased --image_backbone densenet121 --stage s1
Stage 2: Stage 2 with W&B

python train_test_wandb.py --epochs 5 --batch_size 16 --lr 2e-5 --weight_decay 2e-6 --label_smoothing 0.05 --tokenizer_name distilbert-base-uncased --image_backbone mobilenet_v3_large --stage s2
🔑 W&B Setup: Before running the tracking script, log in to your W&B account:

wandb login
Evaluation & Testing
Run test.py after training stage 2:

python test.py --testdata_dir "garbage_data/CVPR_2024_dataset_Test" --checkpoint "saved_model_{backbone_name}/best_model_s2.pt" --tokenizer_name "distilbert-base-uncased"
📊 Automatic Evaluation with W&B Script

If you use train_test_wandb.py for Stage 2, the script automatically performs evaluation on the test dataset after training completes. This includes logging performance metrics and evaluation results directly to Weights & Biases (W&B). one wandb log is on GitHub as an example. No additional testing command is required when using the W&B pipeline for Stage 2.

Outputs:

Overall accuracy located in test_result_resnet18 (for example)
Class-wise accuracy located in test_result_resnet18 (for example)
Average inference time per sample located in test_result_resnet18 (for example)
Confusion matrix located in test_result_resnet18 (for example)
ROC-AUC curves per class located in test_result_resnet18 (for example)
📈 Plots
Training scripts automatically generate:

loss_curve.png → Training vs Validation Loss located in saved_model_resnet18 (for example)
accuracy_curve.png → Training vs Validation Accuracy located in saved_model_resnet18 (for example)
📈 Fusion Strategies
We evaluate three multimodal fusion strategies for combining image and text features.

Fusion Strategy Descriptions

Fusion Strategy	Description
Concatenation	Image and text embeddings are concatenated with equal contribution from each modality.
Attention-based Fusion	Image and text features are treated as a sequence of two tokens (image, text), each of dimension 512, and combined using attention.
Learnable Modality Weights	Each modality is assigned a learnable weight that is optimized during training.
🔬 Experimental Results
The vit_b_16 backbone demonstrated the best overall performance compared to ResNet, DenseNet, and MobileNet variants across the tested fusion strategies.

Performance of the vit_b_16 Backbone Across Fusion Strategies

Fusion Strategy	Accuracy (%)	Precision (%)	Recall (%)	F1-score (%)
Concatenation	86.92	87.00	87.00	87.00
Attention-based	85.64	86.00	86.00	85.00
Learnable Weights	85.98	86.00	86.00	86.00
📊 Results on Self-Gathered Dataset
Accuracy (%)	Precision (%)	Recall (%)	F1-score (%)
93.94	94.36	93.75	93.84
Summary: The model achieves strong performance on the self-gathered dataset, indicating good generalization ability and consistent classification across classes.

💻 Hyperparameters
Parameter	Default	Description
epochs	20	Number of training epochs
batch_size	16	Mini-batch size
lr	2e-4	Learning rate
weight_decay	2e-5	AdamW weight decay
image_backbone	resnet18	Image feature extractor
tokenizer_name	distilbert-base-uncased	Text tokenizer
image_feature_dim	512	Output dimension for image features
text_feature_dim	512	Output dimension for text features
dropout	0.6	Dropout in fusion layers
stage	s1 / s2	Training stage (freeze or fine-tune)
📌 Notes
Stage 1 checkpoint (best_model_s1.pt) is required before Stage 2.
Label smoothing is applied (0.05) to mitigate overconfidence.
Optional: Compute class weights if dataset is heavily imbalanced.
📂 Citation / References
Transformers: HuggingFace Transformers
ResNet / DenseNet / ViT: Torchvision Models

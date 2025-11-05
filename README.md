# GRADIENT: Cross-Domain Implicit Aspect-Based Sentiment Analysis

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**GRADIENT** (Gradient Reversal And Domain-Invariant Extraction Networks for Triplets) is a unified framework for cross-domain implicit sentiment detection in Aspect-Based Sentiment Analysis (ABSA).

📄 **Paper**: [Submitted at Journal of Intelligent Information Systems (JIIS), Springer 2025]

## 🎯 Key Features

- **State-of-the-art Performance**: 54.7-64.0% triplet F1 scores on standard benchmarks
- **Robust Cross-Domain Transfer**: Maintains 47.9-79.1% of source-domain performance when transferred to target domains
- **Implicit Sentiment Detection**: Handles ~38.7% of sentiment expressions that lack explicit indicators
- **Multi-Granularity Processing**: Simultaneous word, phrase, and sentence-level analysis
- **Production Ready**: 78ms inference speed, 127M parameters

## 🚀 Major Contributions

1. **First unified architecture** combining multi-granularity implicit aspect detection with span-level opinion extraction
2. **Novel application** of domain adversarial training to fine-grained ABSA triplet extraction
3. **Multi-scale processing innovation** through Grid Tagging Matrix (GM-GTM) and Span-level Contextual Interaction Network (SCI-Net)
4. **Comprehensive cross-domain evaluation** across 6 domain transfer pairs

## 📊 Performance Highlights

### Single-Domain Performance (Triplet F1)
| Dataset | REST14 | REST15 | REST16 | LAP14 |
|---------|--------|--------|--------|-------|
| GRADIENT | **57.8** | **62.1** | **64.0** | **54.7** |
| Previous SOTA | 54.1 | 57.3 | 60.2 | 50.3 |
| Improvement | +3.7 | +4.8 | +3.8 | +4.4 |

### Cross-Domain Transfer (Zero-Shot)
- **Within-type transfers** (Restaurant→Restaurant): 70.8-79.1% retention
- **Cross-type transfers** (Restaurant↔Laptop): 45.9-49.9% retention
- **2.3× better retention** than baseline methods

## 🛠️ Installation
```bash
# Clone the repository
git clone https://github.com/shreya-sk/GRADIENT.git
cd GRADIENT

# Create conda environment
conda create -n gradient python=3.8
conda activate gradient

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PyTorch 1.10+
- Transformers 4.20+
- NumPy, Pandas, Scikit-learn
- CUDA 11.0+ (for GPU support)

## 📁 Project Structure
```
GRADIENT/
├── data/                    # Dataset files
│   ├── REST14/
│   ├── REST15/
│   ├── REST16/
│   └── LAP14/
├── models/                  # Model implementations
│   ├── gradient.py         # Main GRADIENT model
│   ├── gm_gtm.py          # Grid Tagging Matrix module
│   ├── sci_net.py         # SCI-Net module
│   └── domain_adversarial.py
├── utils/                   # Utility functions
│   ├── data_loader.py
│   ├── metrics.py
│   └── implicit_patterns.py
├── configs/                 # Configuration files
├── checkpoints/            # Saved models
├── train.py                # Training script
├── evaluate.py             # Evaluation script
└── README.md
```

## 💻 Usage

### Training
```bash
# Single-domain training
python train.py --dataset REST16 --batch_size 16 --epochs 10 --lr 3e-5

# Cross-domain training with domain adversarial learning
python train.py --source REST14 --target LAP14 --use_adversarial --lambda_domain 0.1
```

### Evaluation
```bash
# Evaluate on single domain
python evaluate.py --dataset REST16 --checkpoint checkpoints/gradient_rest16.pt

# Zero-shot cross-domain evaluation
python evaluate.py --dataset LAP14 --checkpoint checkpoints/gradient_rest16.pt --zero_shot
```

### Quick Start Example
```python
from models.gradient import GRADIENT
from utils.data_loader import load_dataset

# Load pre-trained model
model = GRADIENT.from_pretrained('checkpoints/gradient_rest16.pt')

# Example sentence
text = "If only the appetizers were as good as the main course"

# Extract sentiment triplets
triplets = model.extract_triplets(text)
# Output: [('appetizers', 'not as good', 'negative')]
```

## 📈 Architecture Components

### 1. Grid Tagging Matrix (GM-GTM)
- Multi-granularity aspect detection
- Processes word, phrase, and sentence levels simultaneously
- Dynamic n×c probability grid construction

### 2. Span-level Contextual Interaction Network (SCI-Net)
- Aspect-conditioned opinion extraction
- Multi-head cross-attention mechanisms
- Span boundary prediction

### 3. Domain Adversarial Learning
- Gradient reversal layer with progressive scheduling
- Orthogonal constraints for sentiment preservation
- Domain-invariant representation learning

## 📝 Implicit Sentiment Patterns

GRADIENT handles four types of implicit sentiment patterns:

1. **Comparative** (13.2%): "not as good as expected"
2. **Temporal** (9.3%): "used to be better"
3. **Conditional** (5.8%): "if only it were faster"
4. **Evaluative** (10.6%): "expensive for what you get"

## 🗂️ Datasets

Download preprocessed datasets:
```bash
bash scripts/download_data.sh
```

| Dataset | Train | Test | Domain |
|---------|-------|------|--------|
| REST14 | 3,041 | 800 | Restaurant |
| REST15 | 1,315 | 685 | Restaurant |
| REST16 | 2,000 | 676 | Restaurant |
| LAP14 | 3,045 | 800 | Laptop |

## 📖 Citation

If you use this code in your research, please cite:
```bibtex
@article{kothari2025gradient,
  title={GRADIENT: Gradient Reversal And Domain-Invariant Extraction Networks for Cross-Domain Implicit Aspect-Based Sentiment Analysis},
  author={Kothari, Shreya and Najafabadi, Maryam Khanian},
  journal={Journal of Intelligent Information Systems},
  year={2025},
  publisher={Springer}
}
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📧 Contact

- **Shreya Kothari**: shreya.kothari@sydney.edu.au
- **Maryam Khanian Najafabadi**: maryam64266@yahoo.com

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- SemEval shared tasks for providing the datasets
- The University of Sydney and Australian Catholic University for research support

---

**Note**: Code and trained models will be released upon paper acceptance.

# GRADIENT: Cross-Domain Implicit Aspect-Based Sentiment Analysis

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-under%20review-orange)](https://github.com/shreya-sk/GRADIENT)

**GRADIENT** (Gradient Reversal And Domain-Invariant Extraction Networks for Triplets) is a unified framework for cross-domain implicit sentiment detection in Aspect-Based Sentiment Analysis (ABSA).

📄 **Paper**: Under review at Journal of Intelligent Information Systems (JIIS), Springer 2025


## 🎯 Overview

GRADIENT addresses two critical challenges in Aspect-Based Sentiment Analysis:
1. **Implicit Sentiment Detection**: Handles sentiment expressions without explicit lexical indicators (temporal comparisons, conditional statements, evaluative constructs)
2. **Cross-Domain Transfer**: Enables robust performance across different domains without target-domain training data

## 🚀 Key Contributions

- **First unified architecture** combining multi-granularity implicit aspect detection with span-level opinion extraction
- **Novel application** of domain adversarial training with gradient reversal to fine-grained ABSA triplet extraction
- **Multi-scale processing innovation** through Grid Tagging Matrix (GM-GTM) and Span-level Contextual Interaction Network (SCI-Net)
- **Comprehensive cross-domain evaluation** across 6 domain transfer pairs with 2.3× better retention than baselines

## 📊 Performance Highlights

### Single-Domain Performance (Triplet F1)
| Dataset | REST14 | REST15 | REST16 | LAP14 | Average |
|---------|--------|--------|--------|-------|---------|
| GRADIENT | **57.8±1.3** | **62.1±1.2** | **64.0±1.4** | **54.7±1.5** | **59.7** |
| LSEMH-GCN (Previous SOTA) | 54.1 | 57.3 | 60.2 | 50.3 | 55.5 |
| **Improvement** | **+3.7** | **+4.8** | **+3.8** | **+4.4** | **+4.2** |

### Cross-Domain Transfer Performance (Zero-Shot)
| Transfer Type | Retention Rate | Absolute F1 | vs. Baseline |
|---------------|----------------|-------------|--------------|
| Within-type (Restaurant→Restaurant) | **71.8%** | 43.0 | +40.5% |
| Cross-type (Restaurant↔Laptop) | **47.9%** | 28.4 | +25.4% |
| **Average** | **57.1%** | **35.7** | **2.3× better** |

### Component-Level Performance
| Component | REST14 | REST15 | REST16 | LAP14 | Average |
|-----------|--------|--------|--------|-------|---------|
| Aspect F1 | 88.2 | 91.6 | 89.0 | 85.4 | **88.6** |
| Opinion F1 | 92.3 | 86.6 | 88.5 | 81.9 | **87.3** |
| Sentiment F1 | 95.6 | 85.0 | 90.2 | 86.4 | **89.3** |

## 🔍 What Makes GRADIENT Different?

### 1. Handles Implicit Sentiment Patterns (~38.7% of expressions)

**Traditional ABSA** misses expressions like:
- ❌ "If only the appetizers were as good as the main course" (conditional + comparative)
- ❌ "The battery life used to be much better" (temporal comparison)
- ❌ "Expensive for what you get" (evaluative without explicit sentiment)

**GRADIENT** detects these through:
- Multi-granularity processing (word, phrase, sentence levels simultaneously)
- Span-level contextual interaction modeling
- Pattern-aware architecture for comparative, temporal, conditional, and evaluative patterns

### 2. Robust Cross-Domain Transfer

| Scenario | BERT Baseline | GRADIENT | Improvement |
|----------|---------------|----------|-------------|
| REST14→REST16 | 33.5% retention | **79.1% retention** | +2.4× |
| REST15→REST16 | 36.5% retention | **74.9% retention** | +2.1× |
| REST16→LAP14 | 21.8% retention | **45.9% retention** | +2.1× |

## 🏗️ Architecture
```
Input Text → RoBERTa Encoder → Domain-Aware Encoding
                                        ↓
                    ┌───────────────────┴───────────────────┐
                    ↓                                       ↓
            GM-GTM (Aspects)                         SCI-Net (Opinions)
         Multi-Granularity Grid                   Span-Level Extraction
                    ↓                                       ↓
                    └───────────────────┬───────────────────┘
                                        ↓
                            Sentiment Classification
                                        ↓
                            Domain Adversarial Layer
                          (Gradient Reversal + Orthogonal)
                                        ↓
                        Triplet Output: (aspect, opinion, sentiment)
```

### Core Components

1. **Grid Tagging Matrix (GM-GTM)**: +8.1 F1 contribution
   - Dynamic n×c probability grid construction
   - Simultaneous word/phrase/sentence-level processing
   - Handles variable-granularity implicit aspects

2. **Span-level Contextual Interaction Network (SCI-Net)**: +6.6 F1 contribution
   - Aspect-conditioned query generation
   - Multi-head cross-attention mechanisms
   - Long-range opinion-aspect relationship modeling

3. **Domain Adversarial Training**: +5.3 F1 contribution
   - Gradient reversal layer with progressive scheduling
   - Orthogonal constraints (sentiment vs. domain subspaces)
   - Enables 2.3× better cross-domain retention

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
- PyTorch 1.10+ with CUDA 11.0+
- Transformers 4.20+ (HuggingFace)
- NumPy, Pandas, Scikit-learn
- Weights & Biases (optional, for experiment tracking)

## 💻 Quick Start

### Training
```bash
# Single-domain training on REST16
python train.py \
    --dataset REST16 \
    --batch_size 16 \
    --epochs 10 \
    --lr 3e-5 \
    --model_name roberta-base

# Cross-domain training (REST14 → LAP14)
python train.py \
    --source REST14 \
    --target LAP14 \
    --use_adversarial \
    --lambda_domain 0.1 \
    --lambda_orth 0.1
```

### Evaluation
```bash
# Single-domain evaluation
python evaluate.py \
    --dataset REST16 \
    --checkpoint checkpoints/gradient_rest16.pt

# Zero-shot cross-domain evaluation
python evaluate.py \
    --dataset LAP14 \
    --checkpoint checkpoints/gradient_rest16.pt \
    --zero_shot
```

### Inference Example
```python
from models.gradient import GRADIENT

# Load pre-trained model
model = GRADIENT.from_pretrained('checkpoints/gradient_rest16.pt')

# Example with implicit sentiment
text = "If only the appetizers were as good as the main course"
triplets = model.extract_triplets(text)

print(triplets)
# Output: [
#   {
#     'aspect': 'appetizers',
#     'opinion': 'not as good',
#     'sentiment': 'negative',
#     'pattern_type': 'conditional+comparative'
#   }
# ]
```

## 📁 Project Structure
```
GRADIENT/
├── data/                          # Dataset files and loaders
│   ├── REST14/
│   ├── REST15/
│   ├── REST16/
│   ├── LAP14/
│   └── data_loader.py
├── models/                        # Model implementations
│   ├── gradient.py               # Main GRADIENT model
│   ├── gm_gtm.py                # Grid Tagging Matrix
│   ├── sci_net.py               # SCI-Net module
│   ├── domain_adversarial.py    # GRL + orthogonal constraints
│   └── roberta_encoder.py       # Base encoder
├── utils/                         # Utilities
│   ├── metrics.py               # Evaluation metrics
│   ├── implicit_patterns.py     # Pattern detection
│   └── visualization.py         # Result plotting
├── configs/                       # Experiment configurations
│   ├── rest16.yaml
│   ├── cross_domain.yaml
│   └── ablation.yaml
├── scripts/                       # Helper scripts
│   ├── download_data.sh
│   ├── run_ablation.sh
│   └── evaluate_all.sh
├── checkpoints/                   # Saved models (upon release)
├── train.py                      # Training script
├── evaluate.py                   # Evaluation script
├── requirements.txt              # Dependencies
└── README.md
```

## 📊 Datasets

We evaluate on four standard SemEval ABSA benchmarks:

| Dataset | Domain | Train | Test | Aspect Categories | Source |
|---------|--------|-------|------|-------------------|--------|
| **REST14** | Restaurant | 3,041 | 800 | 18-22 | SemEval 2014 Task 4 |
| **REST15** | Restaurant | 1,315 | 685 | 18-22 | SemEval 2015 Task 12 |
| **REST16** | Restaurant | 2,000 | 676 | 18-22 | SemEval 2016 Task 5 |
| **LAP14** | Laptop | 3,045 | 800 | 15-20 | SemEval 2014 Task 4 |

### Download Preprocessed Data
```bash
bash scripts/download_data.sh
```

## 🧪 Reproducibility

All experiments are conducted with:
- **Random seeds**: [42, 123, 456, 789, 2024]
- **Hardware**: NVIDIA V100 GPU (32GB)
- **Reported metrics**: Mean ± SD over 5 runs
- **Hyperparameters**: Grid search on REST16 validation set

### Key Hyperparameters
```yaml
learning_rate: 3e-5
batch_size: 16 (effective: 32 with gradient accumulation)
epochs: 10
warmup_ratio: 0.1
weight_decay: 0.01
lambda_sparsity: 0.01
lambda_smoothness: 0.001
lambda_domain: 0.0 → 0.1 (progressive)
lambda_orth: 0.1
```

## 🔬 Ablation Study Results

| Component | Contribution | Consistency |
|-----------|--------------|-------------|
| GM-GTM | **+8.1 F1** | Universal (7.9-8.3 across datasets) |
| SCI-Net | **+6.6 F1** | Universal (6.3-7.1 across datasets) |
| Gradient Reversal | **+5.3 F1** | Universal (4.7-5.7 across datasets) |
| Multi-Granularity | **+4.5 F1** | Universal (4.2-4.9 across datasets) |
| Orthogonal Constraints | **+2.6 F1** | Domain-dependent (2.1-3.2) |

**Key Finding**: Different components show domain-specific vs. universal benefits, challenging assumptions about architectural improvements.

## 📈 Computational Efficiency

| Metric | GRADIENT | RoBERTa Baseline | Overhead |
|--------|----------|------------------|----------|
| Parameters | 127M | 125M | +1.6% |
| Training Time | 4.8h | 4.1h | +17% |
| Training Memory | 12.1 GB | 10.5 GB | +15% |
| Inference Speed | 78 ms | 71 ms | +10% |
| Inference Memory | 3.2 GB | 2.8 GB | +14% |

**Value Proposition**: +15% overhead for +4.2 F1 single-domain gain and 2.3× cross-domain improvement.

## 🎯 Use Cases

GRADIENT is particularly effective for:

✅ **Multi-domain sentiment analysis systems** requiring consistent performance
✅ **Applications with limited target-domain labeled data** (transfer learning scenarios)
✅ **Review analysis platforms** (e-commerce, hospitality, electronics)
✅ **Social media monitoring** with implicit sentiment expressions
✅ **Enterprise feedback systems** across multiple product lines

⚠️ **Exercise caution for**:
- Extreme domain shifts (e.g., medical ↔ social media)
- Real-time applications requiring <50ms latency
- Resource-constrained environments (<3GB GPU memory)

## 📖 Citation

If you find this work useful, please cite:
```bibtex
@article{kothari2025gradient,
  title={GRADIENT: Gradient Reversal And Domain-Invariant Extraction Networks for Cross-Domain Implicit Aspect-Based Sentiment Analysis},
  author={Kothari, Shreya and Najafabadi, Maryam Khanian},
  journal={Journal of Intelligent Information Systems},
  note={Under review},
  year={2025}
}
```

## 🤝 Contributing

We welcome contributions! Areas for potential contributions:
- Additional implicit pattern detectors
- Cross-lingual transfer experiments
- Efficient model variants (distillation, pruning)
- Integration with instruction-learning frameworks

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📧 Contact

**Corresponding Authors:**
- **Shreya Kothari** - shreya.kothari@sydney.edu.au
  - School of Computer Science, The University of Sydney
- **Maryam Khanian Najafabadi** - maryam64266@yahoo.com, maryam.khaniannajafabadi@acu.edu.au
  - Computer and Data Science Discipline, Australian Catholic University

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- SemEval shared tasks for providing the benchmark datasets
- The open-source NLP community for foundational tools and libraries

---

**Last Updated**: November 2025
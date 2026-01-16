# Advanced Algorithm Design - KDAO Team

Evaluating Cost-Efficiency and Accuracy in Graph-Augmented Recommendation Systems: A Comparative Study of LLMRec with Self-Hosted and Lightweight Models.


## Requirements

- Python 3.10.19
- CUDA-enabled GPU (recommended)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

Place your dataset in the `data/` directory with the following structure:
```
data/
└── <dataset_name>/
    ├── image_feat.npy
    ├── text_feat.npy
    ├── train_mat
    ├── augmented_user_init_embedding
    ├── augmented_atttribute_embedding_dict
    └── augmented_sample_dict
```

Supported datasets: `netflix`

## Usage

### Example Commands

```bash
# Train on Netflix dataset
python main.py --dataset netflix --epoch 50 --aug_sample_rate 0.1

```

## Project Structure

```
├── main.py              # Main training script
├── Models.py            
├── utility/
│   ├── parser.py        
│   ├── batch_test.py    
│   ├── load_data.py     
│   ├── logging.py       
│   ├── metrics.py       
│   └── norm.py          
├── LLM_augmentation_construct_prompt/  # LLM augmentation scripts
├── LATTICE/             
├── MMSSL/               
└── data/                
```

## Evaluation Metrics

- Recall@K (K=10, 20, 50)
- Precision@K
- Hit Ratio@K
- NDCG@K

Training logs are saved in the `logs/` directory.
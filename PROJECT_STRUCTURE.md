# LLM-MARL IDS Project Structure

```
llm-marl-ids/
│
├── README.md                          # Project overview
├── pyproject.toml                     # Python dependencies
├── requirements.txt                   # Pip requirements
│
├── config/                            # Configuration files
│   ├── training_config.yaml          # MAPPO training config
│   ├── edge_config.yaml              # Edge agent settings
│   └── cloud_config.yaml             # Cloud LLM settings
│
├── data/                              # Datasets (not in git)
│   ├── iot23_processed.h5
│   ├── ton_iot_processed.h5
│   ├── bot_iot_processed.h5
│   └── metadata.json
│
├── src/                               # Source code
│   ├── __init__.py
│   │
│   ├── edge/                          # Edge layer
│   │   ├── __init__.py
│   │   ├── edge_agent.py             # Main edge agent
│   │   ├── semantic_encoder.py       # MiniLM wrapper
│   │   └── feature_extractor.py      # Statistical features
│   │
│   ├── fog/                           # Fog layer (MARL)
│   │   ├── __init__.py
│   │   ├── marl_environment.py       # RL environment
│   │   ├── train_mappo.py            # Training script
│   │   └── eval_mappo.py             # Evaluation script
│   │
│   ├── cloud/                         # Cloud layer
│   │   ├── __init__.py
│   │   ├── explanation_generator.py  # LLM explanations
│   │   ├── mitre_mapper.py           # ATT&CK mapping
│   │   └── api.py                    # REST API (FastAPI)
│   │
│   ├── training/                      # Training utilities
│   │   ├── __init__.py
│   │   ├── attacker_agent.py         # Self-play attacker
│   │   ├── scenario_generator.py     # LLM scenario gen
│   │   └── self_play_trainer.py      # Self-play loop
│   │
│   └── utils/                         # Utilities
│       ├── __init__.py
│       ├── preprocessing.py          # Data preprocessing
│       ├── metrics.py                # Evaluation metrics
│       └── visualization.py          # Plotting utils
│
├── scripts/                           # Standalone scripts
│   ├── preprocess_datasets.py        # Data preprocessing
│   ├── download_datasets.sh          # Dataset download
│   ├── train.sh                      # Training launcher
│   └── deploy_edge.sh                # Edge deployment
│
├── experiments/                       # Experiment scripts
│   ├── exp1_concept_drift.py         # Experiment 1
│   ├── exp2_generalization.py        # Experiment 2
│   ├── exp3_self_play.py             # Experiment 3
│   └── ablations/
│       ├── ablation_minilm.py
│       ├── ablation_marl.py
│       ├── ablation_selfplay.py
│       └── ablation_llm_exp.py
│
├── checkpoints/                       # Model checkpoints (not in git)
│   ├── mappo/
│   ├── attacker/
│   └── best/
│
├── results/                           # Experimental results
│   ├── figures/
│   ├── tables/
│   └── logs/
│
├── tests/                             # Unit tests
│   ├── test_edge_agent.py
│   ├── test_marl_env.py
│   ├── test_explanation.py
│   └── test_attacker.py
│
├── docs/                              # Documentation
│   ├── architecture.md
│   ├── api_reference.md
│   ├── deployment_guide.md
│   └── figures/
│
├── paper/                             # LaTeX paper
│   ├── main.tex
│   ├── sections/
│   ├── figures/
│   └── bibliography.bib
│
└── docker/                            # Docker deployment
    ├── Dockerfile.edge
    ├── Dockerfile.fog
    ├── Dockerfile.cloud
    └── docker-compose.yml
```

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/yourusername/llm-marl-ids.git
cd llm-marl-ids

# Install dependencies
pip install -e .

# Or with poetry
poetry install
```

### 2. Data Preparation

```bash
# Download datasets
bash scripts/download_datasets.sh

# Preprocess
python scripts/preprocess_datasets.py \
    --input data/raw \
    --output data/iot23_processed.h5
```

### 3. Training

```bash
# Train MAPPO (no self-play)
python src/fog/train_mappo.py \
    --config config/training_config.yaml \
    --name mappo_baseline

# Train with self-play
python src/training/self_play_trainer.py \
    --config config/training_config.yaml \
    --self-play-ratio 0.5
```

### 4. Evaluation

```bash
# Evaluate trained model
python src/fog/train_mappo.py \
    --mode evaluate \
    --checkpoint checkpoints/mappo/final
```

### 5. Deployment

```bash
# Deploy edge agent on Raspberry Pi
bash scripts/deploy_edge.sh \
    --device edge_001 \
    --model checkpoints/best

# Start cloud API
python src/cloud/api.py \
    --port 8000 \
    --llm-model gpt-4
```

## File Descriptions

**Core Implementation:**
- `edge_agent.py`: Edge layer with MiniLM semantic encoding
- `marl_environment.py`: Multi-agent RL environment (Dec-POMDP)
- `train_mappo.py`: MAPPO training with RLlib
- `explanation_generator.py`: GPT-4 explanation generation
- `attacker_agent.py`: Self-play attacker (training only)

**Configuration:**
- `training_config.yaml`: MAPPO hyperparameters, reward weights
- `edge_config.yaml`: Edge device constraints, MiniLM settings
- `cloud_config.yaml`: LLM API settings, MITRE mapping

**Experiments:**
- `exp1_concept_drift.py`: Test robustness to streaming data
- `exp2_generalization.py`: Cross-dataset zero-shot transfer
- `exp3_self_play.py`: Self-play vs non-self-play comparison
- `ablations/`: 4 ablation studies

**Documentation:**
- `../LLM_MARL_IDS_Specification.md`: Complete technical spec (50+ pages)
- `../implementation_roadmap.md`: 8-phase implementation plan (20+ pages)
- `../Quick_Reference_Guide.md`: Quick reference (10+ pages)

## Development Workflow

1. **Feature Development**: Create branch, implement in `src/`
2. **Testing**: Add tests in `tests/`, run `pytest`
3. **Experiment**: Add script to `experiments/`
4. **Documentation**: Update `docs/`
5. **Pull Request**: Review and merge

## Citation

If you use this code, please cite:

```bibtex
@article{yourname2026llm,
  title={LLM-Enhanced Multi-Agent Reinforcement Learning for 
         Intrusion Detection in IoT: A Self-Play Adversarial 
         Training Approach},
  author={[Your Name] and [Co-authors]},
  journal={IEEE Transactions on Dependable and Secure Computing},
  year={2026}
}
```

## License

MIT License (update as appropriate)

## Contact

- **Author**: [Your Name]
- **Email**: your.email@institution.edu
- **Lab**: [Your Research Group]

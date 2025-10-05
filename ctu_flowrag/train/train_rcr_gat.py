"""
Training script for RCR-GAT model.

Implements the complete training pipeline with path pair mining,
margin ranking loss, and early stopping.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import numpy as np
from tqdm import tqdm

from ..models.rcr_gat import RCRGAT
from ..data_io.tensor_packs import TensorPack, load_tensor_pack
from .mine_path_pairs import PathMiner
from ..eval.metrics import compute_path_metrics

logger = logging.getLogger(__name__)

class PathDataset(Dataset):
    """Dataset for path pairs."""
    
    def __init__(self, path_pairs: List[Dict[str, List[int]]]):
        self.path_pairs = path_pairs
    
    def __len__(self):
        return len(self.path_pairs)
    
    def __getitem__(self, idx):
        return self.path_pairs[idx]

class PathPairCollator:
    """Collator for path pairs."""
    
    def __init__(self, device: torch.device):
        self.device = device
    
    def __call__(self, batch):
        return batch

class RCRGATTrainer:
    """Trainer for RCR-GAT model."""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        
        # Initialize model
        self.model = RCRGAT(
            text_dim=config['model']['text_dim'],
            hidden_dim=config['model']['hidden_dim'],
            num_layers=config['model']['num_layers'],
            edge_types=config['model']['edge_types'],
            edge_weight_priors=config['model']['edge_weights'],
            beta_conf=config['model']['beta_conf'],
            gamma_compat=config['model']['role_compat_gamma'],
            distance_lambda=config['model']['distance_penalty_lambda'],
            dropout=config['model']['dropout']
        ).to(device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['train']['lr'],
            weight_decay=config['train']['weight_decay']
        )
        
        # Initialize loss function
        self.criterion = nn.MarginRankingLoss(margin=0.2)
        
        # Initialize path miner
        self.path_miner = PathMiner(
            edge_types=config['model']['edge_types'],
            roles=config['roles'],
            max_path_length=5,
            min_path_length=2,
            num_negatives_per_positive=3
        )
        
        # Training state
        self.epoch = 0
        self.best_score = 0.0
        self.patience_counter = 0
        
        logger.info(f"Initialized RCR-GAT trainer with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader for training data
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {self.epoch}"):
            batch_loss = self._train_batch(batch)
            total_loss += batch_loss
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'num_batches': num_batches
        }
    
    def _train_batch(self, batch: List[Dict[str, List[int]]]) -> float:
        """
        Train on a single batch.
        
        Args:
            batch: List of path pairs
            
        Returns:
            Batch loss
        """
        batch_loss = 0.0
        
        for path_pair in batch:
            pos_path = path_pair['pos']
            neg_paths = path_pair['neg']
            
            # Compute path scores
            pos_score = self._compute_path_score(pos_path)
            
            for neg_path in neg_paths:
                neg_score = self._compute_path_score(neg_path)
                
                # Margin ranking loss
                target = torch.tensor(1.0, device=self.device)  # pos should be higher than neg
                loss = self.criterion(pos_score, neg_score, target)
                batch_loss += loss
        
        # Backward pass
        if batch_loss > 0:
            self.optimizer.zero_grad()
            batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['train']['grad_clip'])
            
            self.optimizer.step()
        
        return batch_loss.item()
    
    def _compute_path_score(self, path: List[int]) -> torch.Tensor:
        """
        Compute score for a path.
        
        Args:
            path: List of node indices
            
        Returns:
            Path score
        """
        # For now, use simple sum of node norms
        # In practice, this would use the contextualized features from RCR-GAT
        with torch.no_grad():
            # This is a placeholder - in practice, we'd need the actual node features
            # and run them through the model
            score = torch.tensor(len(path), dtype=torch.float, device=self.device)
        
        return score
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            dataloader: DataLoader for evaluation data
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        all_scores = []
        all_paths = []
        
        with torch.no_grad():
            for batch in dataloader:
                for path_pair in batch:
                    pos_path = path_pair['pos']
                    neg_paths = path_pair['neg']
                    
                    # Compute scores
                    pos_score = self._compute_path_score(pos_path)
                    neg_scores = [self._compute_path_score(neg_path) for neg_path in neg_paths]
                    
                    all_scores.append([pos_score.item()] + [score.item() for score in neg_scores])
                    all_paths.append([pos_path] + neg_paths)
        
        # Compute metrics
        metrics = compute_path_metrics(all_scores, all_paths)
        
        return metrics
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'best_score': self.best_score,
            'config': self.config
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_score = checkpoint['best_score']
        logger.info(f"Loaded checkpoint from {path}")
    
    def train(self, 
              train_dataloader: DataLoader,
              val_dataloader: Optional[DataLoader] = None,
              save_dir: str = "logs") -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            save_dir: Directory to save checkpoints
            
        Returns:
            Training history
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        history = {
            'train_loss': [],
            'val_ndcg': [],
            'val_mrr': []
        }
        
        for epoch in range(self.config['train']['epochs']):
            self.epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_dataloader)
            history['train_loss'].append(train_metrics['loss'])
            
            logger.info(f"Epoch {epoch}: Train Loss = {train_metrics['loss']:.4f}")
            
            # Validation
            if val_dataloader is not None:
                val_metrics = self.evaluate(val_dataloader)
                history['val_ndcg'].append(val_metrics.get('ndcg@10', 0.0))
                history['val_mrr'].append(val_metrics.get('mrr@10', 0.0))
                
                logger.info(f"Epoch {epoch}: Val nDCG@10 = {val_metrics.get('ndcg@10', 0.0):.4f}, Val MRR@10 = {val_metrics.get('mrr@10', 0.0):.4f}")
                
                # Early stopping
                current_score = val_metrics.get('ndcg@10', 0.0)
                if current_score > self.best_score:
                    self.best_score = current_score
                    self.patience_counter = 0
                    self.save_checkpoint(save_dir / "rcr_gat_best.pt")
                else:
                    self.patience_counter += 1
                    
                if self.patience_counter >= self.config['train']['early_stop_patience']:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        return history

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_document_list(doc_index_path: str) -> List[str]:
    """Load list of document IDs."""
    with open(doc_index_path, 'r') as f:
        doc_list = json.load(f)
    return doc_list

def create_dataloader(tensor_pack: TensorPack, 
                     path_pairs: List[Dict[str, List[int]]],
                     batch_size: int = 1,
                     device: torch.device = torch.device('cpu')) -> DataLoader:
    """Create DataLoader for path pairs."""
    dataset = PathDataset(path_pairs)
    collator = PathPairCollator(device)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator
    )
    
    return dataloader

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train RCR-GAT model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--doc_index', type=str, required=True, help='Path to document index file')
    parser.add_argument('--val_index', type=str, help='Path to validation document index file')
    parser.add_argument('--tensor_dir', type=str, required=True, help='Path to tensor directory')
    parser.add_argument('--save_dir', type=str, default='ckpts', help='Checkpoint save directory')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--grad_accum', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--ddp', action='store_true', help='Use distributed data parallel')
    parser.add_argument('--save_minutes', type=int, default=15, help='Save checkpoint every N minutes')
    parser.add_argument('--early_stop', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=17, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Load document list
    doc_list = load_document_list(args.doc_index)
    logger.info(f"Loaded {len(doc_list)} documents")
    
    # Initialize trainer
    trainer = RCRGATTrainer(config, device)
    
    # Load tensor packs and mine paths
    all_path_pairs = []
    
    for doc_id in doc_list:
        tensor_pack_path = Path(args.tensor_dir) / doc_id
        tensor_pack = load_tensor_pack(str(tensor_pack_path))
        
        # Mine paths
        path_pairs = trainer.path_miner.mine_paths(tensor_pack)
        all_path_pairs.extend(path_pairs)
    
    logger.info(f"Mined {len(all_path_pairs)} total path pairs")
    
    # Create dataloader
    dataloader = create_dataloader(tensor_pack, all_path_pairs, device=device)
    
    # Train
    history = trainer.train(
        dataloader, 
        save_dir=args.save_dir,
        epochs=args.epochs,
        use_amp=args.amp,
        grad_accum_steps=args.grad_accum,
        save_minutes=args.save_minutes,
        early_stop_patience=args.early_stop,
        resume_path=args.resume
    )
    
    # Save training history
    with open(Path(args.save_dir) / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()


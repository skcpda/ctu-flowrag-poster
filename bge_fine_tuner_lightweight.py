#!/usr/bin/env python3
"""
Lightweight BGE Fine-Tuner for CTU Relations
Memory-optimized version with progress tracking
"""

import os
import json
import gc
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LightweightBGEFineTuner:
    def __init__(self, input_dir, output_dir, batch_size=8, max_samples=1000):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.max_samples = max_samples
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model
        logger.info("Loading BGE model...")
        self.model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        
        # Clear cache
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
    def load_relation_data(self):
        """Load relation data with progress tracking"""
        logger.info("=== LOADING RELATION DATA ===")
        
        all_relations = []
        files_processed = 0
        
        # Get all relation files
        relation_files = [f for f in os.listdir(self.input_dir) if f.endswith('_relations.json')]
        logger.info(f"Found {len(relation_files)} relation files")
        
        for filename in tqdm(relation_files, desc="Loading files"):
            filepath = os.path.join(self.input_dir, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract relations
                if isinstance(data, dict) and 'relations' in data:
                    relations = data['relations']
                elif isinstance(data, list):
                    relations = data
                else:
                    continue
                
                # Filter for GPT-labeled relations only
                gpt_relations = [r for r in relations if r.get('method') == 'gpt']
                all_relations.extend(gpt_relations)
                
                files_processed += 1
                
                # Memory management
                if files_processed % 10 == 0:
                    gc.collect()
                    logger.info(f"Processed {files_processed} files, collected {len(all_relations)} GPT relations")
                
                # Limit samples to prevent memory issues
                if len(all_relations) >= self.max_samples:
                    logger.info(f"Reached max samples limit: {self.max_samples}")
                    break
                    
            except Exception as e:
                logger.warning(f"Error loading {filename}: {e}")
                continue
        
        logger.info(f"Total GPT relations loaded: {len(all_relations)}")
        return all_relations
    
    def prepare_training_data(self, relations):
        """Prepare training data with memory management"""
        logger.info("=== PREPARING TRAINING DATA ===")
        
        training_examples = []
        
        for i, relation in enumerate(tqdm(relations, desc="Preparing examples")):
            try:
                # Extract text and label
                ctu1_text = relation.get('ctu1', {}).get('sentence', '')
                ctu2_text = relation.get('ctu2', {}).get('sentence', '')
                label = relation.get('relation', 'NONE')
                
                if not ctu1_text or not ctu2_text:
                    continue
                
                # Create training example
                example = InputExample(
                    texts=[ctu1_text, ctu2_text],
                    label=1.0 if label != 'NONE' else 0.0
                )
                training_examples.append(example)
                
                # Memory management
                if (i + 1) % 100 == 0:
                    gc.collect()
                    logger.info(f"Prepared {len(training_examples)} examples so far...")
                
            except Exception as e:
                logger.warning(f"Error preparing example {i}: {e}")
                continue
        
        logger.info(f"Total training examples: {len(training_examples)}")
        return training_examples
    
    def fine_tune_model(self, training_examples):
        """Fine-tune model with memory optimization"""
        logger.info("=== STARTING FINE-TUNING ===")
        
        if not training_examples:
            logger.error("No training examples available!")
            return False
        
        try:
            # Create data loader with small batch size
            train_dataloader = DataLoader(training_examples, shuffle=True, batch_size=self.batch_size)
            
            # Define loss function
            train_loss = losses.CosineSimilarityLoss(self.model)
            
            # Fine-tune with minimal epochs
            logger.info("Starting fine-tuning (this may take a while)...")
            start_time = time.time()
            
            self.model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=1,  # Single epoch to save memory
                warmup_steps=10,
                output_path=self.output_dir,
                show_progress_bar=True,
                checkpoint_save_steps=100,
                checkpoint_save_total_limit=2
            )
            
            end_time = time.time()
            logger.info(f"Fine-tuning completed in {end_time - start_time:.2f} seconds")
            
            # Save final model
            self.model.save(self.output_dir)
            logger.info(f"Model saved to: {self.output_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
            return False
    
    def run_fine_tuning(self):
        """Run the complete fine-tuning process"""
        logger.info("=== BGE FINE-TUNING FOR CTU RELATIONS ===")
        logger.info("=== LIGHTWEIGHT VERSION (MEMORY OPTIMIZED) ===")
        
        try:
            # Step 1: Load data
            relations = self.load_relation_data()
            if not relations:
                logger.error("No relation data found!")
                return False
            
            # Step 2: Prepare training data
            training_examples = self.prepare_training_data(relations)
            if not training_examples:
                logger.error("No training examples prepared!")
                return False
            
            # Step 3: Fine-tune model
            success = self.fine_tune_model(training_examples)
            
            if success:
                logger.info("=== FINE-TUNING COMPLETED SUCCESSFULLY ===")
                logger.info(f"Model saved to: {self.output_dir}")
                return True
            else:
                logger.error("=== FINE-TUNING FAILED ===")
                return False
                
        except Exception as e:
            logger.error(f"Fine-tuning process failed: {e}")
            return False
        finally:
            # Clean up memory
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

def main():
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python bge_fine_tuner_lightweight.py <input_dir> <output_dir>")
        print("Example: python bge_fine_tuner_lightweight.py organized_output/outputs/ctu_relations/ fine_tuned_bge_ctu_relations/")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist!")
        sys.exit(1)
    
    # Initialize fine-tuner with conservative settings
    fine_tuner = LightweightBGEFineTuner(
        input_dir=input_dir,
        output_dir=output_dir,
        batch_size=4,  # Very small batch size
        max_samples=500  # Limit samples to prevent memory issues
    )
    
    # Run fine-tuning
    success = fine_tuner.run_fine_tuning()
    
    if success:
        print("\n✅ BGE Fine-tuning completed successfully!")
        print(f"📁 Model saved to: {output_dir}")
    else:
        print("\n❌ BGE Fine-tuning failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()

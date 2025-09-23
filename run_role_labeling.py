#!/usr/bin/env python3
"""
Practical Role Labeling Implementation
Processes all sentence files with role labeling and slot extraction
"""

import os
import json
import glob
import time
from role_labeling_system import RoleLabelingSystem

def setup_environment():
    """Setup the environment for role labeling"""
    print("=== SETTING UP ROLE LABELING ENVIRONMENT ===")
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        print("Please set your API key: export OPENAI_API_KEY='your-key-here'")
        return False
    
    print("✅ OpenAI API key found")
    
    # Create output directory
    output_dir = "organized_output/outputs/labeled_sentences"
    os.makedirs(output_dir, exist_ok=True)
    print(f"✅ Output directory created: {output_dir}")
    
    return True

def process_all_sentences(model: str = "gpt-3.5-turbo", batch_size: int = 15):
    """Process all sentence files with role labeling"""
    
    if not setup_environment():
        return
    
    # Initialize the labeling system
    api_key = os.getenv("OPENAI_API_KEY")
    labeler = RoleLabelingSystem(api_key=api_key, model=model)
    
    # Get all sentence files
    input_dir = "organized_output/outputs/split_sentences"
    output_dir = "organized_output/outputs/labeled_sentences"
    
    sentence_files = glob.glob(f"{input_dir}/*.json")
    total_files = len(sentence_files)
    
    print(f"=== PROCESSING {total_files} SENTENCE FILES ===")
    print(f"Model: {model}")
    print(f"Batch size: {batch_size}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    processed_count = 0
    failed_count = 0
    total_sentences = 0
    
    for i, input_file in enumerate(sentence_files, 1):
        filename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, filename.replace("_sentences.json", "_labeled.json"))
        
        print(f"[{i}/{total_files}] Processing {filename}...")
        
        try:
            # Process the file
            result = labeler.batch_process(input_file, output_file, batch_size)
            
            if "error" not in result:
                sentence_count = len(result.get("labels", []))
                total_sentences += sentence_count
                processed_count += 1
                print(f"  ✅ Success: {sentence_count} sentences labeled")
            else:
                print(f"  ❌ Failed: {result['error']}")
                failed_count += 1
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            failed_count += 1
        
        # Rate limiting
        time.sleep(2)
        
        # Progress update every 10 files
        if i % 10 == 0:
            print(f"  📊 Progress: {i}/{total_files} files processed")
            print(f"  📊 Total sentences: {total_sentences:,}")
            print()
    
    print("=== PROCESSING COMPLETE ===")
    print(f"Total files: {total_files}")
    print(f"Successfully processed: {processed_count}")
    print(f"Failed: {failed_count}")
    print(f"Total sentences labeled: {total_sentences:,}")
    
    # Calculate estimated cost
    if model == "gpt-3.5-turbo":
        estimated_cost = total_sentences * 0.0001  # Rough estimate
    else:  # gpt-4o
        estimated_cost = total_sentences * 0.0004  # Rough estimate
    
    print(f"Estimated cost: ${estimated_cost:.2f}")

def process_sample_files(model: str = "gpt-3.5-turbo", sample_size: int = 5):
    """Process a sample of files for testing"""
    
    if not setup_environment():
        return
    
    # Initialize the labeling system
    api_key = os.getenv("OPENAI_API_KEY")
    labeler = RoleLabelingSystem(api_key=api_key, model=model)
    
    # Get sample files
    input_dir = "organized_output/outputs/split_sentences"
    output_dir = "organized_output/outputs/labeled_sentences"
    
    sentence_files = glob.glob(f"{input_dir}/*.json")[:sample_size]
    
    print(f"=== PROCESSING SAMPLE OF {len(sentence_files)} FILES ===")
    print(f"Model: {model}")
    print()
    
    for i, input_file in enumerate(sentence_files, 1):
        filename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, filename.replace("_sentences.json", "_labeled.json"))
        
        print(f"[{i}/{len(sentence_files)}] Processing {filename}...")
        
        try:
            result = labeler.batch_process(input_file, output_file, batch_size=10)
            
            if "error" not in result:
                sentence_count = len(result.get("labels", []))
                print(f"  ✅ Success: {sentence_count} sentences labeled")
            else:
                print(f"  ❌ Failed: {result['error']}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        time.sleep(1)
    
    print("=== SAMPLE PROCESSING COMPLETE ===")

def analyze_labeled_results():
    """Analyze the results of role labeling"""
    
    output_dir = "organized_output/outputs/labeled_sentences"
    labeled_files = glob.glob(f"{output_dir}/*.json")
    
    if not labeled_files:
        print("No labeled files found. Run the labeling process first.")
        return
    
    print("=== ANALYZING LABELED RESULTS ===")
    
    role_counts = {}
    total_sentences = 0
    total_documents = 0
    
    for file_path in labeled_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            total_documents += 1
            labels = data.get("labels", [])
            total_sentences += len(labels)
            
            for label in labels:
                role = label.get("role", "Unknown")
                role_counts[role] = role_counts.get(role, 0) + 1
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    print(f"Total documents processed: {total_documents}")
    print(f"Total sentences labeled: {total_sentences:,}")
    print()
    
    print("Role distribution:")
    for role, count in sorted(role_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_sentences) * 100
        print(f"  {role}: {count:,} ({percentage:.1f}%)")
    
    print()
    print("Top 5 roles:")
    for role, count in list(sorted(role_counts.items(), key=lambda x: x[1], reverse=True))[:5]:
        percentage = (count / total_sentences) * 100
        print(f"  {role}: {count:,} ({percentage:.1f}%)")

def main():
    """Main function with menu options"""
    print("=== GOVERNMENT SCHEME ROLE LABELING SYSTEM ===")
    print()
    print("Options:")
    print("1. Process sample files (5 files) - for testing")
    print("2. Process all files - full processing")
    print("3. Analyze existing results")
    print("4. Exit")
    print()
    
    choice = input("Enter your choice (1-4): ").strip()
    
    if choice == "1":
        model = input("Enter model (gpt-3.5-turbo or gpt-4o) [gpt-3.5-turbo]: ").strip() or "gpt-3.5-turbo"
        process_sample_files(model)
    elif choice == "2":
        model = input("Enter model (gpt-3.5-turbo or gpt-4o) [gpt-3.5-turbo]: ").strip() or "gpt-3.5-turbo"
        batch_size = int(input("Enter batch size (5-50) [15]: ").strip() or "15")
        process_all_sentences(model, batch_size)
    elif choice == "3":
        analyze_labeled_results()
    elif choice == "4":
        print("Goodbye!")
    else:
        print("Invalid choice. Please run again.")

if __name__ == "__main__":
    main()

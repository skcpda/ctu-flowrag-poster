# src/image_gen/generate.py
import json
import openai
import base64
import requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import time
import subprocess
import sys
import os

# Configure OpenAI client
client = openai.OpenAI()

class ImageGenerator:
    """Generate images from prompts using DALL-E or Stable Diffusion."""
    
    def __init__(self, method: str = "dalle", api_key: Optional[str] = None):
        """
        Initialize image generator.
        
        Args:
            method: "dalle" or "stable_diffusion"
            api_key: OpenAI API key (optional, will use env var)
        """
        self.method = method
        if method == "dalle" and api_key:
            client.api_key = api_key
    
    def generate_with_dalle(self, prompt: str, size: str = "1024x1024") -> Dict:
        """
        Generate image using DALL-E 3.
        
        Args:
            prompt: Image prompt
            size: Image size (1024x1024, 1792x1024, 1024x1792)
            
        Returns:
            Dictionary with image data and metadata
        """
        # Short-circuit when no valid API key is set to avoid unnecessary errors/costs
        if not os.getenv("OPENAI_API_KEY"):
            return {
                'success': False,
                'error': 'OPENAI_API_KEY not set – skipping DALL-E call',
                'method': 'dalle',
                'original_prompt': prompt
            }

        try:
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=size,
                quality="standard",
                n=1
            )
            image_url = response.data[0].url
            revised_prompt = response.data[0].revised_prompt

            return {
                'success': True,
                'image_url': image_url,
                'revised_prompt': revised_prompt,
                'method': 'dalle',
                'original_prompt': prompt
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'method': 'dalle',
                'original_prompt': prompt
            }
    
    def generate_with_stable_diffusion(self, prompt: str, output_path: Path) -> Dict:
        """
        Generate image using local Stable Diffusion.
        
        Args:
            prompt: Image prompt
            output_path: Path to save generated image
            
        Returns:
            Dictionary with generation results
        """
        try:
            # This would require local SD installation
            # For now, return a placeholder
            return {
                'success': False,
                'error': 'Stable Diffusion not implemented - requires local installation',
                'method': 'stable_diffusion',
                'original_prompt': prompt
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'method': 'stable_diffusion',
                'original_prompt': prompt
            }
    
    def generate_image(self, prompt: str, output_path: Optional[Path] = None) -> Dict:
        """
        Generate image from prompt.
        
        Args:
            prompt: Image prompt
            output_path: Path to save image (for SD)
            
        Returns:
            Dictionary with generation results
        """
        if self.method == "dalle":
            result = self.generate_with_dalle(prompt)
            # If caller provided desired save location, download image
            if output_path and result.get("success") and result.get("image_url"):
                # Coerce to Path in case a string was supplied
                out_path = Path(output_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if download_image(result["image_url"], out_path):
                    result["local_path"] = str(out_path)
                else:
                    # flag failure to save locally but keep remote URL
                    result["save_error"] = "Failed to download image"
            return result
        elif self.method == "stable_diffusion":
            if not output_path:
                output_path = Path("generated_image.png")
            return self.generate_with_stable_diffusion(prompt, output_path)
        else:
            return {
                'success': False,
                'error': f'Unknown method: {self.method}',
                'method': self.method,
                'original_prompt': prompt
            }
    
    def batch_generate(self, prompts: List[str], output_dir: Path) -> List[Dict]:
        """
        Generate multiple images in batch.
        
        Args:
            prompts: List of image prompts
            output_dir: Directory to save images
            
        Returns:
            List of generation results
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []
        
        for i, prompt in enumerate(prompts):
            print(f"Generating image {i+1}/{len(prompts)}...")
            
            if self.method == "dalle":
                result = self.generate_with_dalle(prompt)
            else:
                output_path = output_dir / f"image_{i+1}.png"
                result = self.generate_with_stable_diffusion(prompt, output_path)
            
            result['prompt_index'] = i
            results.append(result)
            
            # Rate limiting for API calls
            if self.method == "dalle":
                time.sleep(1)
        
        return results

def download_image(url: str, output_path: Path) -> bool:
    """
    Download image from URL.
    
    Args:
        url: Image URL
        output_path: Path to save image
        
    Returns:
        True if successful, False otherwise
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        return True
        
    except Exception as e:
        print(f"Error downloading image: {e}")
        return False

def generate_poster_images(poster_data: List[Dict], output_dir: Path,
                          method: str = "dalle") -> List[Dict]:
    """
    Generate images for all posters.
    
    Args:
        poster_data: List of poster data dictionaries
        output_dir: Directory to save images
        method: Generation method ("dalle" or "stable_diffusion")
        
    Returns:
        List of generation results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    generator = ImageGenerator(method=method)
    
    results = []
    
    for poster in poster_data:
        poster_id = poster['poster_id']
        prompt = poster['image_prompt']
        
        print(f"Generating poster {poster_id}...")
        
        if method == "dalle":
            result = generator.generate_with_dalle(prompt)
            
            # Download image if successful
            if result['success'] and 'image_url' in result:
                image_path = output_dir / f"poster_{poster_id}.png"
                if download_image(result['image_url'], image_path):
                    result['local_path'] = str(image_path)
                else:
                    result['success'] = False
                    result['error'] = 'Failed to download image'
        else:
            image_path = output_dir / f"poster_{poster_id}.png"
            result = generator.generate_with_stable_diffusion(prompt, image_path)
            if result['success']:
                result['local_path'] = str(image_path)
        
        result['poster_id'] = poster_id
        result['role'] = poster['role']
        results.append(result)
        
        # Rate limiting
        if method == "dalle":
            time.sleep(1)
    
    return results

def main():
    """CLI interface for image generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate poster images")
    parser.add_argument("--input", required=True, help="Input poster JSON file")
    parser.add_argument("--output-dir", required=True, help="Output directory for images")
    parser.add_argument("--method", default="dalle", choices=["dalle", "stable_diffusion"],
                       help="Generation method")
    parser.add_argument("--results", help="Output JSON file for generation results")
    
    args = parser.parse_args()
    
    # Load poster data
    with open(args.input, 'r') as f:
        poster_data = json.load(f)
    
    # Generate images
    results = generate_poster_images(
        poster_data, 
        Path(args.output_dir), 
        method=args.method
    )
    
    # Save results
    if args.results:
        with open(args.results, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Generation results saved to {args.results}")
    
    # Print summary
    successful = sum(1 for r in results if r['success'])
    print(f"\nGeneration Summary:")
    print(f"Total posters: {len(poster_data)}")
    print(f"Successful generations: {successful}")
    print(f"Failed generations: {len(poster_data) - successful}")

if __name__ == "__main__":
    main() 
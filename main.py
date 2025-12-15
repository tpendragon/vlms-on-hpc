"""
OCR Processing Script for Vision-Language Models on HPC

This script processes images using vLLM and a vision-language model for OCR tasks.
It supports batch processing, concurrent job handling, and various image formats.
"""

import io
import base64
import sys
import logging
from pathlib import Path
from typing import Any, Dict, List, Union, Generator

from PIL import Image
from vllm import LLM, SamplingParams
from tqdm import tqdm
import srsly
from pillow_heif import register_heif_opener

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Register the HEIF opener for iOS image support
register_heif_opener()

# Configuration constants
INPUT_PATH = "images"
OUTPUT_PATH = "markdown"
MODEL_REPO = "nanonets/Nanonets-OCR-s"
CURRENT_FILES_JSON = "current_files.json"
MODEL_INFO_JSON = "model_info.json"

# Model parameters
BATCH_SIZE = 32
MAX_TOKENS = 4096
MAX_MODEL_LEN = 8192
GPU_MEMORY_UTILIZATION = 0.9

# Default OCR prompt
DEFAULT_OCR_PROMPT = Path('prompt.txt').read_text(encoding='utf-8')


def initialize_directories():
    """Create necessary directories if they don't exist."""
    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory '{OUTPUT_PATH}' is ready")


def get_model_path() -> str:
    """
    Retrieve the model path from model_info.json or use fallback.
    
    Returns:
        str: Path to the model
        
    Raises:
        FileNotFoundError: If model path doesn't exist
        ValueError: If model repo not found in model_info.json
    """
    if Path(MODEL_INFO_JSON).exists():
        logger.info(f"Reading model info from {MODEL_INFO_JSON}")
        model_info = srsly.read_json(MODEL_INFO_JSON)
        model_path = model_info.get(MODEL_REPO)
        
        if model_path is None:
            raise ValueError(f"Model path for {MODEL_REPO} not found in {MODEL_INFO_JSON}")
        
        logger.info(f"Using model from: {model_path}")
        return model_path
    else:
        # Fallback path - update this to your actual model path
        fallback_path = '/scratch/network/aj7878/.cache/huggingface/hub/models--nanonets--Nanonets-OCR-s/snapshots/3baad182cc87c65a1861f0c30357d3467e978172'
        logger.warning(f"{MODEL_INFO_JSON} not found, using fallback path: {fallback_path}")
        
        if not Path(fallback_path).exists():
            raise FileNotFoundError(f"Model path {fallback_path} does not exist. Run 'python fetch.py model {MODEL_REPO}' first.")
        
        return fallback_path


def initialize_llm(model_path: str) -> LLM:
    """
    Initialize the vLLM model.
    
    Args:
        model_path: Path to the model
        
    Returns:
        LLM: Initialized vLLM instance
        
    Raises:
        RuntimeError: If GPU is not available
    """
    try:
        logger.info("Initializing vLLM model...")
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            limit_mm_per_prompt={"image": 1},
        )
        logger.info("vLLM model initialized successfully")
        return llm
    except RuntimeError as e:
        logger.error(f"Failed to initialize vLLM: {e}")
        logger.error("vLLM requires GPU. Make sure you're running on a compute node with GPU access.")
        raise


sampling_params = SamplingParams(
    temperature=0.0,  # Deterministic for OCR
    max_tokens=MAX_TOKENS,
)


def make_ocr_message(
    image: Union[Image.Image, Dict[str, Any], str],
    prompt: str = DEFAULT_OCR_PROMPT,
) -> List[Dict]:
    """
    Create chat message for OCR processing.
    
    Args:
        image: Input image (PIL Image, dict with bytes, or file path)
        prompt: OCR instruction prompt
        
    Returns:
        List of message dictionaries in vLLM format
        
    Raises:
        ValueError: If image type is not supported
    """
    # Convert to PIL Image if needed
    if isinstance(image, Image.Image):
        pil_img = image
    elif isinstance(image, dict) and "bytes" in image:
        pil_img = Image.open(io.BytesIO(image["bytes"]))
    elif isinstance(image, str):
        pil_img = Image.open(image)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    # Convert to base64 data URI
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    data_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    # Return message in vLLM format
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def load_current_files() -> List[str]:
    """
    Load the list of currently processing files.
    
    Returns:
        List of file stems being processed
    """
    if Path(CURRENT_FILES_JSON).exists():
        return srsly.read_json(CURRENT_FILES_JSON)
    return []


def save_current_files(current_files: List[str]):
    """
    Save the list of currently processing files.
    
    Args:
        current_files: List of file stems being processed
    """
    srsly.write_json(CURRENT_FILES_JSON, current_files)


def files_to_process(input_path: str) -> List[Dict[str, Path]]:
    """
    Identify files that need to be processed.
    
    Skip files that already have markdown or are being processed by other jobs.
    
    Args:
        input_path: Path to the input files directory
        
    Returns:
        List of dictionaries with file_path and md_file Path objects
    """
    file_paths = Path(input_path).glob('*')
    images_to_process = []
    current_files = load_current_files()
    
    for file_path in file_paths:
        # Skip directories
        if not file_path.is_file():
            continue
            
        # Check if markdown file already exists
        md_file = Path(OUTPUT_PATH) / f"{file_path.stem}.md"
        if md_file.exists():
            logger.debug(f"Skipping {file_path.name}: markdown already exists")
            continue

        # Check if file is already being processed by other jobs
        if file_path.stem in current_files:
            logger.debug(f"Skipping {file_path.name}: already being processed")
            continue
        
        # Add to processing list and mark as current
        current_files.append(file_path.stem)
        images_to_process.append({
            "file_path": file_path,
            "md_file": md_file
        })
    
    # Save updated current files
    save_current_files(current_files)
    logger.info(f"Found {len(images_to_process)} images to process")
    
    return images_to_process


def batch_generator(
    images_to_process: List[Dict[str, Path]], 
    batch_size: int
) -> Generator[List[Dict[str, Any]], None, None]:
    """
    Generate batches of images for processing.
    
    Args:
        images_to_process: List of image information dictionaries
        batch_size: Number of images per batch
        
    Yields:
        Batches of dictionaries containing opened images and file paths
    """
    for i in range(0, len(images_to_process), batch_size):
        batch = images_to_process[i:i + batch_size]
        try:
            yield [
                {
                    "image": Image.open(img["file_path"]),
                    "file_path": img["file_path"],
                    "md_file": img["md_file"]
                }
                for img in batch
            ]
        except Exception as e:
            logger.error(f"Error loading batch starting at index {i}: {e}")
            # Try to process images 
            for img in batch:
                try:
                    yield [{
                        "image": Image.open(img["file_path"]),
                        "file_path": img["file_path"],
                        "md_file": img["md_file"]
                    }]
                except Exception as img_error:
                    logger.error(f"Failed to load image {img['file_path']}: {img_error}")


def process_batch(
    batch: List[Dict[str, Any]],
    llm: LLM,
    sampling_params: SamplingParams
):
    """
    Process a batch of images through the OCR model.
    
    Args:
        batch: List of image dictionaries to process
        llm: Initialized vLLM instance
        sampling_params: Sampling parameters for generation
    """
    batch_messages = [make_ocr_message(page["image"]) for page in batch]
    
    # Process with vLLM
    outputs = llm.chat(batch_messages, sampling_params)

    # Extract markdown from outputs and save
    current_files = load_current_files()
    
    for img, output in zip(batch, outputs):
        try:
            markdown_text = output.outputs[0].text.strip()
            img["md_file"].write_text(markdown_text, encoding='utf-8')
            logger.info(f"Processed: {img['file_path'].name} -> {img['md_file'].name}")
            
            # Remove from current files list
            if img["file_path"].stem in current_files:
                current_files.remove(img["file_path"].stem)
        except Exception as e:
            logger.error(f"Failed to save output for {img['file_path'].name}: {e}")
    
    # Update current files
    save_current_files(current_files)


def main():
    """Main execution function."""
    try:
        # Initialize
        initialize_directories()
        model_path = get_model_path()
        llm = initialize_llm(model_path)
        
        # Get files to process
        images = files_to_process(INPUT_PATH)
        
        if not images:
            logger.info("No images to process. All images may already be processed.")
            return
        
        # Process in batches
        total_batches = (len(images) + BATCH_SIZE - 1) // BATCH_SIZE
        logger.info(f"Processing {len(images)} images in {total_batches} batches")
        
        for batch in tqdm(
            batch_generator(images, BATCH_SIZE),
            total=total_batches,
            desc="Processing batches"
        ):
            process_batch(batch, llm, sampling_params)
        
        logger.info("Processing complete!")
        
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

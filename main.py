from PIL import Image
from vllm import LLM, SamplingParams
from pathlib import Path 
from tqdm import tqdm
from typing import Any, Dict, List, Union
import io
import base64
import srsly
from pillow_heif import register_heif_opener

# Register the HEIF opener
register_heif_opener()

input_path = "images"
output_path = "markdown"
model_repo = "nanonets/Nanonets-OCR-s"

if Path('model_info.json').exists():
    # Read the model_info.json file created by fetch.py model command
    model_info = srsly.read_json("model_info.json")
    model = model_info.get(model_repo)
    assert model is not None, f"Model path for {model_repo} not found in model_info.json"
else:
    # Enter model path directly
    model = '/scratch/network/aj7878/.cache/huggingface/hub/models--nanonets--Nanonets-OCR-s/snapshots/3baad182cc87c65a1861f0c30357d3467e978172'

batch_size = 32
max_tokens: int = 4096
max_model_len: int = 8192
gpu_memory_utilization: float = 0.9

llm = LLM(
    model=model,
    trust_remote_code=True,
    max_model_len=max_model_len,
    gpu_memory_utilization=gpu_memory_utilization,
    limit_mm_per_prompt={"image": 1},
)

sampling_params = SamplingParams(
    temperature=0.0,  # Deterministic for OCR
    max_tokens=max_tokens,
)

def make_ocr_message(
    image: Union[Image.Image, Dict[str, Any], str],
    prompt: str = "Extract the text from the above document as if you were reading it naturally. Return the tables in markdown format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes.",
) -> List[Dict]:
    """Create chat message for OCR processing."""
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

def files_to_process(input_path: str):
    """Identify files that need to be processed.
     Skip files that already have markdown or are being processed by other jobs.
     
     Args:
         input_path (str): Path to the input files.
         
         Returns:
         List of file paths to process.
    """
    file_paths = Path(input_path).glob('*')
    images_to_process = []
    for file_path in file_paths:
        # Check if markdown file already exists
        md_file = Path(output_path) / f"{file_path.stem}.md"
        if md_file.exists():
            continue

        # Check if file is already being processed by other jobs
        current_files = srsly.read_json("current_files.json")
        if file_path.stem in current_files:
            continue
        else:
            current_files.append(file_path.stem)
            srsly.write_json("current_files.json", current_files)

        images_to_process.append({
                    "file_path": file_path,
                    "md_file": md_file
                })
    return images_to_process


def batch_generator(images_to_process: list, batch_size: int):
    
    image_batches = [
        images_to_process[i:i + batch_size] for i in range(0, len(images_to_process), batch_size)
    ]
    for batch in image_batches:
        yield [{"image": Image.open(img["file_path"]), "file_path": img["file_path"], "md_file": img["md_file"]} for img in batch]                
        
images_to_process = files_to_process(input_path)

for batch in tqdm(batch_generator(images_to_process, batch_size), desc=f"Processing batches"):
    batch_messages = [make_ocr_message(page["image"]) for page in batch]
    
    # Process with vLLM
    outputs = llm.chat(batch_messages, sampling_params)

    # Extract markdown from outputs
    for img, output in zip(batch, outputs):
        markdown_text = output.outputs[0].text.strip()
        img["md_file"].write_text(markdown_text, encoding='utf-8')
        current_files = srsly.read_json("current_files.json")
        if img["file_path"].stem in current_files:
            current_files.remove(img["file_path"].stem)
            srsly.write_json("current_files.json", current_files)

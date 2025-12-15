from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, pipeline, Qwen2VLForConditionalGeneration
from vllm import LLM, SamplingParams
from pathlib import Path 
import pymupdf
from tqdm import tqdm
from typing import Any, Dict, List, Union
import io
import base64
import srsly
from pillow_heif import register_heif_opener

# Register the HEIF opener
register_heif_opener()

input_path = "input"
output_path = "markdown"
model: str = '/scratch/network/aj7878/.cache/huggingface/hub/models--nanonets--Nanonets-OCR-s/snapshots/3baad182cc87c65a1861f0c30357d3467e978172'
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
    #TODO add support for PDF and HEIC files
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

# create a generator that will yield the images 

def img_gen(input_path: str, pdf_dpi: int = 150):
    file_paths = Path(input_path).glob('*')
    images_to_process = []
    for file_path in file_paths:
        md_file = Path(output_path) / f"{file_path.stem}.md"
        if md_file.exists():
            continue

        current_files = srsly.read_json("current_files.json")
        if file_path.stem in current_files:
            #print(f"Skipping {pdf.stem} as it is already being processed by another job.")
            continue
        else:
            current_files.append(file_path.stem)
            srsly.write_json("current_files.json", current_files)
        
        
        
        if file_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif", ".heic"]:
            try:
                img = Image.open(file_path)
                images_to_process.append({
                    "image": img,
                    "page": file_path.name,
                })
            except Exception as e:
                print(f"Error opening {file_path}: {e}")
                continue
        
        if file_path.suffix.lower() == ".pdf":
            
            pdf_images = []
            try:
                doc = pymupdf.open(file_path)
                for i, page in tqdm(enumerate(doc)):  # iterate through the pages
                    pix = page.get_pixmap(dpi=pdf_dpi)  
                    img = pix.pil_image()
                    pdf_images.append({
                        "image": img,
                        "page": file_path.stem + "_" + str(i + 1),
                    })
            except Exception as e:
                print(f"Error opening {file_path}: {e}")
                continue

            pdf_images.sort(key=lambda x: x["page"])
            images_to_process.extend(pdf_images)
    total = len(images_to_process)
    for img in images_to_process:
        yield img, total
       
image_batches = [
    images_to_process[i:i + batch_size] for i in range(0, len(images_to_process), batch_size)
]

pdf_text = """"""
for batch in tqdm(image_batches, desc=f"Processing batches"):
    batch_messages = [make_ocr_message(page["image"]) for page in batch]
    
    # Process with vLLM
    outputs = llm.chat(batch_messages, sampling_params)

    # Extract markdown from outputs
    for output in outputs:
        markdown_text = output.outputs[0].text.strip()
        pdf_text += markdown_text + "\n\n"     
md_file.write_text(pdf_text, encoding='utf-8')
current_files = srsly.read_json("current_files.json")
if pdf.stem in current_files:
    current_files.remove(pdf.stem)
    srsly.write_json("current_files.json", current_files)

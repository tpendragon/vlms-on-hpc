import os
import srsly
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import math
import json 
import httpx
import asyncio
from IIIFTileSource import zoom_to_scale

def iiif_tile_download(manifest_url: str, output_folder:str, scale_factor: float = 0.8, testing: bool = False):
    # Fetch the manifest
    try:
        if manifest_url.startswith(('http://', 'https://')):
            header = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/11"
            }
            response = httpx.get(manifest_url, headers=header)
            response.raise_for_status()
            manifest = response.json()
        else:
            # Treat as local file path
            with open(manifest_url, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load manifest from {manifest_url}: {e}")
    
    
    image_tile_uris = zoom_to_scale(manifest_url, scale_factor)     
    # no output_folder folder create one
    os.makedirs(output_folder, exist_ok=True)

    images = image_tile_uris.get('images', [])
    if testing:
        images = images[:10]
    info = {}
    info['url'] = manifest_url
    info['images'] = {}
    info['metadata'] = manifest.get('metadata', {})
    info['label'] = manifest.get('label', '')
    
    for image in tqdm(images):
        image_filename = "_".join(image['image_id'].split('/')[4:6])
        if '.jp2' in image_filename:
            image_filename = image_filename.replace('.jp2', '.jpg')
            info['images'][image_filename] = image['image_id']
        else:
            image_filename = image_filename + ".jpg"
            info['images'][image_filename] = image['image_id']

        # Get image info to determine proper grid dimensions
        tile_urls = image.get("tile_urls", [] )
        img_width = image['width']
        img_height = image['height']
        scale_factor = image['scaleFactor']

        # Calculate the actual tile grid dimensions for this zoom level
        tile_size = 256  # Standard IIIF tile size
        level_width = math.ceil(img_width * scale_factor)
        level_height = math.ceil(img_height * scale_factor)
        tiles_x = math.ceil(level_width / tile_size)
        tiles_y = math.ceil(level_height / tile_size)

        # Fetch all tiles
        tile_images = []
        
        # Fetch all tiles
        async def fetch_tile(client, url, idx):
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                return idx, resp.content
            except Exception:
                return idx, None

        async def fetch_all(urls):
            limits = httpx.Limits(max_connections=20)
            timeout = httpx.Timeout(30.0)
            async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
                tasks = [fetch_tile(client, u, i) for i, u in enumerate(urls)]
                return await asyncio.gather(*tasks)
        results = asyncio.run(fetch_all(tile_urls))

        # Recreate tile_images in the original order; insert a blank tile on failure
        tile_images = [None] * len(tile_urls)
        for idx, content in results:
            if content is None:
                tile = Image.new("RGB", (tile_size, tile_size), (255, 255, 255))
            else:
                tile = Image.open(BytesIO(content)).convert("RGB")
            tile_images[idx] = tile

        # Create the combined image with proper dimensions
        # Use the actual scaled dimensions, not tile_size multiples
        combined_image = Image.new("RGB", (level_width, level_height))

        # Place tiles in their correct positions
        tile_index = 0
        for y in range(tiles_y):
            for x in range(tiles_x):
                if tile_index < len(tile_images):
                    tile = tile_images[tile_index]

                    # Calculate position in the combined image
                    pos_x = x * tile_size
                    pos_y = y * tile_size

                    # Handle edge tiles that might be smaller
                    tile_width = min(tile_size, level_width - pos_x)
                    tile_height = min(tile_size, level_height - pos_y)

                    # Resize tile if it's at the edge and smaller than expected
                    if tile.size != (tile_width, tile_height):
                        tile = tile.resize((tile_width, tile_height))

                    combined_image.paste(tile, (pos_x, pos_y))
                    tile_index += 1
        combined_image.save(f"{output_folder}/{image_filename}")

    srsly.write_json(f'{output_folder}/info.json',info)
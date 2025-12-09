import json
import requests
import math
from typing import Dict, List, Tuple, Any


class IIIFTileSource:
    """IIIF Tile URL generator based on OpenSeadragon IIIFTileSource logic.
    https://github.com/openseadragon/openseadragon/blob/5ff35848d4960ff4d57e2a8979ce6ea780ad4122/src/iiiftilesource.js
    """

    def __init__(self):
        self.default_tile_size = 256

    def load_manifest(self, manifest_source: str) -> Dict[str, Any]:
        """
        Load IIIF manifest from URL or local file path.

        Args:
            manifest_source: URL or file path to IIIF manifest

        Returns:
            Parsed IIIF manifest as dictionary
        """
        try:
            # Check if it's a URL
            if manifest_source.startswith(('http://', 'https://')):
              header = {
               "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/11"
              }
              response = requests.get(manifest_source, headers=header)
              response.raise_for_status()
              return response.json()
            else:
                # Treat as local file path
                with open(manifest_source, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load manifest from {manifest_source}: {e}")

    def detect_iiif_version(self, service_info: Dict[str, Any]) -> int:
        """
        Detect IIIF version from service information.

        Args:
            service_info: IIIF image service info

        Returns:
            IIIF version (1, 2, or 3)
        """
        context = service_info.get('@context', '')
        profile = service_info.get('profile', '')

        if isinstance(context, list):
            for ctx in context:
                if 'image/3/context.json' in str(ctx):
                    return 3
                elif 'image/2/context.json' in str(ctx):
                    return 2
                elif 'image-api/1.1/context.json' in str(ctx) or 'image/1/context.json' in str(ctx):
                    return 1
        elif isinstance(context, str):
            if 'image/3/context.json' in context:
                return 3
            elif 'image/2/context.json' in context:
                return 2
            elif 'image-api/1.1/context.json' in context or 'image/1/context.json' in context:
                return 1

        # Check profile for version clues
        if isinstance(profile, list) and len(profile) > 0:
            profile_str = str(profile[0])
        else:
            profile_str = str(profile)

        if 'level0.json' in profile_str or 'level1.json' in profile_str or 'level2.json' in profile_str:
            return 2
        elif 'level0' in profile_str or 'level1' in profile_str or 'level2' in profile_str:
            return 3
        elif 'compliance.html' in profile_str:
            return 1

        # Default to version 2 if uncertain
        return 2

    def get_tile_info(self, service_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract tile configuration from IIIF service info.

        Args:
            service_info: IIIF image service info

        Returns:
            Dictionary with tile configuration
        """
        tile_info = {
            'width': self.default_tile_size,
            'height': self.default_tile_size,
            'scale_factors': [1, 2, 4, 8, 16, 32]
        }

        # IIIF 2.0+ tiles array
        if 'tiles' in service_info and service_info['tiles']:
            tiles = service_info['tiles'][0]  # Use first tile config
            tile_info['width'] = tiles.get('width', self.default_tile_size)
            tile_info['height'] = tiles.get('height', tile_info['width'])
            if 'scaleFactors' in tiles:
                tile_info['scale_factors'] = tiles['scaleFactors']

        # IIIF 1.x tile_width/tile_height
        elif 'tile_width' in service_info:
            tile_info['width'] = service_info['tile_width']
            tile_info['height'] = service_info.get('tile_height', tile_info['width'])

        return tile_info

    def calculate_max_level(self, width: int, height: int, scale_factors: List[int] = None) -> int:
        """
        Calculate maximum zoom level for the image.

        Args:
            width: Image width
            height: Image height
            scale_factors: Available scale factors

        Returns:
            Maximum level
        """
        if scale_factors:
            max_scale_factor = max(scale_factors)
            return round(math.log(max_scale_factor, 2))
        else:
            return round(math.log(max(width, height), 2))

    def scale_factor_to_level(self, scale_factor: float, max_level: int) -> int:
        """
        Convert scale factor to zoom level.

        Args:
            scale_factor: Scale factor (e.g., 0.25, 0.0625)
            max_level: Maximum zoom level for the image

        Returns:
            Corresponding zoom level
        """
        if scale_factor <= 0 or scale_factor > 1:
            raise ValueError("Scale factor must be between 0 and 1")

        # Formula: scale = 0.5^(max_level - level)
        # Solving for level: level = max_level - log2(1/scale) = max_level + log2(scale)
        level = max_level + math.log2(scale_factor)
        return max(0, min(round(level), max_level))

    def level_to_scale_factor(self, level: int, max_level: int) -> float:
        """
        Convert zoom level to scale factor.

        Args:
            level: Zoom level
            max_level: Maximum zoom level for the image

        Returns:
            Corresponding scale factor
        """
        return math.pow(0.5, max_level - level)

    def get_num_tiles(self, level: int, width: int, height: int, tile_width: int, tile_height: int, max_level: int) -> Tuple[int, int]:
        """
        Calculate number of tiles at given level.

        Args:
            level: Zoom level
            width: Image width
            height: Image height
            tile_width: Tile width
            tile_height: Tile height
            max_level: Maximum level

        Returns:
            Tuple of (tiles_x, tiles_y)
        """
        scale = math.pow(0.5, max_level - level)
        level_width = math.ceil(width * scale)
        level_height = math.ceil(height * scale)

        tiles_x = math.ceil(level_width / tile_width)
        tiles_y = math.ceil(level_height / tile_height)

        return tiles_x, tiles_y

    def get_tile_url(self, service_id: str, level: int, x: int, y: int,
                     width: int, height: int, tile_width: int, tile_height: int,
                     max_level: int, version: int, tile_format: str = 'jpg') -> str:
        """
        Generate IIIF tile URL for specific tile coordinates.

        Args:
            service_id: IIIF service base URL
            level: Zoom level
            x: Tile x coordinate
            y: Tile y coordinate
            width: Full image width
            height: Full image height
            tile_width: Tile width
            tile_height: Tile height
            max_level: Maximum zoom level
            version: IIIF version
            tile_format: Image format

        Returns:
            Complete IIIF tile URL
        """
        # Constants
        IIIF_ROTATION = '0'

        # Calculate scale for this level
        scale = math.pow(0.5, max_level - level)

        # Level dimensions
        level_width = math.ceil(width * scale)
        level_height = math.ceil(height * scale)

        # Tile size in original image coordinates
        iiif_tile_size_width = round(tile_width / scale)
        iiif_tile_size_height = round(tile_height / scale)

        # Quality parameter
        if version == 1:
            iiif_quality = f"native.{tile_format}"
        else:
            iiif_quality = f"default.{tile_format}"

        # Handle single tile case
        if level_width < tile_width and level_height < tile_height:
            if version == 2 and level_width == width:
                iiif_size = "full"
            elif version == 3 and level_width == width and level_height == height:
                iiif_size = "max"
            elif version == 3:
                iiif_size = f"{level_width},{level_height}"
            else:
                iiif_size = f"{level_width},"
            iiif_region = 'full'
        else:
            # Calculate tile position and dimensions
            iiif_tile_x = x * iiif_tile_size_width
            iiif_tile_y = y * iiif_tile_size_height
            iiif_tile_w = min(iiif_tile_size_width, width - iiif_tile_x)
            iiif_tile_h = min(iiif_tile_size_height, height - iiif_tile_y)

            if x == 0 and y == 0 and iiif_tile_w == width and iiif_tile_h == height:
                iiif_region = "full"
            else:
                iiif_region = f"{iiif_tile_x},{iiif_tile_y},{iiif_tile_w},{iiif_tile_h}"

            # Calculate output size
            iiif_size_w = min(tile_width, level_width - (x * tile_width))
            iiif_size_h = min(tile_height, level_height - (y * tile_height))

            if version == 2 and iiif_size_w == width:
                iiif_size = "full"
            elif version == 3 and iiif_size_w == width and iiif_size_h == height:
                iiif_size = "max"
            else:
                if version < 3:
                    iiif_size = f"{iiif_size_w},"
                else:
                    iiif_size = f"{iiif_size_w},{iiif_size_h}"

        # Construct URL: {scheme}://{server}/{prefix}/{identifier}/{region}/{size}/{rotation}/{quality}
        return f"{service_id}/{iiif_region}/{iiif_size}/{IIIF_ROTATION}/{iiif_quality}"

    def extract_image_info(self, canvas: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract image service information from a canvas.
        Supports both IIIF Presentation API v2 and v3.

        Args:
            canvas: IIIF canvas object

        Returns:
            List of image info dictionaries
        """
        images = []

        # Presentation API v2: uses 'images' property
        if 'images' in canvas:
            for image_annotation in canvas['images']:
                resource = image_annotation.get('resource', {})
                service = resource.get('service', {})

                if not service:
                    continue

                # Handle case where service is a list
                if isinstance(service, list):
                    service = service[0]

                service_id = service.get('@id') or service.get('id')
                if not service_id:
                    continue

                # Get image dimensions from canvas (preferred) or service as fallback
                img_width = canvas.get('width') or service.get('width')
                img_height = canvas.get('height') or service.get('height')

                if not (img_width and img_height):
                    continue

                version = self.detect_iiif_version(service)
                tile_info = self.get_tile_info(service)

                # Handle label (v2 can be string or object)
                canvas_label = canvas.get('label', '')
                if isinstance(canvas_label, dict):
                    # Extract first language value if label is an object
                    canvas_label = next(iter(canvas_label.values()), [''])[0] if canvas_label else ''

                images.append({
                    'service_id': service_id,
                    'width': img_width,
                    'height': img_height,
                    'version': version,
                    'tile_info': tile_info,
                    'canvas_label': canvas_label,
                    'image_id': resource.get('@id') or resource.get('id', '')
                })

        # Presentation API v3: uses 'items' property
        elif 'items' in canvas:
            for annotation_page in canvas['items']:
                # annotation_page could be direct items array or have items property
                annotations = annotation_page if isinstance(annotation_page, list) else annotation_page.get('items', [])

                for annotation in annotations:
                    body = annotation.get('body', {})

                    # Body could be a list
                    if isinstance(body, list):
                        body = body[0] if body else {}

                    service = body.get('service')

                    if not service:
                        continue

                    # Handle case where service is a list
                    if isinstance(service, list):
                        service = service[0]

                    service_id = service.get('@id') or service.get('id')
                    if not service_id:
                        continue

                    # Get image dimensions from canvas (preferred) or service as fallback
                    img_width = canvas.get('width') or service.get('width')
                    img_height = canvas.get('height') or service.get('height')

                    if not (img_width and img_height):
                        continue

                    version = self.detect_iiif_version(service)
                    tile_info = self.get_tile_info(service)

                    # Handle label (v3 uses language map)
                    canvas_label = canvas.get('label', '')
                    if isinstance(canvas_label, dict):
                        # Extract first language value
                        canvas_label = next(iter(canvas_label.values()), [''])[0] if canvas_label else ''

                    images.append({
                        'service_id': service_id,
                        'width': img_width,
                        'height': img_height,
                        'version': version,
                        'tile_info': tile_info,
                        'canvas_label': canvas_label,
                        'image_id': body.get('@id') or body.get('id', '')
                    })

        return images


def zoom_tiles(manifest_source: str, scale_factor: float = None, zoom_level: int = None,
               all_levels: bool = True) -> Dict[str, Any]:
    """
    Main function to generate IIIF tile URLs from a manifest.
    Supports both IIIF Presentation API v2 and v3.

    Args:
        manifest_source: URL or file path to IIIF manifest
        scale_factor: Specific scale factor to generate tiles for (e.g., 0.25 for 25%)
        zoom_level: Specific zoom level to generate tiles for (alternative to scale_factor)
        all_levels: If True, generate tiles for all zoom levels (default behavior)

    Returns:
        Dictionary with manifest URI and list of images with tile URLs

    Note:
        - If scale_factor is provided, only that scale factor will be used
        - If zoom_level is provided, only that level will be used
        - scale_factor takes precedence over zoom_level if both are provided
        - If neither is provided, all_levels=True generates all levels
    """
    fetcher = IIIFTileSource()

    # Load manifest
    manifest = fetcher.load_manifest(manifest_source)

    result = {
        "manifest_uri": manifest_source,
        "images": []
    }

    # Detect manifest version and get canvases
    canvases = []

    # Presentation API v3: uses 'items' directly
    if 'items' in manifest:
        canvases = manifest['items']
    # Presentation API v2: uses 'sequences' -> 'canvases'
    elif 'sequences' in manifest:
        sequences = manifest.get('sequences', [])
        if sequences:
            canvases = sequences[0].get('canvases', [])

    if not canvases:
        return result

    for canvas in canvases:
        images_info = fetcher.extract_image_info(canvas)

        for img_info in images_info:
            service_id = img_info['service_id']
            width = img_info['width']
            height = img_info['height']
            version = img_info['version']
            tile_info = img_info['tile_info']

            tile_width = tile_info['width']
            tile_height = tile_info['height']
            scale_factors = tile_info['scale_factors']

            max_level = fetcher.calculate_max_level(width, height, scale_factors)

            # Determine which levels to generate
            if scale_factor is not None:
                # Use specific scale factor
                target_level = fetcher.scale_factor_to_level(scale_factor, max_level)
                levels_to_generate = [target_level]
                actual_scale_factor = fetcher.level_to_scale_factor(target_level, max_level)
            elif zoom_level is not None:
                # Use specific zoom level
                target_level = max(0, min(zoom_level, max_level))
                levels_to_generate = [target_level]
                actual_scale_factor = fetcher.level_to_scale_factor(target_level, max_level)
            else:
                # Generate all levels (default behavior)
                levels_to_generate = list(range(max_level + 1))
                actual_scale_factor = 1.0  # Base scale factor for full resolution

            # Generate tile URLs for specified levels
            all_tile_urls = []

            for level in levels_to_generate:
                tiles_x, tiles_y = fetcher.get_num_tiles(
                    level, width, height, tile_width, tile_height, max_level
                )

                for y in range(tiles_y):
                    for x in range(tiles_x):
                        tile_url = fetcher.get_tile_url(
                            service_id, level, x, y, width, height,
                            tile_width, tile_height, max_level, version
                        )
                        all_tile_urls.append(tile_url)

            # Create image entry
            image_entry = {
                "image_id": img_info['image_id'],
                "service_id": service_id,
                "width": width,
                "height": height,
                "version": version,
                "canvas_label": img_info['canvas_label'],
                "scaleFactor": actual_scale_factor,
                "zoom_levels": levels_to_generate,
                "max_level": max_level,
                "tile_urls": all_tile_urls
            }

            result["images"].append(image_entry)

    return result


def zoom_to_scale(manifest_source: str, scale_factor: float) -> Dict[str, Any]:
    """
    Convenience function to get tiles for a specific scale factor.

    Args:
        manifest_source: URL or file path to IIIF manifest
        scale_factor: Scale factor (e.g., 0.25 for 25%, 0.0625 for 6.25%)

    Returns:
        Dictionary with tile URLs for the specified scale factor
    """
    return zoom_tiles(manifest_source, scale_factor=scale_factor)


def zoom_to_level(manifest_source: str, zoom_level: int) -> Dict[str, Any]:
    """
    Convenience function to get tiles for a specific zoom level.

    Args:
        manifest_source: URL or file path to IIIF manifest
        zoom_level: Zoom level (0 = most zoomed out, max_level = full resolution)

    Returns:
        Dictionary with tile URLs for the specified zoom level
    """
    return zoom_tiles(manifest_source, zoom_level=zoom_level)



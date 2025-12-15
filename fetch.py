import typer
import srsly
from huggingface_hub import snapshot_download
from pathlib import Path
from datasets import Dataset

app = typer.Typer()


@app.command()
def model(
        repo_id: str
    ):
    """
    Downloads a model from Hugging Face Hub.
    """
    output_path = snapshot_download(repo_id=repo_id, repo_type="model")
    srsly.write_json("model_info.json", {repo_id: output_path})
    print(f"Model downloaded to: {output_path}")

@app.command()
def images(
    manifest_url: str,
):
    """
    Downloads images from a IIIF manifest.
    """
    print(manifest_url)
    print('in progress...need to add this code')

@app.command()
def to_hub(
    repo_id: str,
    public: bool = typer.Option(
        False, "--public", help="Make the dataset public (default: private)"
    ),
):
    """
    Uploads the results to Hugging Face Hub.
    """
    data = []
    info = srsly.read_json("img/info.json")
    images = Path('img').glob('*.jpg')
    for img_path in images:
        img = {}
        img["name"] = img_path.name
        img['manifest_url'] = info['url']
        img["image_url"] = info.get('images').get(img_path.name)
        md_path = img_path.with_suffix(".md")
        if md_path.exists():
            img["text"] = md_path.read_text()
        data.append(img)
    dataset = Dataset.from_list(data)
    dataset.push_to_hub(repo_id, private=not public)
        

if __name__ == "__main__":
    app()

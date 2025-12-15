import re
import typer
import json
from pathlib import Path

app = typer.Typer()

def remove_code_tags(text):
    """Remove code tags (triple backticks) from the text. Also remove tags with language specifiers such as ```yaml."""
    return re.sub(r"```[a-zA-Z]*\n(.*?)\n```", r"\1", text, flags=re.DOTALL)

def remove_repeated_phrases(text):
    """Remove repeated phrases in the text."""
    # This regex looks for any phrase (sequence of words) that is repeated consecutively
    pattern = r"(\b\w+(?: \w+){0,5}\b)( \1)+"
    return re.sub(pattern, r"\1", text)

def clean_json(invalid_json:str) -> json:
    output = []

    # Remove leading/trailing fence lines like ```json ... ``` (tolerant)
    s = invalid_json.strip()
    if s.startswith('```'):
        # drop first fence line
        parts = s.split('\n', 1)
        if len(parts) == 2:
            s = parts[1]
        # drop trailing fence if present
        if s.endswith('```'):
            s = s[:-3].rstrip()

    # Try a strict JSON parse first (fast and correct when valid)
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
        return []
    except Exception:
        # fall through to tolerant parsing
        pass

    # Fallback: extract {...} blocks and parse fields tolerant to unescaped quotes
    blocks = re.findall(r'\{.*?\}', s, re.DOTALL)
    for block in blocks:
        ent = {}
        # entity and type are usually safe to capture with a simple regex
        m_entity = re.search(r'"entity"\s*:\s*"([^"]*)"', block)
        if m_entity:
            ent['entity'] = m_entity.group(1).strip()
        m_type = re.search(r'"type"\s*:\s*"([^"]*)"', block)
        if m_type:
            ent['type'] = m_type.group(1).strip()
        # context may contain unescaped quotes; capture between the first quote after the colon
        # and the last quote in the block (heuristic but works for this data)
        idx = block.find('"context"')
        if idx != -1:
            pos_col = block.find(':', idx)
            first_q = block.find('"', pos_col + 1)
            last_q = block.rfind('"')
            if first_q != -1 and last_q > first_q:
                context = block[first_q + 1:last_q]
                ent['context'] = context.strip()
        if ent:
            output.append(ent)

    return output

@app.command()
def clean(
        markdown_dir: str
    ):
    """
    Processes markdown files in the specified directory to clean text and extract JSON data.
    """
    md_path = Path(markdown_dir)
    md_files = list(md_path.glob('*.md'))
    for md_file in md_files:
        text = md_file.read_text(encoding='utf-8')
        # Clean the text
        cleaned_text = remove_code_tags(text)
        cleaned_text = remove_repeated_phrases(cleaned_text)
        md_file.write_text(cleaned_text, encoding='utf-8')    

if __name__ == "__main__":
    app()

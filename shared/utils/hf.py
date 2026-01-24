"""HuggingFace utilities for LTX-2 pipeline."""


def build_hf_url(repo_id: str, folder: str, filename: str) -> str:
    """Build a HuggingFace URL for a file."""
    if folder:
        return f"https://huggingface.co/{repo_id}/resolve/main/{folder}/{filename}"
    return f"https://huggingface.co/{repo_id}/resolve/main/{filename}"

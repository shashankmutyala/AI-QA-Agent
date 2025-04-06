from huggingface_hub import snapshot_download

# Download with more detailed progress and error reporting
model_path = snapshot_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    local_dir_use_symlinks=False,
    local_dir="./downloaded_model"
)
print(f"Model downloaded to: {model_path}")
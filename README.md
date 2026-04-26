# SemFile

Semantic file search using multimodal embeddings. Find images, videos, audio, and documents by describing what you're looking for in natural language.

Uses [Gemini Embedding 2](https://ai.google.dev/gemini-api/docs/embeddings) to embed files into a shared vector space, enabling cross-modal search (e.g., find images by describing them with text).

## Setup

Requires Python 3.13+, [uv](https://docs.astral.sh/uv/), and [ffmpeg](https://ffmpeg.org/) (for video thumbnails).

```bash
# Clone and install
cd SemFile
uv sync

# Set your Gemini API key
echo "GEMINI_API_KEY='your-key-here'" > .env
```

## Usage

### Index files

```bash
# Index all configured watch directories
semfile index

# Index a specific directory
semfile index --path ~/Photos/Iceland

# Index only certain file types
semfile index --type image
semfile index --type video --type image
semfile index --path ~/Media --type video
```

### Search

```bash
# Basic search
semfile search "sunset on the beach"

# Filter by file type
semfile search "dogs playing" --type image

# Scope to specific directories
semfile search "drone footage" --dir ~/Videos/Iceland
semfile search "waves" --dir ~/Videos/Bali --dir ~/Photos/Bali

# Limit results
semfile search "mountains" --limit 5
```

### Other commands

```bash
# Show index stats and storage usage
semfile status

# Show or create config file
semfile config

# Clear all indexed data
semfile reset
```

## Configuration

Config lives at `~/.config/semfile/config.toml`. Created automatically on first run.

```toml
[semfile]
db_path = "~/.local/share/semfile/db"
thumbnail_dir = "~/.local/share/semfile/thumbnails"
embedding_dimensions = 768

[[semfile.watch]]
path = "~/Pictures"
extensions = ["png", "jpg", "jpeg", "webp"]

[[semfile.watch]]
path = "~/Videos"
extensions = ["mp4", "mov"]

[[semfile.watch]]
path = "~/Documents"
extensions = ["pdf", "txt", "md"]
```

## Supported file types

| Type | Extensions |
|---|---|
| Image | png, jpg, jpeg, webp, gif, bmp, tiff |
| Video | mp4, mov, avi, mkv, webm |
| Audio | mp3, wav, flac, aac, ogg, m4a |
| Document | pdf |
| Text | txt, md, rst, csv, json, xml, html |

## Storage

Embedding database and thumbnails are stored in `~/.local/share/semfile/`. At 768 dimensions, expect roughly **~4 GB per 100K files** (embeddings + thumbnails).

## How it works

1. **Index**: Scans configured directories, sends files to Gemini Embedding 2 API, stores 768-dim vectors in ChromaDB with file metadata
2. **Search**: Embeds your text query into the same vector space, finds nearest neighbors via cosine similarity
3. **Incremental**: Only re-embeds files that have changed (based on modification time)

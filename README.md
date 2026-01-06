<div align="center">
<a href="https://github.com/artpods56/KUL_Notarius" title="TrendRadar">
  <img src="docs/assets/logo.png" alt="Notarius Logo" width="50%">
</a>

**Historical Schematism Indexing & Extraction Engine**

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![beartype](https://raw.githubusercontent.com/beartype/beartype-assets/main/badge/bear-ified.svg)](https://github.com/beartype/beartype)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

## About

**Notarius** is a specialized  data extraction service built at the Centre for Medieval Studies in Lublin. 

## Quick Start

**Prerequisites:** Python 3.12+, Docker, CUDA capable GPU (for local OCR inference)

```bash
# Install dependencies
uv sync

# Set up environment
cp .env.example .env

# Install pre-commit hooks
pre-commit install

# Run tests
uv run pytest

# Run dagster
dagster dev
```

## Documentation

- [Installation Guide](docs/INSTALLATION.md) - Detailed setup instructions for different environments
- [Technical documentation](docs/TECHNICAL_DOCUMENTATION.md) - Explanation of the indexing and extraction strategies
- [Contributing](docs/CONTRIBUTING.md) - How to contribute to the project
- [Conventional Commits](docs/CONVENTIONAL_COMMITS.md) - Commit message conventions
- [Architecture](docs/ARCHITECTURE.md) - System design and ETL pipeline structure

## License

MIT License - see [LICENSE](LICENSE) file for details.
```
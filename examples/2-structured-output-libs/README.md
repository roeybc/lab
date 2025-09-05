# Structured Output Overview

## Requirements

1. call `uv sync`
2. run `cp .env.cp .env`
3. Get an OpenIA key and copy it to the file
4. Initialize baml: `uv run baml-cli generate

## Sanity-check each framework

```bash
uv run instructor_demo_basic
uv run pydanticai_demo_basic
uv run baml_demo_basic
```

## Run the accuracy benchmark

```basg
uv run benchmark_accuracy
```
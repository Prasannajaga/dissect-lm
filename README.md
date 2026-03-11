# dissectlm

`dissectlm` is a fast, metadata-first CLI for inspecting model architecture and parameter distribution without loading full model weights.

## Build

```bash
cargo build --release
```

## Usage

```bash
dissectlm gpt2
dissectlm gpt2 --params
dissectlm gpt2 --graph
dissectlm gpt2 --attention-breakdown
dissectlm compare gpt2 meta-llama/Llama-2-7b-hf
```

JSON output:

```bash
dissectlm gpt2 --json
dissectlm compare gpt2 gpt2-medium --json
```

Deep inspection (optional Python environment):

```bash
dissectlm gpt2 --deep
```

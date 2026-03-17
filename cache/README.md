# cache

This folder is the cache directory for Hugging Face (HF).

When using online mode, downloaded models will be cached in this folder.

For [offline mode](https://huggingface.co/docs/transformers/main/installation#offline-mode) use, please download the models in advance and specify the model directory.


## Directory structures

**There are currently two directory structures:**
- The Huggingface CLI cache structure

### Huggingface CLI cache structure

use this command `tree -a ./cache/huggingface/hub`

```
./cache/
├── .gitignore
├── huggingface
│   └── hub
│       └── models--xiaoyao9184--dall-e
│           └── refs
│               └── master
├── models
│   ├── .cache
│   │   └── huggingface
│   │       ├── download
│   │       │   ├── decoder.pkl.lock
│   │       │   ├── decoder.pkl.metadata
│   │       │   ├── encoder.pkl.lock
│   │       │   ├── encoder.pkl.metadata
│   │       │   ├── .gitattributes.lock
│   │       │   └── .gitattributes.metadata
│   │       └── .gitignore
│   ├── decoder.pkl
│   ├── encoder.pkl
│   ├── .gitattributes
│   └── .gitkeep
└── README.md

9 directories, 14 files
```

## Pre-download for offline mode

Running in online mode will automatically download the model.

### use huggingface-cli

install cli

```bash
pip install -U "huggingface_hub[cli]"
```

download model

```bash
hf download --revision master --cache-dir ./cache/huggingface/hub --local-dir ./cache/models xiaoyao9184/dall-e
```

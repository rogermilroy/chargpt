# CharGPT

---

A little project to build a simple character-level GPT (Generative Pretrained Transformer).

The purpose was for fun and to familiarise myself with the Transformer architecture a little better for other projects.

## Getting Started

You will need `poetry` as that is the dependency manager I have used. If there is demand (unlikely I know) I can add
setup.py and requirements.txt for people who prefer venv or some other dependency manager - open an Issue if you want that.

There is a `main.py` file which contains the training code - I may restructure this in the future.

## Under Development

There are a few things that I will be adding, basically I am using this repo as a test bed for nice-to-haves for
ML development. A non-exhaustive list is below, it probably will become out of date.

- Config management - I will probably use Hydra and yaml files to manage configs. This will also produce the structure
  for the run logs.
- Improved logging - I want to track more things better for each run. I may use Tensorboard in some capacity for this.
- Checkpoint management - This will come in with the above two but the additional will be adding loading of past checkpoints.

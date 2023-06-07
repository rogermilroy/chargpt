# CharGPT

---

## A simple character level GPT for learning and experimentation. 

A little project to build a simple character-level GPT (Generative Pretrained Transformer).
This repo was created originally following along with Andrej Karpathy's video 
[Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY) and subsequently continued with. 

The goal of the repo is to learn more about Transformers in general, test out good training pipelines, 
experiment with different techniques and generally to explore but within the context of the original model - 
ie. character level GPTs. I may add some different tokenizers or datasets for testing and comparison at some stage though.


## Getting Started

You will need `poetry` as that is the dependency manager I have used. If there is demand (unlikely I know) I can add
setup.py and requirements.txt for people who prefer venv or some other dependency manager - open an Issue if you want that.

There is a `main.py` file which contains the training code - I may restructure this in the future.

## Under Development

There are a few things that I will be adding, basically I am using this repo as a test bed for nice-to-haves for
ML development. A non-exhaustive list is below, it probably will become out of date.

- Config management - I have a v0 using hydra to manage configuration.
- Improved logging - I want to track more things better for each run. I may use Tensorboard in some capacity for this.
- Checkpoint management - I have a v0 now implemented that builds on hydra to organise the outputs within the hydra created
output folders.

# Socratic Models: Composing Zero-Shot Multimodal Reasoning with Language

**Abstract** Socratic models is a modular framework in which multiple pre-trained models are composed zeroshot via multimodal informed prompting. This is done to exchange information between models and capture new multimodal capabilities, without requiring finetuning. As a proof of concept, we modify the Socratic models framework such that it is entirely open-source and attempt to achieve the same results as the original version. Additionally, we investigate the capabilities of Socratic models on multimodal reasoning tasks such as chain-of-thought reasoning and visual question-answering in zeroshot and few-shot settings.


## Install
To install the environment, run:

`conda env create -f environment.yml`  
`conda activate socratic`  
`python -m spacy download en`

## Instructions
This repository provides scripts for CLIP with GPT-3, FLAN-T5, GitVision, BLIP and BLIP2 prompting, and self-contained ipython notebooks with prototype implementations of Socratic Models for image captioning geenration, chain-of-thought and visual question answering.

## Notes on files in this repository
* `blogpost.md`: the research corresponding to experiments documented in this repository

* **blogpost_images**: contains images for the `blogpost.md` report

* **scripts**
  * `coco_captioning_baseline.py`
  * `coco_captioning_baseline_test.py`
  * `coco_captioning_improved.py`
  * `coco_captioning_improved_test.py`
  * `coco_captioning_gpt.py`
  * `vlm_alternative-gitvision_test.py`
  * `vlm_alternative-blip_test.py`
  * `vlm_alternative-blip2_test.py`
  * `image_captioning.py`
  * `utils.py`
  * `coco_evaluation.py`

* **notebooks**
    * `demo_baseline.ipynb`
    * `demo_improved.ipynb`
    * `demo_gpt.ipynb`
    * `demo_gitvision.ipynb`
    * `demo_blip.ipynb`
    * `demo_blip2.ipynb`
    * `display_images_captions.ipynb`
    * `visualise_CLIP.ipynb`
    * `socratic_mm_reasoning.ipynb`


## License
This project is licensed under the terms of the [MIT License](https://opensource.org/licenses/MIT), allowing free use of the code.


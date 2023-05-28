# Socratic Models: Composing Zero-Shot Multimodal Reasoning with Language, Free of Charge

> Ryan Amaudruz, Abhinav Bhuyan, Milena Kapralova, Bogdan Palfi, Alexandru Turcu <br>
>  Project Report, Deep Learning 2 University of Amsterdam

## Abstract
Socratic models [1] is a modular framework in which multiple pre-trained models are composed zeroshot via multimodal informed prompting. This is done to exchange information between models and
capture new multimodal capabilities, without requiring finetuning. As a proof of concept, we modify
the Socratic models framework such that it is entirely open-source and attempt to achieve the same
results as the original version. Additionally, we investigate the capabilities of Socratic models on
multimodal reasoning tasks such as chain-of-thought reasoning and visual question-answering in zeroshot and few-shot settings.

## Introduction
Socratic models (SMs) [1] are a fairly new addition to the field of deep learning and comprise a modular framework, which employs multiple pre-trained deep learning (DL) models to solve specific tasks.
Such models range from pure language models (LM), whose input and output are exclusively textual,
to visual-language (VLM) and audio-language models (ALM), which transform visual or audio information into text. In addition to these DL models, the framework can also incorporate modules that
rely on the user’s input or on specific APIs (e.g. robot actions). The framework uses prompting as a
form of exchanging information between these models, a technique known as the ’Socratic method’.
The main benefit of such an approach is that it offers zero-shot multimodal capabilities without requiring task-specific fine-tuning. This has far-reaching consequences for the future development of more
generalized artificial intelligent systems.

Because of their ability to communicate across multiple modalities and the dominant role of LMs
in the framework, SMs have been hypothesized to perform well on reasoning tasks. Various authors
have explored the ability of LMs to perform tasks such as arithmetic [2] and symbolic reasoning, with
promising results. Notably, [3] introduced chain-of-thought (CoT) reasoning, a method for prompting
LMs to perform reasoning tasks such as solving algebraic questions by generating the intermediate
steps or rationale for the problem. While this is usually done in a few-shot setting by prompting the
LM with exemplar rationales and answers [3, 4], this effect has also been demonstrated in a zero-shot
manner, by carefully designing the question prompt [5]. While current studies have relied on the LM
to perform reasoning, [6] also explore the possibility of CoT reasoning in a multimodal setting, using
a combination of text and image prompts.

All tasks presented in [1] employ the VLM CLIP [7] to extract information from the images, which
is then passed to a GPT-3 [8] LM via prompting, whose role is to create a fitting caption or description.
A quantitative analysis shows that SM have a higher performance on the zero-shot image captioning
task compared to the state-of-the-art (SoTA) ZeroCap [9] but highly under-perform compared to finetuned methods such as ClipCap [10], which uses a CLIP [7] encoding as a prefix to a caption and then
fine-tunes an LM (GPT2, [11]) to generate the image caption. A similar trend can be seen for videoto-text retrieval, where SM outperform the zero-shot SoTA algorithms but under-perform when being
compared to fine-tuned methods such as CLIP2Video [12]. As for the contextual image description
task, SM managed to outperform even the fine-tuned method introduced by [13].
Therefore, the aim of this project is to build on top of the model proposed by [1] in the following
ways:

1. **Using FLAN-T5 XL instead of GPT-3**. The model proposed by [1] uses the GPT-3 LM,
   which is a proprietary API of OpenAI. We will refer to this model as the *Original Socratic model*.
   To make the SM framework truly free and open-source, we replace the costly GPT-3 LM model
   with a freely accessible although less capable language model FLAN-T5 [14, 15] developed by
   Google, trained using instruction fine-tuning. This will comprise our *Baseline model*.
2. **Improving the performance on FLAN-T5 XL**.
   GPT-3 has demonstrated a strong ability to summarise and paraphrase information, which allows
   it to create a clear and concise caption, being less affected by sub-optimal prompts. On the
   other hand, FLAN-T5 seems to be much more affected by the given prompt, being less capable
   of paraphrasing information in a realistic way. For instance, FLAN-T5 tends to struggle when
   the provided prompt contains a large number of similar terms - e.g. an image portraying a
   monkey would have a prompt containing the words monkey, ape, chimpanzee, primate etc. which
   are synonyms -. Therefore, we propose a new prompt pre-processing method, called Synonym
   Exclusion (SE), which is based upon the PCA analysis of CLIP’s embedding space. Additional
   prompt engineering methods were tested but were not included since no performance increase
   was seen. We will refer to the model where SE is employed as the *Improved model*.

3. Extending the evaluation on the image captioning task.
   The original paper used qualitative and lexical-based quantitative metrics. However, those often
   do not correlate with human judgments, and have blind spots to syntactically pathological caption
   constructions [17], taking into account only information such as n-gram matching, word order,
   TF-IDF weights, and overlapping sequences of words. We therefore also utilize embedding- and
   learning-based metrics that better correlate with human judgement [18] such as BERT scores to
   evaluate the capabilities of the image captioners.
4. Comparing the models’ performance to GIT and BLIP.
   We also aimed to compare the performance of SM to non-Socratic models such as GIT and BLIP.
   Generative Image-to-Text Transformer (GIT) [16] is a generative VLM with a simplified pipeline
   that achieves high performance in image/video captioning and question-answering tasks. On
   the other hand, BLIP [17] and BLIP-2 [18] involve bootstrapping vision-language pre-training
   from frozen pre-trained image encoders and frozen large LMs, achieving improved performance
   on various vision-language tasks including zero-shot image-to-text generation and image/video
   captioning.

5. Applying the SM framework to multimodal reasoning tasks. We additionally explore
   the capabilities of Socratic models on 2 types of multimodal reasoning tasks: chain-of-thought
   (CoT) reasoning and visual question-answering, in both zero-shot and few-shot settings.
   Intuitively, we would expect the SM framework to excel in reasoning tasks due to their ability
   to exchange information across modalities, leading to more efficient cross-modal discourse and
   knowledge sharing. While the original authors perform a range of open-ended reasoning tasks in-
   cluding (video) summarization, open-ended Q&A, forecasting, image/audio retrieval from video,
   and reasoning for robotic actions, their experiments are limited to the domain of (egocentric)
   video. We extend upon their work by applying reasoning for visual question answering on im-
   ages, and further applying chain-of-thought reasoning as a mechanism to improve their reasoning
   abilities. Our efforts would serve as a handy framework and proof-of-concept for the application
   of SMs in multimodal reasoning tasks, leading to more robust, intelligent systems which can
   apply reasoning and generalize across multiple domains.
6. Making the pipeline more flexible, reproducible and efficient.
   We bring forth a modular codebase that makes it easy to build upon and test different captioning
   methods, including the usage of seeds, split between train, valid and test sets. We also provide a
   random and grid search pipeline to find the best hyperparameters. Finally, as the loading of the
   files and the generation of the embeddings is quite consuming, we have implemented a caching
   functionality that speeds up the development and testing process.


## 1 Method
### 1.1 Image captioning
#### 1.1.1 The Socratic method
The general pipeline for the Socratic image captioning follows the formula: caption = fV LM (fLM (fV LM (image))).
Specifically:
1. The VLM (CLIP) is fed an image and is used zero-shot to detect variables of interest: place
   categories (Places365[19]), object categories (from Tencent ML-Images [20]), image type and
   the number of people. The top-k ranked in each category can then be substituted into an LM
   prompt.
2. Given the VLM-informed language prompt, the LM generates several n candidate captions. For
   this step, we use a non-zero next-token sampling temperature (e.g. 0.9), to return sufficiently
   diverse, but reasonable results across the n candidates.
3. Finally, these n captions are then ranked by the VLM based on their cosine similarity to the
   image, the highest-scoring caption being returned.


#### 1.1.2 The Synonym Exclusion algorithm

The reason for this method is the observation that FLAN-T5 produces low-quality captions com-
pared to GPT-3 when the VLM-informed prompt contains too many similar words referring to the
same object.

For example, when given the wedding image seen above, The VLM prompt contains the sentence:
I think there might be a dress suit, full dress, tailcoat, tail coat, (etc.) in this photo.” and FLAN-T5
might generate this caption: ”A wedding dress is paired with a tuxedo for an elegant wedding.”

Our method creates prompts that are more suitable for FLAN-T5 by paying closer attention to
the words that are passed onto the prompt. In this way, the goal would be to not have similar terms
that might be redundant and thus confuse the model. To this end, we build a list of candidate terms
that have a high cosine similarity with the image, but a low cosine similarity with the other terms
in the candidate list. This is done by looping through the first 100 terms and considering the terms
in succession. The first term is included as a default, as it has the highest cosine similarity and is
therefore assumed to be the most relevant. The subsequent terms are then compared to previously
included candidate terms and are added only if they fall below a predefined cosine similarity threshold.
The list of candidate terms also has a predefined maximum number of allowed candidates. In this way,
the top 10 positions are more likely to contain relevant and distinctive terms.
The threshold for cosine similarities was determined by performing the PCA on the CLIP embed-
dings of both images and object categories. We noticed that even though the ground truth captions
have similarity between the corresponding image of around 0.25, using this threshold for our SE al-
gorithm did not filter synonyms effectively and produced subpar captions. Since both images and text share embedding space in CLIP, we analyzed this space using Principal Component Analysis
(PCA), a dimensionality reduction technique that identifies the most significant directions of variation
in a dataset. This enables the representation of data with fewer dimensions while preserving its key
features.

We visualized 25 random images (in yellow) and their corresponding 25 random object categories (in
blue) by reducing their CLIP embeddings from 768 to 3 dimensions (left). Additionally, we examined
the best-matching object category for each of 10 random images and represented them with matching
colors (right). In both cases, we saw that image and text cluster together, even in the best-matching
scenario. This indicates that texts exhibit greater similarity among themselves than with images,
emphasizing the need for higher thresholds to filter out text-text synonyms.



#### 1.1.3 Hyperparameter search







#### 1.1.4 Dataset



#### 1.1.5 Evaluation metrics

### 1.2 Chain-of-Thought and Visual Question Answering

#### 1.2.1 Model


#### 1.2.2 Dataset


#### 1.2.3 Evaluation

## 2 Results

### 2.1 Image captioning

#### 2.1.1 Qualitative demonstrations
![qualitative_results](blogpost_images/qualitative_results.png)

#### 2.1.2 Quantitative comparisons

| Approach                          | Bleu 4 | METEOR | ROUGE L | CIDEr | SPICE | BERT p | BERT r   | Cosine Sim |
|-----------------------------------|--------------|--------------|----------|-------------|-------------|--------|----------|------------|
| GITVision                         | 37.1 ± 32.7  | 31.5 ± 8.8   | 61.0 ± 14.0 | 162.0 ± 81.4 | 24.8 ± 10.7 | 93.4 ± 1.7 | 87.4 ± 1.6 | 25.5 ± 3.8 |
| BLIP                              | 12.9 ± 20.7  | 23.0 ± 8.9   | 49.4 ± 15.1 | 106.2 ± 62.7 | 17.7 ± 9.0  | 91.5 ± 1.9 | 85.6 ± 1.5 | 24.7 ± 3.6 |
| BLIP2                             | 23.9 ± 31.4  | 29.6 ± 12.2  | 58.1 ± 15.4 | 142.6 ± 72.2 | 22.1 ± 8.8  | 92.5 ± 1.7 | 86.6 ± 1.5 | 25.1 ± 3.7 |
| Original Socratic                 | 2.0 ± 9.4    | 15.4 ± 7.4   | 34.4 ± 15.1 | 45.4 ± 50.3  | 9.6 ± 6.6   | 89.8 ± 3.4 | 85.2 ± 1.8 | 25.8 ± 3.2 |
| Baseline Socratic with best params | 6.8 ± 17.4   | 16.8 ± 8.4   | 38.5 ± 15.9 | 57.6 ± 57.8  | 11.9 ± 8.9  | 90.7 ± 3.0 | 85.4 ± 1.7 | 25.3 ± 2.9 |
| Improved Socratic with best params | 2.4 ± 9.9    | 15.1 ± 6.5   | 34.8 ± 14.4 | 49.4 ± 41.7  | 9.7 ± 8.1   | 90.2 ± 2.9 | 84.7 ± 1.7 | 24.6 ± 2.6 |



### 2.2 Chain-of-Thought and Visual Question Answering

#### 2.2.1 Zero-shot CoT
<div style="text-align:center;">
  <img src="blogpost_images/spring.png" alt="Image" style="width:300px;height:300px;">
    <figcaption>Figure 2: Zero-shot CoT</figcaption>
</div>

#### 2.2.2 Few-shot CoT
<div style="display:flex; justify-content:center;">
  <div style="flex: 0 0 50%;">
    <figure>
      <img src="blogpost_images/spring.png" alt="Image 1" style="width:400px;height:300px;">
      <figcaption>(a) Example sample</figcaption>
    </figure>
  </div>
  <div style="flex: 0 0 50%;">
    <figure>
      <img src="blogpost_images/lemon.png" alt="Image 2" style="width:400px;height:300px;">
      <figcaption>(b) Target sample</figcaption>
    </figure>
  </div>
</div>


#### 2.2.3 Zero-shot VQA

## 3 Discussion


#### Image captioning


### 3.1 Limitations and future research


## 4 Conclusion


### 4.1 Individual contributions and conflict of interest

## References

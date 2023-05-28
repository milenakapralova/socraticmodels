# Socratic Models: Composing Zero-Shot Multimodal Reasoning with Language, Free of Charge

> Ryan Amaudruz, Abhinav Bhuyan, Milena Kapralova, Bogdan Palfi, Alexandru Turcu <br>
>  Project Report, Deep Learning 2 University of Amsterdam

## Abstract


## 1 Introduction




<div style="text-align:center;">
  <img src="blogpost_images/wedding.jpg" alt="Image" style="width:400px;height:300px;">
    <figcaption>Figure 1: Image for which CLIP produces too many synonyms</figcaption>
</div>

## 2 Method
### 2.1 Image captioning

#### 2.1.1 The Socratic method


#### 2.1.2 Synonym exclusion algorith
<div style="text-align:center;">
  <img src="blogpost_images/pca.png" alt="Image" style="width:700px;height:420px;">
    <figcaption>Figure 2: Image for which CLIP produces too many synonyms</figcaption>
</div>

#### 2.1.3 Hyperparameter search

#### 2.1.4 Dataset



#### 2.1.5 Evaluation metrics

### 2.2 Chain-of-Thought and Visual Question Answering

#### 2.2.1 Model


#### 2.2.2 Dataset


#### 2.2.3 Evaluation

## 3 Results

### 3.1 Image captioning

#### 3.1.1 Qualitative demonstrations
![qualitative_results](blogpost_images/qualitative_results.png)

#### 3.1.2 Quantitative comparisons

| Approach                          | Bleu 4 | METEOR | ROUGE L | CIDEr | SPICE | BERT p | BERT r   | Cosine Sim |
|-----------------------------------|--------------|--------------|----------|-------------|-------------|--------|----------|------------|
| GITVision                         | 37.1 ± 32.7  | 31.5 ± 8.8   | 61.0 ± 14.0 | 162.0 ± 81.4 | 24.8 ± 10.7 | 93.4 ± 1.7 | 87.4 ± 1.6 | 25.5 ± 3.8 |
| BLIP                              | 12.9 ± 20.7  | 23.0 ± 8.9   | 49.4 ± 15.1 | 106.2 ± 62.7 | 17.7 ± 9.0  | 91.5 ± 1.9 | 85.6 ± 1.5 | 24.7 ± 3.6 |
| BLIP2                             | 23.9 ± 31.4  | 29.6 ± 12.2  | 58.1 ± 15.4 | 142.6 ± 72.2 | 22.1 ± 8.8  | 92.5 ± 1.7 | 86.6 ± 1.5 | 25.1 ± 3.7 |
| Original Socratic                 | 2.0 ± 9.4    | 15.4 ± 7.4   | 34.4 ± 15.1 | 45.4 ± 50.3  | 9.6 ± 6.6   | 89.8 ± 3.4 | 85.2 ± 1.8 | 25.8 ± 3.2 |
| Baseline Socratic with best params | 6.8 ± 17.4   | 16.8 ± 8.4   | 38.5 ± 15.9 | 57.6 ± 57.8  | 11.9 ± 8.9  | 90.7 ± 3.0 | 85.4 ± 1.7 | 25.3 ± 2.9 |
| Improved Socratic with best params | 2.4 ± 9.9    | 15.1 ± 6.5   | 34.8 ± 14.4 | 49.4 ± 41.7  | 9.7 ± 8.1   | 90.2 ± 2.9 | 84.7 ± 1.7 | 24.6 ± 2.6 |



### 3.2 Chain-of-Thought and Visual Question Answering

#### 3.2.1 Zero-shot CoT
<div style="text-align:center;">
  <img src="blogpost_images/spring.png" alt="Image" style="width:300px;height:300px;">
    <figcaption>Figure 2: Zero-shot CoT</figcaption>
</div>

#### 3.2.2 Few-shot CoT

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
<div align="center">
  Figure 3: Few-shot CoT
</div>


#### 3.2.3 Zero-shot VQA
<div style="text-align:center;">
  <img src="blogpost_images/africa.png" alt="Image" style="width:300px;height:300px;">
    <figcaption>Figure 4: Zero-shot VQA</figcaption>
</div>


#### 3.2.3 Few-shot VQA

## 4 Discussion


#### Image captioning


### 4.1 Limitations and future research


## 5 Conclusion


### 5.1 Individual contributions and conflict of interest

## References

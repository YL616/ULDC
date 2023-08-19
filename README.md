## Introduction
This is the source code of "Uncertainty-based Learning for Deep Clustering"

[comment]: <> (## Abstract)

[comment]: <> (>Due to its impressive capabilities in handling high-dimensional real-world data, such as images, deep clustering is garnering heightened interest. Current  methods predominantly  focus on exploring the self-supervised information within the input data, which tend to facilitate accurate clustering  for data with prominent visual features. However, when dealing with data possessing ambiguous visual features that exhibit similarity with multiple clusters, these methods can result in unreliable feature projections, consequently diminishing the overall clustering performance. To address the challenge, we propose a novel Uncertainty-based Learning for Deep Clustering &#40;ULDC&#41; framework. Specifically, we first model the uncertainty of feature projection to obtain accurate uncertainty estimations. Subsequently, based on similarity and uncertainty, we construct an indicator matrix that possesses resilience to perturbations against  unreliable data correspondence, thereby mitigating the negative impact of unreliable data. Moreover, to ensure the effective aggregation of reliable data, we employ similarity distribution calibration to minimize the distance between data within the same cluster. Extensive experimental results demonstrate that ULDC achieves state-of-the-art or highly competitive clustering performance across five challenging benchmark datasets. Particularly noteworthy is ULDC's accomplishment of an ACC of 0.551 on the Image-Dog dataset, exhibiting a remarkable 26\% improvement compared to the best baseline. )


## Uncertainty-based Learning for Deep Clustering
Our AUL model comprises three components: 1) Uncertainty-aware Matching Filtration that leverages Subjective Logic that can effectively mitigate the disturbance of unreliable matching pairs and select high-confidence cross-modal matches for training; 2) Uncertainty-based Alignment Refinement, which not only simulates coarse-grained alignments by constructing uncertainty representations, but also performs progressive learning to incorporate coarse- and fine-grained alignments properly; 3) Cross-modal Masked Modeling that aims at exploring more comprehensive relations between vision and language.
![CMAP](figs/fig1.png)


## Proposed Model (AUL)
* Projection Uncertainty Modeling
* Deep Uncertainty-based Learning
* Similarity Distribution Calibration

## Motivation
![CMAP](figs/fig2.png)
Sample ⑤ in the airliner category exhibits similarities with both the airliner and dirigible categories, resulting in a higher level of uncertainty (U=0.4). In this case, the feature projection of by sample ⑤ is considered unreliable. Clustering methods that disregard uncertainty would erroneously group example ⑤ with the dirigible category. This issue arises due to the lack of discernment regarding the reliability of information carried by the data, leading to erroneous clustering of data with ambiguous features. Such misclustering exerts a dual adverse impact on both intra-cluster and inter-cluster relationships.

## Results
![Result](fig/fig3.png)


## Implementation
Please first downlaod the **`ImageNet-Dogs.zip`** from [this link](https://github.com/ANONYanonymous/ULDC) and put it in the same folder as **`main.py`** and then run the code.


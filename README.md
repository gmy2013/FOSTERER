<!-- # FOSTERER
Fine-Grained Code-Comment Semantic Interaction Analysis

Our paper is available on -->


# [Fine-Grained Code-Comment Semantic Interaction Analysis (ICPC 2022)](https://shangwenwang.github.io/files/ICPC-22.pdf)

Our paper is available on [ResearchGate](https://shangwenwang.github.io/files/ICPC-22.pdf).


## Introduction

Code comment, i.e., the natural language text to describe code, is considered as a killer for program comprehension. Current literature approaches mainly focus on comment generation or comment update, and thus fall short on explaining which part of the code leads to a specific content in the comment. In this paper, we propose that addressing such a challenge can better facilitate code understanding. We propose FOSTERER, a Fine-grained code-comment Semantic interaction analyzer , which can build fine-grained semantic interactions between code statements and comment tokens. It not only leverages the advanced deep learning techniques like cross-modal learning and contrastive learning, but also borrows the weapon of pre-trained vision models. Specifically, it mimics the comprehension practice of developers, treating code statements as image patches and comments as texts, and uses contrastive learning to match the semantically-related part between the visual and textual information. Experiments on a large-scale manually-labelled dataset show that our approach can achieve an F1-score around 80%, and such a performance exceeds a heuristic-based baseline to a large extent. We also find that FOSTERER can work with a high efficiency, i.e., it only needs 1.5 second for inferring the results for a code-comment pair. Furthermore, a user study demonstrates its usability: for 65% cases, its prediction results are considered as useful for improving code understanding. Therefore, our research sheds light on a promising direction for program comprehension.


## Get Started

Install PyTorch, prettify_js, cv2, html2image. 
The code has been tested with CUDA 11.2/CuDNN 8.1.0, PyTorch 1.8.1.

First, prepare pre-training datasets through [CodeSearchNet](https://github.com/github/CodeSearchNet). 

#### 1. Pre-process

The codes for rendering the source code texts, highlighting syntax with colors and patching are shown in visualize_patching directory.
First, read the data from the jsonl files of CodeSearchNet and convert them to images.
Then, split the visual renderings of the images by line.
```
python render_highlight_patching.py
```
Then, the visual renderings of the codes will be stored in the 'renderings' folder.
Inside each folder, the corresponding code snippet is stored in 'X_0.jpg', with the following 
line patches following, i.e., 'X_1.jpg ... X_len.jpg'.
#### 2. Model

The core model of our FOSTERER is shown in foster.py.





## Citation
```
@inproceedings{geng2022fine,
      title={Fine-Grained Code-Comment Semantic Interaction Analysis},
      author={Geng, Mingyang and Wang, Shangwen and Dong, Dezun and Gu, Shanzhi and Peng, Fang and Ruan, Weijian and Liao, Xiangke},
      booktitle={International Conference on Program Comprehension},
      year={2022},
      doi={https://doi.org/10.1145/3524610.3527887}
}
```





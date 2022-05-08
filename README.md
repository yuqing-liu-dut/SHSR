# Sequential Hierarchical Learning with Distribution Transformation for Image Super-Resolution
## Abstract
Multi-scale design has been considered in recent image super-resolution (SR) works to explore the hierarchical feature information. Existing multi-scale networks aim to build elaborate blocks or progressive architecture for restoration. In general, larger scale features concentrate more on structural and high-level information, while smaller scale features contain plentiful details and textured information. In this point of view, information from larger scale features can be derived from smaller ones. Based on the observation, in this paper, we build a sequential hierarchical learning super-resolution network (SHSR) for effective image SR. Specially, we consider the inter-scale correlations of features, and devise a sequential multi-scale block (SMB) to progressively explore the hierarchical information. SMB is designed in a recursive way based on the linearity of convolution with restricted parameters. Besides the sequential hierarchical learning, we also investigate the correlations among the feature maps and devise a distribution transformation block (DTB). Different from attention-based methods, DTB regards the transformation in a normalization manner, and jointly considers the spatial and channel-wise correlations with scaling and bias factors. Experiment results show SHSR achieves superior quantitative performance and visual quality to state-of-the-art methods with near 34% parameters and 50% MACs off when scaling factor is x4. To boost the performance without further training, the extension model SHSR+ with self-ensemble achieves competitive performance than larger networks with near 92% parameters and 42% MACs off with scaling factor x4.

## Code
You can easily train and test our model with the MSRN(https://github.com/MIVRC/MSRN-PyTorch) architecture.

## Citation
@article{10.1145/3532864,
    author = {Liu, Yuqing and Zhang, Xinfeng and Wang, Shanshe and Ma, Siwei and Gao, Wen},
    title = {Sequential Hierarchical Learning with Distribution Transformation for Image Super-Resolution},
    year = {2022},
    doi = {10.1145/3532864},
    journal = {ACM Trans. Multimedia Comput. Commun. Appl.},
}

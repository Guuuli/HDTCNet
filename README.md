# HDTCNet: A Convolution-based Hybrid-Dimensional Temporal Network for Multivariate Time Series Classification [[Link to paper](https://www.sciencedirect.com/science/article/pii/S0031320325004972)]

![framework](https://github.com/Guuuli/HDTCNet/blob/main/figures/framework.png)

Abstract: *Currently, transformer-based models dominate time series classification due to their ability to capture global data dependencies. However, their large memory footprint and high computational demands make them less desirable for long multivariate time series modeling. At the same time, convolution-based models promise a desirable performance versus efficiency trade-off for the task. Nevertheless, the popularity of transformer models has impeded exploration of sophisticated convolution-based modeling for time series classification. In this work, we introduce a novel convolution-based model, called HDTCNet, that alternates between 1D and 2D convolutional modes to capture temporal pattern correlations between intra- and inter-sensor variations. We develop a Wavelet Time Convolution block to expand the effective receptive field of the underlying neural model with a small convolutional kernel that embeds a set of 1D convolution operators between a cascade of wavelet decomposition and reconstruction of the input time series. To efficiently represent the local features of multivariate time series, we devise a hierarchical difference convolution network to capture intra- and inter-sensor gradient variations. To refine the feature representation of sensor-level time series and achieve multi-view self-enhancement, we design a self-enhancing module. These innovations enable HDTCNet to achieve superior performance while maintaining efficiency as a purely convolutional model. Experimental results demonstrate that HDTCNet achieves 2.1% accuracy and 1% F1-score performance improvements over state-of-the-art methods on 25 UEA classification datasets and 3 anomaly detection datasets, respectively, while reducing parameter counts by two orders of magnitude.*

# Dataset

Prepare Data. You can obtain the well-preprocessed datasets from **TSLib** [[Link](https://github.com/thuml/Time-Series-Library)]

# Train 

`python run.py`

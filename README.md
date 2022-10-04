# Content
This is a pytorch implementation of the paper ***Periodicity Counting in Videos with Unsupervised Learning of
Cyclic Embeddings*** (preprint version available [here](https://hal.archives-ouvertes.fr/hal-03738161/document)).
This implementation is oriented for regular videos. To adapt it to other input natures, change the architecture in *training_utils/triplet_based_encoder.py* so that it fits your input.
Here are some examples.
 * For 4D MRI, change the 2D convolution layers to 3D ones.
 * For complex time series, use MLP layers instead of convolutions.


# Use
1. in the file *count_periodicities.py*, write the path to your video in the variable *video_path*.
2. define the number of epochs you need for training with the variable *epochs_nb*.
3. you can change the parameters of the *Max Detector* algorithm at line 16 when calling the function count_repetitions¹²
4. launch *count_periodicities.py*

¹: period_range is the maximum variation of duration froma a cycle to the next\
²: N represents the number of frequencies to evaluate.

# Citation
If this was useful to you, please cite the article:

    @article{jacquelin:hal-03738161,
      TITLE = {Periodicity Counting in Videos with Unsupervised Learning of Cyclic Embeddings},
      AUTHOR = {Jacquelin, Nicolas and Vuillemot, Romain and Duffner, Stefan},
      URL = {https://hal.archives-ouvertes.fr/hal-03738161},
      JOURNAL = {Pattern Recognition Letters},
      PUBLISHER = {Elsevier},
      YEAR = {2022},
      DOI = {10.1016/j.patrec.2022.07.013},
    }


# ENDOLAS

This repository contains a deep learning based imaging pipeline, which is able to detect and match laser points from endoscopic laryngeal recordings. The name "endolas" is a mix of the words "endoscopy" and "laser".

![Pipeline](https://github.com/engineerByNature/endolas/blob/master/endolas_doc/pipeline.png)

## Installation

1) Download the repository.
2) Activate your desired python environment containing at least Python 3.7. 
3) Within the repository, run the setup.py with:

```
pip install . 
```

The package 'endolas' will now be installed in your environment including resources and additionally required packages.

## Documentation

See the file 'endolas_doc.html' for a code API documentation.

## Guide

* The prediction of keypoints from images by the pipeline can be carried out with:

  'jupyter/pipeline_prediction.ipynb'

* If a data set should be created from raw data, the following notebook can be used:

  'jupyter/ground_truth_preprocessing.ipynb'
  
* A data set can be enlarged with the use of:

  'jupyter/synthesis_or_augmentation.ipynb'
  'jupyter/generation_by_gan.ipynb'
  'jupyter/generation_by_vae.ipynb'

* If the networks should be retrained, the following notebooks can be used:

  'jupyter/train_segmentation.ipynb'
  'jupyter/train_registration.ipynb'

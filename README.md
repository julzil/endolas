# Feature-based image registration in structured light endoscopy

This repository contains a deep learning based image registration for structured light endoscopy. The method was developed with the use of laryngeal recordings to classify keypoints projected by a laser (features). The approach contains a preprocessing step, in which a semantic segmentation is performed to localize keypoints. The image registration is then performed to transform the irregulary placed keypoints into a regularly placed pattern. In a postprocessing step a nearest neighbor approach and a sorting algorithm are used to classifiy individual keypoints. The implementation resides in the package **endolas** (**endo**scopy + **las**er) and demonstration is provided in **demo**. Further, the dataset LASTEN, which was used for training and evaluation is given in **data**.

![Pipeline](https://github.com/engineerByNature/endolas/blob/master/endolas_doc/pipeline.png)

## Installation

1) Download the repository.
2) Activate your desired python environment containing at least Python 3.7. 
3) Within the repository, run the setup.py with:

```
pip install . 
```

The package **endolas** will now be installed in your environment including resources and additionally required packages.

## Demo

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
  
## Documentation

See the file 'endolas_doc.html' for a code API documentation.

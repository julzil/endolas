This repository contains a deep learning based imaging pipeline, which is able to detect and match laser points from endoscopic laryngeal recordings.

* The prediction of keypoints from images by the pipeline can be carried out with:

  'jupyter/pipeline_prediction.ipynb'

* If a data set should be created from raw data, the following notebook can be used:

  'ground_truth_preprocessing.ipynb'
  
* A data set can be enlarged with the use of:

  'synthesis_or_augmentation.ipynb'
  'generation_by_gan.ipynb'
  'generation_by_vae.ipynb'

* If the networks should be retrained the following notebooks can be used:

  'train_segmentation.ipynb'
  'train_registration.ipynb'

# garbage_classification
Kaggle-Garbage Classification
[Link](https://www.kaggle.com/asdasdasasdas/garbage-classification, "Kaggle link")

## Preprocessing
   Convert from image to numpy array and label the classes
   
## 1. modified VGGNet16
  
  # Model  
<img width="584" alt="스크린샷 2021-01-31 오후 9 06 27" src="https://user-images.githubusercontent.com/54826050/106384558-ce655800-640e-11eb-9ef3-96565f1c4843.png">

  Just shorten the Dense layer's parameter!
  
  # Result
  - Validation Accuracy
  <img width="728" alt="스크린샷 2021-01-31 오후 9 07 23" src="https://user-images.githubusercontent.com/54826050/106384598-14222080-640f-11eb-8e4f-c76f36566bcd.png">
  
  - Validation loss
  <img width="739" alt="스크린샷 2021-01-31 오후 9 07 33" src="https://user-images.githubusercontent.com/54826050/106384619-29974a80-640f-11eb-8987-4c29a2cdc94d.png">
  
  *Terribly big loss*
  
  
## 2. Smaller Model

  # Model
  <img width="560" alt="스크린샷 2021-01-31 오후 9 21 57" src="https://user-images.githubusercontent.com/54826050/106384731-afb39100-640f-11eb-804e-8159ce75d9b2.png">
  
  Only 4 Conv layers included.

 # Result
  - Validation Accuracy
  <img width="726" alt="스크린샷 2021-01-31 오후 9 33 38" src="https://user-images.githubusercontent.com/54826050/106384766-e5f11080-640f-11eb-9d3c-d45f34060cb9.png">

  - Validation loss
  <img width="715" alt="스크린샷 2021-01-31 오후 9 33 46" src="https://user-images.githubusercontent.com/54826050/106384779-f3a69600-640f-11eb-9a79-ed2e03181bcf.png">
  
  *Much better loss, but lower Acc*


## 3. Small but Stronger Model

  # Model
  <img width="560" alt="스크린샷 2021-01-31 오후 9 35 39" src="https://user-images.githubusercontent.com/54826050/106384804-1769dc00-6410-11eb-98e2-291c01adb1e5.png">
  
  *5 Conv layers are included but, strides=2 => smaller size of images(feature maps)*
  
   # Result
   - Validation Accuracy
  <img width="715" alt="스크린샷 2021-01-31 오후 9 39 35" src="https://user-images.githubusercontent.com/54826050/106384841-5ef06800-6410-11eb-98a9-fc63cb06811a.png">

  - Validation loss
  <img width="714" alt="스크린샷 2021-01-31 오후 9 39 50" src="https://user-images.githubusercontent.com/54826050/106384853-74fe2880-6410-11eb-9dde-38593baebd7d.png">
  
  ### Conclusion
   - The smaller the size(parameters) of the model, the faster the training speed is.
   - The stride of the filter contributes importantly to accuracy.
   - The depth of the layer should be set by empirical inference.

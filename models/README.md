# Description

* **Downsampling**："resnet18_ghost.py", "resnet18_mobilenet.py", "resnet18_shuffle.py", "resnet18_squeeze.py", "resnet18_squeeze_next.py", and "resnet_18.py".

* **Upsampling**："unet_right_half.py" and "unet_parts.py"

### Others
"model.py" contain all the models that are combine downsampling models and upsampling models.

"quantized_model.py" quaantized the model 'resnet18 + SqueezeNet + UNet'.

"unet_model.py" is complete UNet model.

"model_rebuild.py" is a model which downsampling is resnet with deformable convolution and upsampling is UNet.

"model_voting" is the model used to do voting strategies.

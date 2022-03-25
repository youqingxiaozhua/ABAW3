
# Coarse-to-Fine Cascaded Networks with Smooth Predicting for Video Facial Expression Recognition

This repo contains our solution for the [3rd ABAW Challenge](https://ibug.doc.ic.ac.uk/resources/cvpr-2022-3rd-abaw/).


# Models

coming soon...

# Requirements

Our implementation is based on [mmclassification](https://github.com/youqingxiaozhua/myclassification), and mmcv is enough to run our model.

# Usage

To train the IR-50 model with all the 8 classes, run:

```
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    python tools/train.py configs/affwild2/ir50.py
```



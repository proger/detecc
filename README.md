# detecc: object detection inference as packages

[tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn) packaged as something you can `setup.py install`.

```
python ./setup.py install --user
# to use locally do:
python ./setup.py build_ext --inplace

# you will need to download model checkpoints from somewhere:

export TFMODEL=$HOME/datasets/output/res152/coco_2014_train+coco_2014_valminusminival

# detect 'person' class:
python -m detecc *.jpg > objects.json
```

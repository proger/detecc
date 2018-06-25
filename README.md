# detecc

[tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn) packaged as something you can `setup.py install`

```
python ./setup.py install
export TFMODEL=$HOME/datasets/output/res152/coco_2014_train+coco_2014_valminusminival
# detects 'person' class
detecc *.jpg > objects.json
```

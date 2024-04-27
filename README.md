# PointCNN.PyTorch
This is a PyTorch implementation of [PointCNN](https://github.com/yangyanli/PointCNN). It is as efficent as the origin Tensorflow implemetation and achieves same accuracy on both classification and segmentaion jobs. See the following references for more information:
```
"PointCNN"
Yangyan Li, Rui Bu, Mingchao Sun, Baoquan Chen
arXiv preprint arXiv:1801.07791, 2018.
```
[https://arxiv.org/abs/1801.07791](https://arxiv.org/abs/1801.07791)


# Usage
We've tested code on ModelNet40 only.

```python
python train_pytorch.py
```

# License
Our code is released under MIT License (see LICENSE file for details).



# Custom (Vaishnav)
```
python3 --version # 3.10.12
python3 -m venv venvgpu
. venvgpu/bin/activate
pip install -r requirements.txt

mkdir data
mkdir data/GrabCad34L_hdf5_2048
mkdir data/GrabCad34_hdf5_2048
mkdir data/GrabCad67L_hdf5_2048
mkdir data/GrabCad67_hdf5_2048
```



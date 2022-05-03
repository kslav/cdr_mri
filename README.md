# ConvDecoder with Physics based regularization, implemented using DeepInPy.
Package for training deep inverse problems in Python by Jonathan I. Tamir, repo located at https://github.com/utcsilab/deepinpy.git. We strongly encourage the user to review the original DeepInPy repository and the instructions provided for its use. The [getting_started.md](https://github.com/kslav/cdr_mri/tree/master/docs/getting_started.md) document is borrowed from the DeepInPy repo and explains data formatting for compatibility with DeepInPy. 

The manuscript associated with this work can be at arxiv:xxxx and has been submitted to Magnetic Resonance in Medicine for review. 

## Example Usage of DeepInPy:
- Example for running a single experiment:
```bash
python main.py --config configs/example.json
```

## Purpose:
The purpose of this project is to apply physics-based regulization in training a modified ConvDecoder architecture (G(w))[Darestani et al. 2021. arXiv:2007.02471v3] for accelerated dynamic MRI. 

The proposed regularization term provides an early stopping condition that does not require access to ground truth data. This allows for automated early stoping that yields reconstructed images and corresponding quantitative parameter maps at a high resolution. The cost function for this training is defined in the following figure.

<img src="docs/images/costfunction.png" width="256">


### Test dataset
https://utexas.box.com/s/f1rpp5wvpzqorthxg98rbpszc5816c2f

[sigpy]: https://github.com/mikgroup/sigpy
[torchkbnufft]: https://github.com/mmuckley/torchkbnufft
[pytl]: https://github.com/PyTorchLightning/pytorch-lightning/
[pytorch]: https://pytorch.org/
[testtube]: https://github.com/williamFalcon/test-tube

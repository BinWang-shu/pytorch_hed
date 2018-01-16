# pytorch_hed
* The implementation of hed using pytorch.
* Thanks for  https://github.com/xlliu7/hed.pytorch


You can get vgg16.pth by:
```python
   wget https://download.pytorch.org/models/vgg16-397923af.pth
```
  You also need make the dirs:
```Bash
   mkdir checkpoints
   mkdir tmp
```
I have got ODS=0.771 on BSDS500 dataset with Adam.

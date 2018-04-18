# Chainer MNIST exsample using CNN


「[Chainerによる畳み込みニューラルネットワークの実装](http://aidiary.hatenablog.com/entry/20151007/1444223445)」をモダンに書きなおしました。


![CNN](http://cdn-ak.f.st-hatena.com/images/fotolife/a/aidiary/20150626/20150626203849.png)



### 予測精度

```
> python test.py

     0    1    2    3    4    5    6    7    8    9   accuracy        
0  977    0    0    0    0    0    1    1    1    0   0.99694         
1    0 1131    1    0    0    1    1    0    1    0   0.99648         
2    0    0 1028    0    0    0    0    4    0    0   0.99612         
3    0    0    1 1003    0    4    0    1    1    0   0.99307         
4    0    0    1    0  976    0    1    0    0    4   0.99389         
5    1    0    0    3    0  886    1    0    1    0   0.99327         
6    4    2    1    0    1    1  949    0    0    0   0.99061         
7    0    1    1    0    0    1    0 1024    0    1   0.99611         
8    3    0    1    1    0    6    3    1  956    3   0.98152         
9    0    0    0    0    6    5    0    4    0  994   0.98513         
accuracy all: 0.99240
Epoch 6: early stopping
```

### References

[1] [Chainerによる畳み込みニューラルネットワークの実装](http://aidiary.hatenablog.com/entry/20151007/1444223445)  
[2] [github:chainer/chainer](https://github.com/chainer/chainer/tree/master/examples/mnist)

### Environment
This implementation has been tested with Chainer 4.0.0 / Anaconda on MacOS.


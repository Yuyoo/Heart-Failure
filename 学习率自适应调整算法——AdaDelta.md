# AdaDelta  
## 一、超参数调整  
超参数（Hyper-Parameter)是困扰神经网络训练的问题之一，因为这些参数不可通过常规方法学习获得。

神经网络经典五大超参数:
* 学习率(Leraning Rate)  
* 权值初始化(Weight Initialization)
* 网络层数(Layers)
* 单层神经元数(Units)
* 正则惩罚项（Regularizer|Normalization)  

传统对抗这些超参数的方法是经验规则（Rules of Thumb)。  

这几年，随着深度学习的推进，全球神经网络研究者人数剧增，已经有大量研究组着手超参数优化问题：
* 深度学习先锋的RBM就利用Pre-Traning自适应调出合适的权值初始化值。
* 上个世纪末的LSTM长短期记忆网络，可视为“神经网络嵌套神经网络”，自适应动态优化层数。
* 2010年Duchi et.al 则推出AdaGrad，自适应来调整学习率。  

自适应调整学习率的方法，目前研究火热。一个经典之作，是 Matthew D. Zeiler 2012年在Google实习时，提出的AdaDelta。  

本文主要解析AdaDelta的代码实现。  

## 二、梯度更新
### 2.1 【一阶方法】随机梯度
SGD(Stochastic Gradient Descent)是相对于BGD(Batch Gradient Descent)而生的。

BGD要求每次正反向传播，计算所有Examples的Error，这在大数据情况下是不现实的。

最初的使用的SGD，每次正反向传播，只计算一个Example，串行太明显，硬件利用率不高。

后续SGD衍生出Mini-Batch Gradient Descent，每次大概推进100个Example，介于BGD和SGD之间。

现在，SGD通常是指Mini-Batch方法，而不是早期单Example的方法。

一次梯度更新，可视为：  

```math
x_{t+1}=x_{t}+\Delta x_{t}    

where \Delta x_{t}=-\eta \cdot g_{t}
```
`$x$`为参数，`$t$`为时序，`$\Delta$`为更新量，`$\eta$`为学习率，`$g_{t}$`为梯度  
### 2.2 【二阶方法】牛顿法
二阶牛顿法替换梯度更新量：  
```math
\Delta x_{t}=H^{-1}\cdot g_{t}
```
`$H$`为参数的二阶导矩阵，即Hessian矩阵。  

牛顿法使用Hessian矩阵替代人工设置的学习率，在梯度下降的时候，可以完美的找出下降方向，不会陷入局部最小值当中，是理想的方法。

但是，求逆矩阵的时间复杂度近似`$O(n^3)$`，计算代价太高，不适合大数据。

## 三、AdaDelta  
AdaDelta基本思想是用一阶的方法，近似模拟二阶牛顿法。
### 3.1 矩阵对角线近似逆矩阵
1988年，Becker&LeCun提出一种用矩阵对角线元素来近似逆矩阵的方法：
```math
\Delta x_{t}=-\frac{1}{\left |diag(H_{t})  \right |+\mu }\cdot g_{t}
```
`$diag$` 指的是构造Hessian矩阵的对角矩阵，`$\mu$` 是常数项，防止分母为 `$0$`。

2012年，[Schaul&S. Zhang&LeCun]借鉴了AdaGrad的做法，提出了更精确的近似：
```math
\Delta x_{t}=-\frac{1}{\left |diag(H_{t})  \right |+\mu }\cdot \frac{E[g_{t}-w:t]^2}{E[g_{t}^2-w:t]}\cdot g_{t}
```
`$E[g_{t}-w:t]$` 指的是从当前 `$t$` 开始的前 `$w$` 个梯度状态的期望值  
`$E[g_{t}^2-w:t]$` 指的是从当前 `$t$` 开始的前 `$w$` 个梯度状态平方的期望值  

同样是基于Gradient的Regularizer，不过只取最近的 `$w$` 个状态，这样不会让梯度被惩罚至0。
### 3.2 窗口和近似概率期望 
计算 `$E[g_{t}-w:t]$`，需要存储前 `$w$` 个状态，比较麻烦。

AdaDelta使用了类似动量因子的平均方法：
```math
E[g^2]_{t}=\rho \cdot E[g^2]_{t-1} + (1-\rho )\cdot g_{t}^2
```
当 `$\rho =0.5$` 时，这个式子就变成了求梯度平方和的平均数。  

如果再求根的话，就变成了RMS(均方根)：
```math
RMS[g]_{t}=\sqrt{E[g^2]_{t}+\varepsilon }
```  

再把这个RMS作为Gradient的Regularizer：
```math
\Delta x_{t}=-\frac{\eta }{RMS[g]_{t}}\cdot g_{t}
```  
其中， `$\varepsilon $` 是防止分母爆0的常数。 

这样，就有了一个改进版的AdaGrad。
该方法即Tieleman&Hinton的RMSProp。

RMSProp利用了二阶信息做了Gradient优化，在BatchNorm之后，对其需求不是很大。

但是没有根本实现自适应的学习率，依然需要线性搜索初始学习率，然后对其逐数量级下降。

另外，RMSProp的学习率数值与MomentumSGD差别甚大，需要重新线性搜索初始值。

注：`$\varepsilon $` 的建议取值为1，出处是Inception V3，不要参考V3的初始学习率。  

### 3.3 Hessian方法与正确的更新单元  
Zeiler用了两个反复近似的式子来说明，一阶方法到底在哪里输给了二阶方法。  

首先，考虑SGD和动量法： 
```math
\Delta x_{t}\propto g\propto \frac{\partial f}{\partial x}\propto \frac{1}{x}
```  

 `$\Delta x_{t} $` 可以正比到梯度 `$g $`问题，再正比到一阶导数。而 `$log $` 一阶导又可正比于 `$\frac{1}{x} $` 。  
 
再考虑二阶导Hessian矩阵法：

这里为了对比观察，使用了Becker&LeCun 1988 的近似方法，让求逆矩阵近似于求对角阵的倒数：
```math
\Delta x\propto H^{-1}g\propto \frac{\frac{\partial f}{\partial x}}{\frac{\partial ^2f}{\partial x^2}}\propto \frac{\frac{1}{x}}{\frac{1}{x}\cdot \frac{1}{x}}\propto x
```    

`$\Delta x_{t} $` 可以正比到Hessian逆矩阵`$H^{-1}g $` 问题，再正比到二阶导数。而`$log $` 二阶导又可正比于`$x $`。

可以看到，一阶方法最终正比于`$\frac{1}{x}$` ，即与参数逆相关：参数逐渐变大的时候，梯度反而成倍缩小。

而二阶方法最终正比于`$x$`，即与参数正相关：参数逐渐变大的时候，梯度不受影响。

因此，Zeiler称Hessian方法得到了Correct Units(正确的更新单元)。

### 3.4 由Hessian方法推导出一阶近似Hessian方法  
基于[Becker&LeCun 1988]的近似方法，有：
```math
\Delta x\approx \frac{\frac{\partial f}{\partial x}}{\frac{\partial ^2f}{\partial x^2}}
```

进而又有：
```math
\frac{\frac{\partial f}{\partial x}}{\frac{\partial ^2f}{\partial x^2}}=\frac{1}{\frac{\partial ^2f}{\partial x^2}}\cdot \frac{\partial f}{\partial x}=\frac{1}{\frac{\partial ^2f}{\partial x^2}}\cdot g_{t}
```

简单收束变形一下, 然后用RMS来近似：  
```math
\frac{1}{\frac{\partial ^2f}{\partial x^2}}=\frac{\Delta x}{\frac{\partial f}{\partial x}}\approx -\frac{RMS[\Delta x]_{t-1}}{RMS[g]_{t}}
```  

最后，一阶完整近似式：  
```math
\Delta x= -\frac{RMS[\Delta x]_{t-1}}{RMS[g]_{t}}\cdot g_{t}
```  

值得注意的是，使用了 `$RMS[\Delta x]_{t-1}$` 而不是 `$RMS[\Delta x]_{t}$`，因为此时 `$\Delta x_{t}$` 还没算出来。

### 3.5 算法流程

> 计算 `$x$` 在 `$t$` 时刻的更新值  
> 参数: 衰减率 `$\rho $` , Constant `$\varepsilon $`   
> 参数: 初始参数 `$x_{1} $`  
> 1：初始化累积变量 `$E[g2]_{0} = 0 $` ， `$E[g2]_{0} = 0 $`   
> 2：for t = 1 : T ，执行以下更新：  
> 3：计算梯度： `$g_{t} $`  
> 4：计算累积梯度： `$E[g^2]_{t}=\rho \cdot E[g^2]_{t-1} + (1-\rho )\cdot g_{t}^2 $`   
> 5: 计算 `$\Delta x$` ： `$\Delta x= -\frac{RMS[\Delta x]_{t-1}}{RMS[g]_{t}}\cdot g_{t} $`  
> 6: 计算 `$\Delta x$` 累积： `$E[\Delta x^2]_{t}=\rho \cdot E[\Delta x^2]_{t-1} + (1-\rho )\cdot \Delta x_{t}^2 $`  
> 7: 计算更新值： `$x_{t+1}=x_{t}+\Delta x_{t} $`  
> 8: end for

## 四、theano实现  
### 4.1 代码
```python
def adadelta(lr, tparams, grads, x, mask, y, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """
    
    # 定义共享变量zg，ru2，rg2
    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    # 定义梯度和梯度期望更新的元组
    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    """
    梯度函数映射
    param:
        input：输入是一个Python的列表list，里面存放的是将要传递给outputs的参数，这里inputs不用是共享变量shared variables.
        output：输出由输入inputs，updates以后的shared_variable的值和givens的值共同计算得到。
        updates：这里的updates存放的是一组可迭代更新的量，是(shared_variable, new_expression)的形式。
    return:
        f_grad_shared：梯度共享参数和梯度期望共享参数
    """ 
    f_grad_shared = theano.function([x, mask, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    # 通过均方根计算updir，即delta_x
    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    # 计算delta_x期望（累积）
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    # 更新权值
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]
    # 函数映射返回更新的权值
    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update

```
### 4.2 theano.function(input, output, updates, givens)
> function是一个由inputs计算outputs的对象，它关于怎么计算的定义一般在outputs里面，这里outputs一般会是一个符号表达式。 
> * inputs:输入是一个Python的列表list，里面存放的是将要传递给outputs的参数，这里inputs不用是共享变量shared variables.
> 
> * outputs: 输出是一个存放变量的列表list或者字典dict，如果是字典的话，keys必须是字符串。这里代码段中的outputs是cost，可以把它看成是一个损失函数值，它由输入inputs，updates以后的shared_variable的值和givens的值共同计算得到。在这里inputs只是在采取minibatch算法时准备抽取的样本集的索引，根据这个索引得到givens数据，是模型的输入变量即输入样本集，而updates中的shared_variable是模型的参数，所以最后由模型的输入和模型参数得到了模型输出就是cost。
> 
> * updates: 这里的updates存放的是一组可迭代更新的量，是(shared_variable, new_expression)的形式，对其中的shared_variable输入用new_expression表达式更新，而这个形式可以是列表，元组或者有序字典，这几乎是整个算法的关键也就是梯度下降算法的关键实现的地方。 看示例代码段1中updates是怎么来的，cost最后计算出来的可以看作是损失函数，是关于所有模型参数的一个函数，其中的模型参数是self.params，所以gparams是求cost关于所有模型参数的偏导数，其中模型参数params存放在一个列表里面，所有偏导数gparams也存放在一个列表里面，然后用来一个for循环，每次从两个列表里面各取一个量，则是一个模型参数和这个参数之于cost的偏导数，然后把它们存放在updates字典里面，字典的关键字就是一个param，这里一开始声明的所有params都是shared_variable，对应的这个关键字的值就是这个参数的梯度更新，即param-gparam*lr,其实这里的param-gparam*lr就是new_expression，所以这个updates的字典就构成了一对(shared_variable, new_expression)的形式。所以这里updates其实也是每次调用function都会执行一次，则所有的shared_variable都会根据new_expression更新一次值。
> 
> * givens：这里存放的也是一个可迭代量，可以是列表，元组或者字典，即每次调用function，givens的量都会迭代变化，但是比如上面的示例代码，是一个字典，不论值是否变化，都是x，字典的关键字是不变的，这个x值也是和input一样，传递到outputs的表达式里面的，用于最后计算结果。所以其实givens可以用inputs参数来代替，但是givens的效率更高。


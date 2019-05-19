import torch
from torch import tensor
import torch.nn.functional as F


def shuffle_demo():
    idx = torch.randperm(3)
    print(idx)
    data = torch.randint(0, 100, (3 ,2))
    print(data)
    data = data[idx]
    print(data)
def index_slip_vieworshape_demo():
    img = torch.rand(4,1,28,28)
    # print(img[0].shape)
    # print(img[0, 0].shape)
    # print(img[0, 0, 0].shape)
    # print(img[0, 0, 0, 0].shape)
    # print(img[0, 0, 0])
    # print(img[:,2,:,:].shape)
    # print(img[:, 1:2, ].shape)
    # # 每间隔两个像素采样一次
    # print(img[:, :, ::2, ::2].shape)
    # 在某个维度上采样,第二个参数要Tensor
    # print(img.index_select(1, torch.tensor(range(0,3))).shape)
    # # or
    # print(img.index_select(1, torch.tensor([0,1,2])).shape)
    # img[1:3,...] = img[1:3,：,：,：],后面维度保持不变
    # print(img[1:3,...].shape)
    #找出对应位置大于0.5的矩阵，对应位置全是1
    # ge_t = img.ge(0.5)
    # print(img.masked_select(ge_t))
    # take，有点类似index—select，但是先把矩阵flatten，然后取出flatten之后对应的位置
    # print(torch.take(img,torch.tensor([0,1])))
    #注意不会改变原有数据，正确的做法是重新赋值  如： img = img.view(4,784)
    # 若用 - 1 代替shape中的指定位置， 表示保持原来的数据不变，这样就可以省去不用计算
    img = img.view(4,-1)
    print(img.shape)

    pass


def squeeze_unsqueeze():
    img = torch.rand(1, 28, 1, 28)
    #          0   1    2    3    4
    #         [4,  1,  28,  28]
    #    -5   -4   -3  -2   -1
    #正数插在前面， 负数插在后面， ！！！！！！！
    new_shape = img.unsqueeze(-1).shape
    print(new_shape)
    data = tensor([1, 2]) # shap = ([2])
    new_data = data.unsqueeze(-1)  # shape = ([2, 1])
    print(new_data)
    # 如果某个维度不是 1 不可以压缩维度， 如果压缩，则数据的shape无变化，也不会报错
    new_img = img.squeeze(2).squeeze(0)
    print(new_img.shape)
    pass

def expand_repeat():
    # 只有shape中为 1 的才可以扩， 如若不然会报错
    # 若用 -1 代替shape中的指定位置， 表示保持原来的数据不变，
    # 这样就可以不用计算其他的维度值， 特别实在view操作中
    img = torch.rand(1, 1, 28, 28)
    new_img = img.expand(4, 4, -1, -1)
    print(new_img.shape)

    # repeat是在某个维度上重复的次数, 要注意下面的区别
    rep_img = img.repeat(4, 4, 1, 1)
    rep_img1 = img.repeat(4, 4, 28, 28)
    print(rep_img1.shape)
    pass

def broatcasting():
    img = torch.randint(1,2,(4,18,28,28),dtype=torch.float32)
    print(img)
    bias = torch.rand(28)
    print(bias.shape)
    bias = bias.expand_as(img)
    print(img+bias)
    pass

def cat_stack_split_chunk():
    img0 = torch.randint(1, 10,(1, 3, 2))
    img1 = torch.randint(1, 10, (1, 5, 2))
    print(img0)
    print(img1)
    img = torch.cat([img0, img1], dim=1)
    print(img)


    # stack 是创建一个新的维度， 要求两个tensor的shape完全相同
    img0 = torch.randint(1, 10, (3, 2))
    img1 = torch.randint(1, 10, (3, 2))
    img = torch.stack([img0, img1], dim=1)
    print(img.shape)

    # split 拆分一个向量,这个方法的第一个参数可以是列表，也可以是一个数字， 具体可以看下面的例子
    img0 = torch.randint(1, 10, (10, 16, 32, 32))
    tensor_list = img0.split([3,3,4], dim=0) #type item:tensor
    for item in tensor_list:
        print(item.shape)
        pass

    img0 = torch.randint(1, 10, (10, 16, 32, 32))
    tensor_list = img0.split(5, dim=0)  # type item:tensor
    for item in tensor_list:
        print(item.shape)
        pass
    pass


def operation_tensor():
    # 要区分对应元素相乘还是矩阵相乘
    # 对应的元素相乘，相加，相减， 相加，需要两个矩阵的shape完全相同
    # 相乘的例子
    a = torch.full([2, 2], 3)
    b = torch.full([2, 2], 4)
    print(a*b)
    # 单个矩阵的幂操作
    a = torch.full([2, 2], 3)
    a = a.pow(2)
    print(a)

    # 矩阵的指数操作
    a = torch.full([2, 2], 1)
    a = torch.exp(a)
    a = torch.log10(a)
    print(a)


    # 矩阵的相乘， 要符合数学的定义
    a = torch.full([2,2], 3)
    b = torch.full([2,3], 4)
    print(a@b)
    # 复杂的高纬向量相乘，默认后两个维度相乘，前面的维度运用broatcasting
    c = torch.randint(1, 10, [4, 2, 3, 3])
    d = torch.randint(1, 10, [4, 2, 3, 5])
    e = c@d
    print(e.shape)
    pass

def norm_argmax_opt():
    a = torch.rand(2,5)*10
    print(a)
    # 可以理解为行求和列求和
    # print(a.norm(1, dim=1))
    # print(a.norm(1, dim=0))
    # print(a.topk(2, dim=1))
    print(a.max(dim=1, keepdim=True))
    pass


def where_gather_opt():
    # where(cond, a, b), cond矩阵和a，b矩阵shape一样
    # cond对应的位置为1 取a的对应位置元素，为0 取b的对应位置的元素
    # a = torch.rand(3,3)
    # print(a)
    # d = torch.full([3,3],1)
    # e = torch.full([3,3],0)
    # r = torch.where(a>0.5, d, e)
    # print(r)


    # 这一段代码不太好理解，需要多看看，注意维度的信息
    a = torch.randint(1,10,[4,3]).float()
    b = torch.randint(0, 4, [5, 3])
    print(a)
    print()
    print(b)
    print()
    c = torch.gather(a, dim=0, index=b)
    print(c)
    pass

def mean_opt():
    a = torch.rand(2,2,2)
    print(a)
    print(a.mean(dim=2))


def grad_opt():
    x = torch.ones([1,3])
    w = torch.full([3,1], 10, requires_grad=True)
    print(w)
    y = torch.full([1,1], 3)
    m = F.mse_loss(x@w, y)
    m.backward()
    print(w.grad)
    pass

def grad_softmax():
    a = torch.rand([5], requires_grad=True)
    b = F.softmax(a, dim=0)
    c =  b[0].backward()
    print(a)
    print()
    print(a.grad)

def my_loss():
    a = torch.rand([3,4], requires_grad=True)
    print(a)
    soft_a = F.softmax(a, dim=0)
    print()
    s = soft_a.size()
    # print(a.grad)


def first_demo(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
    pass


def demo_t():
    r = []
    x = torch.tensor([5., 0.], requires_grad=True)
    opter = torch.optim.Adam([x], lr = 1e-1)
    for i in range(50000):
        pre_y = first_demo(x)
        r.append(pre_y)
        opter.zero_grad()
        pre_y.backward()
        opter.step()
        print("step {}, x = {}, y = {} ".format(i, x.tolist(), pre_y.tolist()))






r = demo_t()


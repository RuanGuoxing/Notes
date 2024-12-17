# BEVFormer

论文地址：[BEVFormer](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690001.pdf)

## SCA(Spatial Cross Attention)

### 解释

对于若干个images features， 对所有的points进行attention计算开销非常大。因此SCA
仅对部分点进行deformable attention计算。

对一个BEV Query $Q_t \in \mathbb{R}^{H \times W \times C}$， H，W为BEV特征平面的尺寸，
C为特征通道维数。假如$Q_p \in \mathbb{R}^{1 \times C}$是位置$p = (x,y)$的Query。
对于该Query， 

1. 我们首先根据其在BEV plane上的坐标$(x, y)$计算真实世界的坐标$(x^{\prime}, y^{\prime})$。$$
x^{\prime} = (x - \frac{W}{2}) \times s;\space y^{\prime} = (y - \frac{H}{2}) \times s$$ $s$是BEV plane格子的分辨率大小。

2. 然后采样该位置不同高度的若干个点（一般为4个）去得query高度信息。
3. 将这些点根据外参和内参信息投影到2D图像中，注意由于一个点不可能会出现在环视相机的所有视角中，因此
有些视角的投影会越界，因此在计算Query时就可以忽略， 有效的视野表示为 $\mathcal{V}_{hit}$ 。

4. 最后对不同视野的不同高度的点做Deformable Attention.$$\mathsf{SCA}(Q_p, F_t) = \frac{1}{|\mathcal{V}_{hit}|} \sum_{i \in \mathcal{V}_{hit}} \sum_{j=1}^{N_{ref}}\mathsf{DeformAttn}(Q_p, \mathcal{P}(p, i, j), F_t^i)$$$i$表示某个相机视野，j表示参考点，$N_{ref}$表示这个query投影后有效的参考点总数。$F_t^i$表示某个相机视野的图像特征。

### Code

#### 初始化

```py

@ATTENTION.register_module()
class SpatialCrossAttention(BaseModule):
    """An attention module used in BEVFormer.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_cams (int): The number of cameras
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        deformable_attention: (dict): The config for the deformable attention used in SCA.
    """

    def __init__(self,
                 embed_dims=256,
                 num_cams=6,
                 pc_range=None,
                 dropout=0.1,
                 init_cfg=None,
                 batch_first=False,
                 deformable_attention=dict(
                     type='MSDeformableAttention3D',
                     embed_dims=256,
                     num_levels=4),
                 **kwargs
                 ):
        super(SpatialCrossAttention, self).__init__(init_cfg)

        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.deformable_attention = build_attention(deformable_attention)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first
        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

```

#### 前向传播

##### 前向传播参数

```py

@force_fp32(apply_to=('query', 'key', 'value', 'query_pos', 'reference_points_cam'))
def forward(self,
            query,
            key,
            value,
            residual=None,
            query_pos=None,
            key_padding_mask=None,
            reference_points=None,
            spatial_shapes=None,
            reference_points_cam=None,
            bev_mask=None,
            level_start_index=None,
            flag='encoder',
            **kwargs):
    """Forward Function of Detr3DCrossAtten.
    Args:
        query (Tensor): Query of Transformer with shape
            (bs, num_query, embed_dims).
        key (Tensor): The key tensor with shape
            `(num_key, bs, embed_dims)`.
        value (Tensor): The value tensor with shape
            `(num_key, bs, embed_dims)`. (B, N, C, H, W)
        residual (Tensor): The tensor used for addition, with the
            same shape as `x`. Default None. If None, `x` will be used.
        query_pos (Tensor): The positional encoding for `query`.
            Default: None.
        key_pos (Tensor): The positional encoding for  `key`. Default
            None.
        reference_points (Tensor):  The normalized reference
            points with shape (bs, num_query, 4),
            all elements is range in [0, 1], top-left (0,0),
            bottom-right (1, 1), including padding area.
            or (N, Length_{query}, num_levels, 4), add
            additional two dimensions is (w, h) to
            form reference boxes.
        key_padding_mask (Tensor): ByteTensor for `query`, with
            shape [bs, num_key].
        spatial_shapes (Tensor): Spatial shape of features in
            different level. With shape  (num_levels, 2),
            last dimension represent (h, w).
        level_start_index (Tensor): The start index of each level.
            A tensor has shape (num_levels) and can be represented
            as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
    Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
    """

```

##### 初始化部分参数

设置key, value, 残差值，融合位置编码。

```py

    if key is None:
        key = query
    if value is None:
        value = key

    if residual is None:
        inp_residual = query
        slots = torch.zeros_like(query)
    if query_pos is not None:
        query = query + query_pos

    bs, num_query, _ = query.size()
    # heigh axis anchors. default is 4
    D = reference_points_cam.size(3)

```

##### 获取最大reference points的数量

这里原作代码可能会有一些问题，其假设每个视角的相机的不同batch的图像内参一样。
bev_mask的维度是：$[num\_cams, batch\_size, bev_h*bev_w, depth]$。表示该点是否有效。
reference_points_cam的维度是：$[num\_cams, batch\_size, num\_queries, 4]$
表示每个相机每个query的4个参考点。

```py

indexes = []
for i, mask_per_img in enumerate(bev_mask):
    index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
    indexes.append(index_query_per_img)
# 获取每个视角下的相机最大的参考点数量
# index_query_per_img为每个相机的某个query是否有效的mask（T or F）。size为[Q]
max_len = max([len(each) for each in indexes])

# each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
queries_rebatch = query.new_zeros(
    [bs, self.num_cams, max_len, self.embed_dims])
reference_points_rebatch = reference_points_cam.new_zeros(
    [bs, self.num_cams, max_len, D, 2])
# 2 表示x，y坐标

for j in range(bs):
    for i, reference_points_per_img in enumerate(reference_points_cam):   
        index_query_per_img = indexes[i]
        # query[j, index_query_per_img]获取需要的query
        queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
        reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]

```

##### MSDA（Multi-Scale Deformable Attention）

4个高度当作4个MSDA的level

```py

    num_cams, l, bs, embed_dims = key.shape

    key = key.permute(2, 0, 1, 3).reshape(
        bs * self.num_cams, l, self.embed_dims)
    value = value.permute(2, 0, 1, 3).reshape(
        bs * self.num_cams, l, self.embed_dims)

    queries = self.deformable_attention(query=queries_rebatch.view(bs*self.num_cams, max_len, self.embed_dims), key=key, value=value,
                                        reference_points=reference_points_rebatch.view(bs*self.num_cams, max_len, D, 2), spatial_shapes=spatial_shapes,
                                        level_start_index=level_start_index).view(bs, self.num_cams, max_len, self.embed_dims)

```

##### 根据有效图像视野数特征平均，残差连接

```py

    # 根据MSDA的结果保留部分query后的结果
    for j in range(bs):
        for i, index_query_per_img in enumerate(indexes):
            slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]

    count = bev_mask.sum(-1) > 0
    count = count.permute(1, 2, 0).sum(-1)
    count = torch.clamp(count, min=1.0) # minimum camera number set to 1.
    slots = slots / count[..., None]    # Average feature.
    slots = self.output_proj(slots)     # Linear Transform

    return self.dropout(slots) + inp_residual


```

参考：[SCA](https://blog.csdn.net/mybest4545/article/details/139610477)

## TSA(Temporal Self-Attention)

根据时序信息检测动态障碍物和遮挡障碍物。

$$
TSA(Q_p, {Q, B^\prime_{t-1}}) = \sum_{V \in \{Q, B^\prime_{t-1}\}} DeformAttn(Q_p, p, V)
$$

1. 首先在ego坐标系下对齐BEV特征B和Query Q。
$B_{t-1}$表示前一时刻预测的对齐后的BEV特征，Q_p p=(x, y)表示由Q和B_{t-1} concat后预测得道的offsets，位于
BEV Query Q中。

### 细节

#### Training Phase

计算当前时刻的前3个时刻的BEV特征${B_{t-3}, B_{t-2}, B_{t-1}, B_t}$，其中前三个时刻的BEV只前向计算，
不需要梯度，第一个时刻的特征和Query相同（没有前面的BEV特征了）。
最后只用当前时刻的BEV特征$B_t$用作检测和分割头的输入，并计算loss。

#### Inference Phase

保存前一时刻的BEV特征，用于后一时刻的前向推理计算。

## Detection and Segmentation Head

### Detect

based Deformable DETR. L1 loss

### map segmentation

Panoptic SegFormer


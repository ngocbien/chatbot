��
l��F� j�P.�M�.�}q (X   protocol_versionqM�X   little_endianq�X
   type_sizesq}q(X   shortqKX   intqKX   longqKuu.�(X   moduleq c__main__
DecoderRNN
qNNtqQ)�q}q(X   _backendqctorch.nn.backends.thnn
_get_thnn_function_backend
q)RqX   _parametersqccollections
OrderedDict
q	)Rq
X   _buffersqh	)RqX   _backward_hooksqh	)RqX   _forward_hooksqh	)RqX   _forward_pre_hooksqh	)RqX   _modulesqh	)Rq(X	   embeddingq(h ctorch.nn.modules.sparse
Embedding
qX^   /home/ubuntu/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages/torch/nn/modules/sparse.pyqX�  class Embedding(Module):
    r"""A simple lookup table that stores embeddings of a fixed dictionary and size.

    This module is often used to store word embeddings and retrieve them using indices.
    The input to the module is a list of indices, and the output is the corresponding
    word embeddings.

    Args:
        num_embeddings (int): size of the dictionary of embeddings
        embedding_dim (int): the size of each embedding vector
        padding_idx (int, optional): If given, pads the output with the embedding vector at :attr:`padding_idx`
                                         (initialized to zeros) whenever it encounters the index.
        max_norm (float, optional): If given, will renormalize the embeddings to always have a norm lesser than this
        norm_type (float, optional): The p of the p-norm to compute for the max_norm option
        scale_grad_by_freq (bool, optional): if given, this will scale gradients by the frequency of
                                                the words in the mini-batch.
        sparse (bool, optional): if ``True``, gradient w.r.t. weight matrix will be a sparse tensor. See Notes for
                                    more details regarding sparse gradients.

    Attributes:
        weight (Tensor): the learnable weights of the module of shape (num_embeddings, embedding_dim)

    Shape:
        - Input: LongTensor of arbitrary shape containing the indices to extract
        - Output: `(*, embedding_dim)`, where `*` is the input shape

    .. note::
        Keep in mind that only a limited number of optimizers support
        sparse gradients: currently it's :class:`optim.SGD` (`CUDA` and `CPU`),
        :class:`optim.SparseAdam` (`CUDA` and `CPU`) and :class:`optim.Adagrad` (`CPU`)

    .. note::
        With :attr:`padding_idx` set, the embedding vector at
        :attr:`padding_idx` is initialized to all zeros. However, note that this
        vector can be modified afterwards, e.g., using a customized
        initialization method, and thus changing the vector used to pad the
        output. The gradient for this vector from :class:`~torch.nn.Embedding`
        is always zero.

    Examples::

        >>> # an Embedding module containing 10 tensors of size 3
        >>> embedding = nn.Embedding(10, 3)
        >>> # a batch of 2 samples of 4 indices each
        >>> input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
        >>> embedding(input)
        tensor([[[-0.0251, -1.6902,  0.7172],
                 [-0.6431,  0.0748,  0.6969],
                 [ 1.4970,  1.3448, -0.9685],
                 [-0.3677, -2.7265, -0.1685]],

                [[ 1.4970,  1.3448, -0.9685],
                 [ 0.4362, -0.4004,  0.9400],
                 [-0.6431,  0.0748,  0.6969],
                 [ 0.9124, -2.3616,  1.1151]]])


        >>> # example with padding_idx
        >>> embedding = nn.Embedding(10, 3, padding_idx=0)
        >>> input = torch.LongTensor([[0,2,0,5]])
        >>> embedding(input)
        tensor([[[ 0.0000,  0.0000,  0.0000],
                 [ 0.1535, -2.0309,  0.9315],
                 [ 0.0000,  0.0000,  0.0000],
                 [-0.1655,  0.9897,  0.0635]]])
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2, scale_grad_by_freq=False,
                 sparse=False, _weight=None):
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        if _weight is None:
            self.weight = Parameter(torch.Tensor(num_embeddings, embedding_dim))
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim], \
                'Shape of weight does not match num_embeddings and embedding_dim'
            self.weight = Parameter(_weight)
        self.sparse = sparse

    def reset_parameters(self):
        self.weight.data.normal_(0, 1)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)

    def forward(self, input):
        return F.embedding(
            input, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)

    def extra_repr(self):
        s = '{num_embeddings}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        return s.format(**self.__dict__)

    @classmethod
    def from_pretrained(cls, embeddings, freeze=True):
        r"""Creates Embedding instance from given 2-dimensional FloatTensor.

        Args:
            embeddings (Tensor): FloatTensor containing weights for the Embedding.
                First dimension is being passed to Embedding as 'num_embeddings', second as 'embedding_dim'.
            freeze (boolean, optional): If ``True``, the tensor does not get updated in the learning process.
                Equivalent to ``embedding.weight.requires_grad = False``. Default: ``True``

        Examples::

            >>> # FloatTensor containing pretrained weights
            >>> weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
            >>> embedding = nn.Embedding.from_pretrained(weight)
            >>> # Get embeddings for index 1
            >>> input = torch.LongTensor([1])
            >>> embedding(input)
            tensor([[ 4.0000,  5.1000,  6.3000]])
        """
        assert embeddings.dim() == 2, \
            'Embeddings parameter is expected to be 2-dimensional'
        rows, cols = embeddings.shape
        embedding = cls(num_embeddings=rows, embedding_dim=cols, _weight=embeddings)
        embedding.weight.requires_grad = not freeze
        return embedding
qtqQ)�q}q(hhhh	)RqX   weightqctorch._utils
_rebuild_tensor_v2
q((X   storageqctorch
FloatStorage
q X   93973190002544q!X   cpuq"K-Ntq#QK K	K�q$KK�q%�Ntq&Rq'shh	)Rq(hh	)Rq)hh	)Rq*hh	)Rq+hh	)Rq,X   trainingq-�X   num_embeddingsq.K	X   embedding_dimq/KX   padding_idxq0NX   max_normq1NX	   norm_typeq2KX   scale_grad_by_freqq3�X   sparseq4�ubX   gruq5(h ctorch.nn.modules.rnn
GRU
q6X[   /home/ubuntu/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages/torch/nn/modules/rnn.pyq7X�  class GRU(RNNBase):
    r"""Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.


    For each element in the input sequence, each layer computes the following
    function:

    .. math::

            \begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) n_t + z_t h_{(t-1)} \\
            \end{array}

    where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is the input
    at time `t`, :math:`h_{(t-1)}` is the hidden state of the previous layer
    at time `t-1` or the initial hidden state at time `0`, and :math:`r_t`,
    :math:`z_t`, :math:`n_t` are the reset, update, and new gates, respectively.
    :math:`\sigma` is the sigmoid function.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two GRUs together to form a `stacked GRU`,
            with the second GRU taking in outputs of the first GRU and
            computing the final results. Default: 1
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature)
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            GRU layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        bidirectional: If ``True``, becomes a bidirectional GRU. Default: ``False``

    Inputs: input, h_0
        - **input** of shape `(seq_len, batch, input_size)`: tensor containing the features
          of the input sequence. The input can also be a packed variable length
          sequence. See :func:`torch.nn.utils.rnn.pack_padded_sequence`
          for details.
        - **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the initial hidden state for each element in the batch.
          Defaults to zero if not provided.

    Outputs: output, h_n
        - **output** of shape `(seq_len, batch, hidden_size * num_directions)`: tensor
          containing the output features h_t from the last layer of the GRU,
          for each t. If a :class:`torch.nn.utils.rnn.PackedSequence` has been
          given as the input, the output will also be a packed sequence.
        - **h_n** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the hidden state for `t = seq_len`

    Attributes:
        weight_ih_l[k] : the learnable input-hidden weights of the :math:`\text{k}^{th}` layer
            (W_ir|W_iz|W_in), of shape `(3*hidden_size x input_size)`
        weight_hh_l[k] : the learnable hidden-hidden weights of the :math:`\text{k}^{th}` layer
            (W_hr|W_hz|W_hn), of shape `(3*hidden_size x hidden_size)`
        bias_ih_l[k] : the learnable input-hidden bias of the :math:`\text{k}^{th}` layer
            (b_ir|b_iz|b_in), of shape `(3*hidden_size)`
        bias_hh_l[k] : the learnable hidden-hidden bias of the :math:`\text{k}^{th}` layer
            (b_hr|b_hz|b_hn), of shape `(3*hidden_size)`
    Examples::

        >>> rnn = nn.GRU(10, 20, 2)
        >>> input = torch.randn(5, 3, 10)
        >>> h0 = torch.randn(2, 3, 20)
        >>> output, hn = rnn(input, h0)
    """

    def __init__(self, *args, **kwargs):
        super(GRU, self).__init__('GRU', *args, **kwargs)
q8tq9Q)�q:}q;(hhhh	)Rq<(X   weight_ih_l0q=h((hh X   93973192096720q>h"KKNtq?QK KK�q@KK�qA�NtqBRqCX   weight_hh_l0qDh((hh X   93973189968256qEh"KKNtqFQK KK�qGKK�qH�NtqIRqJX
   bias_ih_l0qKh((hh X   93973194282752qLh"KNtqMQK K�qNK�qO�NtqPRqQX
   bias_hh_l0qRh((hh X   93973196311168qSh"KNtqTQK K�qUK�qV�NtqWRqXuhh	)RqYhh	)RqZhh	)Rq[hh	)Rq\hh	)Rq]h-�X   modeq^X   GRUq_X
   input_sizeq`KX   hidden_sizeqaKX
   num_layersqbKX   biasqc�X   batch_firstqd�X   dropoutqeK X   dropout_stateqf}qgX   bidirectionalqh�X   _all_weightsqi]qj]qk(X   weight_ih_l0qlX   weight_hh_l0qmX
   bias_ih_l0qnX
   bias_hh_l0qoeaX
   _data_ptrsqp]qqubX   outqr(h ctorch.nn.modules.linear
Linear
qsX^   /home/ubuntu/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages/torch/nn/modules/linear.pyqtX#  class Linear(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = Ax + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, in\_features)` where :math:`*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            `(out_features x in_features)`
        bias:   the learnable bias of the module of shape `(out_features)`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
qutqvQ)�qw}qx(hhhh	)Rqy(hh((hh X   93973190013952qzh"K-Ntq{QK K	K�q|KK�q}�Ntq~Rqhch((hh X   93973173617360q�h"K	Ntq�QK K	�q�K�q��Ntq�Rq�uhh	)Rq�hh	)Rq�hh	)Rq�hh	)Rq�hh	)Rq�h-�X   in_featuresq�KX   out_featuresq�K	ubX   softmaxq�(h ctorch.nn.modules.activation
LogSoftmax
q�Xb   /home/ubuntu/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages/torch/nn/modules/activation.pyq�X  class LogSoftmax(Module):
    r"""Applies the `Log(Softmax(x))` function to an n-dimensional input Tensor.
    The LogSoftmax formulation can be simplified as

    :math:`\text{LogSoftmax}(x_{i}) = \log\left(\frac{\exp(x_i) }{ \sum_j \exp(x_j)} \right)`

    Shape:
        - Input: any shape
        - Output: same as input

    Arguments:
        dim (int): A dimension along which Softmax will be computed (so every slice
            along dim will sum to 1).

    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [-inf, 0)

    Examples::

        >>> m = nn.LogSoftmax()
        >>> input = torch.randn(2, 3)
        >>> output = m(input)
    """

    def __init__(self, dim=None):
        super(LogSoftmax, self).__init__()
        self.dim = dim

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def forward(self, input):
        return F.log_softmax(input, self.dim, _stacklevel=5)
q�tq�Q)�q�}q�(hhhh	)Rq�hh	)Rq�hh	)Rq�hh	)Rq�hh	)Rq�hh	)Rq�h-�X   dimq�Kubuh-�X   n_layersq�KhaKub.�]q (X   93973173617360qX   93973189968256qX   93973190002544qX   93973190013952qX   93973192096720qX   93973194282752qX   93973196311168qe.	       �����)>�)�>DÒ>�eѾ��@������=?��9?K       ��.�7L̾פ]�����}P��\��ɇ?>]����n$>L���J��>���= �6Ҍ>c߆����>8�1>�Ts��9R�*ݾ�\�>���(q��Mv<%���W?�U@��@� 5�)ǁ����A���>Fa��ʵ?�"���[=>��>��c��$��}7��(ʯ>��������?=�?�js?8�>�̉�I}���>�Ce?�?����Y>U��>[69�p:>L^��c��= ��>�"J?�Z,?{4Q�����3=����K ?�p>>�!n�E�J�o�e7�x��>VD�=-       严��W{>n��?��$�t���m������=ֿr�W?Fӌ=^�\�p������@hꈿͻ�?T"���\O����?��h��Y3?'�?$�h?UP�=v�>J<�>�>C��w�?� 3@�s��S�[D���<@���?�x{?���XG?�?"s�Rf�=;��`%ƾ
��?-       �hq�J�">B$�>tGK��=��~�������ѿ��I?>��@�J�?r��>�4�@@���i���F@�X>@Ȃ6�o�?�º�':1@*=?��滿���?C����G� 1@P����?�2�_�f���`��C��v��?����3�?�D @B�J@���	9@�V,@��?�����+@K       g~���=��>�_�>��>����>�XS�0�?��V���7�>�_�l�ѽ�>ॾ�\��{>-�U�Ǹ�=��>H��>��Ⱦ���>Ll��Dp�>�؅?��?>���)�������5=�W*�{�=cD��`��tF���da�?%L�.,F���<) ?�y�?i^���Z>gx��p��>���?,;¼�i;�P=�T8ڽJ�=.��g���u�F�+2�=���>'�?=���^�9q羂A?ߵ �#�V����>�����GB>���>,	?��P>���>�Ը>��?>o�K?       �?E-þ:�2?�,�>_]�>e=y�"�0hV��d�>��?�u�|?j�-�$�?��p?       ��}>��>���=��=�Ʋ>�n[��WM�
�����?�O��W���D��'�A?�V>
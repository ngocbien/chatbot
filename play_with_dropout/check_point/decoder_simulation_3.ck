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
q X   94640648810256q!X   cpuq"M@Ntq#QK KK�q$KK�q%�Ntq&Rq'shh	)Rq(hh	)Rq)hh	)Rq*hh	)Rq+hh	)Rq,X   trainingq-�X   num_embeddingsq.KX   embedding_dimq/KX   padding_idxq0NX   max_normq1NX	   norm_typeq2KX   scale_grad_by_freqq3�X   sparseq4�ubX   gruq5(h ctorch.nn.modules.rnn
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
q8tq9Q)�q:}q;(hhhh	)Rq<(X   weight_ih_l0q=h((hh X   94640630323008q>h"M�Ntq?QK K<K�q@KK�qA�NtqBRqCX   weight_hh_l0qDh((hh X   94640647940752qEh"M�NtqFQK K<K�qGKK�qH�NtqIRqJX
   bias_ih_l0qKh((hh X   94640658126176qLh"K<NtqMQK K<�qNK�qO�NtqPRqQX
   bias_hh_l0qRh((hh X   94640663592256qSh"K<NtqTQK K<�qUK�qV�NtqWRqXuhh	)RqYhh	)RqZhh	)Rq[hh	)Rq\hh	)Rq]h-�X   modeq^X   GRUq_X
   input_sizeq`KX   hidden_sizeqaKX
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
qutqvQ)�qw}qx(hhhh	)Rqy(hh((hh X   94640650265680qzh"M@Ntq{QK KK�q|KK�q}�Ntq~RqX   biasq�h((hh X   94640643827344q�h"KNtq�QK K�q�K�q��Ntq�Rq�uhh	)Rq�hh	)Rq�hh	)Rq�hh	)Rq�hh	)Rq�h-�X   in_featuresq�KX   out_featuresq�KubX   softmaxq�(h ctorch.nn.modules.activation
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
q�tq�Q)�q�}q�(hhhh	)Rq�hh	)Rq�hh	)Rq�hh	)Rq�hh	)Rq�hh	)Rq�h-�X   dimq�Kubuh-�X   n_layersq�KhaKub.�]q (X   94640630323008qX   94640643827344qX   94640647940752qX   94640648810256qX   94640650265680qX   94640658126176qX   94640663592256qe.�      ��>�a��N�=v;�=/��>�?Ӽ�"ʽM�P>{�>��5>���=��V=c��<�zS>�h �g�^�i��>0�=��ݽ�n.��s6�S�8��c�<��>�����a=�8�~�7=�(�=�UA�,-=#��.�Jq�5>K���S��<��-�h>���=��=�T�<�~˻ay�/pb���#�Rw?�G�^���L>�-d�y.=z�R>y[l�p[׽q/�se��_�3Jʽf��=��D<x�.>Yx>C��=H�:L�����:����=�l���Cl�,(&>#�>._0=W�&<�R۽
�>���>3T>�`>I��>I�>>n�>�uU=%�T=������=r�����k>O��v�^�N�r��h8>yo�>�$2�Z�H>���=�-�=ď�>\,���g�>�#u>^�ҽT��<��1>���>��L�¾v=ۋ׽A�>%�J>_'6>��>5�qhN�t�->�f>��>�D�:�� >���>o���/)��[.>��">��L=�J>����kS�=��-�E�>>D�=h�B>�t�=���=�(�<�2:>:F?�I��"\M�Ma�=`N�=��>��x>|ɽ{�=��#d�>}F'>�p��*:�=���=�3,>��W>QВ>8�p����<��$>�E>�@�<~��>� Խb2>�.<>��<�H���z����m�=���Tc�=�S&�$E�>��4��%���D;��>����-L��/>�h����>������3=���i�4���>kȶ=��w��#X>��<�#�<�>�D>4c�=��꽰�_��I�=yÅ>\�A=B���Wg>�D�>k�%��v=TV���=3mǽoi>�xx>c*�E�h>���>��o> 5�>��3>}}>� >�x�>G$��� @=G|�>��>I:4�=��i���=K�e>�Q���a���ҽW�U=`��=��B>��c�2��=�#>�q�>��n>��^����ux�>i�f>-��^=���&�%=8����m>Pk�'D?=�!���=>\>�EY�iO�<�&<W�Ͻ>��[��=��>��󽲿�2"�ȒR��L,�gȸ<p����=% 6>*X>�.=���=z�I=o>JJ�=D��=���\��{�K���<���=���=�z�>�8��.�>I�>���=��=�	�Bb�<�.	�Q)��=��<��Y>�Ӡ�Q8>0샽NBH��P>�(X=\����~=��ɽ�h��>�~�;��o>�@���l>��>�=����[�4�p>�����>���&�L>H�G�B�z=R��=�-@���]>A����.>0�=UV>h�½�#�t&�>j�=�:⽹���=�S>_��>���>��P>���=�0>E��>*}�=��->����R�F>���~����~;�P�q=(@=[>
�*<
��b~�=~=�>�])=���>$R>)=��>��=K��<ѓ���Iֽ�K>ҋ�:��=����I��=F�#�K3�>6�w>[��<uj>)c�=^�>;T�>V*�=˸����>s{�>|��=n�>)�U=v��= �=>S��i8=�y>�`n=Л�>��>C�q>�*b�	��>f�O�>v�<|P�>��>	:L���־�>�����?�4@�K�c��=А����=��K><�;Q䢽驦�e�U���=��>�I�� ���P�=���=^��X��9梽���j
�� �'��U�=Q�U���>�h�a�|Ё��,J>�ƴ��N��agҾ:8��� �����s��3 ��\��Ú=�t�=ʳ(���=�(�=ϝ�<��^<ӽ�t��>H��=��2>�YĻp����t.���<ٯ��Hl��S\��V��]v���><A��h;��U������:=MD��e�G�?e�� ֽ�dL:"����$���:��D>����=8½
�W=�پ+���S�</m��~<����m	=6�|�h�=�߆=t��>��F��*����"½(���v�8�)����D>��M=˾�|�����U�d�~�P�t$c�+"�8���`	�<�68�~�<�w��s7�/2I��n+����y���T��C�������|�N@=�=]ŭ;�:��w�U�*;�3J>G���DQ�mD;�ʷ��������=.����F˽����E=c>�M�=��H��<�zKڽ��<��:@3�>fg����큽Ö>�җ��\t�<���ٚϾt�O�f����'��3m>�?�>6)�=3>�f�>��`��<�U=՘g����>��H�,�*�(%|�8�^>�_�>�͊>`,.?Gл�|�����f��xپn�H�n�ƾ#�=�+>� ��Kڼy��W�z=��*�!���q���>����E^�cW��(q��n"=�]���1��`�>�%��?��!;�Ʃ>�vW��>��|>K[��#롼��q��`�#2�����=�Ͼ�{E��ɼӸ =Jj4��}X�>ST<k�!���@��T|��%e���7��i�=g������Ҿ��k��;0�r���ÃF�ʽ:��쫽j\�<�����&Ľx��~Q� t���U�rC����u='���X=�ɾ��ľT<��#����S�1���m�ƢQ�q��=���=Z���Wݾjy�>�o>�cI>@Z<?�h
�_��={�>���p>��>��p=ų�1G��V�I>Սi?��?�P�>ۈ+>�9��>���=�����W�['�=](V�T��;��=�+�=���=YiG�_�������"V*��W��D����v>�pν��>Hֈ��F��Q��g}���*������N��=��>����7�\�@�������)�E�������v�[���Z�N�վ�S��l���Q¾����0����J��fw>����Y��$ƥ�|�̽n��g�;2վ3��:��< _�=�{�=�;>W�(�2X�>�s�������>��>ѡ�<=�=��0>C�>d>BR��Ճ�gU^�u�f>ʂ�� �)����>&Q>rf�>
�s>��^>I'ܽ�f?��r��Z*�Jt�O��Ì;7~�뾀�أv�!��(���g�=1?3��J��8�lkl�r[��������H^�����|�m����>:s׽��.�bV���2E>>��b>��>������=_Z�=u�>�Gϼ^ʽ>/&���k>A�L�q�����ʾ0|v�1	�Jk!��j���`>yʾ>�Ȼ�x=Us���K���N��F�=����8>��Ӿx!O�C����!���=����C?]�d=	�K���¼8)ýp�_�����N�>��ѽp˼>���_�=��C����>�.���&?Y]����ܽ;�m�=� &��p��)����>�%k� _�&���n
 ��{���gtA��=���CW���!'?nv��m>�G>�R3�#�$������#��;	�`����G>�Yc�V�6���ս�;�>��M>�e�#�ھE���P�G>�b�>=�>Nx���"�𫡽�/O?��v>�h,?����k�>2��P7>���>�
�>��޺P��=H%�>�F>�_�>Ȋ7���X��G�^3ƻT��=;�>H�
?��=$���G<�r`>�1=��ξ8�R>u۩>��=���>}}>v�����>���=<�>!��H��-=��.>�w����=�3L>��>���>4��>���>�ݧ=�q>��>>%�Z>pO>v�>e�ƾΰ�8!=	�R>�>>P'�>�N�� �羲���X꾫�>�>C>����-�>�46�Y��>�m0?�IZ<7NN�W*[��9���0޾b?mSU�糰��<
�%�?F�������wž��о�wվ������.�.9����#���A��3ؾ-�>^y>�ǂ��;Ѿ��1���7���Cq�>U�#=�� �\�z�d�`�7��>ᣬ��;ц�>�Qa;�
h>�K�>���=�k�<�t�>�����뭾�U�>]W�<�mB�U���K�����Ѩ0=� �^�t>L	��n��SP��=����>���=�>p���b">O���w�=��>��'�>��E�+��}���+���=��Q�7װ> �.>�Ǿ�P?��=������=�����~g�M���J�~,>u�E>�K.>D�ʶl><�H?aK>V������y->]�I��ʬ����>������$>m�>������>�ƾ�X¾�覽��=�ʂ�)ؾ(h���%����K�H<��ȽDKE�	<������,V�=�1���9��ؽ��G���=?�M<5����k��T7��M���z>��/�$5>f$I��������>ڸf>�Ó>_/�>/���t��=�>���>ק>��a��b�=}���>�Z>>�y��&%>��?��=�o�>/��.=v1�>�5>ƻ>��\H��>�>��>��>�ͼ�>�}!>�n,�E��>C]�=�6�Pu�>]oܽ�[;�v�LT>J�>q#|= ?/�1��>����egi����>9~$>��н�BC�ek��f&�=��<���>˻Y>�ƺ;��9��1\� ����>�~C;){�����G�6�N9�}�T����=pQ>h�=�P>�ؚ�@�><�>�Ѫ#���>�=�����`=&T���J���>0l�=xΫ=�Dn>�!�l�>�0�=yY>�vн�,S��,����C>��|���s�����p>/z>�#a��F?�wZ���'<&u��Բ+>�i�^�A� ���S��56�>�'M>�N��!J�p���5�r�%>6+>       ,.�Q%?^?E�"?����(���띾�+>�P��\~=Bg�(r�:7_�� ��b��1�I>�      %\�>}��>h����4>W�?'�?h_���es>1%����>�D�>�j���2�=j�)>�ɾ� k�z�;>��>�?]=��������0f_>�Β��!0�w���z=���W�Ah�)笾"��/xV>���=���"9�=m½����D>fn>1�j�6�_<��X>F},=+��<7�n=�>�
@�|A>f,J>�$�=K�>�V��㏾���T���U�_j�~<Y�W�$>ݳ���Κ>{��6��=\D���>�2�=��u>QS%��i�=T5W>��">��Ⱦ�5>�!=�U�=rv���T��{�=����Q+�6Pw�����t?���=��=y�A>�x��A���2D]>F-;x�'bQ�� Խ���=����B3�����<��.>mc�X@>�<�n=���O�j>��R>��_=�����H�>�iu>��>�(����0�8�>�5=<j�p>�*=I��=�/+��B��D�>@��=�?�����>��>]����-�=
i�j�c>K�.��U�<O��=+�>����БT>t����>T��3w�Yzo�.H>����D">����I�.�s�<=(>ԕ_�^۽#�7�H�>��i�\<�N!�_�,=yW>H˃=&��N����-?=H{7<��(��)<>P}�<'{ν�f�>�^>��;�"�=x򝾫�ǻv�������ذ��՚j>V�ž�&�>�e=?.g>�ߠ>�!�=�m伆s�>�T> -`�e�v�7�]��=E���������+̜>�����!�0���J:>Ž�=9'�O�=��A="�T�×>l �����>fh�>�U>:�}�!]>9�>�a>�,8�@��=!F\=F��M>y�5>r$��˽0g��i�>����S���p==��@>7���N>�5�;������>d(.��
��u�>��W���0����x	N>B��=pmϾm��=�J���=�p����=���=c��>���<�y�Ґ7N�~�e_S>��=�Ò�r�$>{k�=s�=�6}���2=�#�>�(���>0��Q�=���1��DY=_�2�����G==�w>��Q��ը=�y0=��>�c&�[�">�p�<�(m>��=9E-=`:<>A�I�ى����>��#��RP>Z> >��j=�A}>M�U�𣱽��Ž����=�C���7����;S_	<�5�������<�j��{l��Wഽ�R2>$�>���="I���Ͻ+��wĊ>v@��4�����^x<���= �>ìx>O�&>�>(�>��=Ԗ�=�S�>goz>��'�d0�<dm>� �<��79��M��G]��/=e�<�~���^���E�f"��-z�u˽X�<>w��=�>H�>���>"����GN<��>�.�ס=��-��G<���>����㌏=;�<R��="� >���@M�v}��h�t��[�=�8v<ר�-F>���J)�=�7c>��m>}>��7�D�=�e�=#>J��8=��=��>��:��}�q��>�d<,p|=x2H����>|=��ļV���мC�\>"��mR��R-�=3���إ��3>�}�=e]e<8L�=k%H�:�p�!h=�~�*`�<��B���n>?�������=��@�n��>N>	��m>F�$��-'=��׾Y�#��?�P�>P��r߼kY =I�=��D�	#3;�}0�ܑ�=��x��%3�!ػ���D�l>a�>6=�%.�
4m>	4�`B�=��6��G�=��㽣�$>�1�#-	�ipd>\E��w|h�� > �0���=�P>̒N���D=��5=���<��6�=��L?c�f>d���<K�>�e3�Na>�.k��&��	;�E�>=�ý񘊾��O���J>��=�@[��p��;����:	gԾ#Ƙ>(y>�w=i��ww~�C�>
@��;D�����:`>p��p{�B$��3d�>+��>�ͼ��>;5��Xj>�>�X%[�R0>���]D_>����L;����Y���쇾v:e>셸��{Ͻ��)=�]8>���>��m>�H�;Sʏ�9�9={����T>f���sP����>A_�>�L��[q�={=;Y`>�y)�6��>�)>O�>ots��v>:&��X���=�O�?O-B>}�>���>�>�?_�[��@�-��=�XN�,gY�O�S>=/l>�4�>�����<�?f���&q̽i����N#�~�=9l8>�((>�Q�9a1�P`����?���>��>�/i�Z)@ʃ���'�ŉ��	x�=dx4�_��=��>�:�����M����#=J�3>��>�
���=�I��3�>��?�R��=�]�<�
8>>} �fPa>&ׁ>�1N>��X��Q>��t<��>�U���>`׽Ȑ5>Ų�>*�ƽP�<��)��F�=	�>{�6���A=�Z⾻�>q%�>�ц>uĽh�X=$�>:�>��=}j�=���S=U��=��r<�3����i�h�ž,�>�QV�^�E�i�>��>݃t�C�=���>�)-�������)�.�k��$%=�ͼ��U>�>>��U>La���a��{�=ń����`>��n��Rl=��x>��c=��G>]����>S�>A6ܽn�>Nh��V�>;%�F���~�A>�K�l��� H����L��(�S��r{����O> H�>.F��U�>�@1<��==����׾�f>�ý��r�����X�+>��y�jm�����iQ>5���(�w�Id�=���>�`>=�>�j	>
���9�!��[��	�.>�O>�&վ�b*=��Y�Y>�>��#��]�����hF>�4n�������W>��=I>�'>QI"=�vU>��><�۪��9>��g��1��|*�t��>���=z�g�*_�A,s<	GV�֗۽6⃾@m�>+�>/�7�Ԝ�=�V>�ڔ=C|����=�����Og�.��<Ʒ���ɾȬ>�3w>�v����e�] ���~��O�U��u��򽾁t��v�>V2��{�>�i��+
h> 6�>�鴾��=�m;ډ߽���=�u��	Ծi�?��<��c���3�~^N>e���wb��zF��������f>%�NN	>U�D?�%�> ��>FA�����>�Pj=-�t��*齣^�?��&ཻ%�)�>��>T?r8��v��6����?!]�;�,h>륨?r�=���=�1ݽ�(?H�>�P.>�;��z�@DR������v5���
!>���<.[0�D+>�����>�u�����<X�x>��7> Z�����BB>	M��k��=��޼�o�=�܉���a�p���y�3>�����p<0R��tb�G�d=F@H�#��hC>�Թ��U�=mT�� �Y���n�;��՝W>���>A1��ö=���;��=�	>��H>�9>e�=} C>���;�A>N精C>)��O?*�>�%>?�ԃ�. o;w�� F~>�ڀ?�fv�ע��:�?�t
���<�l�(�K��?~R���T��?M�>��K��a@�$��C!C>�R�P;���e��Ɂ�>.����'�>�� ?t>��3_�>f�Q?��	�C-?W��=�=ؾ~�>���=D�9?����և���>�!N�*���S�?<Q�	���[�|�x�����u? �E���ʾ�����?�.��t����))��׽��>�����j���>�� ?w��>������=<Y�><����o��a�>��&=��?�ݿ(��>�|A>������\>vp�k����ˊ���%��47?P,~=��<q����׻�z>�ʻ�j?�������=�X���r�\kI�)S?!��ӝ�*C��QT'�W����>������0���ྜk->�K?>
�n<�΍�)[�<�ܽ٬m>οľ���"<7o�WO>�#�=c�D���@��p,�-�=�8�w��>̴�<��>�b�=�U�a�d��¢��*4>���=���̝>c.>��f>$E)>C�<�>��1�b?uf��"=��>�t���ra>�Vq�wwy��h?>���B�Y���þ#Ξ?��%�i��4����=O����O�>�A�Ж\�Kae?�mJ??f��,0?��?bQ�=0V����5i=�W��?�.�ٷ�>-?� >P��<�G����Yq�=ɺ�<�Q>���<��>c�~=m��>$*=|Z���PA>)�B>��9>P���I=C}�>X�*�Z}Ž�m�A�;/�I��<��W۶<����W�0�����o���xi>�k>\��=��p>��=G���4K>�s<�����'�<��F>ƅ�=d?�yӾ�y<��<���>�@�~>3/�>��>Ǳ��x��_μZ�=��>���=3��>���&�a��A)=�\��˙�>$GW���=��>���>����A��>��>uN�>}:���bg=hɽ��2����>
M=q��>/E>��{��ƽ���=>�>cO��ۨ�>�;?��>�{u���?���>���=9̦�i|u�Z�0=Ĩ��!��>|B?�t>?<g����>��i='^Ȼao�p�?�yg>Z�6�U��=q@�?09�����/�L��>A��?��-<	}d>�%-��)�?O���u�e|��t�<&��>=�<�����̠=��:>^�S>�'4�`��>��?��<�Ǿ��?��A}��|Ӽ5
�=���]�>j��~G�"Ns��M�����=�pQ����> �O��珿�ں�g�t��}�?S,�>��E����>�a?�_�>G�����:�|?@      �Vݾ�"=@��?���?�{�u�ƿgJ���5������m�m��\��VI�W���k�?�"[?�Ij�Ia���9>ķ>�@P+?�1B=��P�1�;Lj>�?@.S>è�=NQ���Ǚ?��@R5�?��6�=~��>IN�i:=�?b�!������y�>�ӽ��f?���?�a'��X�jT;?�����?ݰ�?5���y����F?�`�>�O#?錧?�QF�����v�?��b?�٫��<6���4�B��?
(����%�rx(<��~>�����F?a��>.�ܾJ$y=���j���W@��j��?�l@��>Qi忞�ſH�׶�R"�u��t����\��L�V5۾����I?�?�荽'kd�'�S�v��
F�_J@FP�=y"�?�r���?������"9?�����|�+����ﾾ̡>m A? t*�z����C�ƱP�@|��N��?f��?u8�?����k?Y���_t�?����ph@�S���軾�Ծ�G�?T�%@��,�m���C� ����?-ڽ�v����_?�Ӿ}X��$Ŀ���K?�=�R�Fj���<�>�j�|��j���E-�?�Mܾ��Ͼ�\�>���*	�?\΂?�\�?�'����?��>eYܿY� ?���?Lrj<�D�?���=�i�ǹ�>����lƹ�rI?w�J���d=�߿
��?��=�_^?`R�-�� �>�������?A��>*�+?��Q���q?�h�OZ�>����Hc?�����W+?D�,�?�0�>�<G��� ?uH
?�<,��L�=K�?��?p5�/����T��&�����?6�������58��:�?b%	?�t��ae?���)?N�4��2��z�>��>D�?�1{��7?z
���`?ó>� �AD@��?��d?m�"�����)��<	��!���ȑ��&+e�����)ݾ����go�?j������>oz��C�?���>j����;?�.��f�?l4�>�k?��([&?�V>�ڎy�cC����οn��?^Z?��0?�r��E�U>����NG�?珿J�ľ$E?��?�'��^���:?����=;-?�iz?W���d��>�`i�D�����?�[f?Y��=e(��#�;�)0���C�Wx%?�����?����>%�>?��?S�K>���Ps����H����콹k�>,O����?ԅk?"bZ���Ľ�'�?�;��ݾ߭r�Ai������OV�m*��V�
�D�=��kB?��v?���@      ��>�����j�>Row����=�"=����,W��A�=�ޑ;/�ܾ:�����=� ���3b���b~��< �<;��U�׽����,��<��
���~�@��޿螁?����=�>�V���p9�L��>bz7?;]��N�?yO�?�}?����uC�7��?�I\��!0?H2Q�_dR��#�>��V?�X��J>ɿ��=�!ɿB���
z1�R �?���>'݇��U^?������x�S&���>ɾ��4�@��>Gf߿{��j�
@��>�}�>�"�?ՙ����>^W�?���������X=����pٹ��.��,��4?ȿʮa>z�:�F��=>��>	}>	��?u�?pd�<�dw����=�w�>N��?���>ev�?�b>x]Ž�AQ?�y��1�>���]#4?>�m�����;����'ֿ��>"�5?��@�mS?�d��|z�<��?,Y��H�?W��?Z:���O�E�> �]@�O�?^�?��	>lVϾq�7>Ι�?H�?1�L= �7@I[>|�?�G@��>�>,�`wо5�~�8d�?�Ŀ��?>UAR>Ƅ=@1�?�͉����>kF�<Wbk�M�����w?�r;�]��ߵt?ޯ�?�  ?�!�,Z.���-��*v?{�m=喴?�2@������!���|@����'��(S?L1�?�Y
��C?����>��?�cE?	��?�ر?��*>Q�`�?i�����D���׿��>���?�r��	*�?��-��K����:�!�9?���=?�?�i��tn¿���?N��?8[4?�?��9?�I�<�󘾞�$>[�?j�M>�˼�?��}�>;��>���?b�?�E�������0zo��Z��Z�?U�@��=�{��!�? �=�U?]�>SQu�� >���&����r���駾S�y>R̽�٭��(�?SjG�SE&�}^��6g�?ڑ�>�f�JFk>^F�>��V?N�k=���?zZ�?�>��g>΋��0ަ>�/ÿ��G��?�d��#�*��������	?�!������>�??�h�=�a)�A��>�1��JԱ>��??��>c_ ?���BJ�j߿?�I�[A�=;�S��?D��>� 2@D�?��:�+?W�[?�j;���>��?�*�=�n@Q��>v�>�_�=�ܾ@�8�-sY� �?ɕ�>Oo�>�66��k_?N*׽9O%�E� ?���>�?1�B�o+�>���?`/��v�=5'����>5�S?��?`.�� @q�?��ھ<       ��=�ڽ��}<gr�>���-f>̘�>6q>���=��<���=\��>C��={��=��2>��<�@�>��=qq�=rC�>ւ��������<\���������߾����Ӿj�����������������&?�n˽�z�7����;ɼ��P��|aҾժ[>�@�(lľ#�>} �\�?:S=I$��zоZ����<y#�>����`> A�>{��>��)0v�Nf�=<       1<x>(x�=�����+>�ӈ>�$�>�`�P��=�<vJ>S�����>�l��`Lu����>w��=h��>/��>#�G>��>�KѾ��Q�@Rv�g�������,�Rn��50Ⱦ�V�>���uӾ��>#������v>����vξU䀾?��>�6�\>��+>,�R<Җ����M�ׅ>�ԡ��m�>�̎��\���͇<��<`Q=��>ѽdy�>�Xy>�6�>�۾�l#�_��>
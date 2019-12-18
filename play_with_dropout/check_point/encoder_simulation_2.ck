��
l��F� j�P.�M�.�}q (X   protocol_versionqM�X   little_endianq�X
   type_sizesq}q(X   shortqKX   intqKX   longqKuu.�(X   moduleq c__main__
EncoderRNN
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
qtqQ)�q}q(X   _backendqhX   _parametersqh	)RqX   weightqctorch._utils
_rebuild_tensor_v2
q ((X   storageq!ctorch
FloatStorage
q"X   94782106985888q#X   cpuq$M|Ntq%QK MK�q&KK�q'�Ntq(Rq)sX   _buffersq*h	)Rq+X   _backward_hooksq,h	)Rq-X   _forward_hooksq.h	)Rq/X   _forward_pre_hooksq0h	)Rq1X   _modulesq2h	)Rq3X   trainingq4�X   num_embeddingsq5MX   embedding_dimq6KX   padding_idxq7NX   max_normq8NX	   norm_typeq9KX   scale_grad_by_freqq:�X   sparseq;�ubX   gruq<(h ctorch.nn.modules.rnn
GRU
q=X[   /home/ubuntu/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages/torch/nn/modules/rnn.pyq>X�  class GRU(RNNBase):
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
q?tq@Q)�qA}qB(hhhh	)RqC(X   weight_ih_l0qDh ((h!h"X   94782107763680qEh$M�NtqFQK K<K�qGKK�qH�NtqIRqJX   weight_hh_l0qKh ((h!h"X   94782095648672qLh$M�NtqMQK K<K�qNKK�qO�NtqPRqQX
   bias_ih_l0qRh ((h!h"X   94782091472896qSh$K<NtqTQK K<�qUK�qV�NtqWRqXX
   bias_hh_l0qYh ((h!h"X   94782202500144qZh$K<Ntq[QK K<�q\K�q]�Ntq^Rq_uh*h	)Rq`h,h	)Rqah.h	)Rqbh0h	)Rqch2h	)Rqdh4�X   modeqeX   GRUqfX
   input_sizeqgKX   hidden_sizeqhKX
   num_layersqiKX   biasqj�X   batch_firstqk�X   dropoutqlK X   dropout_stateqm}qnX   bidirectionalqo�X   _all_weightsqp]qq]qr(hDhKhRhYeaX
   _data_ptrsqs]qtubuX   trainingqu�X   n_layersqvKX   hidden_sizeqwKub.�]q (X   94782091472896qX   94782095648672qX   94782106985888qX   94782107763680qX   94782202500144qe.<       Œ�>�3M>rAU>a��<ʐ��">/?���=(
?�7�v�? �??��>n�>ð�=�>->Н�>�\>��k?꯻>����mƉ=X`#�k�"�I�w��)�>>�s��2��zO?�ԭ�oY�?=)���(�>5MU�`a�>0#F���S��\þ �?I�x�ׇ?n�μ},>�������*~?�Q�e'ɿ$~�?(q��@b����(S��e�>��a?�A�>=J�4&�<�e?�k���      �����ּ����>�>�A�>�u���/�>|��>1�>u�=���X� >oS�	f����=�vϾ�->�߾��><� �b�w�¢D��򖾥��~aq>���O{�&rY=�l;�8�<=���<�~?��0��z��Pn>!
I>ꤱ���� NG>��>{0>�־�ŽN�I��~�G�$?�%*�sٵ>덶���4%�=fm�=�� �;K�=�'����>r6(>�hf>�0վz�k?&f>��z�ͫü-��05�=�m���+�>�!�>5��>��3�1�v�>}<!�!�D��=���R���̥�>�򮽭a>�#��$����K<bX�=�g�<�u�>˛{<)�y��Y=zz۾s��|X�G��=+FR=���>��Ѽj-���=��?����~)>g5�>�i>UU�=�*���=�N�Cs�=\��>¢^>�}����:�� ?�>�ۼ�G�6����"��fN�W�#��׽>-�پ���´��{J�T�>2�����>�aX?��J>��B?�T?�9��9ȾP�!?�7?���ģ?�F=��
=f .?��?��=w�!?9��<zw̽d�J>�bG<?!��&m<��D��cnR���c>���>���=�c
��ჼ�#E;m�L>��?A�t����>;��>v��e>'��<ix���WJ���w>�U̾&����[�>d�>'�ν�P�=�~L>�>>-S�Yw�=���<ڇ&>��h��`�>��6�;&>�a���I���ű����=�徾`�=!�>�->���@R��M�ξ�s����<� >s�پM!>q�!`	?�G(����.���U��>�Ռ��w)?!t�=�0���$?� �<���ʊ=�X�>��>F��#��>R>+>�mB>���
C���噽<2��I�?���>�B�	��=�`> ���4�4�ΦE>o���H?Ŷ�>0��e�=�
������n�&>V��~k�>�i�v�k=6$�b�=���>���=(����h��4�=������>��F=Ѐؽk�=�0�>~P�>�Y����>���&�;�7�>�lӽ�aɾ�ž0������=!ʭ=[+�x���7�>}�=��>B�ȾR�*>�m{���>'�=�y�e�����>0��<>.Z�>�EA���!B�>^�q���4��.��H%�������ܾ_&ٽ`��=�>f>�j>j}{>�*G��8H>�W�=�&_���.�#�R>`���W��>���<	gJ��w�>4��>����y�^���w���>fxM<gB�>����義�
��[>u'\�����|�k�Q���>륻>�>c~�����6�@��]�>_�¾5P��A>[?��X?��'���D��3>�S=��=����">������=~�<��B�q�=��=��׽xD9>��>φ0>��y>��S�e�]_�=�Į=���>��
��>r>'�+��b�R<���M��j?�?� �>�Ⱦ߿�>�r�觿�m5+>y��>��=1{?J/�>+��>Qc����>�q>���������f�> ��ԏ�>I��>�����.��>�`A�@�-�	?,��p�?��1 �>x;*]��8�*?�>�U>}�>�q>�N?Xxe>�g�;��!�̻�=$�?(	����>�'6���?�"�>�L�>����~R��Ò?��7��W�Y��K?է������7E��������M"�����s=.>��?�=�>�侲m��d�=v��>���>١?�תZ���?�����c�@=?��>�Z�?���= ��N�O��%Q?�»�p�=��_���>��?�{��T4�>K�������4�+h>@����
A>�cu��ט�AK�>[�>��"����>��Ǿ�+?O����?�=��>W9��R� >����<�����v?�-o��#�=���܆?I?K�k�GZ�=�B�XԾƝ�=�\e?d?ӽ $�=��%?3��>�!��'7���>�X�>H�ļ\�н	�#�u�׿
`?Ɉ�����>Uf�?e9>���V�BN�?-�>�ʸ?]o������ �=z�>[E?	�&��* ?��!=�*�><Eh?ǟ
>�,>��>�E�?�#��e����?Pv��ˑ���wAž~\@\� >��?%�>�X��	d>��&?ɩz������f�=�I�?4��?Vh�?K&�?t1z���:���>1'F�-@|>��%pS�3��=��&�_��>�پ�]��^`�|����{�����>�V?��9��\�?�z*�w�	?	�N?N+<?�iӾ\f���]7?��n>.�}=*�a?G���D�\�Y=0���ɾ$�?+�b>%��=܍?E�)?����P�������>� a>���J��qQ���K���?�L��>����j�ٿƪĽ��$�U���!?����>�>?+��=�?;�[��F��AN���P?Nʆ=�h��`?�����>J͙?]K:;���A�˽4?Ĩ(?���i3侉<?@#0?V�C?�ξd P<ӍZ?r"��{�4<,��=���C�� b��˿�ZD>��?��L>e��>y�����>�-�͓��\��c���(�yJ>Z�7>�<�R��6��7j4��@Y?hɑ�ѹ�?~��>�%�?`�d?�s?���>�~a;{�v�mk��j�>;Xh��=�>��&��s?VI��N2�>�Dy�>�R�F��^0�7C��I�>i�~�����q�5U?�5;��ܾ���?� ��B�(!��T�?)n���N>����L��$�Y��?l�=t��=�d�=4�X?1�_?���=�	�>J~o=f��vL�,��>�P?������l����>�	�����>;�<[]>�Ѿ���=��ĺ ���j�H?\}1�]�ݽ	�F`��!(�,��>#&���?=��>P �>�����D�!��=��r=7�= ��M��VdY��^�G!��<&�����f?�(�j��:!�����"�>腦>�p�?��D? �?*�>Ef:>e�*��X�?$�>��>?�*?�iE>�2I���i���g=��>�f��:ٽ^�U������?�rü�>���[���Ծ|a?�=��(-Կ"�=�v?1�?�E�>��>��>�R����>j���
�=�ꏽ���V;@6C>��;�<tU�T�����[B�>�7c��C��7y�>����.?>���/}�>�q?w����>����F���&k>��>�P2?[jž�M=@uܽe%?�G��KOû��>'��>a�>�!�Ңվ��@>���RM>����bd.?ޭB?%?��z?Bg�`3b�(�/�����ܚ>Ge�>�Q�?�#S>Z8�>{D}�-Gm>#A��b��;(��	?k�=<Q'����>AW?�V�?{9�>	�X�65Y><6�_�X?�z �1���H=?�Bj>n*=�=�����?��8��糾n%�������>4}�_�/��9����#	��Vp�����^����>�K.?Ò|��
.?XS6����?hb��Et>B�Y=�2��#U����-00�6������>r�>����MF�.)�>���>���>a?7������ޗ�c{/>||V>�]�?�%>E
�s>>�U��9>����m�־hn��ሾ�K>ð>�5�>�Q�Y�X��u���>{��?_t%�ζ彣��=���?d �?c�c?؝%���?������=
�?p�
�������?�U���cP=�E��*L�>%|*��-�?�>Z�D]μ�F��T�+?�=?|���֮?P�=��=G�?�z\?�?�޾�9q��L��>*�=��?��>��>}O
?�s/? �b�b��B4���b'�w�?�o�L-��sq9����湾�>?��x��:�>�*E>��ܽ�:F�-���:�?�*����>ͰC�𢾟Us>�4��*O�N�?���>uY���?�.ﺮ�=��?����=٫>��B>f�'�����z�sA�>x�D>��?�a�>M�?���G=HV?�4�x�$����0���x����{�1���L���؈>A��?p�<E;�>���>��>�H�>7���s>��������/��]?��r?�D�>O� �	�Ÿ(�f��ݦC�U�
����=��޽�ٻ�v�?�ހ>IX�����=������>�?� ɾ���?�3���Z�?/j6���Ǿ􀮾W
c�:�>�N#?!D���l?1D>jB�>4�U��ݘ����=J�l>5b��sM>�g����k?P��>t�?P�K����=�x�� ?���Y>��<�;���XT?�`���g˾`v/>;n�?���!�˽��=���N-��ϊ���T�?��>�怾�w׾!����t�>��>oL�>�bM>��=�A���%x?Q�)>�/��b\��3`?E�>g���e>�>��? !�>���>p���Y�>k� >N�>`�W>�d|>�^?3e{�a=b�*�ׯ	?eK�=���>Z��5�t��Y��{����^?��C��碽_�?'<=>\W�?X�������|�����=��!��4�=��߽��_>��=Ј=�=㝏>�kU�랾dX;-s���>�(�>$S�>nm3��+�>p��?�Iܽ��?��&�Qz3?��ھm`3=ȱ�:T(?����=��G�˼��C?U�r?�=>����?��>w%=Q��?)�r��'�<�ƥ��װ��ȸ�U�%=^X�=y�>=5>�v�B�C?�$��r4�٭V��=׿�>K��>	�j��>�	��׍?|      {'�?��>WEd>m�z��<�?i�o?��>�P��M��)?]���G<^��~��P8��?���>�i���銽�'G? >�?˯6>�,?��۾y'��#�����?z�=A�����νC��?ii�?N���?[�?^J,?8v�?R$���=�?<G��+~?�?��]�Q�?�5�>$D��K@п�>�&����?`d�m��?:�ܿ�jG���u?:�$�h�@Fh�>CA?p�?�[�>Բ�7��!���<q>4�>��@��	[e�TK�?�ő?�%Z��T?�5Ϳڀ+?~Lj?��)����?�?�?�ҋ?K:I�+���DS�z;=��G�<�"?��G>U٦�{,,��띿 �?m_Y��.�����Q/V�{5?:?�?V1�>��>�h����(u��h>���U$<�Ĺ>���?�Ł� `?���`?�CL?B?D4�����'���F���P���S�Y��r�d�׿�c���?i����V�J㫾�]�[g@f�P?��r?�7�?�H�?�b+�<6?<����l=9�0�L����O���y?IP?�|ɿ�(�W9��w0?����:������6^��XG�H�i���\�N�r�+I�> �e�G�>��P?ڴ���?�<1���?�
&�g�����A?N{�>���;R�>�Q����o���j��v�>9��?���?� H����?��?�Ͻ a�Fj��/����P�BUG��C��F�K����
�Y�>f?�xο�ݘ>�����9*����{�]���˾�]?�AO?Fz4�ݛ��,�b��@��<�1�>p���Q<Y楿կ�q���.j����4?��?@��?�Q=��>�o?�a?�v=i��4~�=YRQ?�Q����r���j?B~�?;*��5��h?��?y�y=�V�?B忛�ƿ9�<�!Ӧ�hzb>�SK�Y��?q��fɾ��$�p)��t���? �?%h+�e9f��?O�?퐭��@B��Ɉ���?ٕ6?��>��2�?�����2L?M=���=J���4$��{���D�?Z}��Xi�ͣ ?����H�Q�g�\�����Ǿ��n�ÿ5j�5&'�$��?UEn��5�>�$��l�e�Y➽&1?��7?Ò���+�?�}�?���?�����Z>�ez>4�?���?�Ag?[��>.>$b5>�����7Z?3�?���s�>
��>�����ξ���>E%�?č���?��t��W�?��>Z�h�ԿЇ?DШ�Baſ����x����оS#?�b�?��*�?�!w? ��?�9�6��?3ԙ�`�U�''��I�����E�?wK$<}�>]7�?��=���=��8?N퍾�?��%?�P��^ț?���>��?�=��<��1�I����>e$�?���<}����A�>�J@M��?@g��N�6�E'?
o�>�M��%v?�J>y���+q��࿞�)@g�=)�����D@��ѿ�򠾻]�?�A7?�f=?�F&����?���n\>}������m����׻�?4�?�Qo�$J�>h1�>�H�>��<(��2�>��Ž��>2����?���>��e����u�?��־W����?5�����?⁧?���>2��H&C>�>���>�)���(Ϳ�^���>{#�ˑ?�ʹ>�=#�$s���N>�����ξp�ʾ\).@]�ӿ�V?�A��J�?E
?à1?tz��ܢ�?
����s?�ͼ��C<6�?ġ�9>!��tJ?�|�?���?A@T?�A+���
�����ݎ=&�?큇���
�GS�?�I�>!����t�I��?�X�O�DG�>q�>p�I� ��?�S�p�(?�q}������p��~��x�=��˼Uc�>����g4�F��a?�r�>m��=��?R����θ�(�����>~m�)y���?�~_�f���v�?�)��i�>m������'o�>�ڽ?���?F�&?�r�$��@�<�7��%|^?����d�?^���_�#?�՘�o�-��Y/?P�?tyF��Q�>�ǉ?�/�hE�<��O?�Ձ?��>����I�}��e���S�R��?�����?�?[y�� wF�z�$�$�=ԛ�S��?��!�~)G�Υ�!�@�?GAr�l�9<�1��� �"ta�ڿ@?�>��ʼݴ��<+"��ӿM����_5�I ���
�?�WB=��k?��S�l��O��n�C��1�9Ȣ?�:?�S(?���?��?��@��>���=��.�3����v ����=�G�>zX�x5��6@]z���Ƿ>��>����-��"w �+��?��y>��!?�H��}?�ĺ?6��T#���m?@E���U?�a�?L]H?�!����=E:V?����g���?!Vw>7�?�0�>W��>�e�?-g?�K@8.��5>=J�� �V�ɿ�<�=C��H}E?B
G>~�;��+>���>q4�=�!<�L˛>kʏ?򭑿uA:?�x�>��>��3�
D�=��C?�q<?t������/����L?��{?�ٙ�s>(?G7��P��?�0����yT>v�>@1ľ%�Y?��X�2�?4ݿ�}�?�#�]%�?�o���ӿ���?�.���l>s�/��z��H�[��≿k�T�h�>1��>z(��;yX?{�>�V��>�50�+/�.$=ͯy����?�����ؾJ��?��Ľx�8?G̃?Ǒ���8�=Aܯ?�>�c�?�?�]�]��?	��c?ON��vu!�dy�>�R>½ݾǝݾ�@�q�� ���{�>;�"�E?�Ŀ`☾����1@m}N�J�A?���?U��QP>����_վ��L�jL ?�9��,�?���>��,��?R1��P/=Lb�=Faɿ�ċ�"�?��V���j>���>H�?Bn���q�2U;���p�=��4��L�?�0?�(۾�p�y���Jh�I���u޾��l��q;��R?�I�?�.�?~{/?woM��s=<�C�G��=۾�I*>>��`����8>D����?�}���r�?�=�:��μ>��?4	ƿ#�?
J�?#��<�z��"/>�<��m	�Ҽ�>֩�=��?ȅ�?�G?Jؙ�T��G�ٿ���=\P�=Drh��`=�V�?`z���s�<!$�?ln��5�U�@�s1��K[? ��cW0����?y��?_��?���.܎>'n���Dc���^��(�?����n��s@�N9�+ ��	��Ѝ	?�=4� �?�o�.�U�=n�Jm�ؿ�����ʿ��?��?��ǿ���1֕�bņ�xݥ?�����?�[?�� ����2�˿��6?���� �?"��#=~X?^�$?c�	���|a�=�L~?���>�6��T�T���슶��v<�x�?�n �eB�?0���<�V߲�m�>9�����?muY�HM�?��>'Q�χ���KD�@���P�?E�6���5� ��.V��B^��O׾��j?^e�<:�>�^@x��>3�D=ɢ�>��ؾ��^?�‿Ճ-?F�?���?щ���οƺ¿Pd¾3@� ���hi?E��?��E?��9?t8��Lv�?�8�f��>Nq{�#S>����2u?r{�=��������U����?ݕ��hN>���"�M��~Ž$J?��?�3�>UV�?�6?�~?J�5?��d�Z7�7�ݿQ������k��>��pR�� �%=l��qӖ���?����4-�?�b��R��?�ɿ��(��Y��Wp��4u�4T�V�>�Q��
9�?�����N�>��L?6ͿZi��O�|?	޴?��>�ҝ�=:�;��?�cW?#��?�䕿�z�?�B�O�'��$�A�/��C�>��?���([>a��<�������+�E��>�������� �~sտ�y��}>��Կa��> �.?�U�G&X>2$����Ҿ��>�D���;����!�q@�W2?�S������y<;�i���=�2?ԌȿpD��È���:�;�=?��t�1��>��?���?�=��Y����{���:0?o��� ��xՉ>bֹ>8�T��F�> d�y!%�0�?�$�?��~?��=yr?p�>jȂ>�H��M��?	����6>�ֽ��.��c�?A�#?T�ȿ@m�>�K@��)�>��<��/�M�?R��?����	>?�Q����>�y�3��_ռ!nt>tM𾯣�>�?6a?��?�y�-��q�ν{��?K�ۼ�@?1�N�0��?MB�T@���<�l	?�ѿ���1J��p�E�>�=��`?��I?sr|?��?ah̿�k>��J?V�U���?����.��>�r>�I�?w��Ф����F[���w���0d?e�����?��λ�X(���?�Ÿ?�R�����r&���?��R��<���E	@�Կ���> ���2���3�>?�r��*��E��k<Z?l>�?��߿Ŵ�?��)�Q7�/�=�[+>�%���?G�f>�m?'��=������O)���j������.z�:q>��H?~���O?0~�L��r��?�X���6?�6Ӿ��T?z�?��?+��?��>�u\�ۛ��u?ŏ��[J�3�@����A���]?���jZt<�J>�X��נ���$�����,}b����?�ɕ?�k���:?��K�@���J>�4�KL?�Ͽ4�����9�����?m?���>�ϓ�l5�?wԾ �ɿ�H�};1�� >���?�;)?Rtf��QG�Pf�?&�ǿ�X�>ȋ��>Cr�,ȇ� ��?�@�?�4�>�� �&0����?I�>���� ���z?�\���)W��@t�5�r��d,�6.�?B0?O�>8@YT������K�?����� �X4��#����o?�x��2쟾v�y?��?��e����? ���p���?z�@f�;@�>l>����DU���:?EM��KC?6:>�V?����?F��<OgC?R��??��?�[�=5 �?��ֆ�?�	�AA�?԰n?=��QS��Y"���?a����>�����aϾdD��B/=�Ja� >d??�G??T]?��?��½���?�����I�QX]�k��>�P���=�y?&�����F>� 	��ڢ?�|�?+;�������c��� �:釾Ãn=�*�?d�%���U?��#?:c2��7��=?A�=��@@��[�؃�?���ƾ���
=j���?�&;?q�#�or���5z?Z�S>�X>�X>z�n�G�=H�W?! �?1|�E�?�㒿�?����Wû>6�k���?@��>����?����@��]��?�m<?�?�w_����?h(M��p�� ���S�Lx8�G2e�3��>�ɸ>�nǾ�Ff��p�Ec�^/>7 ��_�����fH����e���ч>뒵?��]�v���,	�?&�^��c�����uľF�@�76="?��g?�a^>�@)��1?Λ?n�����?���?PR4?7��?�8�>�ǈ?�P�>hj�>/�)�w+����>P�{?E�>�5����ſ��>�f�=�gG��^T?�%?��q�n��?����6�>/2?�@��?�Y=>����C�=)ȧ�S���v>��&�C"z�'��?(��]����uH?W�+?�䰿�	�B����w�W�5�B?[�&M�?���?Y��>:s�?}��~�$�n��<��𾯓)@���~v���E�r>ۿD��:�R>�:8��q4�#��?�a?�u���'@�ѕ��B!��߫>�0@��4�� Y?k��_Ŀ��->���ju���a4����?*�|?��8?kqB�i�@X��>���%��>^�׾~�?�W�?���>v?y8M�d߾�!������G�?r_&�ݻP?�t�=�;�>�
`�,�i?�Z��@Ϯ?ֳ���6�>y�?֖C�d)�?��ܾ�2�>N�3>�?��<���,�����>Y>�?�������?cl��#�?�Ǿo0����?����vu�?o���X���Q�ʼ��տ���='��ъf�;'.����"�m?��?�%���@V��>���>1�Z�<�a?���>�??Ә�?f��>؅l��D�?�Uc���{?��t?�� =���?(�?�t@(澾kʾh%?�W_?7���pR��f2/�_�?n�I��ɍ?�yL>$@=�N���麾�-d>,M忍���x3?���?�ų>Y��2�o�=��?.K�?%�n>�?t$�*m�?��@�?�1`?'�h>��5��1g�E+>&;=�W.?*"����=��/�Ͽa3=�9���h=����'?���W�>1�s?�+ۿ��~�E\�>d?2&e���^�:�+�?�����л�ή�?jߒ�\
$��>_2���=?1+���S?d��>6����H�>z�|�J?�>L��?�.u?��+?�h�?�`�� �j?�vR��x]�܏/�����1�B8��v�?`ӫ��R ��� ����R&^?+5?w�C����?護=GuE?��C�����숾�t�=Hg�?V7a?p�tk��.�?�`?̫�O��?e�6���<��o�=�Yʾ]�)>��5�ؿ�k���~^?���ez6?5�H�I��?�)�>��>��ѿ���>��=~½������2?D@��K?F��?Iʴ�
�?H<&?�g��$$��e���:����@?�9�=�F>1�
���"?�M@��F?6#?�l�?���m�?�'}:��?��?��?�C?̓"?��0=�T�?��9>}��Jtg�"²=����E��J_2@ �>�]^?��?�p?���?��s?�5��8濃�<�x.���������?��l>I�M?Ph?[/�?���#@��/?]V"���m��}��
��9���B�?�)X>�F�?�\�?��?1>?2���~3?��;@�>��@B��?�->D"�=��
?*�$��@2�|��><�X�򅂿V���@,_��,��&�?G: ��X?=�}?^O�=�@��K>5�>�������5b�?m\�?���>�,>/�ھu��Ll�?@l���=)�R���*��}>��ɿ"�?���>�P�����}��d���D��B?I_����?]�A?��6��gt?�����3�>:��������2�qT�>�}�?A�B��?��?��
�7��>�р?���>�j���p?�>�c�>|�~?֋'��h�����ԓ2?h���g;�]�>\e�=ë��Z����>hq��80C��!�?�-������ν."�,я>�[��H�#>�t����=ВF@�6?���?�W����?���8�eN�?��l?��ٽ38�.��?�/'�K3��������?���>�b?r!O�f����C>�>l���?U3�>��#>PǨ�ͷ�>肠?T{5�LJ9�T�㾤n��� �?Ѿ̸��U@K��>t˂?#�H>U���Ja������==�S@�]��>3��?V?�P�����?��?7U�>�82?.���$���L?E?�x?tĲ?{.@3�?Es��[�>�h?�ÿ��~?������˿�0?[�?���?<Y?R	?h�տa�z�*��>�9M���?�b�=SU���Q�?�?ѻ�Tѿ���>��O�=������S��?�'�H ��"���CG����=ɬ���P�?@�>���?������KKA�f�9������?u�>-���?�~{�䊾�|�=n��>�mb�����A�?�
�>
�F�H�}?ބ?�|�����aߏ?][���?׿�
2@����?m��� ~ؾM�@����Ĵ�G�ܿs��Ï=���=gp�?tL?�X+?���Л?��_?�О?�l��"q�?��>B&���+?��V?�Y�>����}K?�ɿm�4>�O??�Ͽ'�3�߱�=KV���8?_o@�]?n��?oT�?��9�׋�AC��G�?v¿k���Y/�y8G?]�@%_?�ǖ>�}d�@+������@?�uw?/i�?g��>�C�?�㵿�W�?��g?�
����?�^O?"�?_mc>��?�g�>h7���>`�žw���p=���J?�kǾ���>��@T��= c½�,�=�䬾��J�k��&���AY?I墿�h��U�>��	?�?&e�?�k?�Y�+F��m�>��N�u�>��>9��?x?iw������:~^?({?��e?�9�?�o��?*�c�>�ǜ�5&ľ�z�>������>���=)?�g�=Z?�ۙ�*��o�}Y�?~��?/?��=��8>�پ�
�]��?I����ڭ?��?��ٿ����1s�ҳ���q.�������>8'�=B���#�<���?�?�@��=�ˏ?�?��=X <?g���o�>��A=�����Hԙ�:��M�=??�x�?���?��?�{9?��w=�?��p<���+@��<�{�0>̂	����=`�]?�G�?]/;?2��?5�?z�׿Rf�>����ƞ?pqD����>	5�;'M׾��?�����f?J��?q!I>!�D?9r\�̧�V�s>ʗ�w���-�>��>�Y�<ꏇ? ��>*@�?LQX�T�>�0���I�>���c:�>��>���H٧?^�T�De�?L�"@X�>�:��+�+���@Mk��-�>��?�E�<2��?�܊��5@�S�1�|?c��>�����e�>P��b$`�j�>Q}?9�;�"1�?!:*����?�~����?��B������'?��T����H�{��Kse?*U��8���>�K4��������i*@q׻?�?GO���2�?�UM?Y����\?�c���g�e*��`f��G��?1>	@|?��_>H�*ł?A>�����?6�?v�V�s���D	[���,?�N�>��<�h@8]?��X? z�m��9�?@�DO?dLb���?W�8�G�7�H?��@������F����u��ơ:>j�E?��?�0�Q�3>@�e.,�2\���ɾS�N?a�,�i�o?X����m?a�?�64>r2ܿ��?� 5������}�>�渾;پ�$s�-P�=!2u�4X)���>�Fs�z/��T�=	w���9<����>�\<��x3?�ǣ>w��?<��Z{=�sԾeÿtp�?���-�u�k��>�s�U���I=M�?D�>��?sJ��9@��BI�*Q/�_ƾ��>�e�>�ԙ�Xϑ>ڨ?�	����?�g?����g��<���>a$
>{ �?F���v?Ǵ3�٠���r�?�@3���?>��>tP�����9���aο�Y?s�?q�����=ȏ'���{�l���'F�?���>��?1�WC?�o�>���?xP�?�}?�3c���w>;�ӿ/6;��;r?��!�0� ? �>�8%����>O�>'�[?㳹�s�����>Ö�>�2�?Q	@�s�=
6*��2�um�=���?�w��#¾�_[?"?�6�>͕�?9�?Aς��-�?��>�v�Pv.?�+?�F=>3��>��2��=��>���g�D>�.��^ �?Jn5?�L?T�? ���̿w��j��b��5{A?/̎�� ��F����?���\?��>#�@<�엿��<E3?�s=?�x��%�?)fx=�?ˇ?(ݝ�D�j�э�;�qG?�٪;��@�v>rNӾ�xZ�6�����ɦ�d�ÿ����!�F=�1?��N��F?�bǿ����bcĿA��>��0�/z1��B-?��?Gv���*���"��?I���>���>��i>��ǿ5qg?j�"�Z�>�'�'�����K��%����]ܼ�W�?�n�>~���������? ��>'Uo�0��?�mu>�k�?u.��ٰ��4�?�V�>��x�/��<�G6��N?)�?��"�u�0¾ �ӿ���?��3@�L?
�M?��?�W3?��b?���?g`�?�أ�ڸ�>��?K�@�G�b�ɾ��f�꿖:k?Bf?Zi��O��Ɨ�?^f�/����R��a��?��m�� ߾3[3���Ҿ�N�?a���Bݿ?���zD�,��=�L��yh��X��?�\��#7]���_�>�@r�?����񊿡��?�����˿è�����*�?�Գ=�LJ?�nۿ�N
>B�'���"�(�?�!�,��?��5?�H<�qD��c?��񾟏)?�3�>b&�>�k�s�A��#,=�>?�V����)?��>�Y+@�郼#�ҿ$
\�#��>��a?��W���$�n�Mt�>0�*?�E`>NO����>����?t>�Y���:]���?�<�X�?F��9�I����;�9>����?%��?Ұ����	֠?-R�n��=����|�?��?͓�??�u�����O+?��?z�?v(ݾf鬿�ǣ?�T�����?��k?��׿�˟>@�
��Z�>m�ÿ���C���[2�Gd�?~}:>��Y�8�+m�?�.��Jz����?��@���?�q��:q>P����%'��>�;�z�>��?9�i?]ɨ?��ѿ.4 @�>�iq�i�+�>u�wV��$@��q?#<Q�g�п_N?��?�.�?>ǿ�.�>ML�Az˿��b?j�.�Ld�>1"�?��I��~����k?�!���$�>c�4�
��8�?Kb�n�P?}���^�<��9!?�h�?�W��7,��n>'���<�?G9��u�wͫ�'�?_��>O@S?eŕ��H�k�X=���>"4@QϮ>�?2���?#@BS}?8��>n)��V��N?G�|��W8>ӽ�>�=˿sDt:�\��}�?>	���qH�
��c���>���>�h@0�2?�E�ER�?O�>��U?���>���D��<��+�~���?��?�S��屾�!����>�0�"�>^uH�$�g�TKͿ��U�4����0�?!<�$���盽�<�?��&?z�P?�M�aL��l���s�;>H�>0����ȑ��S����O3?2��=���?+���C�;�?��=L#2��}>���R�?� b?�Q�?�8*�{]y>�ݵ����>;Q^����?F�俱r�>� ž�-z?� �>N}��;��?�߾�U�d��>I;1�7d]?� �>��]>?��ͭ���B���N{��z]�}'0����?ӫ��Z��?�bn��U?�|4?Z|��q�=���MM=n�>BN�?�T�=&�T. �χ��LŠ?��󾨩d>Mt~?��?m?So?\���ZR���wտ�C��,p�3u�>C��?��>�/?!{b�j����"��n�>u&�>ܕ޿Vn?[�?�nz<�!�-�|�m��?Kȶ>���?�9X?ۗ)�6??��&�e?��߾a��t<ܿ��H�32��p��m�>�����?A.C��!g�U*o>��M��?Al�>��3@T���"c?F��6�i�3A�?�L?Mlѽ��p�ĕ?��=��?#Ï�,�ϾS<���p;�
�c�3�>჆=�u?S!+�˧������#ʵ�d�1�梅��K�=�����=����f�?I����,���\�������<� ��7�9~�=��?~k5>n5�?�� ��?����վ�#�>��_���W>�	�?;���x	v?n)�P�1<�%!�^X��G�?#�?�$��{��>o��>�H������5�3���h��;� ?�j�����ʤ���0>a���B�?�޿�)��$���_V��԰��Ɣ>��==�3>S�y��N?��˿.V8���>�;����%?/ÿu$�f=����W<��?��ɾ_�`�����m���?�+?����;?kL�?�0?&���[��Q��ԡ�> n�>GH,������?�4E&�?H��Y�?�d?Α>YQ?K��>�ُ�ӱ'@�`?F��><��?���V��%�)?��g>��D?����-�?�D��ȓԿə?<�o?0�J���y?}&�?'F?�J?�8�?��޾@'�?��z�"��}�$��z���"�o M���?�R?�?)2?[,V�e2�?`'�=��>�؄?kķ?�p#@_�_��YM?�L�=Ω�?��x�W�}���jܿ,��?�:��&Z>U^?Ɲ�?ԋ�?a�޽O�~�qaq�Ƥ�2:�?�w?�%�!��=��?���q�<;Y?[�;�爋?��?;�s��R+?z��?����W�?����b��?��j?���>��@_�޾�YY��	���g�K��?��\>�/�?��俢d���z?8�?x
?���ȁ�kG���|?�_�?�)�=�А�Yd��fSh>`�?ܳY��w @�r�=� �?��Q�?�&�<�M���">tP�>`�?�J�����>��x?�.���ۙ�?X�t>\e�>U�?�-��I��?���wC�?pC�r�E?}`k;�j��S�>��>�&H?� �����?�>n�'�U�>n��?Ҷ̿~�!��V�m���	§���='�=���><D.?6��>���t�п�]ɾ00��7>�Q�u>�Y5����X�K?@u���>.C��]�H?�板�+�����V��$KV���?��"�5�?Iγ�hݙ?��b�3���ؽ1����?=�?�;!?K�?�f>�G�>)�����?�ӌ�d�$��>�C��x?���O�d?��@��>�h�=�	��~οt�e?�v���z>�P�>oW6���k=�W��q?Q+?���>(5J>O٦>��:��?�����?�|�?y��?�壿	�#�ז�?��>��m�R,������k�>��;�a2?�uA?����=�`��^|��.?�UM?!F?t��?�+�z]?�	�>'�D?��o>+*=�C��?<�~?aK/?k:4��v�>|.�>�j~�G��<k��)k�>���=�?�`?������|}?*Q��c�n�X-���n#?�	�>w�_?ž���>���5AN���>��@�L�?�|?ӯ�����5�>�An?������?�0?�Û���>�Ӿ����I����?Dg�>ٻ?�c?�{��4�%�Ǌ2?/���Z���@ɠz?\1�?�����F��ܿ��Y��̌>35!?&��K�>C(>-K6@7��-�?�:�?>}���>$$�?2u�?�\ @T�-?V��s?�������ش�0�u��	H?yx�?��t������A?�&?��=_��>� ��a2����?���M��?���q�?�n�?�K�>��������
?EC ?'�N>�*@J�@��?i��=<?vv��4q@]S?���>%?9?]8�?�y�?�|�?�@	>MK�?�&z>=�W>�2�?�Z�?І?.2�jF\?Ng��C!?gĿ/�ս:n?;7?�*W?�������)#��ْ=[�
��b�J�Y>�8.��mH>���3ը�S��>���>�Q�=h��<"�<?-o?N�����?7ơ?[�����w�>�%�����hm�����?��J>|!�>��>yL���!�?���thտX��?�h�?pp�>���?�^�>����jfп���*�-�(J�M�!�,��,'�B뾜�y�Ѡ�󖵿������Z�`7?�?	O�>m�?��_>��x�1;���?鍐�ˉ��э�U$U��!5����?	}���V�>k㠿9��>�5���>g�? 9�?�f?;��?'�����?��`'?`&� H?$v��T���v9N?~��]^?���?�]8��J>�`�>�b?�ő?�2�?18�>ד�?����1���j��g�?r�ƭ龄/��GC�>F�Y?o�0?pm�?Oy����_1��Y!�>0���?_r�>J�>�ِ?z0���@�F�?��>?�{����Y?}V�>�o@��?[>ܙ}��׌�V�?y��<<e���*���n����={�>FP?U�(>���?�
�f��?��@d�w>x޸��Nᾳ�$?��*? ��z�����?益���ӿK��?��J��y�>`9����<�<7���>B�a��? ,�=���>b�0>�
k����?/�l?֓�?kˤ?�o$?x��>Yb�>��2?��g���
?����K[�S��?{H�͘�?����D�e?*U��� @���>ZT{=袏����>gH��4M��׃?x�?��V>	+��u�>s�2?Ѳ�=
�b�_V��������?u^��SH@��¿d�����(����\$e��,��w��O�����B>��z^6�I������V�����߿wȾn��?�vu>T(�>���V>V2�>�b?zL}�6�?*�>?���>Fr���G��W(�k95��%���Ю�K�?&W?�qR>��q�R?��b��6��6�?�S����U?g������N���qe���W/�<\ ?T��>��4�-����|?G�����羧T�?� ��+,�~Bݾ����?��V=)�$=��7��Gq���>����QJ?h	?����O�?���3�I?j�ུ�7��ݡ>�y�?�$�?�$��d �?8ʁ> �x�f?o�@T�X>�LQ?�?��	�|=��>
M�*�a���>p��z}\?SRʿ4��?��>�i+���o?;O���S�>�R���V?hq�>8V?��>hn��~�<!��>z��)���$Y�?fՁ���C?���MA��P�����>"WJ�&�;�2��?'O��Yu>�4b�^�?Ȭ����ڿ�B����?Yg?y5�N��?��ݿ�[>���/1��Wc�?(ٽۂ(?E��>�#�<��>Xؕ=3#�>л�<��8@2����?u��?�����u��gV�@׿>�=h��g?K~�>wƇ� )�?)Z^?���>eN�>x�?���>�
���|�?$5�?q��>ܻ�?�����?�þ�&o>!���C?T�����R�͠@$�>uxֿ��&���b�� �>i,�>��?�lㄾoZO�v����>��D=/}�?��x�h���V�?�">�$'�L��?.�r?�ɔ��L!�7�����Z�Q?$Q���@��?c�S?Qb�>��ܿ�u>V�����>����(�?7MW�̆�?�!@ν��j?|%r��q��𻅿�?�lr?�"V�җ�>+�@�*�ߤ�?F:>'���k>�`7=f��;~@\?�md?]��������ӿ�x^?��b?��v���8��E�?�UN�|G?�C����?�
p���>�=i>���>5�?�'$?�=���T�?XJn?��9��L?�o�?�2�>i�?p�~?�"�`���?o���ȿ�Z9�`a�?ͶQ?��-?%�>�Sſ�ѱ��q
��k���&?`�J�J���������Q�y��+
Ŀ#ܫ�'S�>%#�i2-<��<?<����&?5:?4��ě�R��l�?p�>Z�h�� ��羋M߾�7?^��>�I�=�?���?���e�?�&>��[��Y?�H��������?nA��~��>�I�?XC�?��޿S��M��?���>�)Ŀe�����d?E"?B���
>��t?l"���Y#@ut�?Y��Y����-@	�o?$�?/1�?�*���$���Jݾ�}����8��%˾��	�:�?TA�?KaԾg��?��P?�~O?��(�>q�Z���<n�_��?�>d?3]5�{�t�������?L<�>�|�?�����B?i�>$��=��?�&>VR�=�Z忒��>5 >��3@���?��E?��r�[ֿ�t��mE7����>��p�����ߖ?j俟� �=�"��������? �t��1�>��O�+3����>�#>��8���E��>�5R?�8ؿ�O5�:�ƾ#��>��(�b�B=�R�a^#?a����4ȿq�r�D�D�~��׾�>/�>D�D?�{���$�?����_p�u@�齶+��;僿�>R8��m��L�ſ�]Y���>%滾�*�?�5��7����?uQ����?�K�>�S@.�?��
=��޿�@�
5?����4��ز?Ft��b=)@�E�W�?w$�T��>ޖ�?���:(�>�l��ʯR����?nq@���c(?�9�=�C��]]=T ?U��>���?�@;�>���>�#���?޿-�
������|n���>��Ǿ��쾁p��#{,�.�T����?w@H�?��t?��������P�3�?sο,J�?80��c!�?)t{>C��;[�?�Pv>�੿�Mi������
3��=�{	�Gt�=�"?N"��]��ȇ�?Gξ�
?l/�?���"�`'���q�?��{?B�?��H�1y�?�8*?��l��Gp�Ԑ	�G�?�0@�M'�� #>AP�?u��>���=��q���y��?S�z��04�?���=�},���>�ux?���?gE�>�����᱾�>h?j~�?/��>N��.�t���2�w?�I�6�>$8�?��W��Uf�=Y?�?Wa(�j��?� �?�&?� ���5���Ӿ�D(���P��f?������>��p��
���&W��ߺ>�lϾhrg�Ai)�8�%[�ze@e��?�>�?$sj>1�K?�Md>B>;��?�%�?�?G?���?p��?N�ҾY蕾�F�<J_�sSc�ɀ�>�1!žnJ�>ER[?�{�?W�?{u�o��ge^�J�?���=�'4?���?�A�� ?f��{�7>w?࿺?�~@�;#?��X��9�=~6�?�э����k��WY�?@ܿ�݊��U��O���{>{�ҿR�����o�=�&.����I�?���>s5��C��fm@��u���=?�a���$x?\@���k9����>l @�~�?�y���C`�j��=^M���b�>����o�T��?y�>�H���#�iV��	�?_�?��𿟡>�[�����$��	��������?+S���p?�b>n���kh� �=H��8 �+��>�7!?XDX�I;�>��>a��>�b�?{U�>��<�:�j���g=��d��S%�r*�#�Q>XG���?Aaտ6x<����1����<?�>�?�ݭ�'��>Fb�?�>��?L�J��R�>�yR?f��?
Z�>;�������l� @��%��󂿙B/?)�>�O?�ҾtU�?�i&?Ӄ�>�����>�.?�������?� �>t�˾���=�ތ>���j��?�ؠ?*��8j��B?�-?��>c�=�����?��qB?��A�쾰>�d�>
 �4ۣ>>ވ?D�:?&����b���N�>Q�?�+5��~@,��?��=�{?���&�ǿ�W��FĿ]���݅,���6��?�>�jȿL�9?��z?)� �oIP?G�Z?B �g�n?�?|?�;Z?̢#��Z�?�����ݸ?"ŕ?��=]��[g�?�T~��ӎ?>c��W�6?{�.>8�j��jA����?`=�5r	>$�>3�?�P�>7�,@낮?#�N?�6���?S�|��C��'��]�r?ā�>�G|?��?[8�? ⛿�0��@́?���>�Jk�I��P��ӽ=����'�?ϴս�K��<�?}<�>��J���0��9쿖���?��t��?��>�5��dϖ? R]?�Y�?ï�?LӾ)Ǔ���ҿ�?K]�?%(�{Q?��G=3���W�'��@����B�W">2��?	�M=JI)?^6,�����&f�� &?��.?C��>K9�>��ڿ��{��S���#�h��c���!Oƽ9 @F��?�0�?�/?��>?�?޻�`ӿx7?�=��(�?q����l��?5����Zb����)��>����;�?���?{��?��o���A?�r˾�_�?���>%��>>��Go�hA ?�ӛ�5�ľ:L�>o�?��e>�q�>�ؾW�?K����[?���>8Ɏ�#ٗ?�E�?|I+�Ɵ�s�?���֊����=��E�aנ��Gu��O�KQ@8c��儃��Ė��u @�1I�Q���`�{�̿#'�=LR۾�⸾��S>�G>�Z��kx?�gQ?�����f��u(���i���ȝ�6)(?pH4>?��[˿TQ[?��>j���Y�;��@۶����?��?�a>��|?��ľ�?���A������?㊾#ӾS�9=V�-?��{�?)i���=n��[��3�?�s��a	�s�f?��=���?��H?$����u"?��2��Aa>�b���i@�L��h �`H�����>:��>�瘾�N޿||o>=Q���>p�?އ?�YM�Y@k� �R?켗?P1L?��ȿ���??��?%�?��U����>MD�?�o2?��R? T�? ��K9�=�B?U��=P�?eί?ȶ-=SL?��>:`?���?���?	/�=���ks5��2���	?�]?$=�>�'�>Rd޾RM>�q�����O~>�sI? l?�S�>I㤿]�P��u ?��@5�%?����s>
"����=yO�?@m��n�? �p�� �?J/�L��>6梿_Q��Ո?]}�?Γ�>�Z��F�@/�o?�Ͼ�<�a�w�?�Ɇ�{༿Rl����R?��>EV��߲ۿF��?]��?m5��!\?�ѐ�*M�?�=��^�>�9ӿ9/�>Si4���,?��?lQ����m>~\)=�Nɾ���3�'�ȁ��5?A�?��ÿ��$?'����7ξ�~F?\Y�?Τ�?[���f�?ό�?6��?�?@?�Q�=�ʜ�%#���T�<��[��=��:?Kp��2�>=8���K�7 �?Wa4����?�m?�Z�?�t ��Mؽ!�¾�b}?��(?�<�վ_���Y�ھ-��=ޤ>#wp>&�����??u�?���d�P?7Y5?�$&?z��Ǘ)�\��?��u?����X��Y�>��>Q9�?���>�&?T��F"�3�4���i�~d�����e�?�?���@W?�E�?-������oߡ>@R�>�W��s������=��?-���Zm?�)�?�?s��n\���jj?9z�����>ɔ@���?B�?JS?�ݘ��;�¾����S?�T�?L??ӷ��C��>�,?̣�=�!T����?NŎ�p;�?I7��Z��>V�>���>xɈ��Z��r��>��?�l>1t5=����h�?:��O����@
ޔ?Dմ=��ᾇm�����>�^6�(�ҾW�Ϳ+���T>b�?� :?sO���ށ?TP�����>ُ�?~��?�zz?�Q��
�=/y�>fdF�޶&?�િ_����=۾�I����@�@�R��Ϳs�0@"C�� q��Z??J��=½Ц>�;>�s��?��0���1�0�'�(8�?}ø�x�T?���>�6:>��/?^E��)>t&��۴�6��>���AH�>*�>���>�\�?�J�?�⯿v �?����@ C�?g��'�]?[6�Ǫ{?��X�4�V	�?���?�A)?0�(>c�>�>y��k~�qͧ?Ȗ�?gY̿�_ӽ�-C?�)�6Ԇ?)Ed?�4�?V�ɾ���?&��]^?��?���o�>�&�$tÿ���>'�?�i?]ѿ!�6?5������?�����Q��׆>��?�>�WO�4l?UE(@a���6|>T�0��B�?�uL?��>�Z �ߺj?��=�p+��'B���>?�!ﾦAžV�X?����>4��;p�?Kn<��Y�i90?�(���\?���� Y2?'�>ٹ8=�zB�PL����{��BR9?�α�w�޿K� �����t��<0�`;0%���5Z��� ?=�>�U��Ϧ��@k�-?nh,?WP��T�>m\߾�(:?���>����_�r?P֣?�.@��J��"��S���4	>GT�>��&?�)@C��S��ؽZ?�?��_ᘾ���./?;�?s�꽵��<R�>aW���Q�?�e>��-��}�>�>D>��/�+�&?�����@��L&.> X{?���?uI��[��gu�>!ώ�mmn��Ñ?�))�v:d��w��V�?����1, @�x���?^��B�Կ%�3��?m�k�?p;?�Xw?	_�?G�J�>�N>�7?�bE>�c"�⾇���̿�J(�:B�>��7��Pؾ��?삂�r ="q&��F?������?>�(�況?1,<D�>v����J���`0?�ᨿ�-��k�?��?3tW>�]��?CX?�`�>c7s=G վ�K@u ۽��˿�1 ����2�����?���>Pq�� ��YL���#n�Z=����9K��.:sm�6���9�&�g.�d�!@��w�,>����/@��@E6?0�8��Կ�c���3|��f,>uRV?GG�>�Y?��H?�Iu?H;"?n�k�m��?v�����=>;�=�Y)?�?����?]^�?�x���G?���^?����y��?���?1��;KG?��>�sӾ�t?�G?��c�q�H?j�^�j�}?�\?�=>���=� �>F(>9�Z�pƪ> H�?}��=��Z�+�t=M%7?��'@H���{e�5r���kp>e�?&��q�z:��=\?��������9�>��m?�^��ѦH>z�?�J�����>�пއ�>O~b�O��?�V�<�nw�۴k��ۯ���P?o�#?S���0�9qG�H�>�G >[Φ>:��?m��>q��Jl�>F�9��K�k���=��?e鶾���?���>٦��u�5������=[d?͵g>��ľ��y��w���=X��,��q?�x1�_;�{�?�ӎ?����Y���V���>涐?O�:����?ztN���p?�Ŀ:�pX(>�%��2EX>_��WF�?_Q�>�J���>�B��r �?��>5��;Ok?1e��k��>�>�S�>%�%�6�j?���?VM��#����;��f���$��!���E�?jXu��>�]H����-����g�֒B���ο�r���^w>�O�?u��\?��M?�r[>���&6�����$�.���萭?m�4�X�?�A>kmſ�L%�3_�������Q?��s?�bW?�tC�|�?���>#f?�V�>��2��\ֽ�}f��y�>]���@O��E���Y���?]�_M�?P۽8�<n��=�߿3�.�<�����?$�w�[�>��?�<�=��D�x�<��?�;��/q_?��ھgb�>�M�?u�Q%(?�>���M�=��?tŇ���M��?��辏z3��L[>��g��$s�᾿2��>\���)W�s)`�%���?i�?�D��dY?�)>xN⾅Ǌ>���t����?�'�?�En�Ќ�?�����?*?�����>�ǿ����o,?���>k�>�����WX?h}�?��?)i?�/8�>�a1>��}?��?���2�>!�ɾG���<�z�{��]�>U��>��=� �?I��^Fb?�\��������?f(�]*�X�?r@>��>?z���ޘ���Q>�?�{����>���|������?��={.?!�j��'D����>c��,o�?H�_�u[�?�מ=p���"F�>X����d�:aw�lI>E`a?W�p�.���>�����=��=8�<m�?�@5'��9P����?��?9��G(��-+ὦ��=��?k2ԾA(����7�>e�I�;=�>��?�3>;z&?�VοaZ�6��?�5*=t��>�{�񣊿���?L��?X�?>w[���#?V��>T�����ʽZ
м]W?�ڊ?�T/�Yz��x��=��O����>CZ��0?3'�>�Q�>o����?p���@�h	v?Rn��ٹ	=�����>�c>_��+R?��!?����@?�z��]�?Hv�{�X?�dY�@�ɾ�)��8��>ĺ�>|Gھ��[��Vs?��@CU�S�:��L3�� g?���U��?�zD�h�V?���?�X���T�U��>��Y��2��ܻ>�3�?S���N�M��9����E�=�x9�L��>M�ڶ?A4���H�?�ڿ|*�?%pܼ���=���      �-�>W��>����*� >�u���G��Lrg���"����>v?�����[><
?��;(�=�=B2�<�%�>̒�>�3>=r�>T�>"0=�^^=@�=���PQ���*=hS>��ڳ�<�"�=8?l>�*�*$���<�ѿ=f�� ��?���D��U���_?o�Fא>����N?�Ю��t���g�����>V�"�G5�?�彞p>{��>>��M�������V\�=��W�x�O�BL��~�!<�G7<on���2<P�)���|������<>�>I�a>�<D����=,��ǼϾbz>��E���b>�U���ڞ<\�)�%ڳ>P�j���6��z�;�T�>4����=%�>j����r>e���v<T?7����>���G�=��>��Z�������C>�`����=��g��X�=W��Ii6�d�,�ǘ�<o;?�h��󢧾�Y��ʰ�><��=c�����?>_���`�佑H?v���#
>��?��?�&�{�����lV:�r��>���a��ּ��@>z����>�{?���=�cʾ�o�<Q#�o�/>{���>,�.?N�^���پ-D>�K�:�x�M}�>�B���2�� 	q��$���O>�_��e�K�y.v>�;�>��>&J�=*� ��!>��<�H>^��>X.>���>�m��L�>����Z,὞ś>�1���#�R�_c����>®����K��>0
�H��Z)q��p=�?�����?�D��5��>�Ѽ�?\H��ؽ�>b2�=Df>ϰ6>���=��g3>s9˽�
>륽�)�J?�L��e����?�?G��w�>{�K>�w��{��>���>��~?�oo�AVT��A�>϶�?��x>¡�V,v��̽�(��aڽl-w>V����N?�v? �?�n=��>�hO�VϷ�&���e���~�r����>�[����>��|��=l�=4m�=��P�nA�= i+>p�L>%���]>��>A�>��P���?F�=j �� �(>�P�>�Z��|�H>���>s��=�w����=jZ��>>.#C���/���&?�u>\���U?��>��q>��>H\���h����>����8�����>0��=�Ù>5��>��N>W��>�c ?R�R��-�=n��>���>���=�����>�	y�xW�!m>�N��Eɨ�=ށ>�޷=Jܴ=@�2<���e�j=�uR>Ü�=�7߽�8v<x�>��(z#>���>��мz��=���>��о�9�<���>���=��@=�g>N����r3�* C=i�>��t�/>p�?����(�>\kI=�%S�t	?���=���;�!E>C��>"�>�K�<�!z>B��=]�&����=%=�=���[��=��}�!P.<%Z=��]>��Y>�<ӽ���<���>�� >��c���z�q���F�?�$>�bl���輁%���"?Q>�O�=�5�=��>q۾��?�j�>���>�����	��>�����b�>H@z���.��e? ־^�?���?�}D?_>B���>$��=/��>��=B?� �>{�=�a=����<�-? ��q��X�2��=��B�Y�=�?v?�>ⶄ>�d�>�Z?H�;�?@ ?��??�\�>Б4��h�?x?�FU�;3P�&��>6����G�>�������o�ž���>α�><�d?Cb�>p�8�(F�?�,���a=z�{�����H�<����ۀ��@?� ��=Vr!�-f�>���=�9C?�W�>q}�RU����:�>�/�>��=�>c��?��X>��?)N>����7��>���>0	��4=��оn�>B�>e����mL<'���r��q:����p�=��h?ʹ?��15?p��<���)�?���=#�{?;�;�+BR��\��=��4>���>I^�>�T\���׾��?:YZ>�����Q�;{_?�v�J��>�߶>�dԿ���?<��?*�`<Y��r����i>�?�5�>s����g=aꜽ�#��(�!?]��>�6��b�?�.�>����KR?��Z=(����os��_���~�=��
�L�=�aG?@k>w��?��d?���>Y�^?ʅ�rr��	�@?#_�>�Y��r�0>:;��T���v�V?2@�?θ���0���-]���E>H�?a|�=��>=�Q���?�vw�D|�>L����TW=?��k?.�����ྰC?4��Vt�����g��O(�l��>��>s�?�?�u�=Q���o�?^)�?���>����3#?���Q(=�����/����>(�s>�z�>TD�>Ki�~(���h��hO�>ca�=�,?�!�>�RS=n��=�3?��+�+�>Kw����>���lϔ��,>�?�(���6��2�<�Ȫ�Ȧ/�q?�$E?G��?��=�J�?���?wP�<����멹>d���ꂿz�¼�-g?\Z�?�J>?Kc���X�����>��G?���>A��?�����"���J>�}@�r�?EEC>@��؁��>��;I>��1>�&�=���u&�����o~'?���=��@: �<"��c'�?7���߰�`,�?h'q��.�>�7B>�>]��Ǿ>?`�>�Ծ�A�>�p?.UG>�H�<Ƿ��Ή<�AK���E��$�>rqM>�V>��?���> Գ�=�C?�1�?҇�?:�'�=ֲ?�W�?"l���X�?��?��=���>�j�ELC>��u�ߵ8?� .>ƒ���5U����?i���j?�UF�>f��>b�\>׉6?�hG?�1 >�:ԾD�I�}o/��	�>)��=��>���>C��>�Ҫ?#w�<P�D?0�>�ұ=��>�>ɩ־���}E2���2�mX?q3?S;���Qv���;-?-P�?�*��6�ƾ�T�>�x����>���>�%��?��>+|>=y�����$	?9;Q�A�?��{�-:>-�J��SC�~_�IQ�>�Og;�ͨ��X0?�q����b=?���#7�#�8>��?���g�+?�>���=|��? �>���X�?*�E>kP?Gy9;��M?*�?si$>�"
?�&_?���=�(�>�A�>ot��?����*W>zM�>4�F��N�>���>�y2>??k��?%d��7�Ҽ�%?jZ-=Z�>�P>g�ܽ�~��l>�e���8?��ǾN>��
�M憎E�.�� ���Z���9�.�m>��0�e@?S.?۔�?zn�l�E?j��� ��>R)���6�>A^�?~p龙�5>r�9_�?Aa�f��>��Qp`��g=pu�v�j?���>α/?�c��5��h	��LѶ�������v�uo��9���FZ?hu��=R�>�Pz�&þf~?�N����8?�jq>�@������zsE���F>��C?]�-PѼ�	�?�	�+E�>�� ?�v7>�ˠ��U�2;�>���N@�5Sj��_?��G��!���? z$�0C�=	��i���$��l�Y�`�C?8���?��p��Q-��HS?�sS���#^��1���??��!���#[����%?��=��>U�ݾ�y>����x��+�,<O�PAm�Љ4?"�Hb�=��?_�?�1.?/��=K�>�ľ|e�?U!���ؾE͖>��q�o�7?w俌C������7>��=�2�H>�3?v�?c�������[x�E��>���=%�L������j?�`��Z&��5>R��M���'?��=��Z>i��]]�?�q2>md?Kv�>x6�>Ih׿�t�~ҿ?�>����6@?�dX��_�>�{m�iWb�ӯ��ǭR�X�[?��>*,ľ��>��'�)�d��ue?bG\����?���>�9�g�S?�X��䎾�7R�����U���#N��&���7>���x�>wG����
�_
?/�v�o��>�`��C�
:��K?���>�vw?ײ�?m��pA���3=�)�<�c��Tۋ��0?s����=D#�?k$���ϔ�⤿���`�l��l ��p�%cT?VՊ�܆���Ϳ(�_��@�=_�?>�~?4,��6;i�T�$�X��?��>ч�vK���
�?#�O��������x۾3*U����>�L����W��)��櫟>���?yTJ=���>w'��̜?F~���������i�x�t>ːi��Y �n��?��N?l�?�7x�VR��u/�>�:;�_?��?�r��˽�T��>�	e�.�?Fq3��7�7�?�܇=�LڿX~��o����2	A=
Ι?�F��e�H>�l?.�v��"�>g�>���?
Jn��TJ��ܿQd6�QF��i=��>�P>oп?��Fl/?X��?�L��4����^��mͿ9�O?�\C?7�X?i�?�8�<��>�xX?e"�>�O�>g�>�`�09�?Fp?�>F�=͜?Z1>#83?h�-��(>�Q��>�~%>��O�=��>R���#��P-�>�#o>�(|��K�>T?&�Ⱦ�-�O�?�-�=���?C�����N�7�?�j�?��~0>Z�n���׾���\�)>��?�׽i�ɾxS?@M!?��v>���?!�A����2���:�v�M�h>���>,C<>���*c�ڃ�(�<�&p��Ld>�I���Ч>#k��>*���ބ?�ׂ?5�>�]��>$7*�s^�
�q��+�?��>D^A�8*�>FҾF[��q���M?��?X�>:RȾ��=���Ͼ߬)=�*B���=� p?N�>^������?U{[�c��<       !�e=�>\Z�=`'>p)d>L��>A�f?f.�;ֱ>W�v>q�0?�n6?�h>^u?%
�>�;��?�;> �,?6��>�:�n�=��!<�C�������>��H6���GV?
�����?�'���>PYn�׳�>��c�����$�ɝC?3
����>Di*�?��>$͇�:�ھxL <��v�d��=,� ?��<��><6#?h,C��3�>;ۍ>���<�w<>��s���>wr�
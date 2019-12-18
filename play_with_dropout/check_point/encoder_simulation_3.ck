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
qtqQ)�q}q(hhhh	)RqX   weightqctorch._utils
_rebuild_tensor_v2
q((X   storageqctorch
FloatStorage
q X   94640646609008q!X   cpuq"M�Ntq#QK K;K�q$KK�q%�Ntq&Rq'shh	)Rq(hh	)Rq)hh	)Rq*hh	)Rq+hh	)Rq,X   trainingq-�X   num_embeddingsq.K;X   embedding_dimq/KX   padding_idxq0NX   max_normq1NX	   norm_typeq2KX   scale_grad_by_freqq3�X   sparseq4�ubX   gruq5(h ctorch.nn.modules.rnn
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
q8tq9Q)�q:}q;(hhhh	)Rq<(X   weight_ih_l0q=h((hh X   94640630704128q>h"M�Ntq?QK K<K�q@KK�qA�NtqBRqCX   weight_hh_l0qDh((hh X   94640645890736qEh"M�NtqFQK K<K�qGKK�qH�NtqIRqJX
   bias_ih_l0qKh((hh X   94640649734912qLh"K<NtqMQK K<�qNK�qO�NtqPRqQX
   bias_hh_l0qRh((hh X   94640648992976qSh"K<NtqTQK K<�qUK�qV�NtqWRqXuhh	)RqYhh	)RqZhh	)Rq[hh	)Rq\hh	)Rq]h-�X   modeq^X   GRUq_X
   input_sizeq`KX   hidden_sizeqaKX
   num_layersqbKX   biasqc�X   batch_firstqd�X   dropoutqeK X   dropout_stateqf}qgX   bidirectionalqh�X   _all_weightsqi]qj]qk(h=hDhKhReaX
   _data_ptrsql]qmubuh-�X   n_layersqnKhaKub.�]q (X   94640630704128qX   94640645890736qX   94640646609008qX   94640648992976qX   94640649734912qe.�      �;>u�Y>$5M�"u�<�i(>���� ���h>�	�n�v���*Z�=�+���G�+>u�J>�e_�Q�d������>�=�X�=�?O���t��= �/�'S?>#>{�2�ɽh���a=z�&>d#>e�v=�.C><�T>��ҽ9��橼�<>��we��S�>�U�=��2>�'K=��0>0��=o�`�	>\�=Xd\�I;����f�B�_�O*>���=�NG��$W>�a��>��"�ր��dH�P�n>W>!�E>7l"���Ͻ����N��p�����=:r�Z�W�5u�<����'>ѡ�=�>�f�o��8�=<�>*��=Z���*����z���=�`>��>) ��т>wfy=�iD����=������m��<,���H�?�F���><t�=#�[��>�P�Hx-�8}#���>�B���=+�|�[Z�~�>>���\�ý�dv�{'罇�N>a�ͽ��8���(=��G���_>b�*>h�_>10�#��;'>���C��{^�+�Q>�MнԆ۽<z<�^�0�E�>)	=\P*�lQ�=4�=4������YZ��Q!�?�>���=��<�A>j�p>Tp�=@�=>i�=4���,�_��Jq]>aȽ��j�.��ݚ����;�x�x�Y����)M>����F��=[>D�>��;�a���û������,>�7����
<}�B�d����"<lō�ط�=�Bh��Nݼ�����B(;V�F���^=M�߽�*�=�n���Q�=
"> ��K�=��=+=��Y���Wc�0;1�J \=�w|=:`=B=<��S�ru4>w�J>�=m^=a�=��=x���a=>yA�u�ƽ�o�=�\�F􃼊#���B+>d.��uL=��t�K�b> �;>��ʽ�4�=��>q!>�KĽٙJ>�'>�>�E��uS���$*=��>�,>A}�����=o{���H>�L�>�%>�;:>�=\ː=@P��p8Ҽ�����=��s���pp>�j=.�T>�-��r��C��m�5>G��<��=����>GSB>t龐z�������>�X�>H�;���(�fC8=�B=�羖od��⃾���=�B������/#�{<n�@ʤ<+�1�5�d=+�5�vC��
��[��=1>ٱ���a>�z��[�>�:ϣ;����"�PR>�+�
wm��S�P1>�Žo��
�E�{��=�5��@���:3=�6>��3��R�>�H>DX=m��/C�<�.�=��%�8�4>��8���//;>g�=��,:=��=��;>��ż3>]�ͽN�ӽu v<�=�����.����=H=����U=�G���i�<�j>��=h��>�-�=hm������ �<e!>�y�>���=�_ս�����s���Nk��7�=��'��:ݽ��=��۽)��>�j8�	\`>q�K������u��ňG>������S�A�n���Q�_�N<�Jk>s��=���o�����.�<�n=�o>�t�=�~M>	����Z�m>��=��=�*f��S�<��>f�+�<$:�rWO�S m���>>�肽 �`>l��=��X�D�b�z?C>�3�Vl�K�.�4&�J�*>!,�=�4��Z>�/�>�I��1ȽC;k=;�>�*>�JȽ}%���>���>9��=��=ݳ����C����=�a�=�(������9-<��*�<��<��=f�_�& #�<�V��%T�D_�hu���;>�&�>>Ɉ ����E=��>��>�*8=��&>������N!���>D��>;ږ��ھ��RGi=ʅ�<F|�=wN�>���=U	�;�F�=��?��E��kF�>��:>�.	=�b+>n3>	l^�Qo�
��=�����r=qb��r;�=8��=D���w�<!K*?ec> &��o��>�p�=�6���,�d͘>��=�d�>�$ ��w=i�N�p$�=���<8qQ>��M����;3ze���<"�=�i����=n��=Ɗ�=�Q�_@��g}�=Y����>>�㺐vK>Ʈھ��>���=���A5>���=��Z�Ҍ]<ks���ӽ�(>�2�;@0U���>�5&>�n>��5>Q�K:�3��i�����o�7=������=�>�?g=�5=�ݝ>����W���]1>�3W=�nr>�>��a�ב�*�;>�b3>0�>=�� >:pW>�����뽶�[��1>�>�1#���>�8<�N5>��u�-�|�`R��+ C>�2�>��=hc�>�-�ҙ= �ʼX��噽���V�,?��_���k=�p�<��>���>S%�<Z;<�p�������n!�="��=�X,�68�<�)m���p>V�2>j�@�8��g:A�&��>�Æ>�.6�h"�>PsE�M0O��ZB���y=�]��|�>��G�����H����X<a�|>���=]���8$ ������=g�<H.�����>?(�*=(�7B�gK=��&�����<">�k>�2�=#��<P9�>�A���3�L�K��o��L=n�>�2>P����<�l�>�l�=�9�=%��=��x�7� �d䏽rJ�X���Yཨ�=�jh>(}j��{�=v��tu>�n= �ѻ<�P�6��Z�+>������g�>C��>a���I=�+޾���<�Ռ���>ˢ �SwN�Dr�>�7_�e$>��>����.�;b2��?�>)��={�=j�">�"��W�>}��=Z�e��:�礂=3��=8]���ª<���=x��>JYC��>��^7=M�;D��n�>,���R>�=��C��ַ>�>$��;Q1>.*�>J�Z�,L��9�i=��=g&����>��6>^�G����3��Ę�=E��<�Vʾ�H>�4>Ο���$�=�G�>�1��1��C�=��1�5��~>*���=~,���=�>�:i�[��>����t�;p!��`�̽FG���2�>.�X�܄���J?���j3���
?����u%��H�*?t?I�>�m>��h�t0>��u>;��=>�.��8v��ŏ���p��w=�@���<�G��=dN>��U=q�>xP�]�ѽX���ݎ�>�<>�$꽦�1>�m����|�y){>;�?��(�J����V>��\?]�~����<���>;՘�y�4>S��ՇD����FxV�7?ϼ36��O�����=�=u�;�߬�E'�p�=z
�U7�=�)c��C�>���f蓾�|��j3�c&i>���vD<��R�P��o�>$�e=P���>��>�L���4ľH�׽$̢���>7�>D��5���ؾ�8>�B�f�k\V>3�>��;��ک���.�C>q�ռ�w�=��E��q�=⟜��TY������/b���>�Ͼ�X>�-ƾ��};�}=����t>i̯<��>+#~>����P�NpӾ�>�]��>�]���ڔ��mg�ʞ�>��!���=0B��8A>sn={�R�G�>��ͽ�>�� ?!�>�(̾���>M���ku��5��<\�">�B�>0(g��2	=\'��Uql���$�G��=�=�&D��m�>PȾ��>5�t>��b�¼��n�>���=0QM�l��n�m<B
�=*K|>\�龊�?��*>���=�y�� ���e��2X���<+��>!�����e>S� >#ư�qo����o>���5۽p���b�� yk�+�x�N����ڝ>a�>���=�	f>�{?x栾�x>[p�����M�>���1�7�t �>��4����xz������Rܛ=!���%�.%ʾ��>�L�>�k�f�=>�\=�Lt>�����`?aP?`RU�����o�������E3�vs�>%»=��?$4��"S>�g�=r�x���@=s:�eΗ=4�P˻�ʾ�K�Ajb�˼[>[[�>E�'>�9e=���N�>aϻ�����>n�>K]���>B�S>�,&��3I��^ڼ���>���/>�=�6�>A�=�9�>[�>W�����ټ��>Ja>;�����>'���T{�c*i>(��æ%=��Py >�����>�Z|����޲	=�g?
��>s����>{s;�dJ�>���_�=gп>dT�>]b�>𢑾m	�=��"�z��>EO�����==�f�>V�>��;�Ґ�����)�q>�t�>k���nQ�w3�>��=����O�<4�����3�b>���+{���_��پ~���K��]����ABa�,#S�@�}>���>�Q2��: ?�e?ʿ��S=xʷ=o�> Fþ�?��>�v�>��o�5>�ٷ>!Y��2�F=̳�>H�KB$>.�_�g>��H<r�N���!�V�>�;�>y�F<n͋>F��>�i���r�Vnh>쀍��������=��a��W�>.9U�����罏n\�qⒾ��M>6:���E{>��ؾ��]>N��"�,�>�����q�>�
�s�����K�1蹾a�\����>V!�>=�>�ݽ�������Ş=bA�=������"�?�ض��C�_�s>>5��Z0?th����*�C���P���h��>�kl�wB-?[�
�� �>d,R��v@��2����A>)�Խ��L�+c�=�k�� ����vA���?<��p=j�z��*�=,��>���[{=AqF��3�>���E��>V�p�Y��> �H=d)U�jʤ���<�#m?栜�8��>e�O?�����F�=ye_>>ڇ>�c�>m �w<��2+[>V�>�'=�į� �+?�      <�����#���<���=�!�K�H����=�bZ��?t<Ȝ�=07�=����X>��U��(���j��Q=���G�j<�!��P>P���R#��5��s$!��[�=���G�1�!��X�[>����D��w&=a{�r���n彮�X�D�=%��hM>��H>�0 �ij:<�KH���>�|ۼk�=���=�:��^�o|=�;>�����=>�5��ê=]�=̟P���`��D<^N�����)��=�g彮�p� ����7>��Ƚ�j5<[w뽦��=)�f>�D�=a���P?H>�{�=s��R�<�[���\E<w���Q0ּ�mA>z�t�}p��Qi>�c>�ȽM�s������_W���2d>W�p�%��=�[(>��Ἄ3���ݼ�BA��<�q[3>j
>�D1>$�J>o>,>�z����%��$���!>��=�엽s�K���>���=-ソu�\�dCU=MK��~>�D�&+�=+�P>;�ݼY��X��1v>>��=T@$�����ٽ��3���=vnR��N>g��=\�ɽ��f��>$�>P6N��������eA=�Fq�&,H�(h<�0�D�=�=��f���g�=%�I����8<�:�%�l�e�d<����S�(<}?c��۾��;�����<߹��6��K�˾&K$>o�E�����c-�=1��=
�=�ЇG�'d>5>���=�-�7G+��<$>=�u�(@7=2���D�l�J��h�4~C�)أ=[`K��<[�;	}�=�&I��=���91=�\���8�E(�	J̼,Q�� =�P��7�=L�=�u�5Щ;�'��n/�'?a�����c#�����c��<���<}3�A½Ɍ)�2����սH��=e�*<x;�=��;c�A�P5��$��+Ľ��=X����)��,��Қ=q�B��s=�g�=5���C���!�<��M��ýw��=�]�=X詽k&�AW ���-><�z��v��;NU>z ^>�C9��վ=m0>[D>|Ó= �<
��us�E�e���O��5�=_p��w0>Pʽ��J=r=;�������];�C2�>-���nC,�ϸV�t8>�I�=_�"����=��V>�H��q����5�RL�����<1����K��d�Ҫ콞NY��_}=��f>̺&�ϣ齼7>j?��n��=�0C<�2,>qQ>��U>�hF�Z�b���d=l�Y���b���>��\�⽩��=�r�;�ɽ���<v@N>��<�ñ=O\>d�=��)�;� ��=X/R�D�޽}
Q>���=�O���w�z#�=� 
>3>�j�;ʿ>���CtI�9�W>��2>��c���Խ&��>�0�>w��<�
|>�p��n����:f�0>O���q�>�|�><F>Ʈ׽6 �����>�N&=($1��K8�q	��&]=�;�=��B�\e�=/.>,i����1>X�=ٽc���M.>�����+D>�B����Ҿ��̖�8�=�$�<��g=S�.��G�{�a����]���젽)�;�8�}(����T����!>R�/�SW
>�RZ�꼌=ƪ��(J>�k=)~�=d�,���*��
>�J��FcҽIGĽn�M=��>�xe������=*�=Q�"��|�=�Z��I;R>0�=��>>��?>�J3>4}N�-j>�����8=
��>�5�:_��=N���訥�5�`�'Z��`(�P箾�`>�+�=I=��b;��R<ҏ(����=^�f�Y9�>��N=��6=�ѯ=k�J�,=4�߼-�z��.��0>���T�(�	�[=nT̽��r>18㽵>_�3=�b=~��=��={�]=�IȽ3��%,>�ؽI�>�;�=3|��U�.>����<�μ[�e�ݻ��u��N�=�G�85�*�v���	���*����k�����=_�>j���ҋ����^?9��~ڽ���=1��Fƾ5	W>?_���7�G�߽�,>U&˽a@
�`a����>E�?[C<����eԽWD�<���=i>��}X��[˾��q=�]��±>�|��+9Y�en��	���A_:�=���$򄾏��&)�w>>��~�^'>�'���=f�>a�=��kM=�Z�4�<�!f���<ʃR�pw>���=I�U>26[=`��[`��o=z/a>Ѐ<���V����=�3������|qB��U<��ؒӽ���>��<m��a?nrҽD���u/r�@��o���d�潄8����8����> A>��OK>�}#��v�>��k>v?�f�L�4�)H�cݚ=�<弞�>���<O�X<���=��A>A��=�����_��IE� �����<��	���k=+��>��t�21�=�n��qm����=�#�=��>h]�Xq���=��=]g>$�>v�۾� ��B��|�k�����=� ��˘�<��<���q>�/f#���9���f�v�O>�iŽ��>�_�uh�=z���g[� 84�|˺=`�|��y��|@>"�>�<���s7�&[ٽ�"�>vP�=�Ď�=��(�=]�8�YN>S��<+�b�>��,=d�m>Z>s^`>����Em�'�e�p��~�=�>�5�>V+�������ĩ߾�ξ����	��<=��� �>Q�ɾn=�<:��>Ņf�S��"?�=���@��>w�*?D�x�=�5���W�'���� C��mr�j/U>!h.=��y��w��=�"��$=#��j�I>��=�����n�=�8�<��$>�iM��ER����d�<�'��1�Żr'�=[��=�ر��j�@�ѽ|�����=�8��s�>��3<���<d�E=�D>Lh"�2�,= B�=m�	=g�I>5>��ga�#�û��=y>8��=��\��U;=����>|7�=��=d�������X��=AҦ��4?�-����R>/���W">~E����3�,��[�Ȥ�����XI�!'>�y??_�I��>"�Ǿ�>�0⾳�?���iK���~$�wB���^l�?�T��\=qh�=�|��6������`b��" �`x�=�e�="s)>RC����A��qL>y��>a&
?�t�>8?�0H>}��>	�<�"�>)�?q��=.^>|���� �>|[�=I�齢�3?
q>Ӄm?0i[���=�#�q�H>�~��v��= ��>k��=���+��=�8��G-ǽ�!r=O#����ݺײַ>��m>�c�>��J����u�����=H�=�sX=�t���]=&�>Z�=�2N��7޽�J!�r��<��>j�>��=���>� �w�=�lA���>��:>("��+=��_�_L>��=%4��(�=돰=E�b> ��C>ê��/��>�u�=���=di�=��{�%��A=�<��q��I>�1C=�W>��8=��=��ߧ=�4o>���:.4\>/q�>� `�� k>��/��I_=�n>d#�A�u5�=B�;��J�=�7?�1�>N�ĽH?Z�־L��>��>�>J� ��׍�߸q�/>�,?B>%)n��>�I�� �>)e5��e�~F����>��+>-p]>��=[թ:�(�`f>������3����=�d��"��;�UV���R��>HQ�=��Z>*�>ʪ.�=�A>(�=I���`��<��>�f>��Z>)��=ԓ5�Ù4���B>N²=�ve��]�=�W�=�tW=\'�Pʼo�#���=X��>���d�a>c���	^����c�1֜>/�=�{>�t@>��ǩ(>��� �>�h<���Df����=uU���o��Y��>�X>�#C>j�	?2���H	I���>6c?R�?�62���>���>v�E>�֖>G<?Y�G>��j>j���D�������9=�-�;��>=����`��'�=-�S>��Z�E꽘{���w���^����Ole>:1W<�{��<>�T�=T��=��&��&m�s�G��]�=v�>�x~>�gм�ٲ��l={ާ�����gQA�����;^T�>�Du����=q�>��������C<���N��=��P=R��$�8>X}X=�d�=�����y�ne����=�
�=E]��8>L��)8e=v���,{����Ll���?>��>���={��=��_=6M>_+�<�%Ž��=�ث�w���?�=+P/>�T�>r�]��%�����(>��2���}>�/�>��62<>�[��Z���-?�?�|X�Fߕ��U��v>�ℿ%>Z?	 �>6f�,b�'��>�y/>����=z�>�:�=|�=��C��/� �ý5�U�qd�=@
�>��(>�N/���<Y'���C
>,(�<���y'�7�B��F�<$1>C=od����X�>	)B��=�<~Wk=Q�������jC
�D��>o>�;Cȼ�)׼>�x>�-���t�>�4\�*;x�y��\���H��|>bh��!� �{~>�}=D�%��U>�/f>�=�=d�>�ɍ��>VM�>R:��[��6*�"�&>�4~>�M�>�<�>���= � �">G>�f?�ߌ>*��>��=��=ɘ>�Pq>+��;�>
��j�>�Ͱ>1���S6��;�q �\>Z�	>�6>	=��=�}=��O=b��=��p=��>u�#�8�_=}g�=�
�<h��=5s��x�Ţ=x.!���=�[����]����>l�$=�c �8v�</�=`>��!�ܰU>����)p>���=�77���G��+><�H>�      �M���?ai�@>@Et�>�G:���>�nl����e�6?�㎿,g=��?p��ǡ?��=7嵿H*@�:)�H`�>k!?��x?�#�}��?`u?H4ʿ�(I��K�?�	�?��>e��=�E>�>+���k?��n>�F=?ھ�9s��|ľ��ݿ�A>���?���>����y�	��S?V�j?�Ɇ��>l�νeЕ�v���\?g5�>�m�� �/�%?n^�0�#��Ф>���>{�y����?��#?W�w�!��?�J?�v ���=�4zs����೔�_k�>�;�>zN>]�Ҿl��>#��>�S�
⧿+赽�\���$��O����(����?�4�?��>��?�h�&�|��á�(�?ᾍ?7���ȗ����R?�e�?��꿽���q�㾔Zt��P�?r[�𫿱y�D�G?�X�=Rh����5��JK���ؽJ� ?���=w���C>�H꿤9�?#�վ�˿Ֆk�h�!?v�>�|�?X<<���?�?Xڶ��?n�?�a��i ?�j���ľ�`>AE?AFU=o�>m-����?�U����>�K�?2������?��I�4�տ%�0@�O�IDR��K?DC>� ���縿ZO�>-�j���>xAJ?��=��?J>?#����>�=9�?���[�/?ӿ�=���=R0��Iؽ�?X�>pK@�rE���M�f�S?�VQ�;��>�KR�35?L�羿����_M�a�?��a?�s�ݩ>{@�����K�&`��My[>��B�I�`��⡭?����@(X1?K��?�-ƾ��!������鿉��{���?�X@�	>ܣo�#�#�K]��L(.?/�ÿ�zٿ��޿�?��?��G�#*?;un?��?`�?�O�QC�������?k����C�wT�?�����P�?H�?jJ�?|8&��ʈ����>�k�h�b����>Ъ�?�����?٧5?�@�ܼ#M�?�.�?��]��b�>�?����������<j���%���R���������>�e��?����Ĵ��"?߁H��0 @��?����>��>�������=��>�����W�?��?{�%?��">�d��n�?�g�?��h?���?�;>P��;H�?��㾢߳=��9?!:R?>T�?�0#?�u=$M?�)���,󾙂 ?\�>����U�C�5M>Z.Z��q���_�>-!�ʝ�>�^a?�F��"�>H��?#P���膿�=����>P����=��?�n4@���_���J�b?�ֽ�Խ�Q�?��:xq���>^?�3��!;��*�W�>��?Q�?��Ǿ,��JCn?n[�>%2��w=�e�?�R?}�,�ӈ5��C?4� >Jb�>�x?q;v?M����->��=�����S?�)(?���?d�]�+ҿ�q�>�	�J<�=�b	@���>kϿ�b�>?��>�bf�@�=#�?���cû�>����?�S�>3��?v.�>�j?q��>rmd?�1��S�?~y��
����T?�q޾8$�=�Q?��?i�4?�:־�Gu�B�p�k�3?���>x�"?��?9�ϾJ*�j�����?ʛ�>�\����i����E�?���Ր?z`�>>A�?��4�#�O�7Mƾ���?N�>�Y�>�c%�a^}>>N!���?2ك���?@�v���?�����?��x>p�	��>�Z�?0�����>�qi?S��?,��R�@�O:?��-��C��⭿,?Ѡ�>3$�?�@����>����"`��.�?r�����Ľ� \?R=,���?�{?���?Km��|#?�~ܾ��0��䵾+��>2?k?�پ�0�>Z�A?@���Q�8�R����#�>?�le�TN?T�R���]�Y�?B�7��uH?��I?���?P�_>{@�>פ�?��z>DN?l��>bJs�H��<�m~��.?�Rs?/��?4��D���c",>�@�?����v/=zIL���'�1S?O����g?� �?5����?�D��}�>��?4���d��6?d?d�>�����}>���?U�[��®?�Q�?���?p�>�qQ>�%�>�/?-F���?�c�>hX�?p$���9��jN��)��ԥ,>H�	?�VK?��>qN?�s>�O���5P��1�?:�{Κ>w�t�����H5��,>�Q��|�^@ӣq>y�{>'�Q?��:?zK���K;���?��:>eգ?��i>O��?��ɿa�=tc�y���c��?�]׿j�4?�Ta?tᦽbj)?^�>�!�<�1��g>7#~��
� ˈ>�<�nľ�ծ���4��L̿wNʾ8哿O�J�J/ɾ�P>�c�@qd�=�����Z?��o>n-���LA�4)�=:?����ˉ>���l?��?[�п�%h�=��?�x0?���>N��EOj��g��^v���>�q�? <V�
�fa�[}�?���>�
�<�fG��ol? 6ؾ��A?��:>S�?��>ԡ&�5O��w{��Ae)�W�![�?|?�?�� �+y?7R?�Տ>4�?�c�=O�O�.�>N"�?��p?��"?*�@',��4������s�R�����> ���L}{?@��}9�@拿�͋?A�ؾ�C�?iu��q��?��?�ۅ��C��D�1F`@�֐�UP'���I���ʿ�,#>l?�c��h�>(ؿ��=1I���'?W��>����ݢ��S�>X)�?�3�?��=��ǿpk%>��ƾ���=����w����?���ͬI?{$��������>ф> ����>�N�>o]=r�c�Y�?�����>҈J��ɫ�	`;����?|��>,��?��?b��>6�?1��>�7��+
�?��f?w���	��D�?��?��&��Ch?�ǌ>���u����N���� �t��x�?zҾ9�����?� �?�]���w��$P?���?�6ӿ�.�����-~�"/��xծ���Խ��>'���EҊ?^��?��<�ǁ?��ھ������`�?�^��������W�>g�>�Ih�W�.��zA=�bK?��?6׷?�t����9��6����������#@^�*�Y̓>�W���?ܞ�>����[�?8��?�m���E���g�����>� �?�3Z>Zz@�)p��Uk������>{��?�7�������O��?�p>�v�=���?��ϊs?�^>��?�#"?� ?��?�J?X��:��o��\>� ̿!j>�rx?���>2�&?��S?�m󽥦�?�����.?�����ч?�5@!X���X�֪,?�#����k?W�>��??�h??�_u?}�(렾Ŵi�[a�?{�?��I?�'�>���C%?�>����?.�?��x�󟵽3uY?�K�&���V�>`�=�R���R?m:���%�?W+�?zz�i,���� ��?��?�|?�֖�7��?�c�? ��OXS� �>�n�?>�y���>֩�?�T<�O�ξ�I?��>��̄>��>0p�W�T��ϾՉ>7/꽌�v?⅀>�DI���?X~S=ƣv?�̸�6�]��pپ�U>���Z�N�?�;>�����=�z9�ٲ#?�@��e<�:��
9��O��> ��x�? ���V��J�?�A6�7�!���>�"�?�#U>�o�?E����|�붐���	?ay�������>�+��M��������=? pR����?��	?v���$Y(�3zĽv1�,�N>��{���=>N�l?�1!�L�%��9>����V�տ\�b���?S$Կ�*��N�����?�?[�O#%?�7=�	=>"n�GK㿍>i>4
̽�VO?H��=���1���Ǵ?�浾1��>��=�$"��+��<��>ע���
/?
�=1�p�妈?I�1��02?S���t�E��]���X��D����Ӿb|l?_��>s���Ĵ?KH۾�ÿ�?�>߭�������	��~�?�1�O���l׿H�����c@4�����<v]�}2ǿ��n?/�z>����:#>�Y�?ړ?+.���j�?��#?��=\��j�?��Q?�mg�q^�?��Pq�?I��)��>T	ۿ�L�?�w���T�*7_�c�?`��?`�?�? ��>���?�rV>l�?�9��7��́�8�p�B��=�|Q�]��^$@:����)t��+/�u�����fu�>����z�?*��?�G~?�1��r	�C7�<��>�������a�h?+��h3?:!@�ƾ��D?��C?�˿3�?�H4?��,@T��?w�?73�����[aK>��S�6-�>��}�����P��>6?��/?8��?��W�Ƅ�>���?�1u��dX���1>��m?C���)B/�E�����f��c[����?��?�&�<%c?�Y��Kk�Yδ�}�
�@
����,?QS�?�͏?�������}�?%Bſ��3?�r�)�,?b�o[���W~� U;��n�?��C��]�>�M���:?'W򽟑=q%6?�d��Qz�?��p���,@U��QuI>?nk?"�?�m��r���zC��Cվ5���턾���?���=I$���"?h�¾S�>��Y�fF?�(���60�@�Ϳ�X��,��?)W�����>q-?-Bz?v�F�<4r�h?I0[?%�����P�<.g���?Dݥ���ҿ<       vս:��=�9������h>�'�H���Sv�<L[�>�{>��$���z=[�i>zy�>��=��=J{�=�t�>'��=B�e�'�==��8��$7�Y$��ڇ\�������@��w�f=̀������2�����4��j2����l�f��)e<��-�V��l<g�<P%=I�m�V�7>VϼiЭ�O*�&E�<g���=�=�k�	n>Ϯ�=�����<��$�ﱙ�r����"><       �c>M�<��<�w�}�N=�*�>�4߼�a��U�>V�ٽ��0>����2����=rp�=~��<�p>��=���;�=MK.>��l��ǽ������ҽ3��&%6�"����E=z����^���,��ؾ�ق��!�������,���y��7Ys=[�
�t����=�C��  ��"��>~�=�̾�.
��D��6_�� "=դ�=	��L�?���X�v��>���eI�~:��
��
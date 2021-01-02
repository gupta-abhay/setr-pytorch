import torch.nn as nn
from ResNet import ResNetV2Model
from Transformer import TransformerModel
from PositionalEncoding import (
    FixedPositionalEncoding,
    LearnedPositionalEncoding,
)


class HybridSegmentationTransformer(nn.Module):
    def __init__(
        self,
        img_dim,
        patch_dim,
        num_classes,
        num_channels,
        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        include_conv5=False,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        positional_encoding_type="learned",
        backbone='r50x1',
    ):
        super(HybridSegmentationTransformer, self).__init__()

        assert embedding_dim % num_heads == 0

        self.img_dim = img_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_channels = num_channels
        self.include_conv5 = include_conv5
        self.backbone = backbone
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.patch_dim = patch_dim
        self.num_classes = num_classes

        self.backbone_model, self.flatten_dim = self.configure_backbone()
        self.projection_encoding = nn.Linear(self.flatten_dim, embedding_dim)

        self.decoder_dim = int(img_dim / 16.0) ** 2
        if self.include_conv5:
            self.decoder_dim = int(img_dim / 32.0) ** 2

        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.decoder_dim, self.embedding_dim, self.decoder_dim
            )
        elif positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )
        self.pe_dropout = nn.Dropout(p=self.dropout_rate)

        self.transformer = TransformerModel(
            embedding_dim,
            num_layers,
            num_heads,
            hidden_dim,
            self.dropout_rate,
            self.attn_dropout_rate,
        )
        self.pre_head_ln = nn.LayerNorm(embedding_dim)

    def encode(self, x):
        # apply bit backbone
        x = self.backbone_model(x, include_conv5=self.include_conv5)
        x = x.view(x.size(0), -1, self.flatten_dim)

        x = self.projection_encoding(x)
        x = self.position_encoding(x)
        x = self.pe_dropout(x)

        # apply transformer
        x, intmd_x = self.transformer(x)
        x = self.pre_head_ln(x)
        return x, intmd_x

    def decode(self, x, intmd_x, intmd_layers=None):
        x = self._reshape_output(x)
        x = nn.Conv2d(
            in_channels=self.embedding_dim,
            out_channels=self.embedding_dim,
            kernel_size=1,
            stride=1,
            padding=self._get_padding('VALID', (1, 1),),
        )(x)
        x = nn.BatchNorm2d(self.embedding_dim)(x)
        x = nn.ReLU()(x)

        x = nn.Conv2d(
            in_channels=self.embedding_dim,
            out_channels=self.num_classes,
            kernel_size=1,
            stride=1,
            padding=self._get_padding('VALID', (1, 1),),
        )(x)
        x = nn.Upsample(scale_factor=self.patch_dim, mode='bilinear')(x)

        return x

    def forward(self, x, auxillary_output_layers=None):
        encoder_output, intmd_encoder_outputs = self.encode(x)
        decoder_output = self.decode(
            encoder_output, intmd_encoder_outputs, auxillary_output_layers
        )

        if auxillary_output_layers is not None:
            auxillary_outputs = {}
            for i in auxillary_output_layers:
                val = str(2 * i - 1)
                _key = 'Z' + str(i)
                auxillary_outputs[_key] = intmd_encoder_outputs[val]

            return decoder_output, auxillary_outputs

        return decoder_output

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)

    def _reshape_output(self, x):
        x = x.view(
            x.size(0),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            self.embedding_dim,
        )
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def configure_backbone(self):
        """
        Current support offered for all BiT models
        KNOWN_MODELS in https://github.com/google-research/big_transfer/blob/master/bit_pytorch/models.py

        expects model name of style 'r{depth}x{width}'
        where depth in [50, 101, 152]
        where width in [1,2,3,4]
        """
        backbone = self.backbone

        splits = backbone.split('x')
        model_name = splits[0]
        width_factor = int(splits[1])

        if model_name in ['r50', 'r101'] and width_factor in [2, 4]:
            return ValueError(
                "Invalid Configuration of models -- expect 50x1, 50x3, 101x1, 101x3"
            )
        elif model_name == 'r152' and width_factor in [1, 3]:
            return ValueError(
                "Invalid Configuration of models -- expect 152x2, 152x4"
            )

        block_units_dict = {
            'r50': [3, 4, 6, 3],
            'r101': [3, 4, 23, 3],
            'r152': [3, 8, 36, 3],
        }
        block_units = block_units_dict.get(model_name, [3, 4, 6, 3])
        model = ResNetV2Model(
            block_units, width_factor, head_size=self.num_classes
        )

        if self.num_channels == 3:
            flatten_dim = 1024 * width_factor
        if self.include_conv5:
            flatten_dim *= 2

        return model, flatten_dim

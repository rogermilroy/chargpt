from unittest import TestCase

from chargpt.components import SinCosPositionEncoding


class TestComponents(TestCase):
    def test_position_encoding(self):
        pos_enc = SinCosPositionEncoding(context_size=32, embed_size=4)

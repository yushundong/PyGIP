from gnnip.datasets import Cora
from gnnip.core_algo import *


c = Cora()
a = MdoelExtractionAttack0(
    c, 0.25)
a.attack()

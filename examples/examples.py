from gnnip.datasets import Cora
from gnnip.core_algo import MdoelExtractionAttack0


c = Cora()
a = MdoelExtractionAttack0(c, 0.25, 0.8)
a.attack()

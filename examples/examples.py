from gnnip.datasets import Cora
from gnnip.core_algo import *


c = Cora()
a = ModelExtractionAttack4(
    c, 0.25, './gnnip/models/attack_3_subgraph_shadow_model_cora_8159.pkl')
a.attack()

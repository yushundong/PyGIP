from gnnip.datasets import Cora
from gnnip.core_algo import *


c = Cora()
# a = ModelExtractionAttack0(c, 0.25)
# a = ModelExtractionAttack1(
#     c, 0.25, "./gnnip/data/attack2_generated_graph/cora/selected_index.txt",
#     "./gnnip/data/attack2_generated_graph/cora/query_labels_cora.txt",
#     "./gnnip/data/attack2_generated_graph/cora/graph_label0_564_541.txt")
# a = ModelExtractionAttack2(c, 0.25)
# a = ModelExtractionAttack3(c, 0.25)
# a = ModelExtractionAttack4(
#     c, 0.25, './gnnip/models/attack_3_subgraph_shadow_model_cora_8159.pkl')
# a = ModelExtractionAttack5(
#     c, 0.25, './gnnip/models/attack_3_subgraph_shadow_model_cora_8159.pkl')
a.attack()

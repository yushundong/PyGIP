from gnnip.datasets.datasets import *
# from gnnip.datasets.datasets_backup import
from gnnip.core_algo import *


dataset = Cora()
# dataset = Citeseer()
# dataset = PubMed()
a = ModelExtractionAttack0(dataset, 0.25)
# a = ModelExtractionAttack1(
#     Cora(), 0.25, "./gnnip/data/attack2_generated_graph/cora/selected_index.txt",
#     "./gnnip/data/attack2_generated_graph/cora/query_labels_cora.txt",
#     "./gnnip/data/attack2_generated_graph/cora/graph_label0_564_541.txt")
# a = ModelExtractionAttack1(
#     Citeseer(), 0.25, "./gnnip/data/attack2_generated_graph/citeseer/selected_index.txt",
#     "./gnnip/data/attack2_generated_graph/citeseer/query_labels_citeseer.txt",
#     "./gnnip/data/attack2_generated_graph/citeseer/graph_label0_604_525.txt")
# a = ModelExtractionAttack1(
#     PubMed(), 0.25, "./gnnip/data/attack2_generated_graph/pubmed/selected_index.txt",
#     "./gnnip/data/attack2_generated_graph/pubmed/query_labels_pubmed.txt",
#     "./gnnip/data/attack2_generated_graph/pubmed/graph_label0_0.657_667_.txt")
# a = ModelExtractionAttack2(dataset, 0.25)
# a = ModelExtractionAttack3(dataset, 0.25)
# a = ModelExtractionAttack4(
#     Cora(), 0.25, './gnnip/models/attack_3_subgraph_shadow_model_cora_8159.pkl')
# a = ModelExtractionAttack4(
#     Citeseer(), 0.25, './gnnip/models/attack_3_subgraph_shadow_model_citeseer_6966.pkl')
# a = ModelExtractionAttack4(
#     PubMed(), 0.25, './gnnip/models/attack_3_subgraph_shadow_model_pubmed_8063.pkl')
# a = ModelExtractionAttack5(
#     Cora(), 0.25, './gnnip/models/attack_3_subgraph_shadow_model_cora_8159.pkl')
# a = ModelExtractionAttack5(
#     Citeseer(), 0.25, './gnnip/models/attack_3_subgraph_shadow_model_citeseer_6966.pkl')
# a = ModelExtractionAttack5(
#     PubMed(), 0.25, './gnnip/models/attack_3_subgraph_shadow_model_pubmed_8063.pkl')
a.attack()

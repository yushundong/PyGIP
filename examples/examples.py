from gnnip.datasets.datasets import *
from gnnip.core_algo import *
from gnnip.core_algo.Defense import Watermark_sage


# dataset = Cora()
# dataset = Citeseer()
# dataset = PubMed()
# a = ModelExtractionAttack0(dataset, 0.25)
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
# a.attack()


def run_attack(attack_type, dataset_name):
    if dataset_name == "Cora":
        dataset = Cora()
    elif dataset_name == "Citeseer":
        dataset = Citeseer()
    elif dataset_name == "PubMed":
        dataset = PubMed()
    else:
        print("Invalid dataset selected.")
        return

    switch = {
        0: lambda: ModelExtractionAttack0(dataset, 0.25),
        1: lambda: ModelExtractionAttack1(
            dataset, 0.25,
            "./gnnip/data/attack2_generated_graph/{}/selected_index.txt".format(
                dataset_name.lower()),
            "./gnnip/data/attack2_generated_graph/{}/query_labels_{}.txt".format(
                dataset_name.lower(), dataset_name.lower()),
            "./gnnip/data/attack2_generated_graph/{}/graph_label0_657_667.txt".format(dataset_name.lower()) if dataset_name == "PubMed" else
            "./gnnip/data/attack2_generated_graph/{}/graph_label0_604_525.txt".format(dataset_name.lower()) if dataset_name == "Citeseer" else
            "./gnnip/data/attack2_generated_graph/{}/graph_label0_564_541.txt".format(
                dataset_name.lower())
        ),
        2: lambda: ModelExtractionAttack2(dataset, 0.25),
        3: lambda: ModelExtractionAttack3(dataset, 0.25),
        4: lambda: ModelExtractionAttack4(
            dataset, 0.25,
            './gnnip/models/attack_3_subgraph_shadow_model_{}_8159.pkl'.format(dataset_name.lower()) if dataset_name == "Cora" else
            './gnnip/models/attack_3_subgraph_shadow_model_{}_6966.pkl'.format(dataset_name.lower()) if dataset_name == "Citeseer" else
            './gnnip/models/attack_3_subgraph_shadow_model_{}_8063.pkl'.format(
                dataset_name.lower())
        ),
        5: lambda: ModelExtractionAttack5(
            dataset, 0.25,
            './gnnip/models/attack_3_subgraph_shadow_model_{}_8159.pkl'.format(dataset_name.lower()) if dataset_name == "Cora" else
            './gnnip/models/attack_3_subgraph_shadow_model_{}_6966.pkl'.format(dataset_name.lower()) if dataset_name == "Citeseer" else
            './gnnip/models/attack_3_subgraph_shadow_model_{}_8063.pkl'.format(
                dataset_name.lower())
        ),
    }

    attack = switch.get(attack_type, lambda: "Invalid attack type selected.")
    a = attack()

    if hasattr(a, 'attack'):
        a.attack()
    else:
        print(a)


model_type = input("Enter model type (attack or defense): ").strip().lower()

if model_type == "attack":
    dataset_name = input("Enter dataset name (Cora, Citeseer, PubMed): ")
    attack_type = int(input("Enter attack type (0-5): "))
    run_attack(attack_type, dataset_name)
elif model_type == "defense":
    attack_name = int(input("Please choose the number:\n1.ModelExtractionAttack0\n2.ModelExtractionAttack1\n3.ModelExtractionAttack2\n4.ModelExtractionAttack3\n5.ModelExtractionAttack4\n6.ModelExtractionAttack5\n"))
    dataset_name = int(
        input("Please choose the number:\n1.Cora\n2.Citeseer\n3.PubMed\n"))
    if (dataset_name == 1):
        defense = Watermark_sage(Cora(), 0.25)
        defense.watermark_attack(Cora(), attack_name, dataset_name)
    elif (dataset_name == 2):
        defense = Watermark_sage(Citeseer(), 0.25)
        defense.watermark_attack(Citeseer(), attack_name, dataset_name)
    elif (dataset_name == 3):
        defense = Watermark_sage(PubMed(), 0.25)
        defense.watermark_attack(PubMed(), attack_name, dataset_name)

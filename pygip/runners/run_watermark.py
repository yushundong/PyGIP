from pygip.models.defense.gnn_watermark.gnn_watermark_defense import GNNWatermarkDefense
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run GNN Watermark Defense")
    args = parser.parse_args()
    defense = GNNWatermarkDefense(args)
    defense.run()

if __name__ == "__main__":
    main()


import pandas as pd
import json

def get_underperforming_model_from_csv(
    path_to_csv: str,
    min_gap_ratio: float = 0.5, # must have at least 50% lower test loss than the best model
    min_improvement_potential_ratio: float = 0.2) -> int: # must have at least 20% improvement potential
    """
    Get the underperforming model from a CSV file.

    Args:
        path_to_csv: Path to the CSV file.
        min_gap_ratio: Minimum gap ratio between the best model and the current model.
        min_improvement_potential_ratio: Minimum improvement potential ratio between the current model and the next size up model.

    Returns:
        The width of the underperforming model.
        `None` if no underperforming model is found.
    """
    df = pd.read_csv(path_to_csv)
    df = df.sort_values(by='width', ascending=True)

    best_model_index = df[df['best_test_loss'] == df['best_test_loss'].min()].index[0]

    best_model = df.iloc[best_model_index]
    # print(list(range(best_model_index-1, -1, -1)))

    improvement_potential_model_index = best_model_index
    for i in range(best_model_index-1, -1, -1):
        cur_model = df.iloc[i]
        improvement_potential_model = df.iloc[improvement_potential_model_index]

        gap_ratio = (cur_model['best_test_loss'] - best_model['best_test_loss']) / best_model['best_test_loss']
        improvement_potential_ratio = (cur_model['best_test_loss'] - improvement_potential_model['best_test_loss']) / cur_model['best_test_loss']

        if gap_ratio > min_gap_ratio and improvement_potential_ratio > min_improvement_potential_ratio:
            return cur_model['width']
        
        improvement_potential_model_index = i

    return None
    
    # for index, row in df.iterrows():
    #     if row['best_test_loss'] < best_model['best_test_loss'] * (1 - min_gap_ratio) and \
    #         row['best_test_loss'] < best_model['best_test_loss'] * (1 - min_improvement_potential_ratio):
    #         return index

    # return None

    
    return df

if __name__ == "__main__":
    from pathlib import Path
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from my_datasets import _DATASET_REGISTRY
    dataset_names = list(_DATASET_REGISTRY.keys())
    RUN_TIMESTAMP_DIR = Path("experiments/underperforming_models/20250802_231414")
    MIN_GAP_RATIO = 0.5
    MIN_IMPROVEMENT_POTENTIAL_RATIO = 0.2
    
    paths = [Path(f'{RUN_TIMESTAMP_DIR}/{dataset_name}_1_hidden_layer_model_width_vs_test_loss.csv') for dataset_name in dataset_names]

    results = {}
    for path, dataset_name in zip(paths, dataset_names):
        print(f"Dataset: {dataset_name}")
        width = get_underperforming_model_from_csv(
            path,
            min_gap_ratio=0.5,
            min_improvement_potential_ratio=0.2,
        )
        results[dataset_name] = width
        print(f"Underperforming model width: {width}")
        print()

    json_return = {
        "Hyper Params": {
            "min_gap_ratio": MIN_GAP_RATIO,
            "min_improvement_potential_ratio": MIN_IMPROVEMENT_POTENTIAL_RATIO,
        },
        "Results": results,
    }

    with open(f'{RUN_TIMESTAMP_DIR}/underperforming_model_widths.json', 'w') as f:
        # Hyper Params
        json.dump(json_return, f, indent=4)

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def cal_auc(clean_file, poison_file):
    # Load feature consistency values
    with open(clean_file, 'r') as f:
        clean_scores = [float(line.strip()) for line in f if line.strip()]

    with open(poison_file, 'r') as f:
        poison_scores = [float(line.strip()) for line in f if line.strip()]

    # Compute averages
    avg_clean = sum(clean_scores) / len(clean_scores) if clean_scores else 0.0
    avg_poison = sum(poison_scores) / len(poison_scores) if poison_scores else 0.0
    print(f"Average clean score: {avg_clean:.4f}")
    print(f"Average poison score: {avg_poison:.4f}")

    # Combine scores and labels
    scores = clean_scores + poison_scores
    labels = [0] * len(clean_scores) + [1] * len(poison_scores)

    # Calculate AUC
    auc = roc_auc_score(labels, scores)

    # Plot histogram distributions using matplotlib
    plt.figure(figsize=(8, 5))
    plt.hist(clean_scores, label='Clean', color='blue')
    plt.hist(poison_scores, label='Poison', color='red')
    plt.title(f'Feature Consistency Distributions (AUC: {auc:.4f})')
    plt.xlabel('Feature Consistency Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    plt.savefig("dist.png", dpi=300)
    print("Plot saved as dist.png")

    return auc


if __name__ == '__main__':
    clean_file = '/storageA/david_projects/DefTimeSeries/stoa_ebd/saved/backdoored_model/poison_rate_0.05/noTrans_ftsimi/freq_meg_500/resnet18/gridTrigger/9_clean.txt'
    poison_file = '/storageA/david_projects/DefTimeSeries/stoa_ebd/saved/backdoored_model/poison_rate_0.05/noTrans_ftsimi/freq_meg_500/resnet18/gridTrigger/9_poison.txt'
    auc = cal_auc(clean_file, poison_file)
    print(f"AUC: {auc:.4f}")

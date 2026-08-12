import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    attacks_list = ['pattern', 'adaptivecifar10', 'freq']
    for attack in attacks_list:
        clean_file = '{}_clean.npy'.format(attack)
        poison_file = '{}_poison.npy'.format(attack)

        # Load the numpy arrays
        try:
            fc_poison = np.load(poison_file)
            fc_clean = np.load(clean_file)[:len(fc_poison)]

        except FileNotFoundError:
            print(f"Files for attack '{attack}' not found. Skipping...")
            continue

        # Draw the histogram
        plt.figure(figsize=(6, 4))

        # Combine data to determine common bins
        combined = np.concatenate([fc_clean, fc_poison])
        bins = np.linspace(combined.min(), combined.max(), 51)  # 50 bins => 51 edges

        plt.hist(fc_clean, bins=bins, alpha=0.6, label='Clean')
        plt.hist(fc_poison, bins=bins, alpha=0.6, label='Poisoned')
        plt.xlabel('Feature Consistency Metric')
        plt.ylabel('Count')
        plt.title(f'Feature Consistency Histogram — {attack}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{attack}_histogram.png', dpi=150)
        plt.close()
        print(f"Saved histogram for {attack} to {attack}_histogram.png")

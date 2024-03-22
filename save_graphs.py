import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image


def save_plots(log_dir):
    plots_dir = os.path.join(log_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Assuming your CSV file has columns named similarly to the dictionary keys
    df = pd.read_csv(os.path.join(log_dir, 'progress.csv'))

    # Normalize loss
    max_loss = df['loss'].max()
    df['norm_loss'] = df['loss'] / max_loss

    # Creating two plots instead of three
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))  # Adjusting the figure size for 2 plots

    # Loss vs. Step plot
    axs[0].plot(df['step_number'], df['norm_loss'], label='MSE')
    axs[0].set_title('MSE vs. Step Number')
    axs[0].set_xlabel('Step Number')
    axs[0].set_ylabel('MSE')
    axs[0].grid(True)
    axs[0].legend()

    # Training Time vs. Step plot
    axs[1].plot(df['step_number'], df['total_training_time'], label='Training Time', color='g')
    axs[1].set_title('Training Time vs. Step Number')
    axs[1].set_xlabel('Step Number')
    axs[1].set_ylabel('Training Time (s)')
    axs[1].grid(True)
    axs[1].legend()

    # Saving the plots
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'training_plots.png'))

    # Show the plots if needed
    plt.close()


def main(args):
    log_dir = args.log_path
    save_plots(log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp", type=str, default="exp", help="Path for saving running related data."
    )
    parser.add_argument(
        "--doc",
        type=str,
        required=True,
        help="A string for documentation purpose. "
             "Will be the name of the log folder.",
    )
    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, "logs", args.doc)

    main(args)

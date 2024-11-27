import os
import torch
import torchaudio
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from traceback import print_exc


class TRPGroundTruth:
    def __init__(self, response_threshold=0.3, frame_rate=50):
        """
        Initialize TRP ground truth processor

        Args:
            response_threshold: Proportion of participants needed to consider an interval as containing a TRP
            frame_rate: Frame rate in Hz for discretizing responses
        """
        self.threshold = response_threshold
        self.frame_rate = frame_rate

    def load_responses(self, stimulus_path, response_paths):
        """
        Load and align all participant responses for a given stimulus

        Args:
            stimulus_path: Path to stimulus audio file
            response_paths: List of paths to response audio files
        """
        # Load stimulus to get duration
        sample_rate, stimulus = wavfile.read(stimulus_path)
        duration = len(stimulus) / sample_rate

        # Create time bins based on frame rate
        num_frames = int(duration * self.frame_rate)
        response_counts = np.zeros(num_frames)

        # Process each response file
        for response_path in response_paths:
            try:
                # Load response audio
                sample_rate, response = wavfile.read(response_path)

                # Find onset of responses (non-zero values above threshold)
                # This is simplified - in practice you'd want more sophisticated onset detection
                response_onsets = np.where(np.abs(response) > np.max(response) * 0.1)[0]

                # Convert sample indices to frame indices
                frame_indices = (response_onsets / sample_rate * self.frame_rate).astype(int)
                frame_indices = frame_indices[frame_indices < num_frames]

                # Increment counts for frames with responses
                response_counts[frame_indices] += 1

            except Exception as e:
                print(f"Error processing {response_path}: {e}")

        # Normalize by number of participants
        response_proportions = response_counts / len(response_paths)

        # Generate binary ground truth based on threshold
        ground_truth = response_proportions >= self.threshold

        return ground_truth, response_proportions

    def align_with_predictions(self, ground_truth, predictions, window_size=5):
        """
        Align ground truth with model predictions, using a small window to account
        for minor timing differences

        Args:
            ground_truth: Binary array of ground truth TRP locations
            predictions: Model predictions
            window_size: Number of frames to consider for alignment
        """
        aligned_gt = np.zeros_like(predictions, dtype=bool)

        for i in range(len(ground_truth)):
            if ground_truth[i]:
                start_idx = max(0, i - window_size)
                end_idx = min(len(predictions), i + window_size + 1)
                aligned_gt[start_idx:end_idx] = True

        return aligned_gt


def load_stimulus_responses(stimulus_id, base_path):
    """
    Load all responses for a given stimulus ID from the ICC dataset
    """
    # Example path pattern based on the dataset structure
    responses_path = Path(base_path) / "responses" / stimulus_id

    # Get all response wav files for this stimulus
    response_files = list(responses_path.glob("**/*.wav"))

    return response_files


def plot_responses(stimulus_name, ground_truth, response_proportions, save_path=None):
    """Plot ground truth and response proportions for visualization"""
    plt.figure(figsize=(15, 5))

    # Plot response proportions
    plt.plot(response_proportions, 'b-', alpha=0.5, label='Response Proportions')

    # Plot ground truth
    plt.plot(ground_truth.astype(int), 'r-', label='Ground Truth TRPs')

    plt.title(f'Responses and Ground Truth TRPs for {stimulus_name}')
    plt.xlabel('Frame')
    plt.ylabel('Proportion / TRP')
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    plt.close()


def main():
    # Set up absolute paths
    base_path = Path("C:/Users/Harry/PycharmProjects/LLA-VAP/datasets/ICC/Stimulus_List_and_Response_Audios")
    response_catalog_path = base_path / "responsesCatalogSONA.csv"

    # Initialize ground truth processor
    trp_processor = TRPGroundTruth(response_threshold=0.3, frame_rate=50)

    # Read response catalog
    response_catalog = pd.read_csv(response_catalog_path)

    # Process a couple of examples
    examples = [
        ("3-29-2023/study4/249_0316.left", "Stimulus list 0"),
        ("3-30-2023/study1/249_0317.left", "Stimulus list 0")
    ]

    results = {}
    for stimulus_path, stimulus_list in examples:
        print(f"\nProcessing stimulus: {stimulus_path}")

        # Get the stimulus audio path
        stimulus_audio = base_path / "stimulus_lists" / "stimulus lists" / "stimulus_0.wav"
        print(f"Looking for stimulus audio at: {stimulus_audio}")

        # Get all responses for this stimulus
        matching_responses = response_catalog[response_catalog['name'].str.contains(stimulus_path, na=False)]
        print(f"Found {len(matching_responses)} matching entries in response catalog")

        response_paths = []
        for _, row in matching_responses.iterrows():
            # Fix the response filename construction
            name_parts = Path(stimulus_path).stem.split('.')  # Split at dot to remove .left
            response_filename = f"{name_parts[0]}_left.wav"  # Construct correct filename
            response_path = base_path / "responses" / "responses" / stimulus_path / response_filename

            print(f"Checking response path: {response_path}")
            if response_path.exists():
                print(f"Found response file: {response_path}")
                response_paths.append(response_path)
            else:
                print(f"Warning: Response path not found: {response_path}")

        print(f"Found {len(response_paths)} valid response paths")

        if not response_paths:
            print("No valid response paths found, skipping stimulus")
            continue

        try:
            # Process ground truth
            ground_truth, response_proportions = trp_processor.load_responses(
                str(stimulus_audio),
                [str(p) for p in response_paths]
            )

            # Store results
            results[stimulus_path] = {
                'ground_truth': ground_truth,
                'response_proportions': response_proportions
            }

            # Create output directory if it doesn't exist
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)

            # Plot results
            plot_responses(
                stimulus_path,
                ground_truth,
                response_proportions,
                save_path=output_dir / f"trp_visualization_{Path(stimulus_path).stem}_left.png"
            )

            # Print some statistics
            print(f"Total frames: {len(ground_truth)}")
            print(f"Number of TRPs detected: {np.sum(ground_truth)}")
            print(f"Average response proportion: {np.mean(response_proportions):.3f}")

        except Exception as e:
            print(f"Error processing {stimulus_path}: {e}")
            print_exc()
            continue

    return results


if __name__ == "__main__":
    # Set up basic logging
    import logging

    logging.basicConfig(level=logging.INFO)

    try:
        results = main()
        print("\nProcessing completed successfully!")
    except Exception as e:
        print(f"Error in main execution: {e}")
        print_exc()
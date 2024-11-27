from ICC_Ground_Truth_Processing import TRPGroundTruth
import os
import torch
import torchaudio
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from traceback import print_exc
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from vap.model import VapGPT, VapConfig
from vap.audio import load_waveform


class TRPGroundTruth:
    def __init__(self, response_threshold=0.3, frame_rate=50):
        """
        Initialize TRP ground truth processor
        frame_rate=50 means we'll have 50 frames per second (one frame every 20ms)
        """
        self.threshold = response_threshold
        self.frame_rate = frame_rate  # 50Hz means ground truth is sampled every 20ms

    def load_responses(self, stimulus_path, response_paths):
        """
        Creates ground truth at frame_rate frequency (50Hz)
        This means regardless of chunk size used in VAP predictions,
        ground truth will be aligned to 20ms intervals
        """
        sample_rate, stimulus = wavfile.read(stimulus_path)
        duration = len(stimulus) / sample_rate

        # Create frame-aligned bins (at 50Hz, each frame is 20ms)
        num_frames = int(duration * self.frame_rate)
        response_counts = np.zeros(num_frames)

        for response_path in response_paths:
            try:
                sample_rate, response = wavfile.read(response_path)
                response_onsets = np.where(np.abs(response) > np.max(response) * 0.1)[0]

                # Convert sample indices to frame indices at 50Hz
                frame_indices = (response_onsets / sample_rate * self.frame_rate).astype(int)
                frame_indices = frame_indices[frame_indices < num_frames]
                response_counts[frame_indices] += 1

            except Exception as e:
                print(f"Error processing {response_path}: {e}")

        response_proportions = response_counts / len(response_paths)
        ground_truth = response_proportions >= self.threshold
        return ground_truth, response_proportions


def get_vap_predictions(model, audio_path, device="cuda", chunk_duration=0.5):
    """
    Get VAP predictions in chunks
    Note: Even though we process in 0.5s chunks, the model still outputs at 50Hz
    So each chunk will produce frame_rate * chunk_duration predictions
    """
    waveform, _ = load_waveform(audio_path, sample_rate=model.sample_rate)

    total_duration = waveform.shape[-1] / model.sample_rate
    print(f"Total audio duration: {total_duration:.2f} seconds")

    # Calculate chunk size (0.5s * 16000Hz = 8000 samples per chunk)
    chunk_samples = int(chunk_duration * model.sample_rate)
    print(f"Processing in chunks of {chunk_duration} seconds ({chunk_samples} samples)")

    if waveform.shape[0] == 1:
        waveform = torch.cat((waveform, torch.zeros_like(waveform)))

    predictions_list = []
    total_chunks = (waveform.shape[-1] + chunk_samples - 1) // chunk_samples

    for i in range(0, waveform.shape[-1], chunk_samples):
        try:
            if device == "cuda":
                torch.cuda.empty_cache()

            chunk_end = min(i + chunk_samples, waveform.shape[-1])
            chunk = waveform[..., i:chunk_end].contiguous()
            chunk = chunk.unsqueeze(0)

            if device == "cuda":
                chunk = chunk.cuda()

            # Each chunk produces predictions at 50Hz
            with torch.no_grad():
                out = model.probs(chunk)
                pred_chunk = out['vad'].squeeze().cpu().numpy() > 0.5
                predictions_list.append(pred_chunk)

            print(f"Processed chunk {len(predictions_list)}/{total_chunks}, "
                  f"Memory: {torch.cuda.memory_allocated() / 1024 ** 2:.1f}MB" if device == "cuda" else "")

        except Exception as e:
            print(f"Error processing chunk {len(predictions_list) + 1}: {e}")
            if device == "cuda":
                torch.cuda.empty_cache()
            continue

    if predictions_list:
        # Concatenate all chunks - predictions will be at 50Hz like ground truth
        predictions = np.concatenate(predictions_list)
        return predictions
    else:
        raise RuntimeError("Failed to process any chunks")


def load_vap_model(state_dict_path, device="cuda"):
    """Load and initialize the VAP model"""
    conf = VapConfig()
    model = VapGPT(conf)
    sd = torch.load(state_dict_path)
    model.load_state_dict(sd)
    model = model.to(device)
    return model.eval()


def plot_comparison(stimulus_name, ground_truth, predictions, response_proportions, save_path=None):
    """Plot ground truth vs predictions with time axis in seconds"""
    plt.figure(figsize=(15, 8))

    # Convert frame indices to time in seconds (assuming 50Hz)
    time_axis = np.arange(len(ground_truth)) / 50  # 50Hz -> seconds

    plt.subplot(3, 1, 1)
    plt.plot(time_axis, response_proportions, 'b-', alpha=0.5, label='Response Proportions')
    plt.title(f'Response Proportions for {stimulus_name}')
    plt.xlabel('Time (s)')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(time_axis, ground_truth.astype(int), 'g-', label='Ground Truth')
    plt.plot(time_axis, predictions.astype(int), 'r--', alpha=0.5, label='VAP Predictions')
    plt.title('Ground Truth vs Predictions')
    plt.xlabel('Time (s)')
    plt.legend()

    plt.subplot(3, 1, 3)
    correct = ground_truth == predictions
    plt.plot(time_axis, np.where(correct, 1, 0), 'g-', label='Correct')
    plt.plot(time_axis, np.where(~correct, 1, 0), 'r-', label='Incorrect')
    plt.title('Prediction Accuracy')
    plt.xlabel('Time (s)')
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda":
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f}MB")
        torch.cuda.empty_cache()

    base_path = Path("C:/Users/Harry/PycharmProjects/LLA-VAP/datasets/ICC/Stimulus_List_and_Response_Audios")
    state_dict_path = "C:/Users/Harry/PycharmProjects/LLA-VAP/VoiceActivityProjection/example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt"

    try:
        model = load_vap_model(state_dict_path, device)
        print("VAP model loaded successfully")
    except Exception as e:
        print(f"Error loading VAP model: {e}")
        return

    # Initialize ground truth processor with 50Hz frame rate
    trp_processor = TRPGroundTruth(response_threshold=0.3, frame_rate=50)

    examples = [
        ("3-29-2023/study4/249_0316.left", "Stimulus list 0"),
        ("3-30-2023/study1/249_0317.left", "Stimulus list 0")
    ]

    for stimulus_path, stimulus_list in examples:
        print(f"\nProcessing: {stimulus_path}")

        stimulus_audio = base_path / "stimulus_lists" / "stimulus lists" / "stimulus_0.wav"
        name_parts = Path(stimulus_path).stem.split('.')
        response_filename = f"{name_parts[0]}_left.wav"
        response_path = base_path / "responses" / "responses" / stimulus_path / response_filename

        if not all(p.exists() for p in [stimulus_audio, response_path]):
            print("Missing required files, skipping...")
            continue

        try:
            # Get ground truth (will be at 50Hz)
            ground_truth, response_proportions = trp_processor.load_responses(
                str(stimulus_audio),
                [str(response_path)]
            )
            print(f"Ground truth shape: {ground_truth.shape}")

            # Get predictions using 0.5s chunks (output will also be at 50Hz)
            predictions = get_vap_predictions(model, str(stimulus_audio), device, chunk_duration=0.5)
            print(f"Predictions shape: {predictions.shape}")

            # Ensure same length
            min_len = min(len(ground_truth), len(predictions))
            ground_truth = ground_truth[:min_len]
            predictions = predictions[:min_len]

            # Calculate metrics
            bacc = balanced_accuracy_score(ground_truth, predictions)
            conf_matrix = confusion_matrix(ground_truth, predictions)

            print(f"\nResults for {stimulus_path}:")
            print(f"Balanced Accuracy: {bacc:.4f}")
            print("\nConfusion Matrix:")
            print(conf_matrix)

            # Additional metrics
            true_pos = conf_matrix[1][1]
            false_pos = conf_matrix[0][1]
            false_neg = conf_matrix[1][0]
            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0

            print(f"\nAdditional Metrics:")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")

            # Create visualization
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            plot_comparison(
                stimulus_path,
                ground_truth,
                predictions,
                response_proportions,
                save_path=output_dir / f"vap_comparison_{name_parts[0]}_left.png"
            )

        except Exception as e:
            print(f"Error processing {stimulus_path}: {e}")
            print_exc()
            continue


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    try:
        results = main()
        print("\nProcessing completed successfully!")
    except Exception as e:
        print(f"Error in main execution: {e}")
        print_exc()
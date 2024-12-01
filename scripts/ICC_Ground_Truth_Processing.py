# ICC_Ground_Truth_Processing.py
from pydub import AudioSegment
import numpy as np
import warnings


class TRPGroundTruth:
    def __init__(self, response_threshold=0.3, frame_rate=50, window_size_sec=1.5):
        self.threshold = response_threshold
        self.frame_rate = frame_rate
        self.window_frames = int(window_size_sec * frame_rate)
        self.initial_trim_sec = 2.0

    def load_audio(self, file_path):
        """Load audio using pydub and convert to numpy array"""
        audio = AudioSegment.from_wav(file_path)
        samples = np.array(audio.get_array_of_samples())

        if audio.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)

        return samples, audio.frame_rate

    def load_responses(self, stimulus_path, response_paths):
        """Load stimulus and responses, identifying TRPs where enough participants responded"""
        # Load stimulus
        stimulus, sample_rate = self.load_audio(stimulus_path)
        trim_samples = int(self.initial_trim_sec * sample_rate)
        stimulus = stimulus[trim_samples:]
        stimulus_duration = len(stimulus) / sample_rate
        expected_frames = int(stimulus_duration * self.frame_rate)

        print(f"Stimulus duration after trim: {stimulus_duration:.2f}s")

        # Initialize array to count responses per frame
        response_counts = np.zeros(expected_frames)
        valid_responses = 0

        for response_path in response_paths:
            try:
                print(f"\nProcessing: {response_path}")
                samples, sr = self.load_audio(response_path)

                # Trim first 2 seconds
                trim_samples = int(self.initial_trim_sec * sr)
                samples = samples[trim_samples:]

                # If response is longer than stimulus, trim it to match
                expected_samples = int(stimulus_duration * sr)
                if len(samples) >= expected_samples:
                    samples = samples[:expected_samples]
                    print(f"Trimmed response to {len(samples) / sr:.2f}s")

                    # Find response moments using a threshold
                    threshold = np.max(np.abs(samples)) * 0.1  # Threshold to detect verbal responses
                    response_indices = np.where(np.abs(samples) > threshold)[0]

                    # Convert to frames (50Hz)
                    response_frames = (response_indices / sr * self.frame_rate).astype(int)
                    response_frames = response_frames[response_frames < expected_frames]

                    # Add this participant's responses
                    response_counts[response_frames] += 1
                    valid_responses += 1
                    print(f"Successfully processed")
                else:
                    print(f"Response too short: {len(samples) / sr:.2f}s vs required {stimulus_duration:.2f}s")
                    continue

            except Exception as e:
                print(f"Error: {str(e)}")
                continue

        if valid_responses == 0:
            raise ValueError("No valid responses found")

        # Calculate proportion of participants who responded in each frame
        response_proportions = response_counts / valid_responses

        # TRP occurs where proportion exceeds threshold
        ground_truth = response_proportions >= self.threshold  # threshold = 0.3

        print(f"\nProcessed {valid_responses} responses with {np.sum(ground_truth)} TRPs")
        return ground_truth, response_proportions
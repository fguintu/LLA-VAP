# VAP.py
import time
import torch
import numpy as np
from pydub import AudioSegment
from vap.model import VapGPT, VapConfig
from vap.audio import load_waveform
import pandas as pd
from pathlib import Path

def load_vap_model(state_dict_path, device="cuda"):
    """Load and initialize the VAP model"""
    conf = VapConfig()
    if not hasattr(conf, 'frame_hz'):
        conf.frame_hz = 50
    model = VapGPT(conf)
    # Use weights_only=True to avoid warning
    sd = torch.load(state_dict_path, weights_only=True)
    model.load_state_dict(sd)
    model = model.to(device)
    model.eval()
    return model


def load_and_preprocess_waveform(audio_path, sample_rate=16000):
    """Load and preprocess waveform ensuring stereo format required by VAP"""
    # Load with pydub first to ensure proper length
    audio = AudioSegment.from_wav(audio_path)

    # Trim first 2 seconds
    trim_ms = 2000  # 2 seconds in milliseconds
    audio = audio[trim_ms:]

    # Convert to waveform format needed by VAP
    waveform, sr = load_waveform(audio_path, sample_rate=sample_rate)
    waveform = waveform[:, int(trim_ms * sample_rate / 1000):]

    # For mono input, create stereo by duplicating channel
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]

    # Ensure batch dimension
    waveform = waveform.unsqueeze(0)

    # Normalize audio
    if waveform.dtype != torch.float32:
        waveform = waveform.float()
    max_val = torch.abs(waveform).max()
    if max_val > 1.0:
        waveform = waveform / max_val

    return waveform


def get_vap_predictions(model, audio_path, device="cuda", chunk_duration=1.0):
    waveform = load_and_preprocess_waveform(audio_path, sample_rate=model.sample_rate)

    samples_per_frame = model.sample_rate // 50
    chunk_frames = int(chunk_duration * 50)
    chunk_samples = chunk_frames * samples_per_frame

    predictions_list = []
    total_frames = 0
    total_inference_time = 0

    def process_chunk(chunk):
        if chunk.shape[-1] < chunk_samples:
            pad_size = chunk_samples - chunk.shape[-1]
            chunk = torch.nn.functional.pad(chunk, (0, pad_size))
        elif chunk.shape[-1] > chunk_samples:
            chunk = chunk[..., :chunk_samples]
        if device == "cuda":
            chunk = chunk.cuda()
        if chunk.dtype != torch.float32:
            chunk = chunk.float()
        return chunk

    with torch.no_grad():
        if device == "cuda":
            waveform = waveform.cuda()

        for i in range(0, waveform.shape[-1], chunk_samples):
            try:
                chunk_end = min(i + chunk_samples, waveform.shape[-1])
                chunk = waveform[..., i:chunk_end].contiguous()
                chunk = process_chunk(chunk)

                torch.cuda.synchronize()
                start_time = time.time()
                out = model(chunk)
                torch.cuda.synchronize()
                inference_time = time.time() - start_time

                num_frames = out['vad'].shape[1]
                total_frames += num_frames
                total_inference_time += inference_time

                pred_chunk = out['vad'].sigmoid().squeeze().cpu().numpy()[:, 0] > 0.5
                if chunk_end - i < chunk_samples:
                    actual_frames = int((chunk_end - i) / samples_per_frame)
                    pred_chunk = pred_chunk[:actual_frames]
                predictions_list.append(pred_chunk)

            except Exception as e:
                print(f"Error: {str(e)}")
                continue

    if predictions_list:
        predictions = np.concatenate(predictions_list)
        avg_time_per_frame = total_inference_time * 1000 / total_frames
        rtf = avg_time_per_frame / 20
        return predictions, rtf
    else:
        raise RuntimeError("Failed to process any chunks")


# And modify process_dataset in baseline_VAP_ICC_full.py to prefer trimmed files:

def process_dataset(base_path):
    """Process ICC dataset structure and return stimulus-response mappings"""
    base_path = Path(base_path)
    response_catalog = pd.read_csv(base_path / "responsesCatalogSONA.csv")
    print(f"Loaded {len(response_catalog)} entries from catalog")

    stimulus_mappings = []
    for _, row in response_catalog.iterrows():
        stimulus_num = 0 if 'Stimulus list 0' in row['StimulusList'] else 1
        is_reversed = 'reversed' in row['StimulusList'].lower()

        # Stimulus path
        stimulus_file = f"stimulus_{stimulus_num}{'_reversed' if is_reversed else ''}.wav"
        stimulus_path = base_path / "stimulus_lists" / "stimulus lists" / stimulus_file

        # Try trimmed response first
        response_base = row['name'].split('/')[-1]  # e.g. 249_0309.left
        response_name = response_base.replace('.left', '_left')  # e.g. 249_0309_left

        trimmed_path = base_path / "responses" / "responses" / row['name'] / f"{response_name}.trimmed.wav"
        regular_path = base_path / "responses" / "responses" / row['name'] / f"{response_name}.wav"

        # Prefer trimmed version if it exists
        response_path = trimmed_path if trimmed_path.exists() else regular_path

        if stimulus_path.exists() and response_path.exists():
            safe_name = f"stim{stimulus_num}_{'rev' if is_reversed else 'reg'}_" + row['name'].replace('/', '_')
            stimulus_mappings.append({
                'name': safe_name,
                'stimulus_path': stimulus_path,
                'response_path': response_path,
                'stimulus_num': stimulus_num,
                'is_reversed': is_reversed
            })
            print(f"Added valid pair: {safe_name}")

    return stimulus_mappings
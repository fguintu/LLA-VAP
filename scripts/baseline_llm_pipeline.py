import numpy as np
import whisperx
import pandas as pd
from pathlib import Path
import json
import warnings
from typing import List, Tuple, Dict
import time
import torch
import gc
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

from ICC_Ground_Truth_Processing import TRPGroundTruth
from baseline_VAP_pipeline import evaluate_vap_predictions, plot_comparison
from VAP import process_dataset

class RealTimeTRPPredictor:
    def __init__(self,
                 model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
                 chunk_size: float = 1.0,
                 max_seq_len: int = 4096):

        # Check GPU availability and print device info
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name()}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # Load LLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        # Load WhisperX with updated settings
        compute_type = "float16" if torch.cuda.is_available() else "float32"
        self.whisper_model = whisperx.load_model(
            "tiny",
            self.device,
            compute_type=compute_type,
            language="en",
            asr_options={
                "multilingual": True,  # Enable multilingual support
                "hotwords": [],  # No specific hotwords to boost
            }
        )

        self.chunk_size = chunk_size
        self.max_seq_len = max_seq_len
        self.context = []

        print(f"Models loaded successfully")

    def predict_trp(self, context: str, current: str) -> Tuple[bool, float]:
        start_time = time.time()

        messages = [
            {"role": "system",
             "content": "You are having a conversation with a user. you will get a speech segment of the user's turn. If you think the user has finished their turn or it is appropriate for you to take your turn, answer with 'yes'. If you think the user is not done with their turn, answer with 'no'. Answer with only 'yes' or 'no'."},
            {"role": "user",
             "content": f"Speech segment: {current}"}
        ]

        with torch.no_grad():
            response = self.pipe(
                messages,
                max_new_tokens=5,
                temperature=0.1,
                top_p=0.9,
                do_sample=False,
                return_full_text=False
            )[0]['generated_text'].lower()

        inference_time = time.time() - start_time
        return "yes" in response, inference_time

    def transcribe_chunk(self, audio_chunk: np.ndarray) -> Tuple[str, float]:
        start_time = time.time()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self.whisper_model.transcribe(audio_chunk, batch_size=16)
            text = " ".join([segment['text'] for segment in result["segments"]])

        inference_time = time.time() - start_time
        return text, inference_time

    def process_stream(self, audio_path: str) -> Tuple[np.ndarray, List[str], Dict[str, float]]:
        audio = whisperx.load_audio(audio_path)
        chunk_samples = int(self.chunk_size * 16000)  # 1 second = 16000 samples
        predictions = []
        transcripts = []
        audio_context = []  # Store audio chunks instead of text

        total_asr_time = 0
        total_llm_time = 0
        total_chunks = 0

        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i + chunk_samples]
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

            try:
                # Add current chunk to audio context
                audio_context.append(chunk)
                # Keep only last 10 chunks
                audio_context = audio_context[-10:] if len(audio_context) > 10 else audio_context

                # Concatenate audio context
                context_audio = np.concatenate(audio_context)

                print(f"\n{'=' * 80}")
                print(f"Chunk {total_chunks + 1}:")
                print(f"Audio context length: {len(audio_context)} chunks ({len(context_audio) / 16000:.2f} seconds)")

                # Transcribe the entire context audio
                current_text, asr_time = self.transcribe_chunk(context_audio)
                print(f"Context transcription: \"{current_text}\"")

                # For TRP prediction, we're most interested in if there's a TRP at the end
                # of the most recent chunk, given all the context
                has_trp, llm_time = self.predict_trp("", current_text)

                total_asr_time += asr_time
                total_llm_time += llm_time
                total_chunks += 1

                predictions.append(has_trp)
                transcripts.append(current_text)

                print(f"\nTRP Prediction: {has_trp}")
                print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f}MB")
                print(f"{'=' * 80}\n")

            except Exception as e:
                print(f"Error processing chunk {total_chunks}: {str(e)}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

        total_time = total_asr_time + total_llm_time
        total_audio_duration = len(audio) / 16000

        rtf_metrics = {
            'rtf_whisperx': total_asr_time / total_audio_duration,
            'rtf_llm': total_llm_time / total_audio_duration,
            'rtf_combined': total_time / total_audio_duration,
            'avg_asr_time': total_asr_time / total_chunks,
            'avg_llm_time': total_llm_time / total_chunks,
            'total_audio_duration': total_audio_duration,
            'total_processing_time': total_time
        }

        frame_predictions = np.repeat(predictions, int(50 * self.chunk_size))
        return frame_predictions, transcripts, rtf_metrics

    def __del__(self):
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()


def save_predictions(output_dir: Path, predictions: Dict):
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "llama_realtime_results.json", 'w') as f:
        json.dump(predictions, f, indent=4)

    metrics_df = pd.DataFrame([r['metrics'] for r in predictions.values()])
    rtf_df = pd.DataFrame([r['rtf_metrics'] for r in predictions.values()])

    metrics_df.to_csv(output_dir / "metrics.csv")
    rtf_df.to_csv(output_dir / "rtf_metrics.csv")

    return metrics_df, rtf_df


def main():
    base_path = Path("C:/Users/Harry/PycharmProjects/LLA-VAP/datasets/ICC/Stimulus_List_and_Response_Audios")
    output_dir = Path("output")

    print(f"\nLooking for files in: {base_path.absolute()}")
    output_dir.mkdir(exist_ok=True)

    try:
        print("Initializing predictor...")
        predictor = RealTimeTRPPredictor()
        print("Initializing TRP processor...")
        trp_processor = TRPGroundTruth(response_threshold=0.3, frame_rate=50, window_size_sec=1.5)

        # Use VAP's process_dataset function instead of direct glob
        stimulus_groups = process_dataset(base_path)

        all_results = {}
        for stim_key, group in stimulus_groups.items():
            print(f"\nProcessing {stim_key}")
            if not group['response_paths']:
                print(f"No response files found for stimulus {stim_key}")
                continue

            try:
                ground_truth, response_proportions, duration_stats = trp_processor.load_responses(
                    str(group['stimulus_path']),
                    [str(path) for path in group['response_paths']]
                )

                predictions, transcripts, rtf_metrics = predictor.process_stream(str(group['stimulus_path']))

                print("\nReal-time Factors:")
                print(f"WhisperX RTF: {rtf_metrics['rtf_whisperx']:.3f}")
                print(f"LLM RTF: {rtf_metrics['rtf_llm']:.3f}")
                print(f"Combined RTF: {rtf_metrics['rtf_combined']:.3f}")

                min_len = min(len(ground_truth), len(predictions))
                ground_truth = ground_truth[:min_len]
                predictions = predictions[:min_len]
                response_proportions = response_proportions[:min_len]

                window_size = max(25, int(duration_stats['avg_duration']))
                metrics = evaluate_vap_predictions(ground_truth, predictions, window_size=window_size)

                all_results[stim_key] = {
                    'metrics': metrics,
                    'rtf_metrics': rtf_metrics,
                    'ground_truth': ground_truth.tolist(),
                    'predictions': predictions.tolist(),
                    'response_proportions': response_proportions.tolist(),
                    'duration_stats': duration_stats,
                    'transcripts': transcripts
                }

                plot_comparison(
                    stim_key,
                    ground_truth,
                    predictions,
                    response_proportions,
                    save_path=output_dir / f"{stim_key}_analysis.png",
                    window_size=window_size
                )

            except Exception as e:
                print(f"Error processing {stim_key}: {str(e)}")
                continue

        if all_results:
            metrics_df, rtf_df = save_predictions(output_dir, all_results)

            print("\nResults Summary:")
            for metric in ['accuracy', 'balanced_accuracy', 'f1', 'precision', 'sensitivity']:
                mean = metrics_df[metric].mean()
                std = metrics_df[metric].std()
                print(f"{metric}: {mean:.3f} (±{std:.3f})")

            print("\nAverage RTF Metrics:")
            for metric in ['rtf_whisperx', 'rtf_llm', 'rtf_combined']:
                mean = rtf_df[metric].mean()
                std = rtf_df[metric].std()
                print(f"{metric}: {mean:.3f} (±{std:.3f})")
        else:
            print("\nNo results were generated. Please check your input files and paths.")

    except Exception as e:
        print(f"Critical error: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

    finally:
        print("\nProcessing complete.")


if __name__ == "__main__":
    main()
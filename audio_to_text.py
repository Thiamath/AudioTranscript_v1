#!/usr/bin/python

import argparse
import os
import time

import whisper
import torch
from pyannote.audio import Pipeline

print(torch.version.cuda)         # Should show your CUDA version
print(torch.cuda.is_available())  # Should be True

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file and label different speakers."
    )
    parser.add_argument("audio_file", type=str, help="Path to the input audio file.")
    parser.add_argument(
        "-o", "--output", type=str, default="transcript.txt",
        help="Path to the output text file."
    )
    parser.add_argument(
        "-t", "--hf_token", type=str, required=False,
        help="Your Hugging Face token (if not already logged in)."
    )
    parser.add_argument(
        "--whisper_model", type=str, default="base",
        help="Which Whisper model to use (tiny, base, small, medium, large)."
    )
    args = parser.parse_args()

    audio_file = args.audio_file
    output_file = args.output
    hf_token = args.hf_token
    whisper_model_name = args.whisper_model

    # Check if the audio file exists
    if not os.path.exists(audio_file):
        print(f"Error: File not found: {audio_file}")
        return

    # Check if the file extension is .wav or .mp3
    if not (audio_file.lower().endswith('.wav') or audio_file.lower().endswith('.mp3')):
        print("Error: Only .wav and .mp3 files are supported.")
        return

    # Register times for start and end of each step
    start_time = time.perf_counter_ns()
    print("Starting speaker diarization...")
    diarization_segments = diarize_audio(audio_file, hf_token)
    print("Diarization complete.")
    print(f"Time taken for diarization: {(time.perf_counter_ns() - start_time) / 1e9:.2f} seconds")

    start_time = time.perf_counter_ns()
    print("Starting transcription with Whisper...")
    transcription = transcribe_audio(audio_file, whisper_model_name)
    print("Transcription complete.")
    print(f"Time taken for transcription: {(time.perf_counter_ns() - start_time) / 1e9:.2f} seconds")

    start_time = time.perf_counter_ns()
    print("Labeling speakers (this might take a moment)...")
    labeled_transcript = label_speakers(diarization_segments, transcription)
    print("Speaker labeling complete.")
    print(f"Time taken for labeling: {(time.perf_counter_ns() - start_time) / 1e9:.2f} seconds")

    print("Saving transcript...")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(labeled_transcript)
    print(f"Transcript saved to {output_file}")


def diarize_audio(audio_file, hf_token=None):
    """
    Returns a list of tuples: [(start_time, end_time, speaker_label), ...]
    using a pyannote.audio Pipeline.
    """
    if hf_token:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization@2.1",
            use_auth_token=hf_token
        )
    else:
        # If you already did `huggingface-cli login`, you typically don’t
        # need to supply the token here.
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1")

    pipeline.to(torch.device('cuda'))
    diarization = pipeline(audio_file)

    # diarization is a pyannote.core.Annotation object
    # We can iterate over segments
    segments = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        start_time = segment.start
        end_time = segment.end
        segments.append((start_time, end_time, speaker))
    # Sort segments by start time
    segments.sort(key=lambda x: x[0])
    return segments


def transcribe_audio(audio_file, model_name="base"):
    """
    Returns a list of tuples: [(start_time, end_time, text), ...]
    from Whisper.
    Whisper can segment the transcript automatically.
    """
    model = whisper.load_model(model_name, device="cuda")
    print(model.device)  # Should show 'cuda:0'

    # The result includes 'segments': each segment has start, end, text, etc.
    result = model.transcribe(audio_file)
    if "segments" not in result:
        # If for some reason there's no segmentation, return entire text as single segment
        return [(0, None, result["text"])]

    transcribed_segments = []
    for seg in result["segments"]:
        if isinstance(seg, dict):
            start = seg.get("start", 0)
            end = seg.get("end", None)
            text = seg.get("text", "")
            transcribed_segments.append((start, end, text))
        else:
            # If seg is not a dict, skip or handle as needed
            continue
    return transcribed_segments


def label_speakers(diarization_segments, transcribed_segments):
    """
    Combine speaker labels (with time ranges) and the transcribed text (with time ranges).
    The naive approach is to match each transcribed segment to whichever speaker
    segment(s) overlap in time.

    Returns a single string with speaker placeholders and text.
    """
    # 1) We’ll assign each transcribed segment to the speaker who has the greatest time overlap
    #    with that segment. This is a simplified approach.

    # Create a final list of (speaker_label, start_time, end_time, text)
    labeled_data = []

    for t_start, t_end, text in transcribed_segments:
        # Find the speaker segment that best overlaps [t_start, t_end]
        best_speaker = "Unknown"
        best_overlap = 0.0

        for d_start, d_end, speaker_id in diarization_segments:
            overlap = _time_overlap(t_start, t_end, d_start, d_end)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker_id

        labeled_data.append((best_speaker, t_start, t_end, text))

    # 2) Convert speaker IDs like "SPEAKER_00" into placeholders "Speaker 1", "Speaker 2", etc.
    #    We'll keep a mapping from pyannote speaker IDs to a more readable label.
    speaker_map = {}
    speaker_counter = 1

    final_lines = []
    for speaker_id, start, end, text in labeled_data:
        if speaker_id not in speaker_map:
            speaker_map[speaker_id] = f"Speaker {speaker_counter}"
            speaker_counter += 1

        speaker_placeholder = speaker_map[speaker_id]
        final_lines.append(f"{speaker_placeholder}: {text.strip()}")

    # 3) Join everything with new lines
    return "\n".join(final_lines)


def _time_overlap(start1, end1, start2, end2):
    """
    Returns the overlap (in seconds) between time ranges [start1, end1] and [start2, end2].
    If there's no overlap, returns 0.
    """
    if end1 is None:
        # If Whisper didn't give an end time for a segment, treat it as 0 overlap
        return 0
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    return max(0, overlap_end - overlap_start)


if __name__ == "__main__":
    main()

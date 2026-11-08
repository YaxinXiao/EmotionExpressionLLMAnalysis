"""Create short audiovisual excerpts without changing source files (Python >=3.9)."""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inspect_media(ffmpeg, path):
    r = subprocess.run([ffmpeg, '-hide_banner', '-i', str(path)], capture_output=True, text=True)
    m = re.search(r'Duration: (\d+):(\d+):(\d+\.\d+)', r.stderr)
    if not m or 'Video:' not in r.stderr or 'Audio:' not in r.stderr:
        raise ValueError(f'Cannot verify duration, video and audio: {path}\n{r.stderr}')
    h, minute, second = map(float, m.groups())
    return h * 3600 + minute * 60 + second


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path(__file__).parent / 'data')
    p.add_argument('--seconds', type=float, default=20)
    args = p.parse_args()
    if args.seconds <= 0:
        p.error('--seconds must be positive')
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    sources = sorted(x for x in args.source.iterdir() if x.suffix.lower() in {'.mov', '.mp4', '.mkv'})
    if not sources:
        p.error('No video files found')
    args.output.mkdir(parents=True, exist_ok=False)
    media = args.output / 'segments'
    media.mkdir()
    manifest = {'version': 1, 'core_seconds': args.seconds, 'context_seconds': 2,
                'media_note': 'H.264 CRF18, width<=960; PCM audio; no loudness normalization. Source times from FFmpeg (0.01s precision).',
                'videos': [], 'segments': []}
    for vi, source in enumerate(sources, 1):
        duration = inspect_media(ffmpeg, source)
        vid = f'V{vi:02d}'
        manifest['videos'].append({'video_id': vid, 'source_filename': source.name,
                                  'duration_s': duration, 'source_sha256': sha256(source),
                                  'source_bytes': source.stat().st_size})
        start, index = 0.0, 0
        while start < duration - 0.01:
            index += 1
            end = min(start + args.seconds, duration)
            # Merge a very short tail into the preceding core.
            if 0 < duration - end < 2:
                end = duration
            clip_start, clip_end = max(0, start - 2), min(duration, end + 2)
            sid = f'{vid}_S{index:03d}'
            dest = media / f'{sid}.mov'
            command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-n',
                       '-ss', str(clip_start), '-i', str(source), '-t', str(clip_end - clip_start),
                       '-map', '0:v:0', '-map', '0:a:0', '-map_metadata', '-1',
                       '-vf', "scale=w='min(960,iw)':h=-2", '-c:v', 'libx264', '-preset', 'fast',
                       '-crf', '18', '-pix_fmt', 'yuv420p', '-threads', '4',
                       '-c:a', 'pcm_s16le', '-movflags', '+faststart', str(dest)]
            print(f'{sid}: core {start:.2f}-{end:.2f}s', flush=True)
            subprocess.run(command, check=True)
            actual = inspect_media(ffmpeg, dest)
            if abs(actual - (clip_end - clip_start)) > 0.25:
                raise ValueError(f'Extracted duration mismatch: {sid}')
            manifest['segments'].append({'segment_id': sid, 'video_id': vid,
                'path': f'segments/{sid}.mov', 'core_start_s': round(start, 3), 'core_end_s': round(end, 3),
                'clip_start_s': round(clip_start, 3), 'clip_end_s': round(clip_end, 3),
                'sha256': sha256(dest)})
            start = end
    (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Prepared {len(sources)} videos, {len(manifest["segments"])} segments.', flush=True)


if __name__ == '__main__':
    main()

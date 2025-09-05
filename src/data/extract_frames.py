import cv2, os
from typing import Optional
from src.config import load_config

def extract_frames(video_path: str, out_dir: str, sample_rate: int = 2) -> int:
    """Extrai frames do vídeo `video_path` e salva em `out_dir`.
    sample_rate: frames por segundo desejados (ex: 2 fps).
    Retorna número de frames salvos."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f'Vídeo não encontrado: {video_path}')
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / float(sample_rate))))
    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            fname = os.path.join(out_dir, f'frame_{saved:05d}.jpg')
            cv2.imwrite(fname, frame)
            saved += 1
        idx += 1
    cap.release()
    return saved

def frames_to_video(frames_dir: str, out_video: str, fps: int = 2):
    files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    if not files:
        raise RuntimeError('Nenhum frame encontrado em ' + frames_dir)
    first = cv2.imread(os.path.join(frames_dir, files[0]))
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video, fourcc, fps, (w, h))
    for f in files:
        img = cv2.imread(os.path.join(frames_dir, f))
        out.write(img)
    out.release()


if __name__ == '__main__':
    cfg = load_config()
    video = cfg.get('video_path')
    out_dir = cfg.get('frames_dir')
    sample_rate = cfg.get('sample_rate', 2)
    if not video or not out_dir:
        print('video_path or frames_dir not configured. Check configs/config.yaml or environment variables.')
    else:
        n = extract_frames(video, out_dir, sample_rate=sample_rate)
        print('Frames extraidos:', n)

    # small CLI: python -m src.data.extract_frames extract <video> <out_dir> <sample_rate>
    import sys
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == 'extract' and len(sys.argv) >= 4:
            v = sys.argv[2]
            out = sys.argv[3]
            sr = int(sys.argv[4]) if len(sys.argv) > 4 else 2
            print(f'Extracting {v} -> {out} @ {sr}fps')
            print(extract_frames(v, out, sample_rate=sr))
        elif cmd == 'video' and len(sys.argv) >= 4:
            frames = sys.argv[2]
            outv = sys.argv[3]
            fps = int(sys.argv[4]) if len(sys.argv) > 4 else 2
            print(f'Building video {outv} from {frames} @ {fps}fps')
            frames_to_video(frames, outv, fps=fps)
        else:
            print('Usage: python -m src.data.extract_frames extract <video> <out_dir> [sample_rate]')
            print('       python -m src.data.extract_frames video <frames_dir> <out_video> [fps]')

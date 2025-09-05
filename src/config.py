import os
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'config.yaml')

def load_config():
    cfg = {}
    # Load from YAML if available
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        cfg = {}
    # Override with environment variables if present
    cfg['video_path'] = os.environ.get('VIDEO_PATH', cfg.get('video_path'))
    cfg['frames_dir'] = os.environ.get('FRAMES_DIR', cfg.get('frames_dir'))
    cfg['model_dir'] = os.environ.get('MODEL_DIR', cfg.get('model_dir'))
    cfg['sample_rate'] = int(os.environ.get('SAMPLE_RATE', cfg.get('sample_rate', 2)))
    cfg['num_classes'] = int(os.environ.get('NUM_CLASSES', cfg.get('num_classes', 2)))
    return cfg

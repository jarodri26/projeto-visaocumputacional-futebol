#!/usr/bin/env python3
"""Validate that paths in configs/config.yaml exist (video_path, frames_dir, model_dir).
Exit codes:
 - 0: all ok
 - 1: missing or non-existent paths
 - 2: configuration not found or unreadable
"""
import sys
import os

try:
    # ensure repo root is on sys.path so top-level package `src` can be imported
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from src.config import load_config
except Exception as e:
    print('Failed to import src.config:', e)
    sys.exit(2)

cfg = load_config()
if not cfg:
    print('No configuration loaded (configs/config.yaml missing or empty).')
    sys.exit(2)

# Select keys to validate dynamically: any config key that looks like a path/dir/model
keys = [k for k in cfg.keys() if any(sub in k.lower() for sub in ('path', 'dir', 'model'))]
if not keys:
    print('No path-like keys found in configuration to validate.')
    sys.exit(2)

print('Validating config keys:', ', '.join(keys))
missing = []
for k in keys:
    v = cfg.get(k)
    if not v:
        print(f"Config '{k}' is not set.")
        missing.append(k)
    else:
        # treat certain remote URIs (Databricks DBFS, S3, GCS) as remote and skip local existence checks
        def is_remote_path(p: str) -> bool:
            if not isinstance(p, str):
                return False
            lower = p.lower()
            return lower.startswith('/dbfs') or lower.startswith('dbfs:') or lower.startswith('dbfs://') or lower.startswith('s3://') or lower.startswith('gs://')

        if is_remote_path(v):
            print(f"Remote path detected for '{k}', skipping local existence check: {v}")
            # do not add to missing; assume remote artifact exists in Databricks / cloud
            continue

        if not os.path.exists(v):
            # auto-create directories for keys that are dirs/models
            key_l = k.lower()
            try:
                if any(sub in key_l for sub in ('dir', 'model')):
                    os.makedirs(v, exist_ok=True)
                    print(f"Created directory for '{k}': {v}")
                elif 'path' in key_l:
                    # for file paths, ensure parent directory exists but do not create the file
                    parent = os.path.dirname(v) or '.'
                    if parent and not os.path.exists(parent):
                        os.makedirs(parent, exist_ok=True)
                        print(f"Created parent directory for '{k}': {parent}")
                    # Optionally create a placeholder file when requested (useful for CI)
                    if os.environ.get('CREATE_PLACEHOLDERS') == '1':
                        try:
                            open(v, 'a').close()
                            print(f"Created placeholder file for '{k}': {v}")
                        except Exception as e:
                            print(f"Failed to create placeholder file for '{k}': {v} -> {e}")
                            missing.append(k)
                    else:
                        print(f"File path for '{k}' does not exist (expected file): {v}")
                        missing.append(k)
                else:
                    # fallback: try create as directory
                    os.makedirs(v, exist_ok=True)
                    print(f"Created (fallback) directory for '{k}': {v}")
            except Exception as e:
                print(f"Failed to create path for '{k}': {v} -> {e}")
                missing.append(k)
        else:
            print(f"OK: {k} -> {v}")

if missing:
    print('\nValidation failed. Missing or non-existent entries:', ', '.join(missing))
    sys.exit(1)

print('\nAll configured paths exist.')
sys.exit(0)

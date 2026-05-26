#!/usr/bin/env bash
set -e

cd /home/site/wwwroot

# Extract Oryx build output (output.tar.zst) if present
# Oryx stores the build result as a zstd-compressed tar archive;
# our custom --startup-file bypasses the standard extraction step.
if [ -f output.tar.zst ]; then
    echo "(extracting oryx build output using Python zstandard)"
    python3 -c "
import tarfile, zstandard, os, sys
archive = '/home/site/wwwroot/output.tar.zst'
try:
    dctx = zstandard.ZstdDecompressor()
    with open(archive, 'rb') as f:
        with dctx.stream_reader(f) as reader:
            with tarfile.open(fileobj=reader, mode='r|') as tar:
                tar.extractall(path='/home/site/wwwroot')
    os.remove(archive)
    print('Extracted oryx build output successfully')
except Exception as e:
    print(f'Extraction failed (non-fatal): {e}')
    sys.exit(0)
"
fi

# Activate the Oryx virtual environment if it exists
if [ -f antenv/bin/activate ]; then
    . antenv/bin/activate
    echo "Activated antenv virtual environment"
fi

export PYTHONPATH="/home/site/wwwroot:${PYTHONPATH:-}"
export DISABLE_PANDERA_IMPORT_WARNING=True

mkdir -p data/models data/chroma_db mlruns

# Diagnostic: verify files and import
python3 << 'PYEOF' || true
import sys, os
print('CWD:', os.getcwd(), flush=True)
sys.path.insert(0, '/home/site/wwwroot')
print('PYTHONPATH:', os.environ.get('PYTHONPATH', '(not set)'), flush=True)
wwwroot = '/home/site/wwwroot'
for d in [wwwroot]:
    try:
        entries = os.listdir(d)
        print(f'ls {d}: {entries}', flush=True)
    except Exception as e:
        print(f'ls {d}: {e}', flush=True)
for root, dirs, files in os.walk(wwwroot):
    for f in files:
        if f == 'app.py' and 'serving' in root:
            print(f'FOUND src/serving/app.py at {os.path.join(root, f)}', flush=True)
    if '.git' in dirs:
        dirs.remove('.git')
    if 'antenv' in dirs:
        dirs.remove('.antenv')
print('Testing app import...', flush=True)
try:
    from src.serving.app import app
    print(f'App imported OK. Routes: {len(app.routes)}', flush=True)
except Exception as e:
    print(f'App import FAILED: {e}', flush=True)
PYEOF

echo "Starting uvicorn on port ${PORT:-8000}"
exec uvicorn src.serving.app:app --host 0.0.0.0 --port "${PORT:-8000}"

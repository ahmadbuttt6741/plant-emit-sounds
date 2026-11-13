"""
Packages PhytoEcho AI into a clean, optimized ZIP archive ready for Vercel deployment.
Includes all serverless entrypoints, templates, assets, pre-trained ML models, and configuration.
Excludes pycache, bytecodes, and temp files.
"""

import os
import zipfile

OUTPUT_ZIP = "phytoecho_vercel_deployment.zip"

EXCLUDE_DIRS = {
    '__pycache__',
    '.git',
    '.venv',
    'venv',
    '.gemini',
    '.vscode',
    'PlantSounds',
    'plant_sounds',
    'raw_audio'
}

EXCLUDE_EXTENSIONS = {
    '.pyc',
    '.pyo',
    '.pyd',
    '.log',
    '.zip',
    '.npz'
}

# Directories/files to explicitly include
INCLUDED_ROOT_ITEMS = [
    'api',
    'demo',
    'vercel.json',
    'requirements.txt',
    '.vercelignore',
    'README.md',
    'start_demo.bat'
]

def make_zip():
    print(f"Creating Vercel deployment package: {OUTPUT_ZIP} ...")
    total_files = 0
    total_uncompressed_bytes = 0

    with zipfile.ZipFile(OUTPUT_ZIP, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        for item in INCLUDED_ROOT_ITEMS:
            if not os.path.exists(item):
                continue
                
            if os.path.isfile(item):
                zipf.write(item, arcname=item)
                size = os.path.getsize(item)
                total_files += 1
                total_uncompressed_bytes += size
                print(f"  Added file: {item} ({size:,} bytes)")
            elif os.path.isdir(item):
                for root, dirs, files in os.walk(item):
                    # Filter out excluded directories in-place
                    dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
                    
                    for f in files:
                        ext = os.path.splitext(f)[1].lower()
                        if ext in EXCLUDE_EXTENSIONS:
                            continue
                        if f.startswith('.'):
                            if f != '.gitkeep':
                                continue
                                
                        file_path = os.path.join(root, f)
                        arcname = os.path.relpath(file_path, start=".")
                        zipf.write(file_path, arcname=arcname)
                        size = os.path.getsize(file_path)
                        total_files += 1
                        total_uncompressed_bytes += size

    compressed_size = os.path.getsize(OUTPUT_ZIP)
    print("\n" + "=" * 60)
    print(f"ZIP Archive successfully created: {OUTPUT_ZIP}")
    print(f"Total files included: {total_files}")
    print(f"Uncompressed size: {total_uncompressed_bytes / (1024*1024):.2f} MB")
    print(f"Compressed ZIP size: {compressed_size / (1024*1024):.2f} MB")
    print("=" * 60)

if __name__ == '__main__':
    make_zip()

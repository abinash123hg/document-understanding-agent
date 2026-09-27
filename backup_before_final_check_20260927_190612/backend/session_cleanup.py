from pathlib import Path
import atexit
import shutil

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"
UPLOADS=ROOT/"uploads"

def cleanup_session_files():
    targets=[
        DATA/"chunks.json",
        ROOT/"efficiency_report.json",
        ROOT/"capacity_report.py",
        ROOT/"pdf_capacity_test.py"
    ]

    for target in targets:
        try:
            if target.exists():
                if target.is_file():
                    target.unlink()
                else:
                    shutil.rmtree(target)
        except Exception:
            pass

    for folder in [UPLOADS]:
        try:
            if folder.exists():
                for item in folder.iterdir():
                    if item.is_file():
                        item.unlink(missing_ok=True)
                    elif item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
        except Exception:
            pass

atexit.register(cleanup_session_files)

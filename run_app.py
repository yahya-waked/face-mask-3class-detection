import sys
from pathlib import Path

import streamlit.web.cli as stcli


def main():

    base_dir = Path(__file__).resolve().parent

    app_path = base_dir / "src" / "app.py"

    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--global.developmentMode=false",
    ]

    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
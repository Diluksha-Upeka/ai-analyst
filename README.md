## Setup

This project uses a Python virtual environment in `.venv`.

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the app:

```powershell
./.venv/Scripts/python.exe -m streamlit run app.py
```

If you prefer activating the environment first:

```powershell
./.venv/Scripts/Activate.ps1
streamlit run app.py
```

## Common issue

If you see an error like:

> Could not find a version that satisfies the requirement requirements.txt

It means you ran `pip install requirements.txt`. Use `-r` instead:

```powershell
python -m pip install -r requirements.txt
```

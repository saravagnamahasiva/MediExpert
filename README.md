# MediExpert+

MediExpert+ is a Flask health assistant app with signup/login, profile photo storage, AI-assisted symptom diagnosis, disease details from JSON, diagnosis history, symptom check-ins, reminders, chatbot page, health timeline, community trends, QR health report sharing, and PDF report downloads.

Built by Saravagna Mahasiva.

## What Is Included

- `app.py` - main Flask app and SQLite database setup.
- `templates/` - all HTML pages.
- `static/` - images, sounds, data files, and assets.
- `static/data/disease_info.json` - disease descriptions, medicines, precautions, and advice.
- `model/disease_model.pkl` - trained disease prediction model.
- `datasets/` - training/test CSV files.
- `requirements.txt` - Python packages needed to run the app.
- `verify_app.py` - automated app verification script.

The zip does not include `venv`, cache files, or local user database records. The database is created automatically on first run.

## Setup On Your Laptop

1. Install Python 3.11 or newer. Python 3.13 also works.

2. Extract the zip file.

3. Open PowerShell in the extracted `MediExpert` folder.

4. Create a virtual environment:

```powershell
py -m venv venv
```

5. Activate it:

```powershell
venv\Scripts\activate
```

6. Install packages:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

7. Run the app:

```powershell
python app.py
```

8. Open this in your browser:

```text
http://127.0.0.1:5000
```

or:

```text
http://localhost:5000
```

## Login And Data

Create a new account from the signup page. The app stores users, profile photos, diagnosis records, reminders, symptom check-ins, feedback, and activity history in SQLite.

The SQLite database is created here:

```text
instance/mediexpert.db
```

To use a custom database path:

```powershell
$env:MEDIEXPERT_DB_PATH="C:\path\to\mediexpert.db"
python app.py
```

To set a safer secret key:

```powershell
$env:MEDIEXPERT_SECRET_KEY="change-this-to-a-long-random-secret"
python app.py
```

## QR Report On Phone

The QR report works while the Flask app is running.

For phone scanning, connect the phone and laptop to the same Wi-Fi network. The QR page tries to use your laptop network IP automatically.

If phone scanning still does not open:

1. Make sure Windows Firewall allows Python/Flask on port `5000`.
2. Keep the app running in PowerShell.
3. Open the QR page again and generate/download the QR.

## Verify The App

After installing dependencies, run:

```powershell
python verify_app.py
```

Expected output:

```text
MediExpert+ verification passed: pages, APIs, DB storage, diagnosis JSON, and reports are working.
```

## Deploy On Render

1. Push this folder to GitHub.
2. Go to Render and create a new Web Service.
3. Connect your GitHub repository.
4. Use these settings:

```text
Environment: Python
Build Command: pip install -r requirements.txt
Start Command: gunicorn app:app
```

5. Add an environment variable:

```text
MEDIEXPERT_SECRET_KEY = any-long-random-secret
```

6. Deploy and open the Render URL.

The included `Procfile`, `runtime.txt`, and `render.yaml` are provided for deployment platforms that detect them automatically.

Important: SQLite works for demos, but many free deployment platforms reset local files during restart/redeploy. For public real use, connect a hosted database such as PostgreSQL.

## Important Safety Note

The diagnosis feature is an AI-assisted screening aid based on a trained model and stored health guidance. It is not a replacement for professional medical diagnosis, emergency care, or prescription advice.

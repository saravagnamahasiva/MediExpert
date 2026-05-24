import os
import re
import tempfile
from base64 import b64decode
from io import BytesIO

db_fd, db_path = tempfile.mkstemp(prefix="mediexpert-verify-", suffix=".db")
os.close(db_fd)
os.environ["MEDIEXPERT_DB_PATH"] = db_path
os.environ["MEDIEXPERT_SECRET_KEY"] = "verify-secret"

from app import app  # noqa: E402


def ok(response, label):
    if response.status_code != 200:
        raise AssertionError(f"{label} returned {response.status_code}")


def main():
    client = app.test_client()
    email = "verify@example.com"
    password = "pass12345"

    ok(client.get("/"), "home")
    ok(
        client.post(
            "/signup",
            data={
                "name": "Verify User",
                "email": email,
                "age": "29",
                "gender": "Male",
                "password": password,
                "confirm_password": password,
            },
            follow_redirects=True,
        ),
        "signup",
    )
    ok(client.post("/login", data={"email": email, "password": password}, follow_redirects=True), "login")

    pages = [
        "/dashboard",
        "/symptom_diagnosis",
        "/timeline",
        "/health_timeline",
        "/log_symptom",
        "/chatbot",
        "/tips",
        "/reminders",
        "/tracker",
        "/trends",
        "/language",
        "/qr",
        "/export",
        "/download_user_report",
        "/edit_profile",
        "/restart_plan",
        "/view_history",
        "/feedback",
        "/privacy",
    ]
    for page in pages:
        ok(client.get(page, follow_redirects=True), page)

    diagnosis = client.post(
        "/predict",
        json={
            "symptoms": ["fever", "cough", "throat pain"],
            "age": 29,
            "gender": "Male",
            "severity": "Moderate",
        },
    )
    ok(diagnosis, "predict")
    assert diagnosis.json["medicines"], "predict returned no medicines"
    assert diagnosis.json["precautions"], "predict returned no precautions"

    ok(client.get("/symptoms_catalog"), "symptoms_catalog")
    diseases = client.get("/diseases_catalog")
    ok(diseases, "diseases_catalog")
    assert diseases.json, "diseases catalog empty"
    details = client.get("/disease_details?disease=Dengue")
    ok(details, "disease_details")
    assert details.json["medicines"], "disease details returned no medicines"

    reminder = client.post("/add_reminder", json={"text": "Take tablet", "time": "09:00", "days": ["Mon"]})
    ok(reminder, "add_reminder")
    reminder_id = reminder.json["id"]
    ok(client.get("/get_reminders"), "get_reminders")
    ok(client.put(f"/update_reminder/{reminder_id}", json={"text": "Drink water", "time": "10:00", "days": ["Tue"]}), "update_reminder")
    ok(client.delete(f"/delete_reminder/{reminder_id}"), "delete_reminder")

    ok(client.post("/chatbot_api", json={"message": "what is fever"}), "chatbot_api")
    ok(client.post("/log_symptom", data={"symptoms": "fever, cough"}, follow_redirects=True), "log_symptom post")
    ok(client.post("/feedback", data={"name": "Verify User", "email": email, "message": "Working"}, follow_redirects=True), "feedback post")
    tiny_png = b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
    )
    ok(
        client.post(
            "/edit_profile",
            data={
                "name": "Verify User",
                "age": "30",
                "gender": "Other",
                "profile_photo": (BytesIO(tiny_png), "profile.png"),
            },
            content_type="multipart/form-data",
            follow_redirects=True,
        ),
        "edit_profile post",
    )
    dashboard = client.get("/dashboard")
    ok(dashboard, "dashboard after profile photo")
    assert b"data:image/png;base64" in dashboard.data, "profile photo was not saved or shown"
    qr_page = client.get("/qr")
    ok(qr_page, "qr report")
    match = re.search(rb"/shared_report/([A-Za-z0-9_-]+)", qr_page.data)
    assert match, "QR page did not expose a shared report link"
    token = match.group(1).decode()
    ok(client.get(f"/shared_report/{token}"), "shared_report")
    ok(client.get(f"/download_shared_report/{token}"), "download_shared_report")
    qr_download = client.get(f"/download_report_qr/{token}")
    ok(qr_download, "download_report_qr")
    assert qr_download.mimetype == "image/png", "QR download did not return a PNG"
    timeline = client.get("/timeline")
    ok(timeline, "timeline after health activity")
    assert b"MediBot Chat" in timeline.data or b"Diagnosis" in timeline.data, "timeline missing activity entries"
    ok(client.post("/restart_plan", follow_redirects=True), "restart_plan post")

    print("MediExpert+ verification passed: pages, APIs, DB storage, diagnosis JSON, and reports are working.")


if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            os.remove(db_path)
        except OSError:
            pass

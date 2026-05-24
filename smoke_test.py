import os
import tempfile

db_fd, db_path = tempfile.mkstemp(prefix="mediexpert-smoke-", suffix=".db")
os.close(db_fd)
os.environ["MEDIEXPERT_DB_PATH"] = db_path
os.environ["MEDIEXPERT_SECRET_KEY"] = "smoke-test-secret"

from app import app  # noqa: E402


def assert_ok(response, label):
    assert response.status_code == 200, f"{label} returned {response.status_code}"


def main():
    client = app.test_client()

    assert_ok(client.get("/"), "home")
    assert_ok(
        client.post(
            "/signup",
            data={
                "name": "Smoke User",
                "email": "smoke@example.com",
                "password": "pass12345",
                "confirm_password": "pass12345",
            },
            follow_redirects=True,
        ),
        "signup",
    )
    assert_ok(
        client.post(
            "/login",
            data={"email": "smoke@example.com", "password": "pass12345"},
            follow_redirects=True,
        ),
        "login",
    )

    for path in [
        "/dashboard",
        "/symptom_diagnosis",
        "/timeline",
        "/log_symptom",
        "/chatbot",
        "/tips",
        "/reminders",
        "/tracker",
        "/trends",
        "/qr",
        "/export",
        "/edit_profile",
        "/restart_plan",
        "/view_history",
        "/feedback",
        "/privacy",
    ]:
        assert_ok(client.get(path), path)

    prediction = client.post(
        "/predict",
        json={"symptoms": ["fever", "cough", "headache"], "age": 30, "gender": "Male", "severity": "Moderate"},
    )
    assert_ok(prediction, "predict")
    assert prediction.json["disease"], "prediction missing disease"
    history = client.get("/diagnosis_history")
    assert_ok(history, "diagnosis_history")
    assert history.json and history.json[0]["diagnosis"], "diagnosis history missing saved prediction"

    reminder = client.post("/add_reminder", json={"text": "Take medicine", "time": "09:00", "days": ["Mon"]})
    assert_ok(reminder, "add_reminder")
    reminder_id = reminder.json["id"]
    assert_ok(client.get("/get_reminders"), "get_reminders")
    assert_ok(
        client.put(f"/update_reminder/{reminder_id}", json={"text": "Take tablet", "time": "10:00", "days": ["Tue"]}),
        "update_reminder",
    )
    assert_ok(client.delete(f"/delete_reminder/{reminder_id}"), "delete_reminder")

    assert_ok(client.post("/chatbot_api", json={"message": "what is fever"}), "chatbot_api")
    assert_ok(client.post("/log_symptom", data={"symptoms": "fever, cough"}, follow_redirects=True), "log_symptom post")
    assert_ok(
        client.post("/edit_profile", data={"name": "Smoke User", "age": "32", "gender": "Other"}, follow_redirects=True),
        "edit_profile post",
    )
    assert_ok(
        client.post(
            "/feedback",
            data={"name": "Smoke User", "email": "smoke@example.com", "message": "Looks good"},
            follow_redirects=True,
        ),
        "feedback post",
    )
    assert_ok(client.get("/download_user_report"), "download_user_report")
    history_page = client.get("/view_history")
    assert_ok(history_page, "view_history after activity")
    assert b"Login" in history_page.data or b"Diagnosis" in history_page.data, "activity history missing"
    assert_ok(client.post("/restart_plan", follow_redirects=True), "restart_plan post")

    print("MediExpert+ smoke test passed.")


if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            os.remove(db_path)
        except OSError:
            pass

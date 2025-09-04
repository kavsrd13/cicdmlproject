import base64
import requests
import os

def push_file_to_github(file_path, repo, branch="main"):
    token = os.getenv("GITHUB_TOKEN")  # from Streamlit Cloud Secrets
    if not token:
        print("❌ No GitHub token found!")
        return

    url = f"https://api.github.com/repos/{repo}/contents/{file_path}"
    
    # Read file
    with open(file_path, "rb") as f:
        content = f.read()
    encoded_content = base64.b64encode(content).decode("utf-8")

    # Get file info (to fetch SHA if exists)
    r = requests.get(url, headers={"Authorization": f"token {token}"})
    sha = r.json().get("sha") if r.status_code == 200 else None

    data = {
        "message": "Update inference log",
        "content": encoded_content,
        "branch": branch,
    }
    if sha:
        data["sha"] = sha  # required if updating existing file

    r = requests.put(url, headers={"Authorization": f"token {token}"}, json=data)

    if r.status_code in [200, 201]:
        print("✅ Log pushed to GitHub successfully.")
    else:
        print(f"❌ Failed to push log: {r.status_code}, {r.text}")

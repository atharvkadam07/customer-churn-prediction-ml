import os
import re
import base64
import shutil
import tempfile
import subprocess
import uuid
from pathlib import Path

import requests
from flask import (
    Flask,
    render_template,
    redirect,
    url_for,
    session,
    request,
    jsonify,
)
from authlib.integrations.flask_client import OAuth
from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# FLASK CONFIGURATION
# ============================================================

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")


# ============================================================
# UPLOAD LIMITS
# ============================================================

MAX_PROJECT_SIZE = 2 * 1024 * 1024 * 1024      # 2 GB whole project
MAX_SINGLE_FILE_SIZE = 100 * 1024 * 1024       # 100 MB per file
MAX_BATCH_SIZE = 25 * 1024 * 1024              # 25 MB per request

app.config["MAX_CONTENT_LENGTH"] = MAX_BATCH_SIZE


# ============================================================
# TEMPORARY UPLOAD DIRECTORY
# ============================================================

UPLOAD_ROOT = Path(tempfile.gettempdir()) / "gitauto_uploads"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# CHECK ENVIRONMENT
# ============================================================

if not app.secret_key:
    raise RuntimeError("FLASK_SECRET_KEY is missing from .env")

if not os.getenv("GITHUB_CLIENT_ID"):
    raise RuntimeError("GITHUB_CLIENT_ID is missing from .env")

if not os.getenv("GITHUB_CLIENT_SECRET"):
    raise RuntimeError("GITHUB_CLIENT_SECRET is missing from .env")


# ============================================================
# GITHUB OAUTH
# ============================================================

oauth = OAuth(app)

github = oauth.register(
    name="github",
    client_id=os.getenv("GITHUB_CLIENT_ID"),
    client_secret=os.getenv("GITHUB_CLIENT_SECRET"),
    access_token_url="https://github.com/login/oauth/access_token",
    authorize_url="https://github.com/login/oauth/authorize",
    api_base_url="https://api.github.com/",
    client_kwargs={"scope": "repo read:user user:email"},
)


# ============================================================
# IGNORE RULES
# ============================================================

IGNORED_FOLDERS = {
    "venv", ".venv", "env", ".env",
    ".git", ".idea", ".vscode",
    "__pycache__", "node_modules",
    "dist", "build",
    ".pytest_cache", ".mypy_cache",
    ".next", ".nuxt", "coverage",
}

IGNORED_FILES = {".env", ".DS_Store", "Thumbs.db", "desktop.ini"}

IGNORED_SUFFIXES = {".pem", ".key"}

IGNORED_KEY_NAMES = {"id_rsa", "id_ed25519", "id_ecdsa"}


def should_ignore(relative_path: Path) -> bool:
    filename = relative_path.name

    if any(part in IGNORED_FOLDERS for part in relative_path.parts):
        return True

    if filename in IGNORED_FILES:
        return True

    # .env.local, .env.production, etc.
    if filename.lower().startswith(".env"):
        return True

    if filename.lower() in IGNORED_KEY_NAMES:
        return True

    if relative_path.suffix.lower() in IGNORED_SUFFIXES:
        return True

    return False


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():
    github_user = session.get("github_user")
    return render_template("index.html", github_user=github_user)


# ============================================================
# GITHUB LOGIN
# ============================================================

@app.route("/login/github")
def github_login():
    redirect_uri = url_for("github_callback", _external=True)
    # GitHub supports prompt=select_account (lets the user switch accounts)
    return github.authorize_redirect(redirect_uri, prompt="select_account")


# ============================================================
# GITHUB CALLBACK
# ============================================================

@app.route("/github/callback")
def github_callback():
    token = github.authorize_access_token()

    # The token must be passed explicitly, otherwise this request is unauthenticated
    user_response = github.get("user", token=token)
    user_response.raise_for_status()
    user = user_response.json()

    session["github_user"] = {
        "login": user["login"],
        "name": user.get("name"),
        "avatar_url": user.get("avatar_url"),
    }
    session["github_token"] = token["access_token"]

    return redirect(url_for("home"))


# ============================================================
# LOGOUT / SWITCH ACCOUNT
# ============================================================

@app.route("/logout")
def logout():
    upload_id = session.get("gitauto_upload_id")
    if upload_id:
        shutil.rmtree(UPLOAD_ROOT / upload_id, ignore_errors=True)

    session.pop("github_user", None)
    session.pop("github_token", None)
    session.pop("gitauto_upload_id", None)

    return redirect(url_for("home"))


# ============================================================
# GENERATE README
# ============================================================

def generate_readme(repo_name, description, uploaded_files):
    if not description:
        description = "A project automatically published using GitAuto."

    if uploaded_files:
        file_list = "\n".join(f"- `{path}`" for path in uploaded_files[:300])
        if len(uploaded_files) > 300:
            file_list += f"\n\n...and {len(uploaded_files) - 300} additional files."
    else:
        file_list = "No project files were uploaded."

    return f"""# {repo_name}

## Description

{description}

## About the Project

This project was automatically published to GitHub using **GitAuto - GitHub Project Automation Tool**.

GitAuto allows users to connect their GitHub account, select a project folder, create a GitHub repository, publish the project, and generate a README automatically.

## Features

- GitHub OAuth authentication
- Automatic repository creation
- Large project upload support
- Batched file uploading
- Automatic README generation
- Automatic Git commit and push
- Automatic filtering of unnecessary folders
- Individual file size validation

## Project Files

{file_list}

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/{repo_name}.git
cd {repo_name}
```
"""


# ============================================================
# CREATE / GET UPLOAD DIRECTORY  (returns a Path)
# ============================================================

def get_upload_directory() -> Path:
    upload_id = session.get("gitauto_upload_id")

    if not upload_id:
        upload_id = str(uuid.uuid4())
        session["gitauto_upload_id"] = upload_id

    upload_dir = UPLOAD_ROOT / upload_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    return upload_dir


# ============================================================
# CALCULATE DIRECTORY SIZE
# ============================================================

def calculate_directory_size(directory: Path) -> int:
    total_size = 0

    for file_path in directory.rglob("*"):
        if file_path.is_file():
            try:
                total_size += file_path.stat().st_size
            except OSError:
                pass

    return total_size


# ============================================================
# UPLOAD BATCH
# ============================================================

@app.route("/upload-batch", methods=["POST"])
def upload_batch():

    # ---- check login ----
    if not session.get("github_user"):
        return jsonify({
            "success": False,
            "error": "GitHub account not connected."
        }), 401

    # ---- get files ----
    files = request.files.getlist("project_files")

    if not files:
        return jsonify({
            "success": False,
            "error": "No files received."
        }), 400

    # ---- upload directory ----
    try:
        upload_directory = get_upload_directory()
    except Exception as error:
        return jsonify({"success": False, "error": str(error)}), 500

    resolved_root = upload_directory.resolve()

    uploaded_count = 0
    skipped_count = 0
    large_file_count = 0
    large_files = []
    skipped_files = []

    # Calculate once, then keep a running total
    current_project_size = calculate_directory_size(upload_directory)

    for uploaded_file in files:

        original_name = uploaded_file.filename

        if not original_name:
            continue

        # ---- normalize path ----
        normalized_name = original_name.replace("\\", "/")
        relative_path = Path(normalized_name)

        # ---- security check ----
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or not relative_path.parts
        ):
            skipped_count += 1
            skipped_files.append(original_name)
            continue

        # ---- ignored files ----
        if should_ignore(relative_path):
            skipped_count += 1
            skipped_files.append(original_name)
            continue

        # ---- read file ----
        file_content = uploaded_file.read()
        file_size = len(file_content)

        # ---- single file limit ----
        if file_size > MAX_SINGLE_FILE_SIZE:
            large_file_count += 1
            large_files.append(original_name)
            continue

        # ---- total project limit ----
        if current_project_size + file_size > MAX_PROJECT_SIZE:
            return jsonify({
                "success": False,
                "error": "Project exceeds the 2 GB GitAuto limit.",
                "project_size": current_project_size,
            }), 413

        # ---- destination + path traversal protection ----
        destination = upload_directory / relative_path
        resolved_destination = destination.resolve()

        if (
            resolved_destination != resolved_root
            and resolved_root not in resolved_destination.parents
        ):
            skipped_count += 1
            skipped_files.append(original_name)
            continue

        # ---- save file ----
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(file_content)

        current_project_size += file_size
        uploaded_count += 1

    return jsonify({
        "success": True,
        "uploaded": uploaded_count,
        "skipped": skipped_count,
        "large_files": large_file_count,
        "project_size": current_project_size,
        "project_size_mb": round(current_project_size / (1024 * 1024), 2),
        "large_file_names": large_files[:50],
        "skipped_file_names": skipped_files[:50],
        "message": "Batch uploaded successfully.",
    })


# ============================================================
# GIT COMMAND HELPER
# ============================================================

def run_git_command(command, project_directory, auth_header=None):
    full_command = ["git"]

    if auth_header:
        full_command.extend([
            "-c",
            f"http.https://github.com/.extraheader={auth_header}",
        ])

    full_command.extend(command)

    result = subprocess.run(
        full_command,
        cwd=str(project_directory),
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )

    if result.returncode != 0:
        error_message = (
            result.stderr.strip()
            or result.stdout.strip()
            or "Git command failed."
        )
        # Never leak the token in error messages
        if auth_header:
            error_message = error_message.replace(auth_header, "***")
        raise RuntimeError(error_message)

    return result


# ============================================================
# PUBLISH PROJECT
# ============================================================

@app.route("/publish", methods=["POST"])
def publish_project():

    # ---- check login ----
    access_token = session.get("github_token")
    github_user = session.get("github_user")

    if not access_token or not github_user:
        return jsonify({
            "success": False,
            "error": "GitHub account not connected."
        }), 401

    # ---- form data ----
    repo_name = request.form.get("repo_name", "").strip()
    description = request.form.get("description", "").strip()

    # ---- validate repository name ----
    if (
        not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", repo_name)
        or repo_name in {".", ".."}
    ):
        return jsonify({
            "success": False,
            "error": "Invalid repository name."
        }), 400

    # ---- upload directory ----
    upload_id = session.get("gitauto_upload_id")

    if not upload_id:
        return jsonify({
            "success": False,
            "error": "No uploaded project found."
        }), 400

    upload_directory = UPLOAD_ROOT / upload_id

    if not upload_directory.exists():
        return jsonify({
            "success": False,
            "error": "Uploaded project no longer exists."
        }), 400

    # ---- check project ----
    project_files = [p for p in upload_directory.rglob("*") if p.is_file()]

    if not project_files:
        return jsonify({
            "success": False,
            "error": "No valid project files found."
        }), 400

    # ---- GitHub API ----
    username = github_user["login"]

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    # ---- create repository ----
    try:
        create_response = requests.post(
            "https://api.github.com/user/repos",
            headers=headers,
            json={
                "name": repo_name,
                "description": description,
                "private": False,
                "auto_init": False,
            },
            timeout=30,
        )
    except requests.RequestException as error:
        return jsonify({
            "success": False,
            "error": f"GitHub connection failed: {error}"
        }), 502

    if create_response.status_code != 201:
        try:
            error_message = create_response.json().get(
                "message", "Repository creation failed."
            )
        except Exception:
            error_message = "Repository creation failed."

        return jsonify({"success": False, "error": error_message}), 400

    repo_url = f"https://github.com/{username}/{repo_name}"

    # ---- create README (don't overwrite the user's own) ----
    relative_files = [
        p.relative_to(upload_directory).as_posix() for p in project_files
    ]

    readme_path = upload_directory / "README.md"

    if not readme_path.exists():
        readme_path.write_text(
            generate_readme(repo_name, description, relative_files),
            encoding="utf-8",
        )

    # ---- git authentication ----
    credential = base64.b64encode(
        f"x-access-token:{access_token}".encode("utf-8")
    ).decode("ascii")

    auth_header = f"AUTHORIZATION: basic {credential}"

    # ---- publish using git ----
    try:
        run_git_command(["init", "-b", "main"], upload_directory)
        run_git_command(["config", "user.name", "GitAuto"], upload_directory)
        run_git_command(
            ["config", "user.email", "gitauto@example.invalid"],
            upload_directory,
        )
        run_git_command(["add", "-A"], upload_directory)
        run_git_command(
            ["commit", "-m", "Publish project using GitAuto"],
            upload_directory,
        )

        remote_url = f"https://github.com/{username}/{repo_name}.git"

        run_git_command(
            ["remote", "add", "origin", remote_url],
            upload_directory,
        )
        run_git_command(
            ["push", "-u", "origin", "main"],
            upload_directory,
            auth_header=auth_header,
        )

    except FileNotFoundError:
        return jsonify({
            "success": False,
            "error": "Git was not found. Install Git and restart VS Code."
        }), 500

    except subprocess.TimeoutExpired:
        return jsonify({
            "success": False,
            "error": "GitHub publishing timed out.",
            "repository": repo_url,
        }), 504

    except RuntimeError as error:
        return jsonify({
            "success": False,
            "error": str(error),
            "repository": repo_url,
        }), 500

    finally:
        shutil.rmtree(upload_directory, ignore_errors=True)
        session.pop("gitauto_upload_id", None)

    return jsonify({
        "success": True,
        "repository": repo_url,
        "files": len(project_files),
        "message": "Project published successfully.",
    })


# ============================================================
# 413 ERROR
# ============================================================

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        "success": False,
        "error": (
            "A single upload batch is too large. "
            "GitAuto uses small batches; try again."
        ),
    }), 413


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    # Set debug=False in production: Flask's debugger allows remote code execution
    app.run(debug=True)
"""
tools/dev.py — Developer workflow tools for KOBRA.

Functions:
  create_file      — write a file with content
  open_vscode      — open a path in VS Code
  scaffold_project — generate boilerplate project structure
"""

import logging
import os
import subprocess
import textwrap

logger = logging.getLogger(__name__)


def create_file(path: str, content: str) -> str:
    """Create a file at path with the given content, creating parent dirs as needed."""
    logger.info("[TOOL] create_file: %s", path)
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"File created at {path}."
    except Exception as exc:
        return f"Failed to create file at {path}: {exc}"


def open_vscode(path: str = ".") -> str:
    """Open a file or folder in VS Code."""
    logger.info("[TOOL] open_vscode: %s", path)
    try:
        subprocess.Popen(
            ["code", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return f"Opened VS Code at {path}."
    except FileNotFoundError:
        return (
            "VS Code CLI not found. Enable it via VS Code: "
            "Command Palette → 'Shell Command: Install code command in PATH'."
        )
    except Exception as exc:
        return f"Failed to open VS Code: {exc}"


def scaffold_project(
    project_name: str,
    project_type: str,
    location: str = ".",
) -> str:
    """
    Generate a boilerplate folder structure for the requested project type.
    Supported types: python, fastapi, react, node.
    """
    logger.info("[TOOL] scaffold_project: %s (%s) at %s", project_name, project_type, location)

    project_root = os.path.join(os.path.abspath(location), project_name)

    try:
        if project_type == "react":
            result = subprocess.run(
                ["npx", "create-react-app", project_name],
                cwd=os.path.abspath(location),
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                return f"React project '{project_name}' scaffolded at {location}."
            return f"create-react-app failed: {result.stderr[:300]}"

        # Python / FastAPI / Node — manual scaffolding
        _create_dir(project_root)

        if project_type == "python":
            _write(project_root, "main.py", _PYTHON_MAIN)
            _write(project_root, "requirements.txt", "")
            _write(project_root, ".gitignore", _PYTHON_GITIGNORE)
            _write(project_root, "README.md", f"# {project_name}\n")

        elif project_type == "fastapi":
            for sub in ("models", "routes", "core"):
                _create_dir(os.path.join(project_root, sub))
                _write(os.path.join(project_root, sub), "__init__.py", "")
            _write(project_root, "main.py", _FASTAPI_MAIN)
            _write(project_root, "requirements.txt", _FASTAPI_REQUIREMENTS)
            _write(project_root, ".gitignore", _PYTHON_GITIGNORE)

        elif project_type == "node":
            _write(project_root, "index.js", _NODE_INDEX)
            _write(project_root, "package.json", _node_package_json(project_name))
            _write(project_root, ".gitignore", _NODE_GITIGNORE)

        else:
            return f"Unknown project type: '{project_type}'. Use: python, fastapi, react, node."

        return f"Project '{project_name}' ({project_type}) scaffolded at {project_root}."

    except Exception as exc:
        return f"Scaffolding failed: {exc}"


# ── File content templates ─────────────────────────────────────────────────────

_PYTHON_MAIN = textwrap.dedent("""\
    def main():
        print("Hello, world!")


    if __name__ == "__main__":
        main()
""")

_FASTAPI_MAIN = textwrap.dedent("""\
    from fastapi import FastAPI

    app = FastAPI()


    @app.get("/")
    def root():
        return {"message": "Hello, world!"}
""")

_FASTAPI_REQUIREMENTS = textwrap.dedent("""\
    fastapi>=0.110.0
    uvicorn[standard]>=0.29.0
""")

_NODE_INDEX = textwrap.dedent("""\
    const http = require('http');

    const server = http.createServer((req, res) => {
      res.writeHead(200, { 'Content-Type': 'text/plain' });
      res.end('Hello, world!\\n');
    });

    server.listen(3000, () => console.log('Server running on port 3000'));
""")

_PYTHON_GITIGNORE = textwrap.dedent("""\
    __pycache__/
    *.py[cod]
    *.egg-info/
    dist/
    build/
    .venv/
    venv/
    .env
    *.log
""")

_NODE_GITIGNORE = textwrap.dedent("""\
    node_modules/
    .env
    *.log
""")


def _node_package_json(name: str) -> str:
    return textwrap.dedent(f"""\
        {{
          "name": "{name}",
          "version": "1.0.0",
          "main": "index.js",
          "scripts": {{
            "start": "node index.js"
          }}
        }}
    """)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _create_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write(directory: str, filename: str, content: str) -> None:
    with open(os.path.join(directory, filename), "w", encoding="utf-8") as f:
        f.write(content)

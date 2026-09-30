import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


VERSION = "0.2.0"

home = Path.home()
simpleai_home = home / ".simpleai"
models_dir = simpleai_home / "models"

local_app = Path(
    os.environ.get("LOCALAPPDATA", home)
) / "SimpleAI"

settings_path = local_app / "settings.json"
cached_registry = local_app / "registry" / "models.json"

dev_registry = Path(
    r"C:\SimpleAIStudio\registry\models.json"
)


def header():
    print()
    print("SimpleAI")
    print("-" * 44)


def fail(message):
    print()
    print(f"Error: {message}")
    print()
    return 1


def load_settings():
    if not settings_path.exists():
        return {}

    try:
        with open(
            settings_path,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except Exception:
        return {}


def save_settings(settings):
    settings_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        settings_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            settings,
            file,
            indent=2,
        )


def fetch_registry(url):
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent":
                f"SimpleAI/{VERSION}"
        },
    )

    with urllib.request.urlopen(
        request,
        timeout=15,
    ) as response:
        raw = response.read()

    data = json.loads(
        raw.decode("utf-8-sig")
    )

    if not isinstance(
        data.get("models"),
        list,
    ):
        raise RuntimeError(
            "Remote registry is invalid."
        )

    cached_registry.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cached_registry.write_bytes(raw)

    return data


def load_registry():
    settings = load_settings()

    url = settings.get(
        "registry_url"
    )

    if url:
        try:
            return fetch_registry(url)

        except Exception as exc:
            print(
                f"Warning: online registry unavailable ({exc})"
            )

    if dev_registry.exists():
        path = dev_registry

    elif cached_registry.exists():
        path = cached_registry

    else:
        raise RuntimeError(
            "No SimpleAI registry is available."
        )

    with open(
        path,
        "r",
        encoding="utf-8-sig",
    ) as file:
        return json.load(file)


def parse_ref(reference):
    if ":" in reference:
        return tuple(
            reference.rsplit(
                ":",
                1,
            )
        )

    return reference, None


def find_model(registry, reference):
    model_id, version = parse_ref(
        reference
    )

    matches = []

    for model in registry["models"]:
        if model.get("id") != model_id:
            continue

        if (
            version
            and model.get("version") != version
        ):
            continue

        matches.append(model)

    if not matches:
        return None

    return matches[-1]


def format_params(value):
    if value is None:
        return "TBD"

    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"

    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"

    if value >= 1_000:
        return f"{value / 1_000:.2f}K"

    return f"{value:,}"


def format_bytes(value):
    if value is None:
        return "TBD"

    value = float(value)

    units = [
        "B",
        "KB",
        "MB",
        "GB",
    ]

    unit = 0

    while (
        value >= 1024
        and unit < len(units) - 1
    ):
        value /= 1024
        unit += 1

    if unit == 0:
        return f"{int(value)} B"

    return f"{value:.2f} {units[unit]}"


def sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def safe_extract(path, destination):
    destination = destination.resolve()

    with zipfile.ZipFile(
        path,
        "r",
    ) as archive:

        for item in archive.infolist():
            target = (
                destination
                / item.filename
            ).resolve()

            try:
                target.relative_to(
                    destination
                )

            except ValueError:
                raise RuntimeError(
                    "Unsafe path found inside model package."
                )

        archive.extractall(
            destination
        )


def download(url, path):
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": f"SimpleAI/{VERSION}"
        },
    )

    with urllib.request.urlopen(
        request,
        timeout=60,
    ) as response:

        total = response.headers.get(
            "Content-Length"
        )

        total = int(total) if total else None
        received = 0
        bar_width = 24

        if total:
            print(
                "\rdownload      "
                "[------------------------]   0%",
                end="",
                flush=True,
            )

        with open(
            path,
            "wb",
        ) as file:

            while True:
                chunk = response.read(
                    256 * 1024
                )

                if not chunk:
                    break

                file.write(chunk)
                received += len(chunk)

                if total:
                    percent = min(
                        received / total,
                        1.0,
                    )

                    filled = int(
                        percent * bar_width
                    )

                    bar = (
                        "#" * filled
                        + "-"
                        * (bar_width - filled)
                    )

                    print(
                        f"\rdownload      "
                        f"[{bar}] "
                        f"{percent * 100:3.0f}%",
                        end="",
                        flush=True,
                    )

        if total:
            print()
        else:
            print(
                f"download      "
                f"{received:,} bytes"
            )


def installed_models():
    found = []

    if not models_dir.exists():
        return found

    for model_folder in models_dir.iterdir():
        if not model_folder.is_dir():
            continue

        for version_folder in model_folder.iterdir():
            if not version_folder.is_dir():
                continue

            manifest_path = (
                version_folder
                / "simpleai-model.json"
            )

            manifest = {}

            if manifest_path.exists():
                try:
                    with open(
                        manifest_path,
                        "r",
                        encoding="utf-8",
                    ) as file:
                        manifest = json.load(file)

                except Exception:
                    pass

            found.append(
                {
                    "id": model_folder.name,
                    "version": version_folder.name,
                    "path": version_folder,
                    "manifest": manifest,
                }
            )

    return found


def find_installed(reference):
    model_id, version = parse_ref(
        reference
    )

    matches = []

    for item in installed_models():
        if item["id"] != model_id:
            continue

        if (
            version
            and item["version"] != version
        ):
            continue

        matches.append(item)

    if not matches:
        return None

    return matches[-1]


def ensure_dependencies(manifest):
    dependencies = manifest.get(
        "dependencies",
        [],
    )

    allowed = {
        "numpy": "numpy",
    }

    for dependency in dependencies:
        package = allowed.get(
            dependency.lower()
        )

        if package is None:
            raise RuntimeError(
                f"Unsupported dependency: {dependency}"
            )

        if importlib.util.find_spec(
            package
        ) is not None:
            continue

        print()
        print(
            f"Installing required dependency: {package}"
        )
        print()

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                package,
            ]
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Could not install {package}."
            )


def show_help():
    header()

    print()
    print("Usage:")
    print("  simpleai <command>")
    print()
    print("Commands:")
    print("  list")
    print("  info <model>")
    print("  pull <model>")
    print("  run <model>")
    print("  installed")
    print("  remove <model>")
    print("  registry")
    print("  registry <url>")
    print("  version")
    print()


def cmd_list():
    registry = load_registry()

    header()
    print()
    print("Available models")
    print()

    for model in registry["models"]:
        ref = (
            f"{model['id']}:"
            f"{model['version']}"
        )

        print(ref)
        print(
            f"  {model.get('name', model['id'])}"
        )
        print(
            f"  Parameters: "
            f"{format_params(model.get('parameters'))}"
        )
        print(
            f"  Status: "
            f"{model.get('status', 'unknown')}"
        )
        print()


def cmd_info(reference):
    registry = load_registry()
    model = find_model(
        registry,
        reference,
    )

    if model is None:
        return fail(
            f"Model '{reference}' was not found."
        )

    header()

    print()
    print(
        f"{model['name']} "
        f"{model['version']}"
    )
    print()

    fields = [
        (
            "Model",
            model["id"],
        ),
        (
            "Family",
            model.get(
                "family",
                "Unknown",
            ),
        ),
        (
            "Status",
            model.get(
                "status",
                "Unknown",
            ),
        ),
        (
            "Parameters",
            format_params(
                model.get(
                    "parameters"
                )
            ),
        ),
        (
            "Brain size",
            format_bytes(
                model.get(
                    "brain_size_bytes"
                )
            ),
        ),
        (
            "Architecture",
            model.get(
                "architecture",
                "Unknown",
            ),
        ),
        (
            "Runtime",
            model.get(
                "runtime",
                "Unknown",
            ),
        ),
        (
            "Training",
            model.get(
                "training",
                "Unknown",
            ),
        ),
        (
            "Benchmark",
            (
                f"{model.get('benchmark_accuracy')}%"
                if model.get(
                    "benchmark_accuracy"
                ) is not None
                else "TBD"
            ),
        ),
    ]

    for name, value in fields:
        print(
            f"{name:<18} {value}"
        )

    description = model.get(
        "description"
    )

    if description:
        print()
        print(description)

    skills = model.get(
        "skills",
        []
    )

    if skills:
        print()
        print("Skills:")

        for skill in skills:
            print(
                f"  - {skill}"
            )

    print()
    print("Install:")
    print()
    print(
        f"  simpleai pull "
        f"{model['id']}:{model['version']}"
    )
    print()


def cmd_pull(reference):
    registry = load_registry()
    model = find_model(
        registry,
        reference,
    )

    if model is None:
        return fail(
            f"Model '{reference}' was not found."
        )

    url = model.get(
        "download"
    )

    expected_hash = model.get(
        "sha256"
    )

    if not url:
        return fail(
            "This model has not been released yet."
        )

    if not expected_hash:
        return fail(
            "The registry is missing the model checksum."
        )

    target = (
        models_dir
        / model["id"]
        / model["version"]
    )

    if target.exists():
        print()
        print(
            f"{model['id']}:{model['version']} "
            f"is already installed."
        )
        print()
        return 0

    header()
    print()
    print(
        f"Pulling "
        f"{model['id']}:{model['version']}"
    )
    print()

    models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory(
        prefix="simpleai-"
    ) as temp:

        temp = Path(temp)

        package = temp / "model.zip"
        extracted = temp / "model"

        extracted.mkdir()

        print("registry      found")
        print(f"source        {url}")

        try:
            download(
                url,
                package,
            )

        except urllib.error.URLError as exc:
            return fail(
                f"Download failed: {exc}"
            )

        print(
            "verification  checking SHA-256"
        )

        actual_hash = sha256(
            package
        )

        if (
            actual_hash.lower()
            != expected_hash.lower()
        ):
            print()
            print("SHA-256 verification failed.")
            print()
            print(
                f"Expected: {expected_hash}"
            )
            print(
                f"Received: {actual_hash}"
            )
            print()

            return 1

        print(
            "verification  SHA-256 OK"
        )

        safe_extract(
            package,
            extracted,
        )

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copytree(
            extracted,
            target,
        )

    print("install       complete")
    print()
    print(
        f"Installed {model['name']} "
        f"{model['version']}"
    )
    print()
    print(
        f"Run: simpleai run "
        f"{model['id']}:{model['version']}"
    )
    print()

    return 0


def cmd_installed():
    header()
    print()
    print("Installed models")
    print()

    models = installed_models()

    if not models:
        print(
            "No SimpleAI models are installed."
        )
        print()
        return

    for model in models:
        name = model[
            "manifest"
        ].get(
            "name",
            model["id"],
        )

        print(
            f"{model['id']}:"
            f"{model['version']}"
        )

        print(
            f"  {name}"
        )

        print(
            f"  {model['path']}"
        )

        print()


def cmd_run(reference):
    model = find_installed(
        reference
    )

    if model is None:
        return fail(
            f"Model '{reference}' is not installed."
        )

    manifest = model[
        "manifest"
    ]

    ensure_dependencies(
        manifest
    )

    entrypoint = manifest.get(
        "entrypoint",
        "run.py",
    )

    path = (
        model["path"]
        / entrypoint
    ).resolve()

    try:
        path.relative_to(
            model["path"].resolve()
        )

    except ValueError:
        return fail(
            "Invalid model entrypoint."
        )

    if not path.exists():
        return fail(
            f"Entrypoint '{entrypoint}' was not found."
        )

    print()

    result = subprocess.run(
        [
            sys.executable,
            str(path),
        ],
        cwd=model["path"],
    )

    return result.returncode


def cmd_remove(reference):
    model = find_installed(
        reference
    )

    if model is None:
        return fail(
            f"Model '{reference}' is not installed."
        )

    shutil.rmtree(
        model["path"]
    )

    try:
        if not any(
            model["path"].parent.iterdir()
        ):
            model[
                "path"
            ].parent.rmdir()

    except OSError:
        pass

    print()
    print(
        f"Removed "
        f"{model['id']}:{model['version']}"
    )
    print()

    return 0


def cmd_registry(url=None):
    settings = load_settings()

    if url is None:
        header()
        print()

        current = settings.get(
            "registry_url"
        )

        if current:
            print(
                f"Registry: {current}"
            )
        else:
            print(
                "Registry: local development mode"
            )

        print()
        return 0

    if not (
        url.startswith("https://")
        or url.startswith("http://")
    ):
        return fail(
            "Registry URL must use http:// or https://"
        )

    print()
    print("Checking registry...")

    try:
        data = fetch_registry(
            url
        )

    except Exception as exc:
        return fail(
            f"Could not use registry: {exc}"
        )

    settings[
        "registry_url"
    ] = url

    save_settings(
        settings
    )

    print()
    print(
        f"Connected to "
        f"{data.get('name', 'SimpleAI Registry')}."
    )
    print()

    return 0


def main():
    args = sys.argv[1:]

    if not args:
        show_help()
        return 0

    command = args[0].lower()

    try:
        if command in {
            "help",
            "-h",
            "--help",
        }:
            show_help()
            return 0

        if command in {
            "version",
            "-v",
            "--version",
        }:
            print(
                f"SimpleAI CLI {VERSION}"
            )
            return 0

        if command == "list":
            return cmd_list() or 0

        if command == "installed":
            return cmd_installed() or 0

        if command == "info":
            if len(args) < 2:
                return fail(
                    "Usage: simpleai info <model>"
                )

            return cmd_info(
                args[1]
            ) or 0

        if command == "pull":
            if len(args) < 2:
                return fail(
                    "Usage: simpleai pull <model>"
                )

            return cmd_pull(
                args[1]
            ) or 0

        if command == "run":
            if len(args) < 2:
                return fail(
                    "Usage: simpleai run <model>"
                )

            return cmd_run(
                args[1]
            ) or 0

        if command == "remove":
            if len(args) < 2:
                return fail(
                    "Usage: simpleai remove <model>"
                )

            return cmd_remove(
                args[1]
            ) or 0

        if command == "registry":
            return cmd_registry(
                args[1]
                if len(args) >= 2
                else None
            )

        return fail(
            f"Unknown command '{command}'."
        )

    except RuntimeError as exc:
        return fail(
            str(exc)
        )

    except KeyboardInterrupt:
        print()
        return 130


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
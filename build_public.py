import shutil
from pathlib import Path


root = Path(__file__).parent

website = root / "website"
registry = root / "registry"
public = root / "docs"


if public.exists():
    shutil.rmtree(public)


public.mkdir()

(public / "registry").mkdir()


website_files = [
    "index.html",
    "styles.css",
    "app.js",
    "sai-favicon.png",
]


for name in website_files:
    source = website / name
    destination = public / name

    if not source.exists():
        raise FileNotFoundError(
            f"Missing website file: {source}"
        )

    shutil.copy2(
        source,
        destination,
    )


registry_file = (
    registry
    / "models.json"
)


if not registry_file.exists():
    raise FileNotFoundError(
        f"Missing registry file: {registry_file}"
    )


shutil.copy2(
    registry_file,
    public
    / "registry"
    / "models.json",
)


(
    public
    / ".nojekyll"
).write_text(
    "",
    encoding="utf-8",
)


print()
print("SimpleAI public site built.")
print()
print(public)
print()
print("Files:")
print()


for path in sorted(
    public.rglob("*")
):
    if path.is_file():
        print(
            "  "
            + str(
                path.relative_to(
                    public
                )
            )
        )


print()
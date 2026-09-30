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


for name in [
    "index.html",
    "styles.css",
    "app.js",
]:
    shutil.copy2(
        website / name,
        public / name,
    )


shutil.copy2(
    registry / "models.json",
    public / "registry" / "models.json",
)


(public / ".nojekyll").write_text(
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
                path.relative_to(public)
            )
        )

print()
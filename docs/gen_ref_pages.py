"""Generate the code reference pages.

Source: https://mkdocstrings.github.io/recipes/?h=reci
"""

from pathlib import Path

import mkdocs_gen_files

from synthetic_languages.utils.misc import repo_path_to_abs_path


# Get $(pwd) as abspath
root = repo_path_to_abs_path(".")
assert root.is_dir()
# TODO(Adriano) not sure what the desired behavior is here
# assert root.as_posix().endswith("synthetic_languages") or root.as_posix().endswith("synthetic_languages/")

package_name = "synthetic_languages"
src = repo_path_to_abs_path(package_name)

print("root", root)  # Debug
print("src", src)  # Debug

for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)
    print("full doc path", full_doc_path.as_posix())

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
    elif parts[-1] == "__main__":
        continue

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)
        # NOTE that we print INTO the file!
        print(f"::: {package_name}.{identifier}", file=fd)

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

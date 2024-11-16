# custom_sorting.py
import os
import re
from sphinx_gallery.sorting import ExampleTitleSortKey


# https://sphinx-gallery.github.io/stable/gen_modules/sphinx_gallery.sorting.html#module-sphinx_gallery.sorting
class SubSectionTitleOrder:
    """Sort example gallery by title of subsection.
    
    Assumes README.txt exists for all subsections and uses the subsection with
    dashes, '---', as the adornment.
    """
    def __init__(self, src_dir):
        self.src_dir = src_dir
        self.regex = re.compile(r"^([\w ]+)\n-", re.MULTILINE)

    def __repr__(self):
        return f"<{self.__class__.__name__}>"

    def __call__(self, directory):
        src_path = os.path.normpath(os.path.join(self.src_dir, directory))

        # Forces Release Highlights to the top
        if os.path.basename(src_path) == "release_highlights":
            return "0"

        readme = os.path.join(src_path, "README.txt")

        try:
            with open(readme, "r") as f:
                content = f.read()
        except FileNotFoundError:
            return directory

        title_match = self.regex.search(content)
        if title_match is not None:
            return title_match.group(1)
        return directory

class SKExampleTitleSortKey(ExampleTitleSortKey):
    """Sorts release highlights based on version number."""

    def __call__(self, filename):
        title = super().__call__(filename)
        prefix = "plot_release_highlights_"

        # Use title to sort if not a release highlight
        if not str(filename).startswith(prefix):
            return title

        major_minor = filename[len(prefix):].split("_")[:2]
        version_float = float(".".join(major_minor))

        # Negate to place the newest version highlights first
        return -version_float
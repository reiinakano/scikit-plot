"""A directive to generate the list of all the built-in components.

Read the content of the component folder and generate a list of all the components.
This list will display some informations about the component and a link to the
GitHub file.
"""

import re
from pathlib import Path
from typing import Any, Dict, List

from docutils import nodes
from sphinx.application import Sphinx
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective

logger = logging.getLogger(__name__)


class ComponentListDirective(SphinxDirective):
    """A directive to generate the list of all the built-in components.

    Read the content of the component folder and generate a list of all the components.
    This list will display some informations about the component and a link to the
    GitHub file.
    """

    name = "component-list"
    has_content = True
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = True

    def run(self) -> List[nodes.Node]:
        """Create the list."""
        # get the list of all th jinja templates
        # not that to remain compatible with sphinx they are labeled as html files
        root = Path(__file__).parents[2]
        component_dir = (
            root
            / "src"
            / "pydata_sphinx_theme"
            / "theme"
            / "pydata_sphinx_theme"
            / "components"
        )
        if not component_dir.is_dir():
            raise FileNotFoundError(
                f"Could not find component folder at {component_dir}."
            )
        components = sorted(component_dir.glob("*.html"))

        # create the list of all the components description using bs4
        # at the moment we use dummy information
        docs = []
        pattern = re.compile(r"(?<={#).*?(?=#})", flags=re.DOTALL)
        for c in components:
            comment = pattern.findall(c.read_text())
            docs.append(comment[0].strip() if comment else "No description available.")

        # get the urls from the github repo latest branch
        github_url = "https://github.com/pydata/pydata-sphinx-theme/blob/main"
        urls = [
            f"{github_url}/{component.relative_to(root)}" for component in components
        ]

        # build the list of all the components
        items = []
        for component, url, doc in zip(components, urls, docs):
            items.append(
                nodes.list_item(
                    "",
                    nodes.paragraph(
                        "",
                        "",
                        nodes.reference("", component.stem, internal=False, refuri=url),
                        nodes.Text(f": {doc}"),
                    ),
                )
            )

        return [nodes.bullet_list("", *items)]


def setup(app: Sphinx) -> Dict[str, Any]:
    """Add custom configuration to sphinx app.

    Args:
        app: the Sphinx application

    Returns:
        the 2 parallel parameters set to ``True``.
    """
    app.add_directive("component-list", ComponentListDirective)

    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
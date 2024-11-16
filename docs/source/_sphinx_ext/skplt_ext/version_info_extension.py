# skplt_ext/infer_next_release_versions.py

import json
import pprint
import jinja2
import importlib
from pathlib import Path
from urllib.request import urlopen

from scikitplot._externals._packaging.version import parse

from sphinx.application import Sphinx
from sphinx.util.logging import getLogger

logger = getLogger(__name__)

# Default templates and configuration
# Each entry is in the format (template name, target file name, kwargs for rendering)
DEFAULT_VERSION_URL_TEMPLATES = [
  (
    "devel/maintainer",  # rst_template_name
    "devel/maintainer",  # rst_target_name
    {},                  # kwargs
  ),
  # Add more default templates here
  # (
  #     "min_dependency_table",
  #     "min_dependency_table",
  #     {"dependent_packages": {} },
  # ),
  # (
  #     "min_dependency_substitutions",
  #     "min_dependency_substitutions",
  #     {"dependent_packages": {} },
  # ),
]

def get_inferred():
    """
    Default version information (short, full, tag).

    {
      'previous_tag' : {'bf': '0.3.7', 'final': '0.4.0', 'rc': 'unused'  },
      'version_full' : {'bf': '0.4.1', 'final': '0.5.0', 'rc': '0.5.0rc1'},
      'version_short': {'bf': '0.4',   'final': '0.5',   'rc': '0.5'     }
    }
    """
    return {
        "previous_tag" : {"rc": "unused", "final": "0.98.33", "bf": "0.97.22"},
        "version_full" : {"rc": "0.99.0rc1", "final": "0.99.0", "bf": "0.98.1"},
        "version_short": {"rc": "0.99", "final": "0.99", "bf": "0.98"},
    }

def infer_next_release_versions(app: Sphinx):
    """
    Infer the most likely next release versions when the builder is initialized.
    This method is triggered by the 'builder-inited' event in Sphinx.
    """
    logger.info("infer_next_release_versions triggered on builder-inited")
    
    # Default version information
    inferred = get_inferred()
    
    try:
        # Attempt to fetch and parse the JSON as before
        html_theme_options = app.config.html_theme_options
      
        # Fetch the version switcher JSON; see `html_theme_options` for more details
        versions_json = json.loads(
            urlopen(html_theme_options["switcher"]["json_url"], timeout=10).read()
        )
        # See `build_tools/circle/list_versions.py`, stable is always the second entry
        # stable_version = parse(versions_json[1]["version"])
        # stable_prev_version = parse(versions_json[2]["version"])

        # Try to find the stable and stable_prev version entries
        stable_prev_entry = next((entry for entry in versions_json if "stable_prev" in entry.get("name", "").lower()), None)
        stable_entry = next((entry for entry in versions_json if "stable" in entry.get("name", "").lower()), None)

        if stable_entry and stable_prev_entry:
            stable_version_prev = parse(stable_prev_entry["version"])
            stable_version      = parse(stable_entry["version"])
          
            next_major_minor    = f"{stable_version.major}.{stable_version.minor + 1}"
         
            # Update the previous version information
            inferred["previous_tag"]["final"]  = stable_version.base_version
            inferred["previous_tag"]["bf"]     = stable_version_prev.base_version
            # Update the full version information
            inferred["version_full"]["rc"]     = f"{next_major_minor}.0rc1"
            inferred["version_full"]["final"]  = f"{next_major_minor}.0"
            inferred["version_full"]["bf"]     = f"{stable_version.major}.{stable_version.minor}.{stable_version.micro + 1}"
            # Update the short version information
            inferred["version_short"]["rc"]    = next_major_minor
            inferred["version_short"]["final"] = next_major_minor
            inferred["version_short"]["bf"]    = f"{stable_version.major}.{stable_version.minor}"
        else:
            logger.warning("The versions JSON list is missing expected entries; skipping version inference")

    except Exception as e:
        logger.warning(f"Failed to infer next release versions: {type(e).__name__}: {e}") 

    finally:
        # Register inferred context value for access in templates
        app.add_config_value('inferred', inferred, 'env')
        # Correct use of pformat to pretty print the 'inferred' dictionary
        logger.info(f"'inferred' context injected: {pprint.pformat(inferred)}")

def preprocess_templates(app: Sphinx):
    """Convert .template files to .rst before Sphinx processes them."""
    logger.info("Starting version template preprocessing")
  
    # Fetch the templates (default or user-defined)
    release_versions_rst_templates = getattr(app.config, 'release_versions_rst_templates', DEFAULT_VERSION_URL_TEMPLATES)
          
    ######################################################################
    ## jinja2 Template Renderer
    ## https://jinja.palletsprojects.com/en/stable/templates/#import
    ## https://ttl255.com/jinja2-tutorial-part-6-include-and-import/
    ######################################################################    
    # Step 1: Create a Jinja environment instance
    jinja_env = jinja2.Environment(extensions=['jinja2.ext.i18n'])
    # Step 2: Register constants and functions globally
    jinja_env.globals['imp0rt'] = importlib.import_module  # Make available in all templates

    # Get the source directory of the Sphinx documentation project
    srcdir = Path(app.srcdir)    
    # Iterate over templates and render them with the "kwargs" data
    for rst_template_name, rst_target_name, kwargs in release_versions_rst_templates:
        template_path = srcdir / f"{rst_template_name}.rst.template"
        target_path = srcdir / f"{rst_target_name}.rst"

        # Inject 'inferred' into kwargs
        kwargs['inferred'] = app.config.inferred   
        try:
            # Load the .rst.template file and render it using Jinja2
            with template_path.open("r", encoding="utf-8") as f:
                # t = jinja2.Template(f.read())  # Use jinja2.Template to create the template
                t = jinja_env.from_string(f.read())  # Use "from_string" to create the template
          
            # Render the template with kwargs variables and write to the corresponding .rst file
            with target_path.open("w", encoding="utf-8") as f:
                f.write(t.render(**kwargs))
              
            logger.info(f"Successfully processed {template_path} to {target_path}")
  
        except Exception as e:
            logger.warning(f"Failed to process template {template_path}: {e}")

def setup(app: Sphinx):
    """Setup the Sphinx extension."""
    logger.info("Setting up Sphinx extension")

    try:      
        # Fetch the templates (default or user-defined)
        version = getattr(app.config, 'version', 'env')      
        if any(st in version for st in ('post', 'dev', 'alpha', 'a', 'beta', 'b')):
            bulid_type = 'dev'
        else:
            bulid_type = 'rel'          
        app.add_config_value('releaselevel', bulid_type, 'env')

    except Exception as e:
        logger.error(f"Failed to set up Sphinx extension: {e}")

    try:
        # https://www.sphinx-doc.org/en/master/extdev/appapi.html#sphinx.application.Sphinx.add_config_value
        # Ensure that 'release_versions_rst_templates' has a default configuration value if it's not set in conf.py
        app.add_config_value('release_versions_rst_templates', DEFAULT_VERSION_URL_TEMPLATES, 'env')
        logger.info("Added default/defined 'release_versions_rst_templates' configuration value.")
      
        # Connect the `infer_next_release_versions` function to the `builder-inited` event
        app.connect("builder-inited", infer_next_release_versions)
        logger.info("Connected 'infer_next_release_versions' to 'builder-inited' event")

        # Connect the `preprocess_templates` function to ensure templates are rendered after inference
        app.connect("builder-inited", preprocess_templates)
        logger.info("Connected 'preprocess_templates' function to 'builder-inited' event")

    except Exception as e:
        logger.error(f"Failed to set up Sphinx extension: {e}")

    # Return the extension metadata
    return {
        'version': '0.1',
        'parallel_read_safe': True,
    }
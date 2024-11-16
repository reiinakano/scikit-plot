# skplt_ext/url_extension.py

import os
import json
import jinja2
import importlib
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

from scikitplot._externals._packaging.version import parse

from sphinx.application import Sphinx
from sphinx.util.logging import getLogger

logger = getLogger(__name__)

# Default templates and configuration
# Each entry is in the format (template name, target file name, kwargs for rendering)
DEFAULT_URL_RST_TEMPLATES = [
  (
    "index",  # rst_template_name
    "index",  # rst_target_name
    {
      'development_link': "devel/index",
    },       # kwargs
  )
]

def preprocess_templates(app: Sphinx):
    """Convert .template files to .rst before Sphinx processes them."""
    logger.info("Starting url template preprocessing")
  
    # Fetch the templates (default or user-defined)
    # https://www.sphinx-doc.org/en/master/extdev/appapi.html#sphinx.application.Sphinx.add_config_value
    url_rst_templates = getattr(app.config, 'url_rst_templates', DEFAULT_URL_RST_TEMPLATES)

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
    for rst_template_name, rst_target_name, kwargs in url_rst_templates:
        template_path = srcdir / f"{rst_template_name}.rst.template"
        target_path = srcdir / f"{rst_target_name}.rst"
      
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

def get_repl_url():
    """Function to generate the REPL URL from the code in `initial_repl.py`."""
    base_url = "https://scikit-plots.github.io/demo/repl/"

    # Read the code from initial_repl.py
    code_file_path = os.path.join(os.path.dirname(__file__), '_pkg_wasm_webassembly.py')

    try:
        with open(code_file_path, 'r') as f:
            # Read the code from the file (strip extra whitespace)
            code = f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"{code_file_path} not found. Please ensure the file exists.")
    
    # URL-encode the code for inclusion in the URL query parameters
    params = {
        "toolbar": "1",
        "kernel": "python",
        "code": quote(code)  # URL-encode the code content
    }
    
    # Construct the full URL with the parameters
    repl_url = f"{base_url}?toolbar={params['toolbar']}&kernel={params['kernel']}&code={params['code']}"
    
    # Return the repl_url
    return repl_url

# Add the REPL URL to the HTML context, making it available in templates
def add_to_html_context(app: Sphinx, pagename, templatename, context, doctree):
    # Simply use the value stored in the config (don't register it again)
    context['repl_url'] = app.config.repl_url

def setup(app: Sphinx):
    """Setup the Sphinx extension."""
    logger.info("Setting up Sphinx application")
    try:
        # Ensure that 'url_rst_templates' has a default configuration value if it's not set in conf.py
        app.add_config_value('url_rst_templates', DEFAULT_URL_RST_TEMPLATES, 'env')
        logger.info("Added default/defined 'url_rst_templates' configuration value.")

        # Connect the `preprocess_templates` function to ensure templates are rendered
        app.connect("builder-inited", preprocess_templates)
        logger.info("Connected 'preprocess_templates' function to 'builder-inited' event")

      
        # Check if `repl_url` already exists; if not, set a default value
        app.add_config_value('repl_url', get_repl_url(), 'env')

        # Connect the 'add_to_html_context' function to the 'html-page-context' event
        app.connect("html-page-context", add_to_html_context)
        logger.info("Connected 'add_to_html_context' to 'html-page-context' event")
    except Exception as e:
        logger.error(f"Failed to set up Sphinx extension: {e}")

    # Return the extension metadata
    return {
        'version': '0.1',
        'parallel_read_safe': True,
    }
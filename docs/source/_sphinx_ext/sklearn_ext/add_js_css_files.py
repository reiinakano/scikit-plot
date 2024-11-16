import os
import re

import sphinx_gallery
from sphinx.application import Sphinx
from sphinx.util.logging import getLogger
logger = getLogger(__name__)


# https://www.sphinx-doc.org/en/master/_modules/sphinx/application.html#Sphinx
# https://www.sphinx-doc.org/en/master/extdev/appapi.html#sphinx-runtime-information
# https://www.sphinx-doc.org/en/master/_modules/sphinx/builders/html.html


def disable_plot_gallery_for_linkcheck(app: Sphinx):
    # Check if the current builder is 'linkcheck'
    if app.builder.name == "linkcheck":
        # Ensure sphinx_gallery_conf exists in app.config
        if hasattr(app.config, "sphinx_gallery_conf") and "plot_gallery" in app.config.sphinx_gallery_conf:
            # Set plot_gallery to False for linkcheck builds
            app.config.sphinx_gallery_conf["plot_gallery"] = False
            logger.info("Plot gallery disabled for linkcheck builder")


def add_js_css_files(app: Sphinx, pagename, templatename, context, doctree):
    """Load additional JS and CSS files only for certain pages.

    Note that `html_js_files` and `html_css_files` are included in all pages and
    should be used for the ones that are used by multiple pages. All page-specific
    JS and CSS files should be added here instead.
    """
    # Adding a custom variable to the context dictionary for every page
    api_search_js_path = "scripts/api-search.js"
    api_search_css_path = "styles/api-search.css"
    index_css_path = "styles/index.css"
    api_css_path = "styles/api.css"
  
    shell_js_path = "shell_scripts.js"
    shell_css_path = "shell_styles.css"
    shell_code_css_path = "shell_code_styles.css"

    # missing_file = []
    # for file in [api_search_js_path, api_search_css_path, index_css_path, api_css_path]:        
    #     if not os.path.exists(os.path.join(app.builder.outdir, "_static", file)):
    #         missing_file.append(file)
    #         logger.warning("File does not exist: %s", file)
            
    # if len(missing_file):
    #     logger.warning("File does not exist here: %s", app.builder.outdir)
    #     return
    
    if pagename == "api/index":  # "api/index"
        # External: jQuery and DataTables
        app.add_js_file("https://code.jquery.com/jquery-3.7.0.js")
        app.add_js_file("https://cdn.datatables.net/2.0.0/js/dataTables.min.js")
        app.add_css_file("https://cdn.datatables.net/2.0.0/css/dataTables.dataTables.min.css")
        
        # Internal: API search initialization and styling
        app.add_js_file(api_search_js_path)
        app.add_css_file(api_search_css_path)
      
        # logger.info("Adding JS and CSS files for page: %s", pagename)
    elif pagename == "index":
        # External: Include Prism.js for syntax highlighting
        app.add_js_file("https://cdnjs.cloudflare.com/ajax/libs/prism/1.23.0/prism.min.js")
        app.add_js_file("https://cdnjs.cloudflare.com/ajax/libs/prism/1.23.0/components/prism-python.min.js")
        # Internal: Include the modular JavaScript file
        app.add_js_file(shell_js_path)
      
        # External: Link to Prism.js CSS for syntax highlighting
        app.add_css_file("https://cdnjs.cloudflare.com/ajax/libs/prism/1.23.0/themes/prism.min.css")
        # Internal: Link to custom CSS for styling
        app.add_css_file(index_css_path)
        app.add_css_file(shell_css_path)
        app.add_css_file(shell_code_css_path)
      
        # logger.info("Adding JS and CSS files for page: %s", pagename)
    elif pagename.startswith("modules/generated/"):
        app.add_css_file(api_css_path)
        # logger.info("Adding JS and CSS files for page: %s", pagename)


def make_carousel_thumbs(app: Sphinx, exception):
    """Produce final resized carousel images."""
    if exception is not None:
        logger.error("Error during build: %s", exception)
        return

    # The following dictionary contains the information used to create the
    # thumbnails for the front page of the scikit-learn home page.
    # key: first image in set
    # values: (number of plot in set, height of thumbnail)
    carousel_thumbs = {"sphx_glr_plot_classifier_comparison_001.png": 600}

    image_dir = os.path.join(app.builder.outdir, "_images")
    for glr_plot, max_width in carousel_thumbs.items():
        image = os.path.join(image_dir, glr_plot)
        if os.path.exists(image):
            c_thumb = os.path.join(image_dir, glr_plot[:-4] + "_carousel.png")
            sphinx_gallery.gen_rst.scale_image(image, c_thumb, max_width, 190)
            logger.info("Generated carousel thumbnail: %s", c_thumb)


def setup(app: Sphinx):
    """
    Set up the Sphinx application with custom event handlers and configuration.
    """
    logger.info("Setting up Sphinx application")

    try:
        # Connect the function to the 'config-inited' event, which allows access to config values
        # do not run the examples when using linkcheck by using a small priority
        # (default priority is 500 and sphinx-gallery using builder-inited event too)
        app.connect("builder-inited", disable_plot_gallery_for_linkcheck, priority=50)
        logger.info("Connected 'disable_plot_gallery_for_linkcheck' to 'builder-inited' event")
    except Exception as e:
        logger.error(f"Failed to connect 'disable_plot_gallery_for_linkcheck': {e}")

    try:
        app.connect("html-page-context", add_js_css_files)
        logger.info("Connected 'add_js_css_files' to 'html-page-context' event")
    except Exception as e:
        logger.error(f"Failed to connect 'add_js_css_files': {e}")

    try:
        app.connect("build-finished", make_carousel_thumbs)
        logger.info("Connected 'make_carousel_thumbs' to 'build-finished' event")
    except Exception as e:
        logger.error(f"Failed to connect 'make_carousel_thumbs': {e}")
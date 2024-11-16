# skplt_ext/api_extension.py

from sphinx.application import Sphinx
from sphinx.util.logging import getLogger
logger = getLogger(__name__)

def autodoc_process_bases(app, name, obj, options, bases):
    """
    Hide pybind11 base object from inheritance tree.

    Note, *bases* must be modified in place.
    """
    for cls in bases[:]:
        if not isinstance(cls, type):
            continue
        if cls.__module__ == 'pybind11_builtins' and cls.__name__ == 'pybind11_object':
            bases.remove(cls)

# def add_html_cache_busting(app, pagename, templatename, context, doctree):
#     """
#     Add cache busting query on CSS and JavaScript assets.

#     This adds the Matplotlib version as a query to the link reference in the
#     HTML, if the path is not absolute (i.e., it comes from the `_static`
#     directory) and doesn't already have a query.

#     .. note:: Sphinx 7.1 provides asset checksums; so this hook only runs on
#               Sphinx 7.0 and earlier.
#     """
#     from sphinx.builders.html import Stylesheet, JavaScript

#     css_tag = context['css_tag']
#     js_tag = context['js_tag']

#     def css_tag_with_cache_busting(css):
#         if isinstance(css, Stylesheet) and css.filename is not None:
#             url = urlsplit(css.filename)
#             if not url.netloc and not url.query:
#                 url = url._replace(query=SHA)
#                 css = Stylesheet(urlunsplit(url), priority=css.priority,
#                                  **css.attributes)
#         return css_tag(css)

#     def js_tag_with_cache_busting(js):
#         if isinstance(js, JavaScript) and js.filename is not None:
#             url = urlsplit(js.filename)
#             if not url.netloc and not url.query:
#                 url = url._replace(query=SHA)
#                 js = JavaScript(urlunsplit(url), priority=js.priority,
#                                 **js.attributes)
#         return js_tag(js)

#     context['css_tag'] = css_tag_with_cache_busting
#     context['js_tag'] = js_tag_with_cache_busting

# def _parse_skip_subdirs_file():
#     """
#     Read .mpl_skip_subdirs.yaml for subdirectories to not
#     build if we do `make html-skip-subdirs`.  Subdirectories
#     are relative to the toplevel directory.  Note that you
#     cannot skip 'users' as it contains the table of contents,
#     but you can skip subdirectories of 'users'.  Doing this
#     can make partial builds very fast.
#     """
#     default_skip_subdirs = [
#         'users/prev_whats_new/*', 'users/explain/*', 'api/*', 'gallery/*',
#         'tutorials/*', 'plot_types/*', 'devel/*']
#     try:
#         with open(".mpl_skip_subdirs.yaml", 'r') as fin:
#             print('Reading subdirectories to skip from',
#                   '.mpl_skip_subdirs.yaml')
#             out = yaml.full_load(fin)
#         return out['skip_subdirs']
#     except FileNotFoundError:
#         # make a default:
#         with open(".mpl_skip_subdirs.yaml", 'w') as fout:
#             yamldict = {'skip_subdirs': default_skip_subdirs,
#                         'comment': 'For use with make html-skip-subdirs'}
#             yaml.dump(yamldict, fout)
#         print('Skipping subdirectories, but .mpl_skip_subdirs.yaml',
#               'not found so creating a default one. Edit this file',
#               'to customize which directories are included in build.')

#         return default_skip_subdirs

# skip_subdirs = []
# # triggered via make html-skip-subdirs
# if 'skip_sub_dirs=1' in sys.argv:
#     skip_subdirs = _parse_skip_subdirs_file()
  
def setup(app: Sphinx):
    """Setup the Sphinx extension."""
    logger.info("Setting up Sphinx application")
    try:
        app.connect('autodoc-process-bases', autodoc_process_bases)
        logger.info("Connected 'autodoc_process_bases' function to 'autodoc-process-bases' event")

        # Connect the `add_html_cache_busting` function to ensure templates are rendered
        if sphinx.version_info[:2] < (7, 1):
            app.connect('html-page-context', add_html_cache_busting, priority=1000)
            logger.info("Connected 'add_html_cache_busting' function to 'html-page-context' event")
          
        # app.add_config_value('skip_sub_dirs', 0, '')
    except Exception as e:
        logger.error(f"Failed to set up Sphinx extension: {e}")

    # Return the extension metadata
    return {
        'version': '0.1',
        'parallel_read_safe': True,
    }
import inspect
from sphinx.ext.autodoc import Documenter, ModuleLevelDocumenter


class ShortSummaryDocumenter(ModuleLevelDocumenter):
    """An autodocumenter that only renders the short summary of the object."""

    # Defines the usage: .. autoshortsummary:: {{ object }}
    objtype = "shortsummary"

    # Disable content indentation
    content_indent = ""

    # Avoid being selected as the default documenter for some objects, because we are
    # returning `can_document_member` as True for all objects
    priority = -99

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        """Allow documenting any object.
        
        Dynamically checks if an object can be documented based on its type.
        This checks if the member is a function, class, data, module, etc.
        """

        # Check if it's a function
        if inspect.isfunction(member):
            cls.objtype = "function"  # Treat as a function
        # Check if it's a class
        elif inspect.isclass(member):
            cls.objtype = "class"  # Treat as a class
        # Check if it's a module-level object (module)
        elif inspect.ismodule(member):
            cls.objtype = "module"  # Treat as a module
        # Check if it's a method (also caught by isfunction, but explicitly here for clarity)
        elif inspect.ismethod(member):
            cls.objtype = "method"  # Treat as a method
        # Check if it's a property (optional)
        elif isinstance(member, property):
            cls.objtype = "property"  # Treat as a property
        # Check if it's an instance (custom object like tweedie_gen)
        elif isinstance(member, object):
            cls.objtype = "data"  # Treat as data if it's an instance of a class
        # Check if the member is a constant or simple module-level variable (data)
        elif isinstance(member, (int, float, str, bool, complex, dict, list)):
            cls.objtype = "data"  # Treat as data if it's a simple module-level constant or value
        # Default fallback for unrecognized types
        else:
            cls.objtype = "shortsummary"  # Default to short summary for unknown types

        return True  # Allow all objects to be documented

    def get_object_members(self, want_all):
        """Document no members."""
        return (False, [])

    def add_directive_header(self, sig):
        """Override default behavior to add no directive header or options."""
        pass

    def add_content(self, more_content):
        """Override default behavior to add only the first line of the docstring.

        Modified based on the part of processing docstrings in the original
        implementation of this method.

        https://github.com/sphinx-doc/sphinx/blob/faa33a53a389f6f8bc1f6ae97d6015fa92393c4a/sphinx/ext/autodoc/__init__.py#L609-L622
        """
        sourcename = self.get_sourcename()
        docstrings = self.get_doc()

        if docstrings is not None:
            if not docstrings:
                docstrings.append([])
            # Get the first non-empty line of the processed docstring; this could lead
            # to unexpected results if the object does not have a short summary line.
            short_summary = next(
                (s for s in self.process_doc(docstrings) if s), "<no summary>"
            )
            self.add_line(short_summary, sourcename, 0)


def setup(app):
    app.add_autodocumenter(ShortSummaryDocumenter)
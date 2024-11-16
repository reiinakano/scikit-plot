{{ objname | escape | underline(line="=") }}

.. {{ objtype }}
{%- if objtype == "module" %}

.. automodule:: {{ fullname }}
{%- elif objtype == "function" %}

.. currentmodule:: {{ module }}

.. autofunction:: {{ objname }}

.. minigallery:: {{ module }}.{{ objname }}
   :add-heading: Gallery examples
   :heading-level: -

{%- elif objtype == "class" %} {#  or objname.__call__ is not none or is defined #}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :undoc-members:
   :inherited-members:
   :special-members: __call__

.. minigallery:: {{ module }}.{{ objname }} {% for meth in methods %}{{ module }}.{{ objname }}.{{ meth }} {% endfor %}
   :add-heading: Gallery examples
   :heading-level: -

{%- elif objtype == "data" %}
{# Check if it's a class instance or simple data object #}
.. currentmodule:: {{ module }}

.. autodata:: {{ objname }} {# An instance or a data object or value. #}

.. minigallery:: {{ module }}.{{ objname }}
   :add-heading: Gallery examples
   :heading-level: -

{%- else %}
{# General fallback for unrecognized types #}
.. currentmodule:: {{ module }}

.. auto{{ objtype }}:: {{ objname }}
   {# Optional debug message to show unrecognized objtype #}
   :annotation: Unrecognized objtype: `{{ objtype }}`

{%- endif -%}
import os
import glob
from jinja2 import Environment, FileSystemLoader

def render_templates(template_dir, template_vars):
    """
    Renders all .template files in the specified directory and its subdirectories.

    Parameters
    ----------
    template_dir : str
        Path to the directory containing .template files.
    
    template_vars : dict
        Dictionary of variables to be replaced in the templates.
    
    Returns
    -------
    None
    
    Example
    -------
    Suppose you have the following directory structure:

    source/
    ├── example.template
    └── subdir/
        └── another.template

    The content of `example.template` is:
    
    ```
    Title: {{ title }}
    Description: {{ description }}
    ```
    
    You can render these templates with the following code:
    
    >>> template_dir = './'
    >>> template_vars = {
    ...     'title': 'Sample Title',
    ...     'description': 'This is a sample description.'
    ... }
    >>> render_templates(template_dir, template_vars)
    
    This will generate `example` and `another` files in their respective directories 
    with the variables replaced.
    """
    # Set up the Jinja2 environment to load templates from the specified directory
    env = Environment(loader=FileSystemLoader(template_dir))

    # Use glob to find all .template files in the directory and its subdirectories
    template_files = glob.glob(os.path.join(template_dir, '**', '*.template'), recursive=True)

    for template_path in template_files:
        # Load the template
        template_name = os.path.relpath(template_path, template_dir)
        template = env.get_template(template_name)

        # Render the template with the provided variables
        rendered_content = template.render(template_vars)

        # Determine the output filename by removing the .template extension
        output_filename = os.path.join(os.path.dirname(template_path), os.path.basename(template_path).replace('.template', ''))

        # Write the rendered content to the output file
        with open(output_filename, 'w') as f:
            f.write(rendered_content)

        print(f'Rendered {template_path} to {output_filename}')

if __name__ == "__main__":
    # Directory containing the .template files
    template_dir = './'

    # Variables to replace in the templates
    template_vars = {
        'module': 'scikitplot',  # or any relevant value
        'module_info': {
            'description': 'Module description',
            'sections': [
                {
                    'title': 'Section Title',
                    'description': 'Section description',
                    'autosummary': ['item1', 'item2']
                }
            ]
        },
        'inferred': {
            'version_full': {
                'some_key': 'some_value'
            },
            'version_short': {
                'some_key': 'short_version'
            },
            'previous_tag': {
                'some_key': 'some_value'
            }
        },
    }
    # Call the function to render templates
    render_templates(template_dir, template_vars)
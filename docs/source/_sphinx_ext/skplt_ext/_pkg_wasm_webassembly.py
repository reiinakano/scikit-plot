from IPython.display import display, Javascript, HTML
import micropip
import importlib.metadata

display(HTML("<h4>Pyodide Environment preparing...</h4>"))

# Function to trigger clearing after a delay
def clear_console_after_delay():
    # Clear the console after 3 seconds
    display(Javascript("""
        function clearConsole() {
            const consoleContent = document.querySelector('.jp-CodeConsole-content');
            if (consoleContent) {
                consoleContent.innerHTML = '';  // Clear the console content
                console.log('Console cleared after delay!');
            } else {
                console.error('Console not found!');
            }
        }
        setTimeout(clearConsole, 3000);  // Delay 3 seconds before clearing
    """))
# Function to create a new code cell to run micropip.list()
def create_new_cell_and_run():
    # This will create a new code cell and insert micropip.list() in it
    display(HTML("""
        <h4>Packages installed.<br>
          <br>Run <code>micropip.list()</code> to view installed packages.
        </h4>
    """))
# Install packages using micropip
async def install_and_import_packages():    
    # List of packages to install
    packages = [
        'numpy',
        'scipy',
        'pyarrow',
        'fastparquet',
        'pandas',
        'matplotlib',
        'scikit-learn',
        'scikit-plots',
    ]
    # Install each package
    for pkg in packages:
        try:
            await micropip.install(pkg, keep_going=True, index_urls=None)
        except Exception as e:
            if "Python 3 wheel" in str(e):
                print(f"Can't find a pure Python 3 wheel for: '{pkg}'")
            pass

    # Import the installed packages (optional for this example)
    import numpy as np
    import matplotlib as plt
    import scikitplot as skplt

    # Print installed packages and versions after the console is cleared
    print("Installed packages and their versions:")
    for dist in importlib.metadata.distributions():
        print(f"{dist.metadata['Name']:<20} == {dist.metadata['Version']}")

    # Trigger the console clearing after installation and import
    clear_console_after_delay()
    # Create a new cell and run micropip.list() after the console is cleared
    create_new_cell_and_run()

# Run the installation and import process
await install_and_import_packages()
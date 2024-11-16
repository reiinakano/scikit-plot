import os
import sys
import shutil

def clean_up():
    """Remove the temporary pyproject.toml file if it exists."""
    if os.path.exists("pyproject.toml"):
        os.remove("pyproject.toml")

def get_toml_file(build_system):
    """Copy the appropriate pyproject.toml file based on the build system."""
    toml_files = {
        "meson": "pyproject.meson.toml",
        "setuptools": "pyproject.setuptools.toml"
    }

    if build_system not in toml_files:
        print("Invalid option. Use 'meson' or 'setuptools'.")
        sys.exit(1)
    
    toml_file = toml_files[build_system]
    
    try:
        if build_system == "meson":
            import meson
        else:
            import setuptools
            import wheel
    except ImportError:
        os.system(f"pip install {build_system} {'meson-python' if build_system == 'meson' else 'setuptools wheel'}")

    print(f"Using {build_system.capitalize()} pyproject.toml...")
    shutil.copy(toml_file, "pyproject.toml")

def build_with_meson():
    """Build the project using Toml via Meson."""
    get_toml_file("meson")
    os.system("meson setup builddir")
    os.system("meson compile -C builddir")
    os.system("meson install -C builddir")
    clean_up()

def package_with_meson(package_type):
    """Package the project using Toml via Meson."""
    get_toml_file("meson")
    build_commands = {
        "build": "python -m build",
        "sdist": "python -m build --sdist",
        "bdist": "python -m build --wheel"
    }
    
    if package_type not in build_commands:
        print("Invalid package type. Use 'build', 'sdist' or 'bdist'.")
        sys.exit(1)
    
    os.system(build_commands[package_type])
    clean_up()

def install_with_meson():
    """Install the project using Toml via Meson."""
    get_toml_file("meson")
    os.system("pip install .")
    clean_up()

def build_with_setuptools():
    """Build the project using Toml via Setuptools."""
    get_toml_file("setuptools")
    os.system("python setup.py build")
    clean_up()

def package_with_setuptools(package_type):
    """Package the project using Toml via Setuptools."""
    get_toml_file("setuptools")
    build_commands = {
        "build": "python -m build",
        "sdist": "python -m build --sdist",
        "bdist": "python -m build --wheel"
    }
    
    if package_type not in build_commands:
        print("Invalid package type. Use 'build', 'sdist' or 'bdist'.")
        sys.exit(1)
    
    os.system(build_commands[package_type])
    clean_up()

def install_with_setuptools():
    """Install the project using Toml via Setuptools."""
    get_toml_file("setuptools")
    os.system("pip install .")
    clean_up()

def build_with_py():
    """Build the project using setup.py."""
    clean_up()
    os.system("python setup.py build")

def package_with_py(package_type):
    """Package the project using only setup.py."""
    clean_up()
    build_commands = {
        "build": "python setup.py build",
        "sdist": "python setup.py sdist",
        "bdist": "python setup.py bdist_wheel"
    }
    
    if package_type not in build_commands:
        print("Invalid package type. Use 'build', 'sdist' or 'bdist'.")
        sys.exit(1)
    
    os.system(build_commands[package_type])
    clean_up()

def install_with_py():
    """Install the project using setup.py and pip install."""
    clean_up()
    os.system("pip install .")

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("""
Usage: python run_setup.py [py|setuptools|meson] [build|sdist|bdist]

Build and Install [py|setuptools|meson]:
- pip: Builds and installs the package.

Compiles/Packaging [build|sdist|bdist]:
- build: Compiles source code and prepares it for installation folder (e.g., build).
- sdist: Creates a source distribution file (e.g., .tar.gz).
- bdist: Creates a binary distribution file (e.g., .whl).
""")
        sys.exit(1)

    choice = sys.argv[1]
    package_type = sys.argv[2] if len(sys.argv) == 3 else None

    actions = {
        "meson": {
            None: install_with_meson,
            "build": build_with_meson,
            "sdist": lambda: package_with_meson("sdist"),
            "bdist": lambda: package_with_meson("bdist")
        },
        "setuptools": {
            None: install_with_setuptools,
            "build": build_with_setuptools,
            "sdist": lambda: package_with_setuptools("sdist"),
            "bdist": lambda: package_with_setuptools("bdist")
        },
        "py": {
            None: install_with_py,
            "build": lambda: package_with_py("build"),
            "sdist": lambda: package_with_py("sdist"),
            "bdist": lambda: package_with_py("bdist")
        }
    }

    if choice in actions:
        if package_type in actions[choice]:
            actions[choice][package_type]()
        else:
            actions[choice][None]()
    else:
        print("Invalid option. Use 'py', 'meson', or 'setuptools'.")
        sys.exit(1)

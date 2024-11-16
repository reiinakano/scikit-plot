# How effectively delete a git submodule.

To remove a submodule you need to:

- Delete the relevant section from the .gitmodules file.
- Stage the .gitmodules changes git add .gitmodules
- Delete the relevant section from .git/config.
- Run git rm --cached path_to_submodule (no trailing slash).
- Run rm -rf .git/modules/path_to_submodule (no trailing slash).
- Commit git commit -m "Removed submodule "
- Delete the now untracked submodule files rm -rf path_to_submodule

1. Remove the submodule entry from .git/config
    ```
    git submodule deinit -f path/to/submodule
    git submodule deinit -f scikitplot/_xp_core_api/array-api-compat
    ```

2. Remove the submodule directory from the superproject's .git/modules directory
    ```
    rm -rf .git/modules/path/to/submodule
    ```

3. Remove the entry in .gitmodules and remove the submodule directory located at path/to/submodule
    ```
    git rm -f path/to/submodule
    ```
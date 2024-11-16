# GitHub Submodule Guide

## Overview

Submodules are repositories nested inside another Git repository.
They allow you to keep track of versions of external repositories within your main project.

---

## Check for Existing Submodules

To see if your repository contains any submodules, you can use one of the following commands:
```bash
# List all submodules
git submodule

# Alternatively, check the .gitmodules file for submodule information
cat .gitmodules
```


# Initialize Git LFS in your repository, If Needed
```bash
# On Ubuntu or Debian
sudo apt-get install git-lfs

# Initialize Git LFS
git lfs install
```



## Adding a Submodule

To add a new submodule to your repository, use the following command:
```
git submodule add <repository-url> <submodule-path>

# Example
git submodule add -b gh_array_api_compat https://github.com/scikit-plots/array-api-compat.git
git submodule add -b gh_boost_math https://github.com/scikit-plots/math.git
git submodule add -b gh_boost https://github.com/scikit-plots/boost.git
git submodule add -b gh_numcpp https://github.com/scikit-plots/NumCpp.git
git submodule add -b gh_xla https://github.com/scikit-plots/xla.git
```

### Notes:
- Replace `<repository-url>` with the URL of the submodule repository.
- `<submodule-path>` is the directory where the submodule will be stored.

## Cloning a Repository with Submodules

When cloning a repository that contains submodules,
use the --recurse-submodules option to clone the submodules as well:
```
git clone --recurse-submodules <repository-url>

# Example
git clone --recurse-submodules https://github.com/your/repo.git
```

## Updating Submodules

To initialize and update submodules in an existing repository, run:
```
git submodule update --init --recursive
```

## Making Changes in the Submodule

If you want to make changes within the submodule, follow these steps:

1. Navigate to the submodule directory in your main repository:
    ```
    cd submodule-directory
    ```
2. Make the necessary changes in the submodule.
3. Commit and push the changes to the submodule’s repository:
    ```
    git checkout <branch>      # Switch to the appropriate branch
    git add .                  # Stage your changes
    git commit -m "Changes in submodule"  # Commit with a message
    git push origin <branch>   # Push changes to the remote repository
    ```

## Updating the Submodule Reference in the Main Repository

After pushing changes to the submodule, you need to update
the reference in the main repository to reflect these changes:

  1. Go back to the root of the main repository:
      ```
      cd ..
      ```
  2. Update the submodule reference in the main repository:
      ```
      git add submodule-directory
      git commit -m "Update submodule reference"  # Commit the updated submodule reference
      git push origin <branch>   # Push the changes to the main repository
      ```

### Best Practices:
- **Commit Changes Separately**: Always commit changes in the submodule and the main repository separately to maintain clarity.
- **Keep Submodules Updated**: Regularly update your submodules to ensure you have the latest versions and features.
- **Branch Management**: Make sure you are working on the correct branches in both the submodule and main repository to avoid conflicts.

### Additional Comments for Newbies:
- **Understanding Submodules**: It’s crucial to comprehend that submodules are linked to a specific commit of the repository they point to. Changes in the submodule do not automatically update the main repository until you manually update the reference.
- **Use Help Commands**: When in doubt, use `git help submodule` to get more information about submodule commands.
- **Practice on Test Repositories**: Before applying changes to important projects, consider practicing with test repositories to build your confidence.

Feel free to modify any sections or add any additional details you think might be helpful!
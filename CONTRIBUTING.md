# Contribute to Pixeltable

We welcome community contributions to Pixeltable. If there's a new feature or integration that you'd like to see and
you're motivated to make it happen, this guide will help you get started.

Pixeltable uses the standard fork-and-pull contribution model: fork the Pixeltable repo, clone the fork, create a
branch for your changes, and then submit the changes via a pull request. This guide will walk you through how to do
this step-by-step. Here are some guidelines to keep in mind for your first contributions:

* Familiarize yourself with the Pixeltable documentation and codebase. Look through the Pixeltable
    [community issues](https://github.com/pixeltable/pixeltable/issues) and
    [discussions](https://github.com/orgs/pixeltable/discussions) to see if it's a problem or feature that's been
    discussed before. Issues that are marked `good-first-issue` are particularly suitable for first-time contibutors.
* If it's your first or second contribution, it's easiest to start out by adding a new UDF or integration, rather than
    trying to improve some core Pixeltable feature or make changes to our process or workflow. Once you've become more
    familiar with Pixeltable engineering, you'll be able to contribute to those areas as well.
* Pixeltable adheres to rigorous coding and engineering standards. If you submit a PR, expect to see a healthy amount
    of commentary. We strive to ensure that every change or new feature is adequately tested, so it's advisable to
    include unit tests in the `tests` package alongside any code changes in the `pixeltable` package.
* If you're not sure how to proceed or where something should go, or if you have any other questions, don't hesistate
    to open a conversation on the [discussions](https://github.com/orgs/pixeltable/discussions) page. We're here to
    help!

The remainder of this document guides you through setting up your dev environment and creating your first PR.

## Setting up a Dev Environment

Before making a contribution, you'll first need to set up a Pixeltable development environment. It's assumed that you
already have standard developer tools such as `git` and `make` installed on your machine.

1. Set up your Python environment for Pixeltable

    * Install Miniconda:

        * <https://docs.anaconda.com/free/miniconda/index.html>

    * Create your conda environment:

        * `conda create --name pxt python=3.9`
        * For development, we use Python 3.9 (the minimum supported version) to ensure compatibility.

    * Activate the conda environment:

        * `conda activate pxt`

2. Install Pixeltable

    * Fork the `pixeltable` git repo:

        * <https://github.com/pixeltable/pixeltable>

    * Clone your fork locally:

        * `git clone https://github.com/my-username/pixeltable`

    * Install dependencies:

        * `cd pixeltable`
        * `make install`

    * Verify that everything is working:

        * `make test`

We recommend VSCode for development: <https://code.visualstudio.com/>

## Crafting a pull request

Once you've set up your dev environment, you're ready to start contributing PRs.

1. Create a branch for your PR

    * First make sure your `main` branch is up-to-date with the repo:

        * `git checkout main`
        * `git pull home main`

    * Create a branch:

        * `git checkout -b my-branch`

2. Write some code!

    * Don't worry about making small, incremental commits to your branch; they'll be squash-committed when it
        eventually gets merged to `main`.

3. Create a pull request

    * `git checkout my-branch`
    * `git push -u origin my-branch`
    * Now visit the Pixeltable repo on github; you'll see a banner with an option to create a PR. Click it.
    * Once the PR is created, you can continue working; to update the PR with any changes, simply do a
        `git push` from your branch.

4. Periodically sync your branch with `home/main` (you may need to do this occasionally if your branch becomes
    out of sync with other changes to `main`):

    * Update your local main:

        * `git checkout main`
        * `git pull home main`

    * Merge changes from `main` into your PR branch:

        * `git checkout my-branch`
        * `git merge main`

    * Resolve merge conflicts (if any):

        * If there's a merge conflict in `poetry.lock`, follow the steps below.
        * Resolve all other merge conflicts manually.
        * When all conflicts are resolved: `git commit` to complete the process.

    * To resolve a merge conflict in `poetry.lock`:

        * First resolve merge conflicts in `pyproject.toml` (if any).
        * `git checkout --theirs poetry.lock`
        * `poetry lock --no-update`
        * `git add poetry.lock`

5. Code review

    * We use [Reviewable](https://reviewable.io/) for code reviews. You can find a link to the Reviewable page for
        your PR just below the description on your PR page.
    * Respond to any comments on your PR. If you need to make changes, follow the workflow in Steps 3-4 above.
    * Once your PR is approved, click the green "Squash and merge" button on your PR page to squash-commit it into
        `main`.
    * You can now safely delete your PR branch. To delete it in your local clone: `git branch -d my-branch`

6. Congratulations! You're now a Pixeltable contributor.

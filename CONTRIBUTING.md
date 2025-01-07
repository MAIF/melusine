# How to contribute to Melusine Open source

This guide aims to help you contribute to Melusine. If you have found any problems, improvements that can be done, or you have a burning desire to develop new features for Melusine, please make sure to follow the steps bellow.

- [How to open an issue](#how-to-open-an-issue)
- [Create your contribution to submit a pull request](#create-your-contribution-to-submit-a-pull-request)
    - [Fork to code in your personal Melusine repo](#fork-to-code-in-your-personal-melusine-repo)
    - [Clone your forked repository](#clone-your-forked-repository)
    - [Make sure that your repository is up to date](#make-sure-that-your-repository-is-up-to-date)
    - [Start your contribution code](#start-your-contribution-code)
    - [Commit your changes](#commit-your-changes)
    - [Create a pull request](#create-a-pull-request)
    - [Finally submit your pull request](#finally-submit-your-pull-request)

# How to open an issue

**Screenshots are coming soon**

An issue will open a discussion to evaluate if the problem / feature that you submit is eligible, and legitimate for Melusine.

Before opening any new issue, check on the project tab if your issue / feature is not already created. We would like to avoid duplicates. In this tab, you will find the roadmap of Melusine.

A Pull Request must be linked to an issue. Whether it is a bugfix or a feature request, please specify the labels **bug** or **enhancement**.

Next, the Melusine team, or the community, will give you a feedback on whether your issue must be implemented in Melusine, or if it can be resolved easily without a pull request.

# Create your contribution to submit a pull request

## Fork to code in your personal Melusine repo

The first step is to get our MAIF repository on your personal GitHub repositories. To do so, use the "Fork" button on Github landing page of melusine project.


## Clone your forked repository

Click on the "Code" button to copy the url of your repository, and next, you can paste this url to clone your forked repository.

```
git clone https://github.com/<YOUR_GITHUB_PROFILE>/melusine.git
```

## Make sure that your repository is up-to-date

To ensure that your local forked repository is synced with the upstream version available in this repository, you have to update your repo with the `master` branch of Melusine (managed by MAIF). So, go to your repository and as follows:

```
cd melusine
git remote add upstream https://github.com/MAIF/melusine.git
git pull upstream master
```

## Install Melusine into a Python `virtualenv`

Install your local Python environment using the `venv` module of recent Python versions. Read the instructions below to build this environment for local deployments:

First create a Python virtual environment. Python 3.8+ is required.
```
python -m venv melusine-pyenv
```

Activate the environment. This step is required every time you will contribute to the development of Melusine. For more information about Python virtual environments, [read the official documentation](https://docs.python.org/3/library/venv.html). Your environment is next to the Melusine project, stored in the `melusine` directory.
```
source melusine-pyenv/bin/activate
```

Then, go to the melusine directory and install the local packages.
```
cd melusine
```

Then, install the dependencies using `pip`. You can add the `-e` option to keep the last modifications dynamically stored in your virtual env.
```
pip install .
```

There are many other targets available to add dependencies: `dev`, `tests`, `transformers`, `docs`. You can install them like:
```
pip install ".[dev]"
```

## Start your Contribution Code

To contribute to Melusine, you will need to create a personal branch on which you will be responsible for your changes. This means you will have to ensure your tests pass and you only modify the code related to your changes. You have to install the dependencies of the `dev` target.

```
git checkout -b feature/my-contribution-branch
```

We recommend to use a convention of naming branch:
- **feature/your_feature_name** if you are creating a feature.
- **hotfix/your_bug_fix** if you are fixing a bug.

## Commit Your Changes

Before committing your modifications, we have some recommendations:

- Execute `pytest` to check that all tests pass.
```
pytest
```

- Try to build Melusine.
```
python -m build
```

- Use the `pre-commit` configuration.
```
# Installation + auto-configuration of pre-commit
pip install pre-commit
pre-commit install

# Initial run
pre-commit run --all-files
```

- Call `tox` to execute more automated tests.
```
tox
```

In addition, we recommend committing with clear messages and grouping your commits by modifications dependencies. Read this [GitHub Gist](https://gist.github.com/robertpainsi/b632364184e70900af4ab688decf6f53) to have an idea of what we have in mind when speaking of which.

Once all of these steps succeed, push your local modifications to your remote repository.
```
git add YOUR_FILES_WHAT WHERE_MODIFIED
git commit
git push origin feature/my-contribution-branch
```

Your branch is now available on your remote forked repository, with your changes. Next step is now to create a Pull Request so the Melusine Team can review and merge your changes to the official repository.

## Create a Pull Request

A pull request allows you to ask the Melusine team to review your changes, and merge your changes into the master branch of the official repository.

To create one, on the top of your forked repository, you will find a button "Compare & pull request"

<img src="https://raw.githubusercontent.com/MAIF/melusine/master/docs/assets/images/contributing/melusine-compare-pr.png" alt="pull request" />

As you can see, you can select on the right side which branch of your forked repository you want to associate to the pull request.

On the left side, you will find the official Melusine repository. Due to increase of external contributions, we advise you to create a pull request on develop so we can test before integrating it to the final release.

- Base repository: MAIF/melusine
- Base branch: develop
- Head repository: your-github-username/melusine
- Head branch: your-contribution-branch

<img src="https://raw.githubusercontent.com/MAIF/melusine/master/docs/assets/images/contributing/melusine-pr-branch.png" alt="clone your forked repository" />

Once you have selected the right branch, let's create the pull request with the green button "Create pull request".

<img src="https://raw.githubusercontent.com/MAIF/melusine/master/docs/assets/images/contributing/melusine-pr-description.png" alt="clone your forked repository" />

In the description, a template is initialized with all information you have to give about what you are doing on what your PR is doing.

Please follow this to write your PR content.

## Finally, submit your pull request

Your pull request is now ready to be submitted. A member of the Melusine team will contact you and will review your code and contact you if needed.

You have contributed to an Open source project, thank you and congratulations! 🥳

Show your contribution to Melusine in your curriculum, and share it on your social media. Be proud of yourself, you gave some code lines to the entire world!

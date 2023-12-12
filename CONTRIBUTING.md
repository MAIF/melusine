# How to contribute to Melusine Open source

This guide aims to help you contributing to Melusine. If you have found any problems, improvements that can be done, or you have a burning desire to develop new features for Melusine, please make sure to follow the steps bellow.

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

Check on the project tab if your issue / feature is not already created. In this tab, you will find the roadmap of Melusine.

A Pull Request must be linked to an issue.
Before you open an issue, please check the current opened issues to insure there are no duplicate. Define if it's a feature or a bugfix.

Next, the Melusine team, or the community, will give you a feedback on whether your issue must be implemented in Melusine, or if it can be resolved easily without a pull request.

# Create your contribution to submit a pull request
## Fork to code in your personal Melusine repo

The first step is to get our MAIF repository on your personal GitHub repositories. To do so, use the "Fork" button.

<img src="https://raw.githubusercontent.com/MAIF/melusine/master/docs/assets/images/contributing/fork_melusine.PNG" alt="fork this repository" />

## Clone your forked repository

<img align="right" width="300" src="https://raw.githubusercontent.com/MAIF/melusine/master/docs/assets/images/contributing/clone_melusine.PNG" alt="clone your forked repository" />

Click on the "Code" button to copy the url of your repository, and next, you can paste this url to clone your forked repository.

```
git clone https://github.com/YOUR_GITHUB_PROFILE/melusine.git
```

## Make sure that your repository is up to date

To insure that your local forked repository is synced, you have to update your repo with the master branch of Melusine (MAIF). So, go to your repository and as follow :

```
cd melusine
git remote add upstream https://github.com/MAIF/melusine.git
git pull upstream master
```

## Install Melusine into a virtualenv

Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed, this is how you set up your fork for local development::

First create a python virtual environment
```
mkvirtualenv melusine
```
Go in the melusine directory
```
cd melusine
```
Install you local package
```
python setup.py develop
```
or
```
pip install -e .
```
or to have optionnal dependencies
```
pip install ".[transformers]"
```

## Start your contribution code

To contribute to Melusine, you will need to create a personal branch.
```
git checkout -b feature/my-contribution-branch
```
We recommand to use a convention of naming branch.
- **feature/your_feature_name** if you are creating a feature
- **hotfix/your_bug_fix** if you are fixing a bug

## Commit your changes

Before committing your modifications, we have some recommendations :

- Execute pytest to check that all tests pass
```
pytest
```
- Try to build Melusine
```
python setup.py bdist_wheel
```
- Check your code with **flake8**

*We will soon add **pre commit** to automatically check your code quality during commit*

```
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
```

Also for now, try testing tox. It will not be used for much longer as we will be moving to Github Actions and a simpler process soon.

```
tox
```

To get flake8 and tox, just pip install them into your virtualenv.

In addition, we recommend committing with clear messages and grouping your commits by modifications dependencies.

Once all of these steps succeed, push your local modifications to your remote repository.

```
git add .
git commit -m ‘detailed description of your change’
git push origin feature/my-contribution-branch
```

Your branch is now available on your remote forked repository, with your changes.

Next step is now to create a Pull Request so the Melusine Team can add your changes to the official repository.

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

In the description, a template is initialized with all informations you have to give about what you are doing on what your PR is doing.

Please follow this to write your PR content.


## Finally submit your pull request

Your pull request is now ready to be submitted. A member of the Melusine team will contact you and will review your code and contact you if needed.

You have contributed to an Open source project, thank you and congratulations ! 🥳

Show your contribution to Melusine in your curriculum, and share it on your social media. Be proud of yourself, you gave some code lines to the entire world !

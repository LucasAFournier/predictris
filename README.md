# Exemple of minimal git repository

First create your new project, for exemple "project_name" in your team group

Then in a terminal, setup your git global config

```
git config --global user.name "Firstname Name"
git config --global user.email "Firstame.Name@ens-lyon.fr"
```

Copy the depot template in your new project

```{bash}
NEW_DEPOT=git@gitbio.ens-lyon.fr:LBMC/team/project_name.git

NEW_NAME=$(basename $NEW_DEPOT)
NEW_NAME=${NEW_NAME/.git}

git clone git@gitbio.ens-lyon.fr:LBMC/hub/minimal_git_repo.git
git clone $NEW_DEPOT

rm -fr minimal_git_repo/.git
mv $NEW_NAME/.git minimal_git_repo/
rm -r $NEW_NAME

mv minimal_git_repo $NEW_NAME
cd $NEW_NAME

rm README.md
touch README.md

git add -f * .gitignore
git commit -m "initial commit"
```

Finally, push your depot with the initial template

```{bash}
git push

```

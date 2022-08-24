# Applying Neural Network Models for School-aged Children Undernutrition Classification and Feature Selection
This is a thesis project that extends the work of [Siy Van et al. (2022)](https://doi.org/10.1016/j.nut.2021.111571) by applying neural network models to predict undernutrition among school-aged children using a combination of individual and household risk factors.

---

## How to Contribute

### Setting Up

#### For Windows Users:
1. Make sure you have [Python 3.10.6](https://www.python.org/downloads/) installed. Do not install python from the windows store.
2. Clone this repository using `git clone https://github.com/ongyiumark/predicting-undernutrition.git`
3. Go inside the reposisitory using `cd .\predicting-undernutrition\`
4. Create a new python virtual environment using `py -m venv [environment name]`. For example `py -m venv thesisPU`
5. Activate the virtual environment using `.\[environment name]\Scripts\activate`. For example `.\thesisPU\Scripts\activate`
6. Make sure you have the latest version of pip using `py -m pip install --upgrade pip` 
7. Install the project dependencies using `pip install -r .\requirements.txt`

#### For Unix or MacOS Users:
Not sure how this works because I don't have a device to test it on, but it should be pretty similar. The main difference might be that `py` is replaced with `python3`.

### Contribution Workflow
1. Setup the repository by following the instructions above.
2. Create a new branch using `git checkout -b [add your 2-letter initials here]--[branch code]`. For example `git checkout -b mo--edit-readme`
3. Open jupyter using `jupyter notebook`
4. Open a jupyter notebook inside the folder `/code`
5. Add and commit your changes with git
6. Push your local branch to github using `git push -u origin [branch name]`. For example, `git push -u origin mo--edit-readme`
7. Submit a pull request 
8. Tag someone to review your code
9. Merge your PR only after receiving at least 1 approval from a reviewer

## Contributors
- [Mark Kevin A. Ong Yiu](https://github.com/ongyiumark)
- [Carlo Gabriel M. Pastor](https://github.com/AQ51)
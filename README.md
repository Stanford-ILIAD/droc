# DROC

Distilling and retrieving generalizable knowledge from online human language corrections.

## Setup

Create the environment and start using the repo.
```bash
conda env create -f environment.yml
```

## Usage

```bash
python scripts/script.py --realrobot <realrobot> --task <task>
```
where `<realrobot>` is a bool variable that determines whether this experiment is run on a real robot or not, `<task>` is the name of the task to run. You have to pre-define the available tasks along with their clip candidates in `utils/perception/perception_utils.py`.
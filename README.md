# Distilling and Retrieving Generalizable Knowledge for Robot Manipulation via Language Corrections
## [<a href="https://sites.google.com/stanford.edu/droc" target="_blank">Project Page</a>][<a href="https://arxiv.org/abs/2311.10678">Paper</a>]

![til](https://github.com/lihzha/visualizations/blob/main/overview(twitter).gif)

We propose a method for responding to online language corrections, distilling generalizable knowledge from them and retrieving useful knowledge for novel tasks, as described <a href="https://sites.google.com/stanford.edu/droc" target="_blank">here</a>.

[//]: # (### Abstract)
> Today's robot policies exhibit subpar performance when faced with the challenge of generalizing to novel environments. Human corrective feedback is a crucial form of guidance to enable such generalization. However, adapting to and learning from online human corrections is a non-trivial endeavor: not only do robots need to remember human feedback over time to retrieve the right information in new settings and reduce the intervention rate, but also they would need to be able to respond to feedback that can be arbitrary corrections about high-level human preferences to low-level adjustments to skill parameters. In this work, we present Distillation and Retrieval of Online Corrections (DROC), a large language model (LLM)-based system that can respond to arbitrary forms of language feedback, distill generalizable knowledge from corrections, and retrieve relevant past experiences based on textual and visual similarity for improving performance in novel settings. DROC is able to respond to a sequence of online language corrections that address failures in both high-level task plans and low-level skill primitives. We demonstrate that DROC effectively distills the relevant information from the sequence of online corrections in a knowledge base and retrieves that knowledge in settings with new task or object instances. DROC outperforms other techniques that directly generate robot code via LLMs by using only half of the total number of corrections needed in the first round and requires little to no corrections after two iterations.

For more details please refer to our [paper](https://arxiv.org/abs/2311.10678).


## Installation

1. Create a virtual environment (this could take a while) and install the package:
      ```bash
      conda create -n droc python==3.8
      pip install -r requirements.txt
      pip install -e .
      ```

2. (Optional) Download <a href="https://github.com/facebookresearch/fairo/tree/main/polymetis">Polymetis</a> for real robot experiments. Note this only supports PyTorch ~= 1.12. If you are using new versions of PyTorch, please refer to the <a href="https://github.com/facebookresearch/fairo/tree/main/polymetis">monometis</a> fork from Hengyuan Hu.

3. (Optional) Download <a href="https://github.com/Stanford-ILIAD/muse">muse</a> for real robot experiments.

4. Set your OpenAI key in `utils/LLM_utils.py`.

## Code Structure

* **Scripts**: Contains the main script and all baseline scripts.

* **Prompts**: All LLM prompts are stored in the folder `prompts/`. For the function of each prompt, please refer to [here](https://github.com/Stanford-ILIAD/droc/tree/main/prompts/prompt_overview.txt).

* **Utils**: Contains all utilities for running the main script, including I/O, perception, robot control, LLM, exception handling, etc.

* **Cache**: Contains the knowledge base (in .pkl format), calibration results, detection results and other caches.


## Usage

### Real robot experiments

1. Calibrate the cameras and put the calibration results into `cache/calib`. Define your realsense serial numbers in `utils/perception/shared_devices.py`.

2. Define the task name and their clip candidates in `utils/perception/perception_utils.py`.

3. Run the main script.
      ```bash
      python scripts/script.py --realrobot --task <task>
      ```
      where `<task>` is the name of the task to run.

### Dummy testing
If you want to test your prompts, a better choice would be running your code in a dummy mode.
```bash
python scripts/script.py --task <task>
```

## Troubleshooting
Since the released implementation is primarily concered with real world robot experiments, there may be some unexpected bugs specific to indiviual use cases. If you meet any bug depolying this codebase, feel free to contact <lihanzha20@gmail.com>.


## Citation

```
@misc{zha2023distilling,
      title={Distilling and Retrieving Generalizable Knowledge for Robot Manipulation via Language Corrections}, 
      author={Lihan Zha and Yuchen Cui and Li-Heng Lin and Minae Kwon and Montserrat Gonzalez Arenas and Andy Zeng and Fei Xia and Dorsa Sadigh},
      year={2023},
      eprint={2311.10678},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```
# Diffusion course project

- [Diffusion course project](#diffusion-course-project)
  - [Description](#description)
    - [Source code structure](#source-code-structure)
  - [Usage](#usage)
  - [TODO](#todo)


## Description


### Source code structure
```bash
  DiffusionProject
    |-configs       (root of hydra config files)
    |-models
    | |-emd         (to be downloaded)
    | |-solvers     (root of solver files)
    |-outputs       (root of all generated images)
    |-report
    | |-data        (root of all images in report)
    | |-report.pdf  (pdf report file)
    | |-report.tex  (tex report sources)
    |-run_rests.py  (main entry point)
    |-utils.py      (util functions)
```



## Usage
Clone EDM repo into `models` folder
```bash
    cd models & git clone git@github.com:NVlabs/edm.git
```
To run code:
```bash
    PYTHONPATH="$PYTHONPATH:models/edm" python3 run_tests.py
```


## TODO
 - fill readme
 - Add solvers
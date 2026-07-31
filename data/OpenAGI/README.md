# OpenAGI official benchmark data

Official project: https://github.com/agiresearch/OpenAGI

Official paper dataset assets:
https://drive.google.com/drive/folders/1AjT6y7qLIMxcmHhUBG5IE1_5SnCPR57e

The task definitions are copied without modification from the official
`research` branch:

- `train/task_description.txt`: 15 training-task definitions.
- `train/model_sequence.txt`: 15 corresponding training model sequences.
- `test/task_description.txt`: 185 benchmark-task definitions.

The benchmark paper states that each of the 185 tasks has 100 augmented
multimodal samples. Those large input/output assets are hosted only in the
official Google Drive folder. They are not present here yet because this host
cannot currently connect to `drive.google.com`. Run
`scripts/download_openagi_official_assets.py` when Google Drive is reachable.

Do not treat this directory as a complete OpenAGI download until the script
finishes and `official_assets/` is populated. Dataset license: CC BY 4.0.

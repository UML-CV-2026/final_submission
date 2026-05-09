## Evaluating Topology-Aware Losses for Retinal Vessel Segmentation
This is the code for our Computer Vision Spring 2026 Final Project at UMass Lowell

## Models
The models can be found here:
https://huggingface.co/yu-alvin/Computer-Vision-2026
However, our code in evals/ will automatically download the weights for you. 

## Training the Models
If you want to train the models, you NEED to change the cells so that they point at the exact directory in which you downloaded the FIVES dataset.
NOTE: `main.ipynb` is 0.5 Dice + 0.5 BCE. Other files use the same naming convention as the evals.

## Citations
Some code was derived from these repositories:
1. https://github.com/jocpae/clDice - `cldice.py`
2. https://github.com/PengchengShi1220/cbDice - `cbdice.py`

## Demos
Our demos are located in the evals folder where you can evaluate our pretrained models on the FIVES dataset. The evaluation notebooks will automatically fetch the models from huggingface and evaluate them on the test split of the FIVES dataset. It is imperative that you first download the FIVES dataset properly.

You will find 4 different evals:
1. `baseline.ipynb` - 0.5 Dice + 0.5 BCE
2. `betti.ipynb` - 0.5 Dice + 0.5 BCE + 1e-5 Betti
3. `cbDice.ipynb` - 0.4 Dice + 0.4 BCE + 0.2 cbDice
4. `clDice.ipynb` - 0.8 Dice + 0.2 clDice


**Steps:**
1. `curl -L -o fundus.zip https://www.kaggle.com/api/v1/datasets/download/nikitamanaenkov/fundus-image-dataset-for-vessel-segmentation`
2. Then once you unzip the folder, name it "fundus". After downloading the dataset, please install the uv package manager through this command: `curl -LsSf https://astral.sh/uv/install.sh | sh`
3. And then run `uv sync`
4. After this, the notebooks in evals/ should be plug and play in terms of running the experiments. Make sure the run the cells in order or something will break.
5. If the demo doesn't work, we've saved the results of the the demo within the evals notebooks for your convenience. You can also contact us at alvin_yu@student.uml.edu.

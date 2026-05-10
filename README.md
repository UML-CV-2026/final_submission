



## Evaluating Topology-Aware Losses for Retinal Vessel Segmentation
Code for Computer Vision (Spring 2026) Final Project at UMass Lowell

## Models
The models can be found here:
https://huggingface.co/yu-alvin/Computer-Vision-2026
However, our code in `evals/` and `demo/` will automatically download the weights for you. 

## Training the Models
If you want to train the models, you NEED to change the cells so that they point at the exact directory in which you downloaded the FIVES dataset.
NOTE: `main.ipynb` is 0.5 Dice + 0.5 BCE. Other files use the same naming convention as the evals.

## Citations
Some code was derived from these repositories:
1. https://github.com/jocpae/clDice - `cldice.py`
2. https://github.com/PengchengShi1220/cbDice - `cbdice.py`

## Evaluation Notebooks
Our evaluation notebooks are located in the `evals` folder where you can evaluate our pretrained models on the FIVES dataset. The evaluation notebooks will automatically fetch the models from huggingface and evaluate them on the test split of the FIVES dataset. It is imperative that you first download the FIVES dataset properly. You can do this by running `getdataset.sh`.

You will find 4 different evals:
1. `baseline.ipynb` - 0.5 Dice + 0.5 BCE
2. `betti.ipynb` - 0.5 Dice + 0.5 BCE + 1e-5 Betti
3. `cbDice.ipynb` - 0.4 Dice + 0.4 BCE + 0.2 cbDice
4. `clDice.ipynb` - 0.8 Dice + 0.2 clDice

## Demo
We have a streamlit demo at `demo/` folder. There's also two images from the FIVES dataset to try out under the `demos/` folder . Unfortunately the demo only works well with images that are have dimensions of a multiple of 16.

https://github.com/user-attachments/assets/47fd6e66-6526-4f4b-9ceb-6183e4a96ba1

**Steps:**
1. Install the uv package manager through this command: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. And then run `uv sync`
3. Run `uv run streamlit run demo/demo.py`. The output will give you a URL to enter into your browser. Enter that URL to access the demo.
4. Upload your image and mask

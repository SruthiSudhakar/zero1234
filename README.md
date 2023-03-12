Installation: 

conda create -n zero123 python==3.9
conda activate zero123
pip install -r stable-diffusion/requirements.txt
git clone https://github.com/CompVis/taming-transformers.git
pip install -e taming-transformers/
git clone https://github.com/openai/CLIP.git
pip install -e CLIP/

Run gradio demo: 

python gradio_objaverse.py
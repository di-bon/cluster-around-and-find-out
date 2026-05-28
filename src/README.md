# Setup

## Initialize the environment

Inside the `src/` folder, run:

1. `conda create -n sherlock python=3.13 pip -y`
1. `conda activate sherlock`
1. `pip install -r requirements.txt`

## Run the conversational clustering system

To run the system:

1. In a different terminal tab, start ollama with `ollama serve`
1. Run `python main.py`

Note: 
this code uses `qwen3.6:35b`.
To run the code, make surve to have it installed by running `ollama pull qwen3,6:35b`.
If you want to use a different model, pull it using `ollama pull <model>` and update the model used in `src/agents/clustering_interview_agent.py`

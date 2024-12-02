# JobGen

Generating Synthetic Job Ads through LLMs

This repo contains code for JobGen. If you like this paper, please cite us:

```
Samuele Colombo, Simone D’Amico, Lorenzo Malandri, Fabio Mercorio, and Andrea Seveso. 2025. JobSet: Synthetic Job Advertisements Dataset for Labour Market Intelligence. In Proceedings of ACM SAC Conference (SAC’25).
```

## What this repo contains

- The JobGen code
- It's required inputs in `data/`
- Notebook for metrics & experiments, in `experiments/`
- Results of previous Jobset, in `results/`

## Getting the environment set up

I used a vLLM server (https://docs.vllm.ai/en/latest/getting_started/quickstart.html) as inference engine. My recommendation is to install vLLM and run the server with this command `python -m vllm.entrypoints.openai.api_server --model casperhansen/llama-3-70b-instruct-awq --quantization awq --gpu-memory-utilization 0.95 --enable-prefix-caching --enforce-eager` for maximum performance. Then, use the following command to install dependencies:

```
pip install -r requirements.txt
```

At this point, we used to launch two concurrenct worker in order to fully utilize a continuos batching

```
python main.py 1
python main.py 2
```

## Adapting the code

In main.py set `use_online_oja = False` if you want to provide your own custom seed (real-life vacancies) otherwise the system will scrape it from the internet
Also set up where do you want to store the generated OJAs. We used an AWS S3 bucket so you need to provide your access key and bucket name

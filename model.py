from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
import time


class Agent:
    def __init__(self, client, model):
        self.client = client
        self.model = model
        self.memory = []

    def answer(self, prompt, max_tokens):
        # Un po fuorviante puo ritornare una lista di risposte ma anche una singola stringa
        return llm_batch_generate(self.model, self.client, prompt, max_tokens)

    def memorize(self, message):
        self.memory = message

    def remember(self):
        return self.memory

    def forget(self):
        self.memory = []

    def find_similarity_in_memory(self, avg_similarity_threshold, emb_model, emb_tokenizer, similarity_matrix):
        
        n = len(similarity_matrix)
        min_similarity = float('inf')
        max_similarity = float('-inf')

        min_pair = (-1, -1)
        max_pair = (-1, -1)

        for i in range(n):
            for j in range(n):
                if i != j:
                    sim = similarity_matrix[i][j]
                    if sim < min_similarity:
                        min_similarity = sim
                        min_pair = (i, j)
                    if sim > max_similarity:
                        max_similarity = sim
                        max_pair = (i, j)

        mean_similarity = [(sum(row)-1)/(n-1) for row in similarity_matrix]
        avg_pairwise_similarity = sum(mean_similarity) / n

        # in teoria min = max potrebbero coincidere ma
        best_oja = min(range(n), key=lambda doc: mean_similarity[doc])
        # i float fanno casini ;)
        worst_oja = max(range(n), key=lambda doc: mean_similarity[doc])

        if avg_pairwise_similarity > avg_similarity_threshold:  # they can't be too similar

            print("Pair with min similarity: {}, similarity: {}".format(
                min_pair, min_similarity))
            print("Pair with max similarity: {}, similarity: {}".format(
                max_pair, max_similarity))
            print("OJA to use as example (least similar on average): {}".format(best_oja))
            print("OJA to rewrite (most similar on average): {}".format(worst_oja))
            print("Average pairwise similarity: {}".format(
                avg_pairwise_similarity))

            return avg_pairwise_similarity, best_oja, worst_oja, self.memory[worst_oja][1], similarity_matrix
        else:
            return avg_pairwise_similarity, None, None, None, similarity_matrix


def load_model(path):

    start_time = time.time()

    # I love Tim Dettmers
    bnb_config = {'load_in_4bit': True,
                  'bnb_4bit_use_double_quant': True,
                  'bnb_4bit_quant_type': "nf4",
                  'bnb_4bit_compute_dtype': torch.bfloat16}

    model = AutoModelForCausalLM.from_pretrained(path, **bnb_config)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(path)
    end_time = time.time()  # End the timer
    print(f"Model loading: {end_time - start_time} seconds")
    return model, tokenizer


def load_emb_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


def llm_batch_generate_old(model, tokenizer, prompt, device, max_token, num_sequences):

    torch.cuda.empty_cache()

    start_time = time.time()
    chat = [
        {"role": "user", "content": f"{prompt}"},
    ]
    model_inputs = tokenizer.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    model_inputs = model_inputs.to(device)
    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs,
            max_new_tokens=max_token,
            epsilon_cutoff=1.49,  # https://arxiv.org/abs/2210.15191
            eta_cutoff=10.42,
            repetition_penalty=1.17,
            do_sample=True,
            top_k=49,  # https://huggingface.co/blog/how-to-generate
            top_p=0.14,
            temperature=1.31,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=num_sequences,  # 1 perchè sono gpu-poor
        )

    answers = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    # Split answers if the tokenizer includes special tokens that you wish to remove
    # Note: The line below assumes your special token is '[/INST]', adjust if different
    answers = [answer.split(
        '[/INST]')[1] if '[/INST]' in answer else answer for answer in answers]

    # If only one sequence was generated, return a string instead of a list
    if len(answers) == 1:
        answers = answers[0]

    end_time = time.time()  # End the timer
    print(f"Batch inference: {end_time - start_time} seconds")

    return answers


def llm_batch_generate(client, model, prompts, max_token):

    response = client.completions.create(
        model=model,
        max_tokens=max_token,
        prompt= prompts
    )
    answers = [choice.text for choice in response.choices]
    return answers[0] if len(answers) == 1 else answers


if __name__ == '__main__':

    from openai import OpenAI
    # Set OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
    )

    start = time.time()
    prompts = ["write me a realistic job description for a pipe metal operator"] * 10
    max_token_list = 20
    # batched example, with 10 story completions per request
    response = client.completions.create(
        model="TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
        prompt=prompts,
        max_tokens=max_token_list,
    )
    answers = [choice.text for choice in response.choices]

    # Return a single answer if only one prompt was provided or a list of answers otherwise
    print(answers[0] if len(answers) == 1 else answers)
    print(" batched" , time.time() - start) 


    

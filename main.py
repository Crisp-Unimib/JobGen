import ast
from jobspy import scrape_jobs # keep this first idk why
import pyarrow as pa
import pyarrow.parquet as pq
from io import BytesIO
import boto3
from botocore.exceptions import BotoCoreError, ClientError
import sys
from time import time, sleep
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import random
import pandas as pd
from search import job_ad_web_search
from utils import print_colored, pick_seed_offline
from read_esco import read_and_prepare_datasets
from model import load_emb_model, Agent
import os
from openai import OpenAI
import random
import pickle

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:300'
pd.options.mode.chained_assignment = None


def main(do_only_occupation_starting_with):

    # Params
    similarity_threshold = 0.85  # 85% simili tra loro
    standard_job_ad_length = 3000

    delay = 1 # delay per riprovare su s3, esponenziale

    use_online_oja = False

    s3_path = 'output/parquet/2024_05_27'

    # import pickle list  experiments/bad_esco_ids.pickle
    with open('experiments/bad_esco_ids.pickle', 'rb') as handle:
        bad_esco_ids = pickle.load(handle)


    s3 = boto3.client('s3', aws_access_key_id='ACCESS_KEY',
                      aws_secret_access_key='SECRET_KEY')
    bucket_name = 'crisp-projects-llm-benchmark'
    data_path = os.path.dirname(os.path.abspath(
        __file__))  # TODO da standardizzare
    emb_model_name = "BAAI/bge-large-en-v1.5"
    websites = ["indeed", "linkedin", "zip_recruiter", "glassdoor"]

    openai_api_key = "EMPTY" 
    openai_api_base =  "http://localhost:8000/v1" 
    openai_client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    df_oja_to_generate, oja_examples = read_and_prepare_datasets(data_path)
    df_oja_to_generate = df_oja_to_generate[df_oja_to_generate['code'].astype(str).str.startswith(do_only_occupation_starting_with)]
    model =  "casperhansen/llama-3-70b-instruct-awq" #"casperhansen/mixtral-instruct-awq"
    emb_model, emb_tokenizer = load_emb_model(emb_model_name)

    # Initialize agent
    Writer = Agent(model, openai_client)  # create job description
    # quality check of the job description
    Supervisor = Agent(model, openai_client)

    # Iterate over the first subset of occupations

    total_row = len(df_oja_to_generate)
    for i, row in tqdm(df_oja_to_generate.iterrows(), total=total_row):

        print(f'_Digit {row.code} starting...')
        if row.code in bad_esco_ids:
            print(f'EscoID {row.code} is in the bad list, redoing it.')
        else:
            
            print(f'EscoID {row.code} is not in the bad list, skipping it.')
            continue
        subset = df_oja_to_generate[df_oja_to_generate['code'] == row.code] 
        start_time = time()
        try:
                s3.head_object(Bucket=bucket_name, Key=f"{s3_path}/{row.code}.parquet")
                print(f's3://{bucket_name}/{s3_path}/{row.code}.parquet already exists, skipping.')
                continue

        except:
            pass

        # Reset agents memory
        Writer.forget()
        Supervisor.forget()
        j = 0

        print_colored(
            f"[Boss]: incoming task #{i+1}/{total_row} for the team", "red")

        # Look up for job ads
        print_colored(
            f"[Scraper]: looking for job ads about {row['occ_label']} as reference", "blue")
        
        if use_online_oja:
            real_job_ad = job_ad_web_search(row['code'], websites)
            real_job_ad = real_job_ad_list.sample(1)
            max_tokens = len(real_job_ad) if real_job_ad != "No jobs found" else standard_job_ad_length

        else:
            real_job_ad_list = pick_seed_offline(row['code'], oja_examples)

        answers = []
        prompts = []
        skills_picked_list = [] 
        max_job_per_occupation = len(subset)
        
        for k in range(0, max_job_per_occupation):

            if use_online_oja == False:
                real_job_ad = random.choice(real_job_ad_list)
                max_tokens = len(real_job_ad) if real_job_ad != "No jobs found" else standard_job_ad_length

            if model == "casperhansen/llama-3-70b-instruct-awq" and max_tokens > 8192:
                max_tokens = 8192


            # Prompt
            generate_task = f"You are an expert recruiter and your main objective is to generate a realistic, human-like yet anonimized job description for the role of {row['occ_label']}.\n"
            generate_task += f"""Include a brief opening talking about the role loosely based on the following text:\n"{row['occ_description']}"\n"""
            generate_task += f"""You must include only and every skills listed below for this position, even if it seems unusal:\n"""

            # Retrieve all the skills related to the job ## 3 digit
            skills_picked = ast.literal_eval(subset.iloc[k]['skills'])

            # Add skills as reference
            generate_task += "".join([f"{skill}, " for skill in skills_picked]).strip(', ')

            # Add real job ads as reference
            if real_job_ad != "No jobs found":
                 generate_task += f"""\n\nAdditionally, use the following job ad as a reference, but for style and structure only:\n"{real_job_ad}" """
            generate_task += "\nDo not write anything else other than the plain job description itself!!"
            generate_task += "\nAfter you finish writing the job description, output the special token @end!!"
            generate_task += "\n### Job Description:\n"


            print_colored(
                f"[Writer]: receiving task #{i+1}-->{k+1}/{max_job_per_occupation}:", "blue")

            # prepare prompt batch
            prompts.append(generate_task)
            skills_picked_list.append(skills_picked)

        batch_answers = Writer.answer(prompts, max_tokens)
        answers.extend([[answer, skill] for answer, skill in zip(batch_answers, skills_picked_list)])

        print_colored(
            f"\n[Writer]: jd creation task #{i+1} executed {max_job_per_occupation} times", "green")

        # Save job descriptions in memory of agents 2
        Writer.memorize(answers)
        Supervisor.memorize(Writer.remember())

        # make a dict that save how many times a skill is rewrote
        skill_rewrite_count = {}
        time_rewrite_count = [0 for _ in range(max_job_per_occupation)]

        # Iterate over job descriptions
        encoded_input = emb_tokenizer([item[0] for item in Supervisor.memory], padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = emb_model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        similarity_matrix = cosine_similarity(sentence_embeddings)
        while True:

            # Check if job descriptions are too similar
            avg_pairwise_similarity, best_oja, worst_oja, worst_oja_skills, similarity_matrix = Supervisor.find_similarity_in_memory(
                similarity_threshold, emb_model, emb_tokenizer, similarity_matrix)
            if worst_oja == None:
                print_colored(
                    f"[Supervisor]: all the OJA generated are different enough ({avg_pairwise_similarity})", "green")
                break
            rewrite_time_start = time()
            # Construct task string for the supervisor
            # Introduce the task with clear directives
            supervise_task = "Your task is to rewrite a bad job description. Draw inspiration from the top-performing job description and avoid elements typical of the lower-performing description.\n"
            supervise_task += "### Top-performing (Good Example):\n"
            # supervise_task += "This description has been acknowledged for its realistic, unique and well-written content:\n"
            supervise_task += f"#### {answers[best_oja][0]}\n"
            # Providing the bad example
            supervise_task += "### Lower-performing (Bad Example):\n"
            # supervise_task += "The following description is considered less effective due to its lack of distinctiveness and average similarity with many others:\n"
            supervise_task += f"#### {answers[worst_oja][0]}\n"
            # Add instruction to tailor the rewrite to a specific label
            supervise_task += "### Rewriting caveat:\n"
            supervise_task += "Use synonyms and/or different style/tone/structures/descriptions to make the job description different.\n"
            supervise_task += "The position mandates only and only the following skills, do not include any other skills even if it doesn't make sense for you. Here is the list:\n"
            skills_checklist = "\n".join([f"- {skill}" for skill in worst_oja_skills])
            supervise_task += f"{skills_checklist}\n"
            supervise_task += "Do not get too creative with fantasy-themed job descriptions\n"
            supervise_task += "Do not write anything else other than the job description!!\n"
            supervise_task += "After you finish writing the job description, output the special token @end!!\n"
            supervise_task += "### Revised Job Description:\n"

            print_colored(
                f"[Supervisor]: the OJA generated are too similar #{i+1} (Average pairwise similarity: {avg_pairwise_similarity:.3f}%).", "magenta")

            fix = Supervisor.answer(
                supervise_task, max_tokens=max_tokens)
            print_colored(f"\n[Supervisor]: task #{j+1} rewrote", "yellow")

            # Update skill_rewrite_count
            skill_rewrite_count[str(worst_oja)] = skill_rewrite_count.get(
                str(worst_oja), 0) + 1

            # Refresh memory of agents 2
            old_memory = Supervisor.remember()
            new_memory = old_memory
            new_memory[worst_oja][0] = fix
            Supervisor.memorize(new_memory)

            # Update similarity matrix
            encoded_input_fixed_oja = emb_tokenizer(fix, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                model_output_new_oja = emb_model(**encoded_input_fixed_oja)
                new_sentence_embedding = model_output_new_oja[0][0][0] #cls pooling
            new_sentence_embedding = torch.nn.functional.normalize(new_sentence_embedding, p=2, dim=0)
            
            # Update the similarity matrix
            for i in range(len(similarity_matrix)):
                sim_i_rewroted_tensor = torch.cosine_similarity(
                    new_sentence_embedding.unsqueeze(0),  
                    sentence_embeddings[i].unsqueeze(0)  
                )
                if sim_i_rewroted_tensor.numel() == 1:
                    # Extract the scalar value from the tensor
                    sim_i_worst = sim_i_rewroted_tensor.item()  # This should be a scalar now

                    # Update the similarity matrix
                    similarity_matrix[worst_oja][i] = similarity_matrix[i][worst_oja] = sim_i_worst
                else:
                    raise ValueError("Expected a single-element tensor from cosine_similarity.")
                
            rewrite_time_end = time() - rewrite_time_start
            time_rewrite_count[worst_oja] += rewrite_time_end 
            # Update counter
            j += 1

        # Save memory of agent 2 in a huggingface dataset
        final_memory = Supervisor.remember()
        # Add 'preferredLabel_job' to each job ad  # FIX the skills picked will change every time
        final_memory_dicts = [{
            'job_ad': job_ad[0],
            'escoId': row['code'],
            'escoLabel': row['occ_label'],
            'escoSkills': ', '.join(job_ad[1]),
            'seed': real_job_ad,
            'num_rewrites': skill_rewrite_count.get(str(n), 0),
            'time': time() - start_time + time_rewrite_count[n],
        } for n, job_ad in enumerate(final_memory)]

        # Convert the list of memories into a csv and save it to s3
        table = pa.Table.from_pylist(final_memory_dicts)
        for attempt in range(1, 5 + 1):
            try:
                
                csv_buffer=BytesIO()
                pq.write_table(table, csv_buffer)
                csv_buffer.seek(0)

                s3.put_object(Bucket=bucket_name, Body=csv_buffer.getvalue(), Key=f"{s3_path}/{row.code}.parquet")
                print(f"Object uploaded successfully on attempt {attempt}.")
                break  
            except (BotoCoreError, ClientError) as error:
                print(f"Attempt {attempt} failed with error: {str(error)}")
                if attempt < 5:
                    sleep(delay)  
                    delay *= 2  
                else:
                    print("Max retries reached. Operation failed.")
                raise  # Re-raise the last error encountered
        print_colored(f"[Boss]: good job team you finished EscoID {row['code']} in {(time()-start_time)/60:.3f} minutes).", "cyan")


if __name__ == '__main__':
    
    # Ensure the command-line argument is cast to an integer.
    start_type = int(sys.argv[1])
    
    if start_type == 1:
        # Process odd numbers from 1 to 9.
        start = 7
        end = 10
        step = 2
    else:
        # Process even numbers from 2 to 8.
        start = 2
        end = 9
        step = 2
    
    for i in range(start, end, step):
        print(f'python main.py {i}')
        print(f'Starting with digit {i}')
        main(str(i))






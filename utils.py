import numpy as np
import pandas as pd


def print_colored(text, color):
    if color == 'red':
        print(f"\033[91m{text}\033[00m")
    elif color == 'green':
        print(f"\033[92m{text}\033[00m")
    elif color == 'yellow':
        print(f"\033[93m{text}\033[00m")
    elif color == 'blue':
        print(f"\033[94m{text}\033[00m")
    elif color == 'magenta':
        print(f"\033[95m{text}\033[00m")
    elif color == 'cyan':
        print(f"\033[96m{text}\033[00m")
    elif color == 'white':
        print(f"\033[97m{text}\033[00m")
    elif color == 'black':
        print(f"\033[98m{text}\033[00m")
    else:
        print(f"Color {color} not found")
        print(text)


def sample_skills(code_5d, df_occupations_and_skills, df_skills_count_per_occupation, global_skill_frequency):

    code_3d = int(code_5d[:-3])  # 4 digit code
    # Get the occupation-specific dataframe at 3 digit
    occupation_df = df_skills_count_per_occupation[
        df_skills_count_per_occupation['code_3d'] == str(code_3d)]

    # Determine the average number of skills associated with this occupation
    # Divide by 2 to avoid too many skills
    try:
        lambda_poisson = int(occupation_df['skill_to_job_ratio_3d'].iloc[0] / 2) ## il numero di skill da pescare deve essere al 4 digit (la somma dei 5)
    except:
        lambda_poisson = 10
        print("No skill found for", code_3d, "using default value of 10")
    n_skills_to_pick = np.random.poisson(lambda_poisson)


    # Ensure at least 1 skill is picked
    n_skills_to_pick = max(n_skills_to_pick, 1) 

    # Obtain a list of unique skills for the occupation ## modified to sample from all the skills from the upper digit in ESCO ## QUA DEVO FARLA AL TERZO, se filtra MERDA A CASO RIPORTARE AL 4
    skill_list_for_this_3d = df_occupations_and_skills.copy()
    skill_list_for_this_3d['code_3d'] = skill_list_for_this_3d['code_5d'].str[:-3]
    skill_list_for_this_3d = skill_list_for_this_3d[skill_list_for_this_3d['code_3d'] == str(code_3d)]
    unique_skills_list_per_occupation = skill_list_for_this_3d.preferredLabel_skill.unique()

    # Filter the global skills frequency for relevant occupation skills
    global_skill_frequency_filtered = global_skill_frequency[global_skill_frequency['escoskill_level_3'].isin(
        unique_skills_list_per_occupation)]

    if not global_skill_frequency_filtered.empty:

        # compute the distance between 3 digit code and code_5d
        # skill_list_for_this_3d['distance'] = skill_list_for_this_3d['code_5d'].apply(lambda x: abs(int(x.replace('.','')) - int(code_3d)*100))
        skill_list_for_this_3d['distance'] = skill_list_for_this_3d['code_5d'].str[3:].str.replace('.','') # in teoria basta fare cosi

        # merge skill_list_for_this_3d with global_skill_frequency_filtered in preferedLabel_skill and escoskill_level_3
        global_skill_frequency_filtered = global_skill_frequency_filtered.merge(skill_list_for_this_3d[['preferredLabel_skill','distance']], left_on='escoskill_level_3', right_on='preferredLabel_skill', how='inner')
        # drop duplicates in escoskill_level_3
        global_skill_frequency_filtered.drop_duplicates(subset='escoskill_level_3', inplace=True)
        
        # Adjust probabilities so they sum to 1 ## NON E' GLOBAL MA DEL QUARTO DIGIT FILTRATO AL TERZO
        prob_sum = global_skill_frequency_filtered['cnt'].sum()
        global_skill_frequency_filtered['probability_rescaled'] = global_skill_frequency_filtered['cnt'] / prob_sum ## NON E' GLOBAL MA DEL QUARTO DIGIT FILTRATO AL TERZO + PESATURA IN BASE AL DIGIT

        # normalize inverse of distance
        global_skill_frequency_filtered['distance'] = 1 / (global_skill_frequency_filtered['distance'].replace(0, np.inf).astype('float'))
        total_sum = global_skill_frequency_filtered['distance'].sum()
        global_skill_frequency_filtered['distance'] = global_skill_frequency_filtered['distance'] / total_sum
    
        a = 0.9
        global_skill_frequency_filtered['final_importance'] = global_skill_frequency_filtered['probability_rescaled'] * (1 - a) + a * (1 / global_skill_frequency_filtered['distance'].astype('float'))

        # Ensure not to sample more skills than available
        n_skills_to_pick = min(n_skills_to_pick, len(global_skill_frequency_filtered))
        print("Skills available for sampling:", len(global_skill_frequency_filtered)) 
        print(f"λ/2 (Poisson mean): {lambda_poisson}, value drawn: {n_skills_to_pick}") 

        # Sample skills based on their weighted probabilities
        selected_skills = global_skill_frequency_filtered.sample(
            n=n_skills_to_pick, weights='final_importance', replace=False)
    else:
        # If no skills match the filtering criteria, sample directly with equal probability from the df_occupations_and_skills
        selected_skills = occupation_df.sample(
            n=n_skills_to_pick, replace=False)

    return selected_skills['escoskill_level_3'] # if not global_skill_frequency_filtered.empty else selected_skills['preferredLabel_skill']


def rescale_counts(df):
    
    df['log_cnt'] = np.log(df['cnt'])

    # Rescaling function
    def scale_log_cnt(log_cnt, target_min, target_max, desired_mean):

        # Min-Max 
        min_log_cnt = df['log_cnt'].min()
        max_log_cnt = df['log_cnt'].max()
        # X - Xmin / Xmax - Xmin --> [0, 1]
        scaled_cnt = (log_cnt - min_log_cnt) / (max_log_cnt - min_log_cnt)

        # [0,1] --> [target_min, target_max]
        range_scaled = scaled_cnt * (target_max - target_min) + target_min  # [0,1] * 28 + 2 --> [2, 30]

        current_mean = range_scaled.mean()
        # [2, 30] --> [2, 30] + (10 - current_mean) --> [2, 30] + (10 - current_mean) --> [2, 30] con 10 di media
        corrected_scaled_cnt = range_scaled + (desired_mean - current_mean)
        
        # Ensure the corrected values remain within the target range
        corrected_scaled_cnt = np.clip(corrected_scaled_cnt, target_min, target_max)

        return corrected_scaled_cnt

    # Apply the scaling function to log_cnt with specified target range and desired mean
    df['rescaled_cnt'] = scale_log_cnt(df['log_cnt'], 2, 30, 10)

    # Return the DataFrame with the new column 'rescaled_cnt'
    return df

def get_rescaled_count(code_4d_value, rescaled_df):
    matching_row = rescaled_df.loc[rescaled_df['code_4d'] == code_4d_value]
    if not matching_row.empty:
        return int(matching_row['rescaled_cnt'].iloc[0])
    else:
        return 10  # Default value
    
def pick_seed_offline(code, oja_example_df):

    oja_example_df = oja_example_df[oja_example_df['found_code'] == code]
    oja_example_df['job_ad'] = oja_example_df['title'] + '. ' + oja_example_df['description']
    oja_example_list = oja_example_df['job_ad'].tolist()
    return oja_example_list if len(oja_example_list) > 0 else ["No jobs found"]


if __name__ == "__main__":

    df_counts_5d = pd.read_csv("E:/uni/thesis/data/counts_5d.csv")
    rescaled_df = rescale_counts(df_counts_5d)
    for code_5d_value in df_counts_5d.code_5d.sample(5):
        # 1324.3
        print("Rescaled count for", code_5d_value, ":", get_rescaled_count(code_5d_value))
       

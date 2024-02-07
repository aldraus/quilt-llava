# list of ["?", ".", ","] with their indices in the text
def get_punct(text):

    punct_list = [(i, char) for i, char in enumerate(text) if char in ["?", ".", "!"]]

    return punct_list

def get_full_questions(caption):   

    punct_list = get_punct(caption)

    q_list = [item for item in punct_list if item[1] == "?"]

    questions = []

    for q_idx, q_char in q_list:
        punct_list_new = [punct for punct in punct_list if punct[0] < q_idx]
        if len(punct_list_new) == 0:
            start_idx = 0
        else:
            start_idx = punct_list_new[-1][0] + 1
        end_idx = min(q_idx + 1, len(caption))
        questions.append(caption[start_idx:end_idx].strip())
  
    return questions

def dict_to_string(d):
    s = ""
    for key, value in d.items():
        s += '"' + key + '"' + ": " + '"' + value + '"' + ","
    return s[:-1]


def generate_user_msg(chunk_text, q_list):

    user_questions = chunk_text + " Sentences: ["
    for q in q_list:
        user_questions += "'" + q + "',"
        
    user_questions = user_questions[:-1] + "]"
    
    return user_questions
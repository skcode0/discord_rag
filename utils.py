from sqlalchemy_utils import database_exists, create_database
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import declarative_base, Session, sessionmaker
import os
from pathlib import Path
from dotenv import load_dotenv, set_key
import csv
import pickle
from typing import Optional, Type, Any, Dict, List, Union
import ast
import random
import string
from datetime import datetime
import re
import pandas as pd
import numpy as np

# --------------------------
# SQLAlchemy
# --------------------------

def make_pgdb(password: str, db: str, user: str="postgres", host: str="localhost", port: int=5432, add_vectors: bool=False):
    """
    If database doesn't exist, create one.
    Reference: Connect to PostgreSQL Using SQLAlchemy & Python (https://www.youtube.com/watch?v=neW9Y9xh4jc)

    - password: postgres password
    - db: database name
    - user: postgres username
    - host: host network name
    - port: database port number
    - add_vectors: enable vector embedding or not

    Returns postgres url
    """
    
    # postgresql + pschcopg3
    url = f'postgresql+psycopg://{user}:{password}@{host}:{port}/{db}'
    
    if not database_exists(url):
        create_database(url)
        print(f"Database {db} has been sucessfully created.")
    else:
        print(f"The database with '{db}' name already exists.")

    if add_vectors:
        enable_vectors(url)
    
    return url


def enable_vectors(url: str) -> None:
        """
            Add pgvectorscale to db

            - url: postgres db url
        """
        engine = create_engine(url)
        with Session(engine) as session: # will auto close
            session.execute(text("CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;")) # CASCADE will automatically install pgvector
            session.commit()
        print("Vectorscale enabled.")


Base = declarative_base()
def add_row(table: Type[Base], session: Session, data: Dict[str, Any], file_path: Optional[str] = None) -> None:
    """
    Add record.

    - table: table to add row
    - session: pg session
    - data: row data
    - file_path: csv file

    """
    # TODO: spread these and add dynamically
    time_spoken = data['time']
    speaker = data['speaker']
    text = data['text']
    embedding = data['embedding']

    try:
        session.add(table(time_spoken=time_spoken,
                            speaker=speaker,
                            text=text,
                            embedding=embedding))
        session.commit()
    except IntegrityError:
        session.rollback()

        # put failed to add record in text file for later
        #TODO: get full path, not just file path
        #TODO: get env var, or pass in env var (program session)
        write_to_csv(data=data, full_file_path=file_path)

        print(f"Error adding record. Non-added record is stored inside {file_path}.")


#TODO
def query_vector(query: List[Union[int, float]], db_url: str, search_list_size: int=100, rescore: int=50,  top_k: int=5) -> List[Dict]:
    """
    Use streamingDiskAnn to get relevant queries from short-term long-term memory (temporary db)

    - query: vectorized query input
    - db_url: url to postgres database
    - search_list_size: number of additional candidates considered during the graph search
    - rescore: re-evaluating the distances of candidate points to improve the precision of the results
    - top_k: get top k results
    """
    engine = create_engine(db_url) 
    Session = sessionmaker(bind=engine) #todo: need to do something about this session so that it doesn't affect the main session
    session = Session()

    # https://github.com/timescale/pgvectorscale/blob/main/README.md?utm_source=chatgpt.com
    session.execute(text(f"SET diskann.query_search_list_size = {search_list_size}"))
    session.execute(text(f"SET diskann.query_rescore = {rescore}")) 

    # <=> = cosine DISTANCE (1 - cosine similarity); lower the distance, the better
    sql = text("""
                WITH relaxed_results AS MATERIALIZED (
                SELECT 
                    *,
                    embedding <=> :embedding AS distance
                FROM vectors
                ORDER BY distance
                LIMIT :limit)
               
                SELECT * 
                FROM relaxed_results 
                ORDER BY distance;
               """)
    
    params = {
        'embedding': str(query),  # seems like vector embedding needs to be passed in as string
        'limit': top_k
    }

    result = session.execute(sql, params)
    rows = result.fetchall()

    columns = result.keys()
    
    session.close()
    return [dict(zip(columns, row)) for row in rows]

# TODO
def shutdown_protocol():
    """
    When program stops, do this.
    """
    # write all data to long-term db

    # delete temp database rows

    # close session
    pass


# --------------------------
# Folder/File Saving
# --------------------------
def create_program_session_dir() -> None:
    """
    Creates a folder with a program session name.
    """
    path_dir = './db/storage'

    load_dotenv()

    session_name = os.environ.get('PROGRAM_SESSION')

    if session_name:
        acceptable_ans = {"yes", "y", "no", "n"}
        current_str = f"Your current program session name is: {session_name}. If you would like to keep the current session name, type yes (y). If not, type no (n): "

        ans = validate_ans(acceptable_ans=acceptable_ans,
                           question=current_str)

        if ans in {"yes", "y"}:
            print(f"Keeping program session name: {session_name}.")
            return
        else: # {no, n}
            note_str = "Note that session name will be saved in lowercase letters. Also, if left empty, session name will be randomly generated alphanumeric string and today's date (ex. 'as30k1mm_3-27-2025'): "

            session_name = input("Create a session name (for file saving). " + note_str).lower().strip()
            
            if session_name == "":
                session_name = create_session_name()
                # for some reason, there may be existing session folders while PROGRAM_SESSION=''
                session_name = check_dir(path_dir=path_dir,
                                         session_name=session_name)
            else:
                add_date = validate_ans(acceptable_ans=acceptable_ans,
                                    question="Add today's date at the end (y,n)? ")
                
                if add_date in {"yes", "y"}:
                    session_name = append_date(name=session_name)

                validity_message = """
                Your file name is invalid for windows file system. You CANNOT have:
                    - special characters: <>:"/\\|?*
                    - trailing spaces or periods
                    - reserved Windows names (CON, PRN, AUX, NUL, COM1-COM9, LPT1-LPT9)
                """
                while not is_valid_windows_name(session_name):
                    session_name = input(validity_message + "\nCreate a session name (for file saving). " + note_str).lower().strip()

                session_name = check_dir(path_dir=path_dir,
                                         session_name=session_name)
    else:
        session_name = create_session_name()
        # for some reason, there may be existing session folders while PROGRAM_SESSION=''
        session_name = check_dir(path_dir=path_dir,
                                 session_name=session_name)

    full_path = os.path.join(path_dir, session_name)
    if not os.path.isdir(full_path):
        set_key(".env", "PROGRAM_SESSION", session_name)
        os.mkdir(full_path)

        # make pickle to keep track of file numbering
        create_pickle_file(dir_path=full_path,
                           filename="file_num",
                           data={})

        print(f"{session_name} session folder created inside {path_dir}.")
    else:
        if os.environ.get('PROGRAM_SESSION') != session_name:
            set_key(".env", "PROGRAM_SESSION", session_name)
            print(f"Current session changed: {os.environ.get('PROGRAM_SESSION')} --> {session_name}")
        else:
            print(f"Keeping program session name: {os.environ.get('PROGRAM_SESSION')}.")


def create_pickle_file(dir_path:str="/", filename:str="pickle", data:dict={}) -> None:
    """
    Create pickle file.

    - dir_path: directory path
    - filename: pickle file name
    - data: data to save inside pkl   

    """
    try:
        filename = check_filename(filename, ".pkl")
    except Exception as e:
        print(f"{e} Unable to create pickle file.")

    with open(os.path.join(dir_path, filename), 'wb') as file:
        pickle.dump(data, file)


def check_filename(filename:str, correct_ext:str) -> str:
    """
    Checks if file name is valid.

    - filename: name of file (can include extension)
    - correct_ext: correct extension (assumes the correct extension name has been passed)

    Returns checked filename or invalid message.
    """
    filename, ext = os.path.splitext(filename.strip())

    correct_ext = correct_ext.strip()
    if not correct_ext.startswith("."):
        correct_ext = "." + correct_ext

    if ext.lower() != correct_ext.lower():
        ext = correct_ext
    
    if is_valid_windows_name(filename):
        return filename + ext
    else:
        raise ValueError("Invalid file name.")


def create_session_name(str_len: int = 8) -> str:
        """
        Creates a randomly generated alphnumeric session name

        - str_len: length of randomly generated string

        Returns a string of randomly generated name + local datetime
        """
        alphanum_chars = string.ascii_lowercase + string.digits
        today = datetime.today().strftime('%Y-%m-%d')

        session_name = "".join(random.choices(alphanum_chars, k=str_len)) + "_" + today

        return session_name


def validate_ans(acceptable_ans: Union[list, set], question: str) -> str:
    """
    Keep asking for valid user input.

    - acceptable_ans: list/set of acceptable anser choices
    - question: question for user 

    Returns valid user input
    """

    ans = input(question).lower().strip()

    while ans not in acceptable_ans:
        ans = input(f"Only acceptable answers are: {acceptable_ans}. " + question).lower().strip()
    
    return ans


def append_date(name: str, format: str='%Y-%m-%d', date: str=None) -> str:
    """
    Append date to end of name

    - name: string that needs to have date appended at the end
    - format: date/time format
    - date: specific date to be appended. If not specified, it will default to today's datetime.

    Returns name_timedate
    """
    if not date:
        date = datetime.now() # date + time
    
    formatted_date = date.strftime(format)

    return name + "_" + formatted_date


def is_valid_windows_name(name:str) -> bool:
    """
    Validates if the directory/file name is valid

    - name: folder/file name

    Returns bool that check if the name is valid or not
    """
    if not name:
        return False

    if name.endswith(' ') or name.endswith('.'):
        return False
    
    base_name = os.path.splitext(name)[0].upper()
    win_reserved = {"CON", "PRN", "AUX", "NUL",
                "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
                "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"}
    if base_name in win_reserved:
        return False
    
    invalid_chars = set('<>:"/\\|?*')
    if any(char in invalid_chars for char in name):
        return False

    return True


def check_dir(path_dir:str, session_name:str) -> str:
    """
    Check if folder already exist

    - path_dir: path of folder
    - session_name: name of session folder

    Returns valid, unique folder name
    """
    validity_message = """
    Your file name is invalid for windows file systen. You CANNOT have:
        - special characters: <>:"/\\|?*
        - trailing spaces or periods
        - reserved Windows names (CON, PRN, AUX, NUL, COM1-COM9, LPT1-LPT9)
    """

    while os.path.isdir(os.path.join(path_dir, session_name)):
        new_name = input(f"Folder named '{session_name}' already exists in storage folder. If you want to keep using {session_name}, confirm with '/keep', else, give a new session name. Note that session name will be saved in lowercase letters. Also, if left empty, session name will be randomly generated alphanumeric string and today's date (ex. 'as30k1mm_3-27-2025'): ")  

        new_name = new_name.lower().strip()
        if new_name == '/keep':
            break
        else:
            if new_name == "":
                new_name = create_session_name()
            else:
                acceptable_ans = {"yes", "y", "no", "n"}
                add_date = validate_ans(acceptable_ans=acceptable_ans,
                                        question="Add today's date at the end (y,n)? ")
                    
                if add_date in {"yes", "y"}:
                    session_name = append_date(name=new_name)

            while not is_valid_windows_name(new_name):
                    new_name = input(validity_message + "\nGive a valid session name: ").lower().strip()
            session_name = new_name
        
    return session_name


def write_to_csv(full_file_path: str, 
                 data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> None:
    """
    Write (list of) data dict to csv.

    - full_file_path: full file path. Note that csv file doesn't need to exist, but folder(s) must exist.
    - data: (list of) dict to be added
    """

    if not full_file_path.endswith(".csv"):
        full_file_path += ".csv"
    
    single_row = True
    if isinstance(data, list):
        single_row = False
    

    no_data = False
    file_exists = os.path.isfile(full_file_path)

    if single_row:
        no_data = not data
    else:
        no_data = not data[0] # assumes only based on first data (for speed). Use any() or all() for more accurcy
        

    if no_data and not file_exists:
        # create empty file
        with open(full_file_path, 'w', newline='', encoding='utf-8') as f:
            pass
        return
    elif no_data and file_exists:
        return


    # Only reads first line for header
    has_header = False
    if file_exists:
        with open(full_file_path, "r", newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header:
                has_header = True


    if single_row:
        fieldnames = data.keys()
    else:
        fieldnames = data[0].keys()

    # append data
    with open(full_file_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # if file doesn't have header, add header
        if not has_header:
            writer.writeheader()

        if single_row:
            writer.writerow(data) # singular
        else:
            writer.writerows(data) # multiple


def name_and_write_to_csv(data: Union[Dict[str, Any], List[Dict[str, Any]]] = {},
                    file_path: str = "./db/storage",
                    file_name: str = "output.csv",
                    session_name: Optional[str] = None,
                    add_date: bool = False, 
                    auto_increment: bool = False) -> str:
    """
    Creates csv file. Based on the parameters, it will try to automate file naming and add data accordingly. 
    ex. If example_1.csv file exists, example_2.csv will be made if auto_increment=True. If auto_increment=False, it will data append to example_1.csv. 
    Note that if it cannot find directory, it will create a folder. Moreover, by default, to keep track of file numbering, pickle file will be created for each directory.
    HIGHLY RECOMMEND this format if providing date and file numbering manually: filename_YYYY-MM-DD_filenum 

    - data: (list of) data dict to add
    - file_path: path to save csv file
    - file_name: name of csv file. Avoid typing date. Use add_date parametere instead if creating new. If adding date manually, use this format: "YYYY-MM-DD" and append it at the end using "_".
    - session_name: EXISITNG program session name (in .env). If none, current one in .env file will be used. 
    - add_date: add date to file name or not. If session name is not provided and session name is empty inside .env, date will be added (False will be overriden).
    - auto_increment: if file name already exists, append or create a new one. If True, it will create a new file with incremental number at the end. ex. If test_1.csv exists and auto_increment=True, then test_2.csv will be created. If auto_increment=True, it will append to test_1.csv.

    Returns full path of created file.
    """
    load_dotenv()
    
    if not session_name:
        session_name = os.environ.get('PROGRAM_SESSION')

        # unnamed session name will automatically get date appended
        if session_name == "":
            session_name = "unnamed_session"
            unnamed_path = os.path.join(file_path, session_name)
            if not os.path.isdir(unnamed_path):
                os.mkdir(unnamed_path)
            add_date = True
    
    if not os.path.isdir(os.path.join(file_path, session_name)):
        raise FileNotFoundError(f"Program session folder ({os.path.join(file_path, session_name)}) does not exist. Create a session folder at {file_path}")

    file_name = check_filename(file_name, ".csv")
    
    file_name, ext = os.path.splitext(file_name)
    if ext != ".csv":
        ext = ".csv"

    if add_date:
        date_pattern = r'\d{4}-\d{2}-\d{2}'
        match = re.search(date_pattern, file_name)
        if match:
            add_date = False
        else:
            file_name = append_date(file_name)


    file_num = 1
    num_pattern = r'_([0-9]+)$'
    match = re.search(num_pattern, file_name)
    if match:
        file_num = int(match.group(1)) 
        file_name = file_name[:-len(match.group())]

    pkl_name = "file_num.pkl"
    pkl_file_path = os.path.join(file_path, session_name, pkl_name)
    pickle_data = dict()
    
    
    if not match:
        # If the session folder has no pickle file but has .csv files, pickle file created at this point will not know about these csv files. So, the csv files may be overwritten.
        if os.path.isfile(pkl_file_path):
            with open(pkl_file_path, "rb") as file:
                pickle_data = pickle.load(file) # dict {file_path: file_num}
                pickle_name = os.path.join(session_name, file_name+ext)
                if pickle_name in pickle_data:
                    file_num = pickle_data[pickle_name]
    

    full_file_name = None
    if os.path.isfile(os.path.join(file_path, session_name, file_name + ext)) and not add_date and not auto_increment:
        full_file_name = file_name + ext
    else:
        full_file_name = file_name + "_" + str(file_num) + ext

    full_path = os.path.join(file_path, session_name, full_file_name)
    # only increment file number. Don't try to fill the gap in.
    while os.path.isfile(full_path):
        if not auto_increment:
            print(f"Using existing file at {full_path}")
            break
 
        file_num += 1

        full_file_name = file_name + "_" + str(file_num) + ext 
        full_path = os.path.join(file_path, session_name, full_file_name)

    pickle_data[os.path.join(session_name, file_name+ext)] = file_num

    with open(pkl_file_path, "wb") as file:
        pickle.dump(pickle_data, file)

    write_to_csv(full_file_path=full_path,
                 data=data)

    return full_path


def update_file_num_pkl(dir_path: str = './',
                        delimiter: str = "_") -> str:
    """
    Check and update file numbering stored in pickle.
    Useful when file(s)/folder(s) or pickle deleted and need update/reconstruction.
    HIGHLY RECOMMEND files have this format: {file_name}_{date}_{file_num}
    File number MUST be at the end.

    - dir_path: directory to check.
    - delimiter: character used to separate sections (filename, date, file numbering, etc.)
    """
    fs = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]

    not_added = []

    d = dict()
    for f in fs:
        base, ext = os.path.splitext(f)
        
        # checks if there's name and file_num
        # ex. folder 'example' when split will not have file_num
        try:
            file_name, file_num = base.rsplit(delimiter, 1)
            # any decimal numbers (ex. 3.1) will be turned to int
            file_num = int(file_num)
        except:
            file_name = base
            not_added.append(f)
            continue
            

        path_f = Path(rf"{f}")
        direct_parent = path_f.parent.name

        d_name = os.path.join(direct_parent, file_name+ext)
        if d_name not in d:
            d[d_name] = 1
        else:
            d[d_name] = file_num
    
    pkl_name = "file_num.pkl"
    pkl_file_path = os.path.join(dir_path, pkl_name)
    # will replace if there's already a file with same name
    with open(pkl_file_path, "wb") as file:
        pickle.dump(d, file)
    
    print(f"file numbering updated at {pkl_file_path}. {f"These files are not added: {not_added}" if not_added else ""}")

    return pkl_file_path
        

def csv_to_pd(filepath) -> pd.DataFrame:
    """
    Return csv as pandas dataframe

    - filepath: full csv file path

    Returns pandas dataframe
    """
    if not filepath.endswith(".csv"):
         filepath += ".csv"

    df = pd.read_csv(filepath)

    return df

def str_to_vec(s: str) -> np.array:
    """
    Convert to vector (numpy array)

    - s: string version of vectors
    
    Returns numpy array vector
    """
    clean = s.strip("[]")

    return np.fromstring(clean, sep=",")


#TODO pandas --> postgres 
def pd_to_postgres():
    """
    
    """



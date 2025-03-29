from sqlalchemy_utils import database_exists, create_database
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import declarative_base, Session, sessionmaker
import os
from dotenv import load_dotenv, set_key
import csv
import pickle
from typing import Optional, Type, Any, Dict, List, Union
import ast
import random
import string
from datetime import datetime
from collections import defaultdict

# --------------------------
# SQLAlchemy
# --------------------------

def make_pgdb(password: str, db: str, user: Optional[str]="postgres", host: Optional[str]="localhost", port: Optional[int]=5432, add_vectors: Optional[bool]=False):
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
def add_row(table: Type[Base], session: Session, info: Dict[str, Any], file_path: Optional[str] = None) -> None:
    """
    Add record.

    - table: table to add row
    - session: pg session
    - info: row data
    - file_path: csv file

    """
    # TODO: spread these and add dynamically
    time_spoken = info['time']
    speaker = info['speaker']
    text = info['text']
    embedding = info['embedding']

    try:
        session.add(table(time_spoken=time_spoken,
                            speaker=speaker,
                            text=text,
                            embedding=embedding))
        session.commit()
    except IntegrityError:
        session.rollback()

        # put failed to add record in text file for later
        write_row_to_csv(info, file_path)

        print(f"Error adding record. Non-added record is stored inside {file_path}.")


#TODO
import numpy as np
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
def name_program_session() -> None:
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
        os.mkdir(os.path.join(path_dir, session_name))

        # make pickle to keep track of file numbers
        #TODO: replace with func
        d = defaultdict(int)
        pkl_name = "file_num.pkl"
        with open(os.path.join(full_path, pkl_name), 'wb') as file:
            pickle.dump(d, file)

        print(f"{session_name} session folder created inside {path_dir}.")
    else:
        if os.environ.get('PROGRAM_SESSION') != session_name:
            set_key(".env", "PROGRAM_SESSION", session_name)
            print(f"Current session changed: {os.environ.get('PROGRAM_SESSION')} --> {session_name}")
        else:
            print(f"Keeping program session name: {os.environ.get('PROGRAM_SESSION')}.")

#TODO: finish this
def create_pickle_file(dir_path:Optional[str]="/", filename:Optional[str]="pickle", data:Optional[Any]=None) -> None:
    """
    Create pickle file.

    - dir_path: directory path
    - filename: pickle file name
    - data: data to save inside pkl   

    """



def check_filename(filename:str, correct_ext:str) -> str:
    """
    Correct file name

    - filename: name of file (can include extension)
    - correct_ext: correct extension

    Returns corrected file name
    """
    filename, ext = os.path.splitext(filename.strip())

    correct_ext = correct_ext.lower().strip()
    if not correct_ext.startswith("."):
        correct_ext = "." + correct_ext

    if ext.lower() != correct_ext:
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

    Returns valid user input
    """

    ans = input(question).lower().strip()

    while ans not in acceptable_ans:
        ans = input(f"Only acceptable answers are: {acceptable_ans}. " + question).lower().strip()
    
    return ans


def append_date(name: str, format: Optional[str]='%Y-%m-%d', date: Optional[str]=None) -> str:
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


def is_valid_windows_name(name) -> bool:
    """
    Validates if the directory/file name is valid

    - name: folder/file name

    Returns bool that check if the name is valid or not
    """
    invalid_chars = set('<>:"/\\|?*')
    if any(char in invalid_chars for char in name):
        return False
    
    if name.endswith(' ') or name.endswith('.'):
        return False
    
    base_name = os.path.splitext(name)[0].upper()
    win_reserved = {"CON", "PRN", "AUX", "NUL",
                "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
                "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"}
    if base_name in win_reserved:
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


# TODO: get session folder?
def write_row_to_csv(data: Union[Dict[str, Any], List[Dict[str, Any]]],
                     file_path: Optional[str] = "/", 
                     file_name: Optional[str] = "output.csv",
                     session_name: Optional[str] = None,
                     add_date: Optional[bool] = True, 
                     auto_increment: Optional[bool]=False) -> None:
    """
    Write dict of data to csv

    - data: row (dict) to be added
    - file_path: path of csv file
    - file_name: name of csv file
    - session_name: program session name (in .env). If none, current one in .env file will be used.
    - add_date: add date to file name or not. If session name is not provided and session name is empty inside .env, date will be added (False will be overriden).
    - auto_increment: if file name already exists, append or create a new one. If True, it will create a new file with incremental number at the end. ex. If test_1.csv exists and auto_increment=True, then test_2.csv will be created. If auto_increment=True, it will append to test_1.csv.
    """
    load_dotenv()
    
    if not session_name:
        session_name = os.environ.get('PROGRAM_SESSION')

        # unnamed session name will automatically get date appended
        if session_name == "":
            session_name = "unnamed_session"
            add_date = True

    single_row = True
    if isinstance(data, list):
        single_row = False
    
    file_name, ext = os.path.splitext(file_name)
    if ext.lower() != ".csv":
        ext = ".csv"

    if add_date:
        file_name = append_date(file_name)
    
    full_file_name = file_name + "_" + file_num + ext 
    full_path = os.path.join(file_path, session_name, full_file_name)
    

    pkl_file_path
    file_num_dict = 

    new_file = True
    while os.path.isfile(full_path):
        if not auto_increment:
            new_file = False
            break
        
        file_num = int(file_num) + 1
        set_key(".env", "FILE_NUM", str(file_num))

        full_file_name = file_name + "_" + file_num + ext 
        full_path = os.path.join(file_path, session_name, full_file_name)   

    if single_row:
        fieldnames = data.keys()
    else:
        fieldnames = data[0].keys()

    #TODO
    # append data
    with open(full_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # if file doesn't exist, add header
        if not file_exists:
            writer.writeheader()

        if single_row:
            writer.writerow(data) # singular
        else:
            writer.writerows(data) # multiple


def csv_to_dict(file) -> list[dict]:
    """
    Read csv file

    - file: csv file name

    Returns list of dict
    """
    if not file.endswith(".csv"):
         file += ".csv"

    with open(file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
    
        data = []
        for row in reader:
            for k,v in row.items():
                try: 
                    row[k] = ast.literal_eval(v) # convert text back to actual data type
                except (ValueError, SyntaxError):
                    pass # keep as string if conversion fails
        data.append(row)

    return data

#TODO: make another function for csv_to_pandas
#TODO pandas --> postgres 


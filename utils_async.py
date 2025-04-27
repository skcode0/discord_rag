from sqlalchemy import text
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy_utils import database_exists, create_database
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from asyncio.subprocess import PIPE
import asyncio
import aiofiles
import aiocsv
import os
from pathlib import Path
from dotenv import load_dotenv, set_key
import csv
import pickle
from typing import Optional, Type, Any, Dict, List, Union, Hashable, Callable, Literal
import random
import string
from datetime import datetime
import re
import pandas as pd
import numpy as np
from pandas.io.parsers import TextFileReader
import logging
# from sentence_transformers import SentenceTransformer


# --------------------------
# SQLAlchemy
# --------------------------
class AsyncPostgresDataBase:
    def __init__(self,
                 password: str,
                 db_name: str,
                 port: int = 5432,
                 user: str = "postgres",
                 host: str = "localhost",
                 pool_size: int = 50,
                 echo: bool = False,
                 hide_parameters: bool = False):
        self.password = password 
        self.db_name = db_name
        self.port = port
        self.user = user
        self.host = host
        self.pool_size = pool_size
        self.echo = echo
        self.hide_parameters = hide_parameters

        self.url = f'postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.db_name}'

        self.engine = create_async_engine(self.url, pool_size=self.pool_size, echo=self.echo, hide_parameters=self.hide_parameters)

        self.Session = async_sessionmaker(self.engine, expire_on_commit=False)

    #TODO: fix
    async def make_db(self) -> None:
        """
        If database doesn't exist, create one.
        Reference: Connect to PostgreSQL Using SQLAlchemy & Python (https://www.youtube.com/watch?v=neW9Y9xh4jc)

        Returns postgres url
        """
        async with self.engine.connect() as conn:
            result = await conn.execute(
                text("SELECT 1 FROM pg_catalog.pg_database WHERE datname = :dbname"), 
                {"dbname": self.db_name})
            exists = result.scalar() is not None

            if not exists:
                await conn.execute(text(f"CREATE DATABASE {self.db_name}"))
                print(f"Database {self.db_name} has been sucessfully created.")
            else:
                print(f"The database with '{self.db_name}' name already exists.")
    
        
    async def enable_vectors(self) -> None:
        """
        Adds pgvectorscale to db
        """
        async with self.Session() as session:
            await session.execute(text("CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;")) # CASCADE will automatically install pgvector
            await session.commit()
        print("Vectorscale enabled.")

    
    async def add_record(self, table: Type[DeclarativeBase], data: Dict[str, Any]) -> None:
        """
        Add record. If record could not be added, it will raise error. 

        - table: table to add record
        - data: record data. If the data dict's keys don't have the same name as the table name or there's more keys than column names, it will raise error. If there are less keys than columns, then depending on whether the column is nullable or not, it will add null or raise (IntegrityError) error.
        
        """
        try:
            async with self.Session() as session: # auto-closes session
                await session.add(table(**data))
                await session.commit()
        except DBAPIError as e:
            await session.rollback()
            err = format_db_error(e)
            raise RuntimeError(err)


    async def query_vector(self, 
                     query: List[Union[int, float]],
                     join: bool = False,
                     search_list_size: int=100,
                     rescore: int=50,
                     top_k: int=5) -> List[Dict]:
        """
        Uses streamingDiskAnn and cosine distance to get the most relevant query answers.

        - query: vectorized query input
        - join: join 'vectors' and 'transcriptions' table or not
        - search_list_size: number of additional candidates considered during the graph search
        - rescore: re-evaluating the distances of candidate points to improve the precision of the results
        - top_k: get top k results
        """  

        # <=> is cosine DISTANCE (1 - cosine similarity); lower the distance, the better
        # Note: pgvectorscale currently supports: cosine distance (<=>) queries, for indices created with vector_cosine_ops; L2 distance (<->) queries, for indices created with vector_l2_ops; and inner product (<#>) queries, for indices created with vector_ip_ops. This is the same syntax used by pgvector.
        sql = ""
        #! Change table name(s) as needed
        if join:
            sql = text("""
                        WITH relaxed_results AS MATERIALIZED (
                        SELECT 
                            timestamp,
                            speaker,
                            text,
                            embedding <=> :embedding AS distance
                        FROM vectors v INNER JOIN transcriptions t
                            ON v.transcription_id = t.id 
                        ORDER BY distance
                        LIMIT :limit)
                    
                        SELECT * 
                        FROM relaxed_results 
                        ORDER BY distance;
                    """)            
        else:
            sql = text("""
                        WITH relaxed_results AS MATERIALIZED (
                        SELECT 
                            timestamp,
                            speaker,
                            text,
                            embedding <=> :embedding AS distance
                        FROM transcriptionsvectors
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

        async with self.Session() as session:
            # https://github.com/timescale/pgvectorscale/blob/main/README.md?utm_source
            await session.execute(text(
            "SET diskann.query_search_list_size = :size; "
            "SET diskann.query_rescore = :rescore"), {"size": search_list_size, "rescore": rescore})

            result = await session.execute(sql, params)
            rows = await result.fetchall()
            columns = result.keys()
        
        return [dict(zip(columns, row)) for row in rows]

    
    async def delete_all_rows(self, tablename) -> None:
        """
        Deletes all rows in a table. Table won't be deleted. Slower than TRUNCATE because it deletes row by row. However, it's safer for data integrity and triggers.
        
        - tablename: name of table to delete

        """
        async with self.Session() as session:
            async with session.begin():
                await session.execute(text(f"DELETE FROM {tablename};"))


    async def truncate_all_rows(self, tablename) -> None:
        """
        Truncates all rows in a table. Table won't be deleted. Faster than DELETE but can be harder to log or rollback. 

        - tablename: name of table to truncate
        
        """
        async with self.Session() as session:
            async with session.begin():
                await session.execute(text(f"TRUNCATE TABLE {tablename};"))


    async def pandas_to_postgres(self, 
                           df: pd.DataFrame,
                           table_name: str,
                           if_exists: str = "append",
                           index: bool = False,
                           dtype = None,
                           method: Optional[Union[Literal['multi'], Callable]] = "multi") -> None:
        """
        Sends pandas dataframe/iterator to postgres. Also writes a log.

        - df: dataframe
        - table_name: name of table to add data to
        - if_exists: if table exists, "fail", "replace", or "append". Make sure to have a table with proper relationships/contraints or to_sql will create a new table. 
        - index: write index as column or not
        - dtype: dtype of column(s)
        - method: method to insert rows (none = one per row; multi = multiple values in single INSERT; callable with signature (pd_table, conn, keys, data_iter))

        """
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(lambda sync_conn: 
                    df.to_sql(
                        name=table_name,
                        con=sync_conn,
                        if_exists=if_exists,
                        index=index,
                        dtype=dtype,
                        method=method
                    )
                )
        except DBAPIError as e:
            err = format_db_error(e)
            raise RuntimeError(err)
        

    async def postgres_to_csv(self, 
                        table_name: str, 
                        output_path: str) -> None:
        """
        Export Postgres table to csv file.

        - table_name: name of table to export
        - output_path: path for csv output.

        """
        try:
            copy_sql = f"COPY {table_name} TO STDOUT WITH CSV HEADER"

            with self.engine.connect() as conn:
                async with conn.begin():
                    raw_conn = await conn.get_raw_connection()
                
                # write to file
                async with aiofiles.open(output_path, mode="wb") as f:
                    async with raw_conn.cursor() as cur:
                        async with cur.copy(copy_sql) as stream:
                            async for chunk in stream:
                                await f.write(chunk)
            
            print(f"Backup created at {output_path}")
        except Exception as e:
            print("Backup error:", e)


    async def dump_postgres(self, 
                            backup_path: str, 
                            database_name: str, 
                            F: str = "p", 
                            blob: bool = True, 
                            compress: bool = True, compress_level: Optional[int] = None) -> None:
        """
        Uses subprocess to dump postgres database. 

        - backup_path: path for backup file output
        - database_name: name of database to back up
        - F: format of backup. Defaults to (p)lain. Other formats will require pg_restore when restoring db.
        - blob: include large objects (BLOBs) or not
        - compress:  
        - compress_level: If compress = True, what level should the compress be (0-9)? When None, it will default to 6.
            
        """
        cmd = ["pg_dump", "-F", F]

        if blob:
            cmd.append("-b")
        
        if F.lower() in {"c", "d"}:
            if compress:
                if compress_level is not None:
                    cmd += ['-Z', str(compress_level)]
            else:
                cmd += ['-Z', '0']

        cmd += ["-f", backup_path]
        cmd.append(database_name)
        
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=PIPE,
            stderr=PIPE,
            text=True
        )

        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(f"pg_dump failed: {stderr.strip()}")
        else:
             print(f"Backup completed successfully. Output saved to {backup_path}")


# --------------------------
# Folder/File Saving and Logging
# --------------------------
windows_filename_validity_message = """ 
Your file name is invalid for windows file system. You CANNOT have:
- special characters: <>:"/\\|?*
- trailing spaces or periods
- reserved Windows names (CON, PRN, AUX, NUL, COM1-COM9, LPT1-LPT9)
"""

def create_program_session_dir() -> str:
    """
    Creates a folder with a program session name.

    Returns program session name
    """
    path_dir = './db/storage'

    load_dotenv(override=True)

    session_name = os.environ.get('PROGRAM_SESSION')
    
    # session name exists but actual folder doesn't
    if not os.path.isdir(os.path.join(path_dir, session_name)):
        session_name = ""

    true_options = {"yes", "y"}
    false_options= {"no", "n"}
    if session_name:
        current_str = f"Your current program session name is: {session_name}. Would you like to keep the current session name? (y/n): "

        ans = input_to_bool(question=current_str, true_options=true_options, false_options=false_options)

        if ans:
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
                add_date = input_to_bool(question="Add today's date at the end? (y/n): ", true_options=true_options, false_options=false_options)
                
                if add_date:
                    session_name = append_date(name=session_name)

                while not is_valid_windows_name(session_name):
                    session_name = input(windows_filename_validity_message + "\nCreate a session name (for file saving). " + note_str).lower().strip()

                session_name = check_dir(path_dir=path_dir,
                                         session_name=session_name)
    else:
        current_str = "Would you like to create a custom name for the folder? (y/n): "
        ans = input_to_bool(question=current_str, true_options=true_options, false_options=false_options)

        if ans:
            note_str = "Note that session name will be saved in lowercase letters. Also, if left empty, session name will be randomly generated alphanumeric string and today's date (ex. 'as30k1mm_3-27-2025'): "

            session_name = input("Create a session name (for file saving). " + note_str).lower().strip()
            
            if session_name == "":
                session_name = create_session_name()
                # for some reason, there may be existing session folders while PROGRAM_SESSION=''
                session_name = check_dir(path_dir=path_dir,
                                         session_name=session_name)
            else:
                add_date = input_to_bool(question="Add today's date at the end? (y/n): ", true_options=true_options, false_options=false_options)
                
                if add_date:
                    session_name = append_date(name=session_name)

                while not is_valid_windows_name(session_name):
                    session_name = input(windows_filename_validity_message + "\nCreate a session name (for file saving). " + note_str).lower().strip()

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

    return session_name


def create_pickle_file(dir_path: str = "/", filename: str = "pickle", data: dict = {}) -> None:
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


def check_filename(filename: str, correct_ext: str) -> str:
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

    - acceptable_ans: list/set of acceptable answer choices
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
                    new_name = input(windows_filename_validity_message + "\nGive a valid session name: ").lower().strip()
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
        if not data: # empty list
            no_data = True
        else:
            no_data = not data[0] # assumes only based on first data (for speed). Use any() or all() for more accuracy
        

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


async def write_to_csv_async(full_file_path: str, 
                 data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> None:
    """
    Write (list of) data dict to csv in async.
    Note: making this async doesn't necessarily mean it will be faster. It can be slower than syncronous version due to overhead. However, async doesn't block the event loop.

    - full_file_path: full file path. Note that csv file doesn't need to exist, but folder(s) must exist.
    - data: (list of) dict to be added
    """

    if not full_file_path.endswith(".csv"):
        full_file_path += ".csv"
    
    single_row = True
    if isinstance(data, list):
        single_row = False
    
    no_data = False
    file_exists = await asyncio.to_thread(os.path.isfile, full_file_path)

    if single_row:
        no_data = not data
    else:
        if not data: # empty list
            no_data = True
        else:
            no_data = not data[0] # assumes only based on first data (for speed). Use any() or all() for more accuracy
        

    if no_data:
        if not file_exists:
            # create empty file
            async with aiofiles.open(full_file_path, mode='w', newline='', encoding='utf-8') as f:
                pass
        return
    

    # Only reads first line for header
    has_header = False
    if file_exists:
        async with aiofiles.open(full_file_path, mode="r", newline='', encoding='utf-8') as f:
            reader = aiocsv.AsyncReader(f)
            async for row in reader:
                if row:
                    has_header = True
                break

    if single_row:
        fieldnames = data.keys()
    else:
        fieldnames = data[0].keys()

    # append data
    async with aiofiles.open(full_file_path, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = aiocsv.AsyncDictWriter(csvfile, fieldnames=fieldnames)

        # if file doesn't have header, add header
        if not has_header:
            await writer.writeheader()

        if single_row:
            await writer.writerow(data) # singular
        else:
            await writer.writerows(data) # multiple


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
    load_dotenv(override=True)
    
    if not session_name:
        session_name = os.environ.get('PROGRAM_SESSION')

        if session_name == "":
            create_program_session_dir()
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
        except Exception as e:
            print("Error", e)
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
        

def csv_to_pd(filepath: str, 
              chunksize: Optional[int] = None,
              parse_dates: Union[
                  bool,
                  List[Hashable],
                  List[List[Hashable]],
                  Dict[Hashable, List[Hashable]]] = False,
              date_format: Optional[Union[str, dict]] = None,
              compression: Union[str, dict] = "infer") -> Union[pd.DataFrame, TextFileReader]:
    """
    Converts csv to pandas dataframe/iterator.

    - filepath: full csv file path
    - chunksize: how many rows/data per chunk. Default is None. If size defined, it will return TextFileReader, a iterable pandas chunks.
    - parse_dates: parse date or not (auto inference for python 2.0+)
    - date_format: give specific date format to adhere to. Don't use when the dates have multiple formats.
    - compression: file compression

    Returns either pandas dataframe or iterable pandas chunks.
    """
    df = pd.read_csv(filepath, 
                     chunksize=chunksize,
                     parse_dates=parse_dates,
                     date_format=date_format,
                     compression=compression)

    return df


def str_to_vec(s: str, to_list=False) -> Union[np.ndarray, list]:
    """
    Convert to vector (numpy array)

    - s: string version of vectors
    - to_list: convert to list or not
    
    Returns numpy array or list
    """
    clean = s.strip("[]")

    result = np.fromstring(clean, sep=",")

    if to_list:
        result = result.tolist()

    return result


LogLevelStr = Literal["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LogLevelInt = Literal[0, 10, 20, 30, 40, 50]
LogLevel = Union[LogLevelStr, LogLevelInt]
def setLogger(name: str = __name__,
              setLevel: LogLevel = "DEBUG") -> logging.Logger:
    """
    Sets up logger config.

    - name: name for logger
    - setLevel: log level ("NOTSET" (0), "DEBUG" (10), "INFO" (20), "WARNING" (30), "ERROR" (40), "CRITICAL" (50))
    
    Returns configured Logger
    """
    logger = logging.getLogger(name)

    if isinstance(setLevel, str):
        setLevel_num = getattr(logging, setLevel.upper(), None)
        if setLevel_num is None:
            raise ValueError(f"Invalid log level: {setLevel}")
        setLevel = setLevel_num
    else:
        if setLevel not in (0, 10, 20, 30, 40, 50):
            raise ValueError(f"Invalid log level number: {setLevel}")

    logger.setLevel(setLevel)

    return logger


def setLogHandler(log_dir: str = './',
                  log_filename: str = 'log.log',
                  mode: str = 'a',
                  setLevel: LogLevel = 'DEBUG',
                  fmt: str = '%(asctime)s - %(levelname)s - %(message)s') -> logging.FileHandler:
    """
    Sets handler for logger.

    - logger: logger
    - log_dir: directory where log file is saved
    - log_filename: file name of log file
    - mode: how to deal with log file, (a)ppend, over(w)rite, e(x)clusive creation 
    - setLevel: log level ("NOTSET" (0), "DEBUG" (10), "INFO" (20), "WARNING" (30), "ERROR" (40), "CRITICAL" (50))
    - fmt: format string for log messages. See the official documentation: https://docs.python.org/3/library/logging.html#logrecord-attributes

    Returns configured FileHandler
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    log_fullpath = log_path / log_filename

    handler = logging.FileHandler(log_fullpath, mode=mode)
    if isinstance(setLevel, str):
        setLevel_num = getattr(logging, setLevel.upper(), None)
        if setLevel_num is None:
            raise ValueError(f"Invalid log level: {setLevel}")
        setLevel = setLevel_num
    else:
        if setLevel not in (0, 10, 20, 30, 40, 50):
            raise ValueError(f"Invalid log level number: {setLevel}")

    handler.setLevel(setLevel)

    formatter = logging.Formatter(fmt)
    handler.setFormatter(formatter)

    return handler


def format_db_error(e: DBAPIError) -> str:
    """
    Formats DBAPIError for logging

    Returns select error message.
    """
    orig = e.orig
    cls = f"{orig.__class__.__module__}.{orig.__class__.__name__}"
    return f"({cls}) {orig}"

# --------------------------
# Handling shutdown
# --------------------------
async def clean_table(db: AsyncPostgresDataBase, 
                tablename: str,
                truncate: bool = True) -> None:
    """
    
    - db: PostgresDataBase class that controls postgres database
    - tablename: existing table name you want rows deleted/truncated
    - truncate: truncate table. Faster than delete. If False, 'delete' will be used.

    """
    try:
        if truncate:
            await db.truncate_all_rows(tablename=tablename)
        else:
            await db.delete_all_rows(tablename=tablename)
        print(f"All rows successfully removed from '{tablename}' table.")
    except Exception as e:
        print(f"Could not delete all rows from '{tablename}' table. Error: {e}")


#TODO: need to fix
async def close_docker_compose(compose_path: str = "db/compose.yaml", down: bool = True) -> None:
    """
    Closes docker compose container(s).

    - compose_path: path of docker compose yaml file
    - down: should it be compose down or stop. If False, it will stop instead. Down will get rid of container and network.

    """
    choice = ""
    if down:
        choice = "down"
    else:
        choice = "stop"

    command = ["docker", "compose", "-f", compose_path, choice]
    proc = await asyncio.create_subprocess_exec(*command, stdout=PIPE, stderr=PIPE, text=True)

    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(f"Docker stop failed (code={proc.returncode}): {stderr.strip()}")
    else:
        print("Docker Compose stopped successfully.")


def input_to_bool(question: str, true_options: Union[set, list], false_options: Union[set, list]) -> bool:
    """
    Gets user input and output a bool.

    - question: input question for user
    - true_options: list/set of options that will give True
    - false_options: list/set of options that will give False

    Returns a boolean.
    """
    all_options = list(true_options)
    all_options.extend(false_options)
    
    ans = validate_ans(acceptable_ans=all_options, question=question)

    if ans in true_options:
        return True
    else:
        return False


# --------------------------
# Embedding Model
# --------------------------
task = 'Given a user query, retrieve relevant information that answer the query.'
def get_detailed_instruct(query: str,
                          task_description: str = task) -> str:
    """
    Adds instruction to query. Some embedding models like 'intfloat/multilingual-e5-large-instruct' require instructions to be added to query. Documents don't need instructions.

    - task_description: instruction for the query
    - query: input query

    Returns string of instruction + query 
    """
    return f'Instruct: {task_description}\nQuery: {query}'

#TODO: async
def create_embedding(model_name: str, 
                     input: str) -> np.ndarray:
    """
        Creates vector embedding
        
        - model_name: name of embedding model
        - input: input string that needs to be converted to embedding

        Returns NORMALIZED numpy array of vector embedding
    """
    #TODO: use lru cache? don't load this every time func is called 
    model = SentenceTransformer(model_name)

    embeddings = model.encode(input, convert_to_tensor=False, normalize_embeddings=True)

    return embeddings


# --------------------------
# LLM Agents/Tools
# --------------------------




from sqlalchemy_utils import database_exists, create_database
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import declarative_base, Session, sessionmaker
import os
import csv
from typing import Optional, Type, Any, Dict, List, Union
import ast


# transcription
# add to db
# llm output/response (agent: rag search)
# add to db

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


def write_row_to_csv(data: Dict[str, Any], file_path: Optional[str] = "output.csv") -> None:
    """
    Write dict of data to csv

    - data: row (dict) to be added
    - file_path: name of output csv file
    """
    # if path doesn't have .csv
    if not file_path.endswith(".csv"):
         file_path += ".csv"

    file_exists = False
    if os.path.exists(file_path):
        file_exists = True
    
    # append data
    fieldnames = data.keys()
    with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # if file doesn't exist, add header
        if not file_exists:
             writer.writeheader()

        writer.writerow(data) # singular

def write_rows_to_csv(data: List[Dict[str, Any]], file_path: Optional[str] = "output.csv") -> None:
    """
    Write list of dict of data to csv

    - data: list of rows (dict) to be added
    - file_path: name of output csv file
    """
    # if path doesn't have .csv
    if not file_path.endswith(".csv"):
         file_path += ".csv"

    file_exists = False
    if os.path.exists(file_path):
        file_exists = True
    
    # append data
    fieldnames = data[0].keys()
    with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # if file doesn't exist, add header
        if not file_exists:
             writer.writeheader()

        writer.writerows(data) # multiple


def csv_to_dict(file) -> list[dict]:
    """
    Read csv file

    - file: csv file name

    Returns list of dict
    """
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


Base = declarative_base()
def add_row(table: Type[Base], session: Session, info: Dict[str, Any], file_path: Optional[str] = None) -> None:
    """
    Add record.

    - table: table to add row
    - session: pg session
    - info: row data
    - file_path: csv file

    """
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


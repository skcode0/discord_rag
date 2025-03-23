import os
from util_funcs import make_pgdb

# --------------------------
# create vector database
# --------------------------
pg_username = os.environ.get('POSTGRESS_USER')
pg_password = os.environ.get('POSTGRES_PASSWORD')
db_name = os.environ.get('SHORT_TERM_DB')
port = os.environ.get('SHORT_TERM_HOST_PORT')

url = make_pgdb(password=pg_password,
                db=db_name,
                port=port,
                add_vectors=True)



# --------------------------
# start discord and adding vectors
# --------------------------

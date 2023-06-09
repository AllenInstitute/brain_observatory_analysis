import argparse
from pathlib import Path
from datetime import datetime
import sqlite3
import traceback
import pandas as pd

from brain_observatory_analysis.ophys.experiment_loading import get_ophys_expt

parser = argparse.ArgumentParser(description='')
parser.add_argument('--oeid', type=int, default=None, metavar='oeid')
parser.add_argument('--dev', type=bool, default=True, metavar='dev')

EXPT_CHECK_FOLDER = Path("/allen/programs/mindscope/workgroups/learning/"
                         "analysis_data_cache/expt_load_check")


def create_error_db():
    """Create expt_load_check.db, with table BOEO_errors and BOEO_dev_errors
    with columns: oeid, datetime, loaded, exception_type, error"""
    conn = sqlite3.connect(EXPT_CHECK_FOLDER / 'expt_load_check.db')
    c = conn.cursor()

    # create table
    c.execute('''CREATE TABLE BOEO_errors
                    (oeid int, datetime text, loaded bool,
                    exception_type text, error text, traceback text)''')
    c.execute('''CREATE TABLE BOEO_dev_errors
                    (oeid int, datetime text, loaded bool,
                    exception_type text, error text, traceback text)''')
    conn.commit()
    conn.close()

    return


def load_error_db():
    conn = sqlite3.connect(EXPT_CHECK_FOLDER / 'expt_load_check.db')
    c = conn.cursor()
    # check table structure
    c.execute("SELECT * FROM BOEO_dev_errors")
    df = pd.DataFrame(c.fetchall(), columns=["oeid", "datetime", "loaded", "exception_type", "error", "traceback"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")
    conn.close()
    return df


def get_recent_errors_from_db(not_loaded=False):
    """ for each oeid, only keep most recent (index)"""
    df = load_error_db()
    df = df.sort_index()
    df = df.groupby("oeid").last()

    if not_loaded:
        df = df[df["loaded"] == 0]
    return df


def _add_error_to_db(oeid, exception_type, error, traceback, dev=False):
    """When loading failed, add to expt_load_check.db, with loaded = 0
    and error info""" 
    table = 'BOEO_dev_errors' if dev else 'BOEO_errors'

    # escape single quotes
    error = error.replace("'", "''")
    traceback = traceback.replace("'", "''")

    conn = sqlite3.connect(EXPT_CHECK_FOLDER / 'expt_load_check.db')
    c = conn.cursor()
    # create table
    c.execute(f"INSERT INTO {table} VALUES ({oeid}, '{datetime.now()}', 0,"
              f"'{exception_type}', '{error}', '{traceback}')")

    conn.commit()
    conn.close()

    return


def _add_success_to_db(oeid, dev=False):
    """When loading was successful, add to expt_load_check.db, with loaded = 1
    and blank str for error columns"""  
    table = 'BOEO_dev_errors' if dev else 'BOEO_errors'

    conn = sqlite3.connect(EXPT_CHECK_FOLDER / 'expt_load_check.db')
    c = conn.cursor()
    c.execute(f"INSERT INTO {table} VALUES ({oeid}, '{datetime.now()}', 1, '', '', '')")
    conn.commit()
    conn.close()

    return


if __name__ == '__main__':
    args = parser.parse_args()
    oeid = args.oeid
    dev = args.dev
    dev = True
    if dev:
        print(f"Running in dev mode. Will add to BOEO_dev_errors table.")

    try:
        expt = get_ophys_expt(oeid, dev=dev)
        _add_success_to_db(oeid, dev=dev)
    except Exception as e:
        tb = traceback.format_exc()
        _add_error_to_db(oeid, type(e).__name__, str(e), tb, dev=dev)

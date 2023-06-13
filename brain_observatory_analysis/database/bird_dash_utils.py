import functools
import sqlite3
import os


BIRD_DASH_DB_PATH = '/allen/programs/mindscope/workgroups/learning/database/bird_dash/bird_dash.db'

def log_to_database(db_path):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get the metadata and filepath from the function arguments
            metadata = kwargs.pop('metadata', None)
            filepath = kwargs.pop('filepath', None)
            
            # Call the wrapped function
            result = func(*args, **kwargs)
            
            # If metadata and filepath were provided, log them to the database
            if metadata and filepath:
                # Connect to the database
                conn = sqlite3.connect(db_path)
                c = conn.cursor()
                
                # Insert a new record into the image_table
                c.execute("INSERT INTO image_table (mouse_name, experiment_id, session_id, image_path) VALUES (?, ?, ?, ?)",
                          (metadata['mouse_name'], metadata['experiment_id'], metadata['session_id'], filepath))
                
                # Commit the changes and close the connection
                conn.commit()
                conn.close()
            
            return result
        return wrapper
    return decorator



def init_database(db_path):
    if os.path.exists(db_path):
        # If the database already exists, raise an error
        raise ValueError("Database file already exists")
    else:
        # If the database doesn't exist, create it and initialize the tables
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        # Create the image_table
        c.execute('''CREATE TABLE image_table
                     (mouse_name text, experiment_id integer, session_id integer, image_path text, output_path text, 
                      PRIMARY KEY(mouse_name, experiment_id, session_id, image_path))''')
        
        # Commit the changes and close the connection
        conn.commit()
        conn.close()
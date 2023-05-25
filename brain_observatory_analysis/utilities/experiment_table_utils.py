import pandas as pd

########################################################################
# Project Agnostic constants
########################################################################

MOUSE_NAMES = {"603892": "Gold",
               "608368": "Silicon",
               "612764": "Silver",
               "624942": "Bronze",
               "629294": "Copper",
               "632324": "Nickel",
               "633532": "Titanium",
               "636496": "Aluminum",
               "639224": "Mercury",
               "646883": "Iron",
               "648606": "Zinc",
               "662253": "Cobalt",
               "662491": "Palladium",
               "662264": "Dubnium",
               "657850": "Osmium",
               "662680": "Bohrium",
               "623975": "Helium",
               "623972": "Neon",
               "631563": "Argon",
               "637848": "Xenon",
               "637851": "Radon"}

# maps "reporter" column to "gcamp_name" column
gcamp_name_map = {"Ai195(TIT2L-GC7s-ICF-IRES-tTA2)-hyg": 'GCaMP7s',
                  "Ai210(TITL-GC7f-ICF-IRES-tTA2)-hyg": 'GCaMP7f',
                  "Ai93(TITL-GCaMP6f)": 'GCaMP6f'}

########################################################################
# Primary functions
########################################################################


def import_project(project: str):

    if project == "LearningmFISHTask1A":
        import brain_observatory_analysis.projects.LearningmFISHTask1A as project_module
    elif project == "lamf_associative_pilots":
        import brain_observatory_analysis.projects.lamf_associative_pilots as project_module
    elif project == "visual_behavior":
        import brain_observatory_analysis.projects.visual_behavior as project_module
    else:
        raise ValueError(f"project {project} not recognized")

    return project_module


def experiment_table_extended_project(df: pd.DataFrame, project: str) -> pd.DataFrame:
    """Adds extra columns to the expt_table, in project specific manner"""

    if project is not None:
        project_module = import_project(project)

    df = add_short_session_type_column(df, project_module)
    df = add_n_exposure_short_session_type_column(df)
    df = add_short_session_type_num_column(df)
    df = set_cat_and_order_for_short_session_type(df, project_module)
    df = add_tiny_session_type_num(df)

    return df


def experiment_table_extended(df: pd.DataFrame):
    """Adds extra columns to the expt_table #WFDF

    Parameters:
    -----------
    df : pandas.DataFrame
        experiment table
    project : str
        name of project

    Returns:
    --------
    df : pandas.DataFrame
        experiment table with additional columns

    Notes:
    ------
    Adds the following columns:
        - 'session_type_num'
        - 'bisect_layer'
        - 'depth_order'
        - 'n_exposure_session_type'
        - 'n_exposure_session_type_num'
        - 'n_exposure_session_type_num_layer'
        - 'reporter_line'
        - 'reporter'
        - 'cre_name'
        - 'date_string'

    """
    df = add_n_exposure_session_type_column(df)
    df = add_session_number(df)
    df = add_session_type_num_column(df)
    df = add_bisect_layer_column(df)
    df = add_depth_order_column(df)
    df = add_fixed_reporter_line_column(df)
    df = add_mouse_names_columns(df)
    df = add_cre_name_column(df)
    df = add_date_string_column(df)
    df = add_ai_reporter_name_column(df)
    df = add_gcamp_name_column(df)

    return df

########################################################################
# Project agnostic columns
########################################################################


def add_mouse_names_columns(df: pd.DataFrame):
    """Adds a column called 'mouse_name' to expt_table #WKDF

    Parameters:
    -----------
    df : pandas.DataFrame
        expt_table with "mouse_id" (et) column

    Returns:
    --------
    df : pandas.DataFrame

    """

    df["mouse_name"] = df["mouse_id"].map(MOUSE_NAMES)

    return df


def add_session_number(df: pd.DataFrame):
    """Adds a column called 'session_number' to expt_table, sorted by
    date_of_acquisition, for each ophys_session_id. Thus, all experiments
    on the same day will have the same session_number. In general,
    one session is recorded per day, but we can add a day column if needed.

    Examples:
    --------

    session_number  sesssion_type
    --------------  -------------
    1               TRAINING_1_gratings
    2               TRAINING_1_gratings
    3               TRAINING_2_flashed_gratings

    Parameters:
    -----------
    df : pandas.DataFrame
        expt_table

    Returns:
    --------
    df : pandas.DataFrame

    """
    df = df.sort_values(by=["date_of_acquisition"])

    # code seems hacky, but it works (mjd)
    mice = df["mouse_id"].unique()
    for mouse in mice:
        mouse_df = df[df["mouse_id"] == mouse]
        for i, (n, g) in enumerate(mouse_df.groupby("ophys_session_id")):
            # assign session number
            df.loc[g.index, "session_number"] = i + 1

    df["session_number"] = df["session_number"].astype(int)

    return df


def add_session_type_num_column(df: pd.DataFrame):
    """Adds a column called 'session_type_num' to expt_table #WKDF

    Parameters:
    -----------
    df : pandas.DataFrame
        expt_table with "session_type" (et) & "n_exposure_session_type" (ete)
        columns

    Returns:
    --------
    df : pandas.DataFrame

    Examples:
    --------

    "TRAINING_1_gratings" -> "TRAINING_1_gratings_1"
    "TRAINING_1_gratings" -> "TRAINING_1_gratings_2"
    "TRAINING_1_gratings" -> "TRAINING_1_gratings_3"
    "TRAINING_2_gratings_flashed" -> "TRAINING_2_gratings_flashed_1"
    ...

    """
    df["session_type_num"] = df["session_type"] + "_" + \
        df["n_exposure_session_type"].astype(str)

    return df


def add_bisect_layer_column(df, bisecting_depth=220):
    """Adds a column called 'bisect_layer' to expt_table #WKDF

    Parameters:
    -----------
    df : pandas.DataFrame
        expt_table with "imaging_depth" (et) column

    Returns:
    --------
    df : pandas.DataFrame

    """
    df.loc[:, 'bisect_layer'] = None

    indices = df[(df.imaging_depth < bisecting_depth)].index.values
    df.loc[indices, 'bisect_layer'] = 'upper'

    indices = df[(df.imaging_depth > bisecting_depth)].index.values
    df.loc[indices, 'bisect_layer'] = 'lower'

    return df


def add_depth_order_column(df):
    """Adds a column called 'depth_order' to expt_table #WKDF

    Parameters:
    -----------
    df : pandas.DataFrame
        expt_table with "imaging_depth" (et) column

    Returns:
    --------
    df : pandas.DataFrame

    Notes:
    ------
    Will rank the order of the imaging planes from 1 to N, th most dorsal plane
    will be 1. Generally the 1st planes are less than 220 um"""

    gb = ["ophys_session_id", "targeted_structure"]
    df["depth_order"] = (df.groupby(gb)["imaging_depth"]
                         .transform(lambda x: x.rank(method="dense",
                                    ascending=True)))

    return df


def add_n_exposure_session_type_column(df):
    """Adds a column called 'n_exposure_session_type' to expt_table #WKDF

    Parameters:
    -----------
    df : pandas.DataFrame
        expt_table with "mouse_id" (et) & "session_type" (et) columns

    Returns:
    --------
    df : pandas.DataFrame

    Notes:
    ------

    """

    df["n_exposure_session_type"] = (df.groupby(["mouse_id", "session_type"])
                                     ["date_of_acquisition"]
                                     .rank(method="dense", ascending=True)
                                     .astype(int))

    return df


def add_abbreviated_reporter_line_column(df):

    def abbreviate_reporter_line(row):
        return row.reporter_line.split("(")[0]

    df["reporter_line"] = df.apply(abbreviate_reporter_line, axis=1)

    return df


def add_fixed_reporter_line_column(df):

    def fix_reporter_line(row):
        return row.full_genotype.split(';')[-1].split('/')[0]

    df["reporter"] = df.apply(fix_reporter_line, axis=1)

    return df


def add_cre_name_column(df):

    df['cre_line'] = df['cre_line'].fillna('')
    df['cre_name'] = (df['cre_line'].apply(lambda x: x.split('-')[0]))

    return df


def add_date_string_column(df):
    def date_from_dt(x):
        return str(x).split(' ')[0]
    df['date_string'] = (df['date_of_acquisition'].apply(lambda x: date_from_dt(x)))

    return df


def add_ai_reporter_name_column(df):
    df["ai_reporter_name"] = df["reporter"].apply(lambda x: x.split("(")[0])

    return df


def add_gcamp_name_column(df):

    df["gcamp_name"] = df["reporter"].map(gcamp_name_map)

    return df


########################################################################
# Project specific columns
########################################################################

def add_short_session_type_column(df, project_module):
    """Adds a column called 'short_session_type' to expt_table #WKDF

    Parameters:
    -----------
    df : pandas.DataFrame
        expt_table with "session_type" (et) column
    project_module : module
        module containing constants

    Returns:
    --------
    df : pandas.DataFrame

    Notes:
    ------

    """
    assert "short_session_type_map" in dir(project_module), \
        "project module must have a dict called 'short_session_type_map'"

    df["short_session_type"] = df["session_type"].map(
        project_module.short_session_type_map)

    return df


def add_short_session_type_num_column(df):
    """Adds a column called 'short_session_type_num' to expt_table,
     num describes the order of each session type.

    Parameters:
    -----------
    df : pandas.DataFrame
        expt_table with "short_session_type" (etep) & "n_exposure_stimulus" (ete)
        columns

    Returns:
    --------
    df : pandas.DataFrame

    Notes:
    ------

    """

    df["short_session_type_num"] = df["short_session_type"] + " - " + \
        df["n_exposure_short_session_type"].astype(str)

    return df


# TODO: this is an extended column
def add_n_exposure_short_session_type_column(df):
    """stim exposure based on short session name (which groups stim)"""

    df["n_exposure_short_session_type"] = \
        (df.groupby(["mouse_id", "short_session_type"])["date_of_acquisition"]
           .rank(method="dense", ascending=True)
           .astype(int))

    return df


def set_cat_and_order_for_short_session_type(df, project_module):
    """Sets the category and order for the short_session_type column #WKDF

    Parameters:
    -----------
    df : pandas.DataFrame
        expt_table with "short_session_type" (etep) column
    project_module : module
        module containing constants

    Returns:
    --------
    df : pandas.DataFrame

    Notes:
    ------

    """
    assert "short_session_type_order" in dir(project_module), \
        "project module must have a list called 'short_session_type_order'"

    df["short_session_type"] = pd.Categorical(df["short_session_type"],
                                              categories=project_module.short_session_type_order,
                                              ordered=True)

    return df


def add_tiny_session_type_num(df):
    """Creates a column called 'tiny_session_type_num' that is a shortened
    version of the short_session_type_num column. Useful for plotting labels

    Example:
    --------
    "Flashed Images - 1" -> "FI1"
    "Familar Images + Omissions - 2" -> "FIO2"

    Parameters:
    -----------
    df : pandas.DataFrame
        expt_table with "short_session_type_num" (etep) column

    Returns:
    --------
    df : pandas.DataFrame

    """
    def abbrv_text(row):
        text = row.short_session_type_num
        name, num = text.split(' - ')
        split_text = name.split(' ')

        # Only grab first letter of each word
        new_text = []
        for word in split_text:
            if word != '+':
                new_text.append(word[0].upper())

        return ''.join(new_text) + num

    df['tiny_session_type_num'] = df.apply(abbrv_text, axis=1)

    return df

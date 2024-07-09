import pandas as pd
import mindscope_qc.data_access.from_lims_utilities as from_limsu
from allensdk.brain_observatory.behavior.behavior_project_cache \
    import VisualBehaviorOphysProjectCache

from typing import Optional


def container_cells_table_plus_metadata(container_id: int,
                                        expt_table: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """From cache.get_ophsy_cells_table get cells in container_id and
    add experiment metadata

    Parameters
    ----------
    container_id: int
        The ophys_container_id of the container to get cells from
    expt_table: pd.DataFrame
        The ophys_experiment_table to use. If None, use the cache's
        ophys_experiment_table

    Returns
    -------
    container_cells: pd.DataFrame
        A dataframe (ophys_cells_table) of cells from a single container
        with experiment metadata added
    """

    from_limsu.validate_LIMS_id_type("ophys_container_id", container_id)

    cache = VisualBehaviorOphysProjectCache.from_lims()

    # use default expt_table if not provided
    if expt_table is None:
        expt_table = cache.get_ophys_experiment_table()

    # check expt_table for container_id
    if container_id not in expt_table.ophys_container_id.unique():
        raise ValueError(f"container_id {container_id} not in expt_table")

    oct = cache.get_ophys_cells_table().reset_index()
    oct = oct.merge(expt_table, on='ophys_experiment_id')
    container_cells = oct[oct.ophys_container_id == container_id]

    return container_cells


def get_cell_specimens_matched_across_sessions(container_cells: pd.DataFrame,
                                               session_types: Optional[list] = None):
    """
    Get a list of cell_specimen_ids that are matched across all sessions
    in session_types.

    Parameters
    ----------
    container_cells: pd.DataFrame
        A dataframe (ophys_cells_table) of cells from a single container,
        needs to have experiment metadata added
    session_types: list
        A list of session types to match across, defaults to all session types

    Returns:
    -------
    cell_specimens_matched: list
        A list of cell_specimen_ids that are matched across session_types
    expt_ids_filtered: list
        A list of experiment ids for the cell_matched session_types
    """
    # check for session_types col
    if "session_type" not in container_cells.columns:
        raise ValueError("container_cells must have session_type column, use"
                         "container_cells_table_plus_metadata()")

    if session_types is None:
        session_types = container_cells.session_type.unique()

    cells_filtered = container_cells.query("session_type in @session_types")

    # value error if no cells in session_types
    if len(cells_filtered) == 0:
        raise ValueError("No cells found for given session_types")

    n_sessions = len(cells_filtered.session_type_num.unique())

    expt_ids_filtered = cells_filtered.ophys_experiment_id.unique()

    sc = (cells_filtered
          .groupby("cell_specimen_id")
          .size()
          .reset_index()
          .rename(columns={0: "count"}))

    cell_specimen_ids = list(sc[sc["count"] == n_sessions]
                             .cell_specimen_id.values)

    return cell_specimen_ids, expt_ids_filtered

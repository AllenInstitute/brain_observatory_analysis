from typing import Optional, List

import pandas as pd

from allensdk.brain_observatory.behavior.behavior_project_cache.project_apis\
    .data_io import \
    BehaviorProjectLimsApi
from allensdk.brain_observatory.behavior.behavior_project_cache.tables\
    .experiments_table import \
    ExperimentsTable
from allensdk.brain_observatory.behavior.behavior_project_cache.tables\
    .sessions_table import \
    SessionsTable
from allensdk.internal.api import PostgresQueryMixin
from allensdk.internal.api.queries.utils import build_in_list_selector_query, \
    _sanitize_uuid_list


def session_stage_from_foraging_id(
        mtrain_engine: PostgresQueryMixin,
        foraging_ids: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Get DataFrame mapping behavior_sessions.id to session_type
    by querying mtrain for foraging_ids
    """
    # Select fewer rows if possible via behavior_session_id
    if foraging_ids is not None:
        foraging_ids = [f"'{fid}'" for fid in foraging_ids]
    # Otherwise just get the full table from mtrain
    else:
        foraging_ids = None

    foraging_ids_query = build_in_list_selector_query(
        "bs.id", foraging_ids)

    query = f"""
        SELECT
            stages.name as session_type,
            bs.id AS foraging_id
        FROM behavior_sessions bs
        JOIN stages ON stages.id = bs.state_id
        {foraging_ids_query};
        """
    return mtrain_engine.select(query)


def get_ophys_experiment_table():
    api = BehaviorProjectLimsApi.default(passed_only=False)
    experiments = api.get_ophys_experiment_table()\
        .reset_index()
    print(f"Ophys experiments: {len(experiments)}")
    behavior_sessions = api._get_behavior_summary_table()
    print(f"Behavior sessions: {len(behavior_sessions)}")
    foraging_ids = behavior_sessions['foraging_id']
    foraging_ids = foraging_ids[~foraging_ids.isna()]
    foraging_ids = _sanitize_uuid_list(foraging_ids)
    stimulus_names = session_stage_from_foraging_id(
        mtrain_engine=api.mtrain_engine,
        foraging_ids=foraging_ids)
    behavior_sessions['session_type'] = \
        behavior_sessions['foraging_id'].map(
            stimulus_names.set_index('foraging_id')['session_type'])

    behavior_sessions = SessionsTable(
        df=behavior_sessions, fetch_api=api)

    experiments = behavior_sessions.table.merge(
        experiments, on='behavior_session_id',
        suffixes=('_behavior', '_ophys'))

    experiments = experiments.set_index('ophys_experiment_id')
    experiments = ExperimentsTable(df=experiments)

    return experiments.table


if __name__ == '__main__':
    experiments = get_ophys_experiment_table()
    pass
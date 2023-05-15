

def cohort_mouse_ids():

    # AssociativeLearningCohort1 (DATE)
    # 614617 Slc17a7 mouse
    mouse_ids1 = ["614353", "614618", "615873", "615875", "614617"]

    # AssociativeLearningCohort2
    # (621408)?
    mouse_ids2 = ["620864", "621325", "621326", "621409"]

    # AssociativeLearningCohort3 (2022-07-07)
    mouse_ids3 = [
        "629823",  # vip
        "630414",  # slc
        "630417",  # slc
        "631576",  # slc
        "631577",  # slc
        "632294"
        # "629981",
    ]

    # AssociativeLearningCohort4 (DATE)
    mouse_ids4 = [
        "637762",  # errored
        "636506",
        "636497",
        "639425",
        "639424",
        "639423"  # errored
    ]

    # AssociativeLearningCohort5 (DATE)
    mouse_ids5 = [
        "646454",
        "646640",
        "647550",
        "648153",
        "648153",
        "648950"
    ]

    # store in dict for easy access
    cohort_dict = {
        "cohort1": mouse_ids1,
        "cohort2": mouse_ids2,
        "cohort3": mouse_ids3,
        "cohort4": mouse_ids4,
        "cohort5": mouse_ids5
    }

    return cohort_dict

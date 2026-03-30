from enum import Enum


class MAPFErrorCodes(Enum):
    NO_PATH = 0
    NO_INIT_PATH = 1
    NO_CONSTRAINED_PATH = 2
    TIMELIMIT_REACHED = 3
    START_GOAL_DISCONNECT = 4
    INVALID_BUDGET_ALLOCATER = 5
    BUDGET_MISMATCH = 6
    INVALID_SPLITTER = 7


def MAPFError(error_code, low_level_code=None):
    message = ""
    if error_code == MAPFErrorCodes.NO_PATH:
        message = "No path found."
    elif error_code == MAPFErrorCodes.NO_INIT_PATH:
        message = "No initial path found."
    elif error_code == MAPFErrorCodes.NO_CONSTRAINED_PATH:
        message = "No constrained path found."
    elif error_code == MAPFErrorCodes.TIMELIMIT_REACHED:
        message = "Time limit reached."
    elif error_code == MAPFErrorCodes.START_GOAL_DISCONNECT:
        message = "Start and goal are disconnected."
    elif error_code == MAPFErrorCodes.INVALID_BUDGET_ALLOCATER:
        message = "Invalid budget allocator."
    elif error_code == MAPFErrorCodes.BUDGET_MISMATCH:
        message = "Budget mismatch."
    elif error_code == MAPFErrorCodes.INVALID_SPLITTER:
        message = "Invalid constraint splitter."

    if low_level_code is not None:
        if low_level_code == MAPFErrorCodes.NO_PATH:
            message += " Low-Level Search could not find a path."
        elif low_level_code == MAPFErrorCodes.NO_INIT_PATH:
            message += " Low-Level Search could not find an initial path."
        elif low_level_code == MAPFErrorCodes.NO_CONSTRAINED_PATH:
            message += " Lower-Level Search could not find a constrained path."
        elif low_level_code == MAPFErrorCodes.TIMELIMIT_REACHED:
            message += " Low-Level Search exceeded the time-limit."
        elif low_level_code == MAPFErrorCodes.START_GOAL_DISCONNECT:
            message += " Low-Level Search could not find a path between start and goal."

    return {"error_code": error_code, "message": message}

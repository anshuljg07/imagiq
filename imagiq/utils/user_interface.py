import sys


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.
    Credit: http://code.activestate.com/recipes/577058/
    Arguments:
        question: a string to be prompted to the user.
        default: presumed default answer if the user just hits <Enter>.
            Only allowed values are "yes", "no", None
            If None, an answer is required from the user
            (cannot just hit <Enter>).
    Returns:
        True if the answer is yes, False otherwise.
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("Invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no'\n")

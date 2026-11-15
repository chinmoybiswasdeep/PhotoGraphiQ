import importlib.util

import photographiqml as pqml

if importlib.util.find_spec("mentpy"):
    from photographiqml.validation import compare_mentpy

    m = pqml.MuTA(2, one_column=True)
    error = compare_mentpy(m, [1, 0, 0, 0], m.initialize(2))
    assert error < 1e-10
    print("Logical density agreement:", error < 1e-10)
else:
    print("Optional MentPy reference not installed")

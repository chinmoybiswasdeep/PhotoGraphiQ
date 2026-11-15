import photographiqml as pqml

m = pqml.MuTA(1, representation="gkp")
r = pqml.GKPBridge().logical_target(m, [1, 0])
assert r.representation == "logical"
try:
    m.run([1, 0])
except NotImplementedError:
    print("Legacy resource-only execution is unsupported; ideal target remains logical")
else:
    raise AssertionError("Unexpected physical execution")

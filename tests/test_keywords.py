from fortify_scg.keywords import compile_patterns, is_sensitive

def test_keyword_matcher():
    kw = ["strcpy", "*RC6*", "MD5_*"]
    rgx = compile_patterns(kw)
    assert is_sensitive("strcpy", rgx)
    assert is_sensitive("fastRC6encrypt", rgx)
    assert is_sensitive("MD5_Init", rgx)
    assert not is_sensitive("safe_function", rgx)

from upsc_rag.config import PROJECT_ROOT, load_book_config, load_runtime_config


def test_project_root_exists():
    assert PROJECT_ROOT.is_dir()


def test_load_laxmikanth_config():
    runtime = load_runtime_config("laxmikanth_6")
    assert runtime["book"]["id"] == "laxmikanth_6"
    book = load_book_config("laxmikanth_6")
    assert book.edition == 6

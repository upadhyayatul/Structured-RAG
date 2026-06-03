from upsc_rag.chunking.structured import chunk_section_text, extract_entities


def test_extract_entities():
    text = "Article 14 allows equality. Article 15 prohibits discrimination."
    assert "Article 14" in extract_entities(text)
    assert "Article 15" in extract_entities(text)


def test_chunk_section_text_yields_records():
    text = "Para one.\n\nPara two.\n\nPara three."
    chunks = list(
        chunk_section_text(
            text,
            book_id="laxmikanth_6",
            section_id="ch01_test",
            max_tokens=10,
            overlap_tokens=2,
        )
    )
    assert len(chunks) >= 1
    assert chunks[0].book_id == "laxmikanth_6"

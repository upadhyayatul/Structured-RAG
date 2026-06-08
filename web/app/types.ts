// Shared types mirroring the FastAPI /ask response (src/upsc_rag/api/app.py).

export interface Source {
  n: number;
  section_path: string[];
  chapter_title: string;
  page_start: number | null;
  page_end: number | null;
}

export interface AskResponse {
  answer: string;
  sources: Source[];
}

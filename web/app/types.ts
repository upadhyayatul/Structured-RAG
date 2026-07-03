// Shared types mirroring the FastAPI /ask response (src/upsc_rag/api/app.py).

// A source is either a textbook section (type "book") or a web result (type "web").
// The agentic pipeline (UPSC_RAG_PIPELINE=agentic) returns a mix; the direct/graph
// paths return only book sources. Fields are optional per type.
export interface Source {
  n: number;
  type?: "book" | "web";
  // book source
  section_path?: string[] | null;
  chapter_title?: string | null;
  page_start?: number | null;
  page_end?: number | null;
  // web source
  title?: string | null;
  url?: string | null;
  snippet?: string | null;
}

export interface AskResponse {
  answer: string;
  sources: Source[];
}

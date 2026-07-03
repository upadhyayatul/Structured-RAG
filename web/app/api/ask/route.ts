// Backend-for-frontend proxy: forwards the question to the FastAPI streaming
// endpoint and pipes the NDJSON event stream straight back to the browser.
// Keeps the backend URL server-side and avoids CORS.
import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8000";

export async function POST(req: NextRequest) {
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const query = (body as { query?: string })?.query?.trim();
  if (!query) {
    return NextResponse.json({ error: "query is required" }, { status: 400 });
  }
  // Conversation id from the client; forwarded so the backend can group this
  // question's retrieve + answer traces under one Langfuse Session.
  const sessionId = (body as { sessionId?: string })?.sessionId;
  // Recent conversation turns for follow-up resolution; forwarded as-is (the
  // backend condenses them for retrieval and passes them to the answer LLM).
  const history = (body as { history?: unknown })?.history;

  let upstream: Response;
  try {
    upstream = await fetch(`${BACKEND_URL}/ask/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, history, rerank_top_k: 6, session_id: sessionId }),
    });
  } catch {
    return NextResponse.json(
      { error: "Cannot reach the RAG backend. Is the FastAPI server running on port 8000?" },
      { status: 502 },
    );
  }

  if (!upstream.ok || !upstream.body) {
    const detail = await upstream.text().catch(() => "");
    return NextResponse.json(
      { error: `Backend error (${upstream.status}): ${detail}` },
      { status: upstream.status || 502 },
    );
  }

  // Pipe the NDJSON stream through unbuffered.
  return new Response(upstream.body, {
    headers: {
      "Content-Type": "application/x-ndjson",
      "Cache-Control": "no-cache, no-transform",
    },
  });
}

"use client";

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Source } from "@/app/types";

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  quote?: string; // excerpt this user question was asked "about" (reply-style)
  sources?: Source[];
  ttftMs?: number;
  costUsd?: number;
  totalTokens?: number;
  streaming?: boolean;
  error?: boolean;
}

// A live text selection inside an assistant answer, plus where to float the button.
interface AnswerSelection {
  text: string;
  x: number; // viewport px, horizontal center of the selection
  y: number; // viewport px, top of the selection
}

export default function Home() {
  const [dark, setDark] = useState(true);
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  // Pending reply-quote (excerpt the user picked "Ask about this" on), shown above the input.
  const [quote, setQuote] = useState<string | null>(null);
  // Live selection inside an answer → drives the floating "Ask about this" button.
  const [selection, setSelection] = useState<AnswerSelection | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  // One stable id per page load = one conversation = one Langfuse Session.
  const sessionId = useRef<string>(crypto.randomUUID());

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
  }, [dark]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Detect a text selection made INSIDE an assistant answer (marked data-answer),
  // and remember it so we can float an "Ask about this" button near it.
  useEffect(() => {
    function onMouseUp() {
      const sel = window.getSelection();
      const text = sel?.toString().trim() ?? "";
      if (!sel || sel.isCollapsed || !text) {
        setSelection(null);
        return;
      }
      const anchor = sel.anchorNode;
      const el = anchor instanceof Element ? anchor : anchor?.parentElement;
      if (!el?.closest("[data-answer]")) {
        setSelection(null); // selection outside an answer (user bubble, sources, input)
        return;
      }
      const rect = sel.getRangeAt(0).getBoundingClientRect();
      setSelection({ text, x: rect.left + rect.width / 2, y: rect.top });
    }
    document.addEventListener("mouseup", onMouseUp);
    return () => document.removeEventListener("mouseup", onMouseUp);
  }, []);

  // The floating button is anchored to viewport coords, so hide it once the user scrolls.
  useEffect(() => {
    const node = scrollRef.current;
    if (!node) return;
    const hide = () => setSelection(null);
    node.addEventListener("scroll", hide);
    return () => node.removeEventListener("scroll", hide);
  }, []);

  // Promote the live selection to a pending reply-quote and focus the input.
  function askAboutSelection() {
    if (!selection) return;
    setQuote(selection.text);
    setSelection(null);
    window.getSelection()?.removeAllRanges();
    inputRef.current?.focus();
  }

  // Patch the most recent message in place (used while streaming).
  function patchLast(patch: Partial<ChatMessage>) {
    setMessages((m) => {
      const copy = [...m];
      copy[copy.length - 1] = { ...copy[copy.length - 1], ...patch };
      return copy;
    });
  }

  function send(e: React.FormEvent) {
    e.preventDefault();
    submitQuery(input.trim());
  }

  async function submitQuery(q: string) {
    if (!q || loading) return;

    // If a passage was quoted, fold it into the query the backend sees so the
    // follow-up's retrieval + answer take the excerpt into account; keep `quote`
    // on the message separately so the UI can show it reply-style.
    const activeQuote = quote;
    const backendQuery = activeQuote
      ? `Regarding this excerpt from the previous answer:\n"${activeQuote}"\n\n${q}`
      : q;

    // Recent conversation for follow-up resolution: only completed turns, last N
    // exchanges (N questions + N answers), stripped to {role, content}. `messages`
    // here still holds the prior conversation (the new turn is appended below).
    const history = messages
      .filter((m) => !m.streaming && !m.error && m.content)
      .map((m) => ({ role: m.role, content: m.content }))
      .slice(-HISTORY_EXCHANGES * 2);

    setMessages((m) => [
      ...m,
      { role: "user", content: q, quote: activeQuote ?? undefined },
      { role: "assistant", content: "", streaming: true },
    ]);
    setInput("");
    setQuote(null);
    setLoading(true);

    const start = performance.now();
    let answer = "";
    let ttft: number | null = null;

    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: backendQuery, history, sessionId: sessionId.current }),
      });

      if (!res.ok || !res.body) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error ?? "Request failed");
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.trim()) continue;
          const evt = JSON.parse(line);
          if (evt.type === "sources") {
            patchLast({ sources: evt.sources as Source[] });
          } else if (evt.type === "token") {
            if (ttft === null) {
              ttft = performance.now() - start;
              patchLast({ ttftMs: ttft });
            }
            answer += evt.text;
            patchLast({ content: answer });
          } else if (evt.type === "done") {
            patchLast({
              costUsd: typeof evt.cost_usd === "number" ? evt.cost_usd : undefined,
              totalTokens:
                typeof evt.input_tokens === "number" && typeof evt.output_tokens === "number"
                  ? evt.input_tokens + evt.output_tokens
                  : undefined,
            });
          }
        }
      }
      patchLast({ streaming: false });
    } catch (err) {
      patchLast({
        content: err instanceof Error ? err.message : "Something went wrong",
        error: true,
        streaming: false,
      });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex h-full flex-col bg-background text-foreground">
      <div className="flex items-center justify-between px-5 py-3">
        <ThemeToggle dark={dark} onToggle={() => setDark((d) => !d)} />
        <a
          href="https://github.com"
          target="_blank"
          rel="noreferrer"
          className="text-sm text-neutral-500 hover:text-neutral-700 dark:hover:text-neutral-300"
        >
          Source Code
        </a>
      </div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto">
        <div className="mx-auto w-full max-w-3xl px-4 pb-6">
          <header className="py-8 text-center">
            <h1 className="text-3xl font-bold">UPSC Polity Chat</h1>
            <p className="mt-2 text-sm text-neutral-500">
              Grounded RAG over M. Laxmikanth&apos;s <em>Indian Polity</em> (6th ed.) — answers with page citations.
            </p>
          </header>

          {messages.length === 0 && <SampleQuestions onPick={submitQuery} />}

          <div className="flex flex-col gap-6">
            {messages.map((m, i) =>
              m.role === "user" ? (
                <UserBubble key={i} text={m.content} quote={m.quote} />
              ) : (
                <AssistantMessage key={i} message={m} />
              ),
            )}
            <div ref={bottomRef} />
          </div>
        </div>
      </div>

      <div className="border-t border-neutral-200 dark:border-neutral-800">
        <div className="mx-auto w-full max-w-3xl px-4">
          {quote && <ReplyPreview quote={quote} onClear={() => setQuote(null)} />}
          <form onSubmit={send} className="flex w-full items-center gap-2 py-4">
            <input
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={quote ? "Ask about the quoted text…" : "Send a message…"}
              className="flex-1 rounded-lg border border-neutral-300 bg-transparent px-4 py-3 text-sm outline-none focus:border-neutral-500 dark:border-neutral-700"
            />
            <button
              type="submit"
              disabled={loading || !input.trim()}
              aria-label="Send"
              className="rounded-lg border border-neutral-300 p-3 text-neutral-600 hover:bg-neutral-100 disabled:opacity-40 dark:border-neutral-700 dark:text-neutral-300 dark:hover:bg-neutral-900"
            >
              <SendIcon />
            </button>
          </form>
        </div>
      </div>

      {selection && (
        <button
          // preventDefault on mousedown keeps the text selection alive through the click.
          onMouseDown={(e) => {
            e.preventDefault();
            askAboutSelection();
          }}
          style={{
            position: "fixed",
            top: Math.max(8, selection.y - 40),
            left: selection.x,
            transform: "translateX(-50%)",
          }}
          className="z-50 flex items-center gap-1.5 rounded-full border border-neutral-300 bg-white px-3 py-1.5 text-xs font-medium text-neutral-700 shadow-lg hover:bg-neutral-100 dark:border-neutral-700 dark:bg-neutral-900 dark:text-neutral-200 dark:hover:bg-neutral-800"
        >
          <QuoteIcon />
          Ask about this
        </button>
      )}
    </div>
  );
}

// How many recent exchanges (question + answer pairs) to send as conversation
// context for follow-up resolution. Keep in sync with `conversation.history_turns`
// in config/default.yaml.
const HISTORY_EXCHANGES = 3;

// Tappable starter prompts shown on the empty startup state to cue users on what to ask.
const SAMPLE_QUESTIONS = [
  "What is the difference between Fundamental Rights and Directive Principles?",
  "Explain the powers of the President of India.",
  "How is the Prime Minister of India appointed?",
  "What are the key features of Indian federalism?",
];

function SampleQuestions({ onPick }: { onPick: (q: string) => void }) {
  return (
    <div className="mb-2">
      <p className="mb-3 text-center text-xs uppercase tracking-wide text-neutral-400">
        Try asking
      </p>
      <div className="flex flex-wrap justify-center gap-2">
        {SAMPLE_QUESTIONS.map((q) => (
          <button
            key={q}
            type="button"
            onClick={() => onPick(q)}
            className="rounded-full border border-neutral-300 px-4 py-2 text-left text-sm text-neutral-700 transition-colors hover:border-neutral-500 hover:bg-neutral-100 dark:border-neutral-700 dark:text-neutral-300 dark:hover:border-neutral-500 dark:hover:bg-neutral-900"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}

// WhatsApp-style reply preview: the quoted excerpt pinned above the input, with a clear button.
function ReplyPreview({ quote, onClear }: { quote: string; onClear: () => void }) {
  return (
    <div className="mt-3 flex items-start gap-2 rounded-lg border-l-2 border-neutral-400 bg-neutral-100 px-3 py-2 dark:border-neutral-500 dark:bg-neutral-900">
      <QuoteIcon />
      <p className="line-clamp-2 flex-1 text-xs text-neutral-600 dark:text-neutral-400">{quote}</p>
      <button
        type="button"
        onClick={onClear}
        aria-label="Remove quote"
        className="text-neutral-400 hover:text-neutral-700 dark:hover:text-neutral-200"
      >
        <CloseIcon />
      </button>
    </div>
  );
}

function UserBubble({ text, quote }: { text: string; quote?: string }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[80%] rounded-2xl bg-neutral-900 px-4 py-2.5 text-sm text-white dark:bg-white dark:text-black">
        {quote && (
          <div className="mb-1.5 line-clamp-3 border-l-2 border-white/40 pl-2 text-xs italic opacity-70 dark:border-black/40">
            {quote}
          </div>
        )}
        {text}
      </div>
    </div>
  );
}

function AssistantMessage({ message }: { message: ChatMessage }) {
  const waiting = message.streaming && !message.content;

  if (message.error) {
    return (
      <div className="whitespace-pre-wrap rounded-lg border border-red-400/40 bg-red-500/10 px-4 py-3 text-sm text-red-500 dark:text-red-400">
        {message.content}
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {(message.ttftMs != null || message.costUsd != null) && (
        <div className="flex flex-wrap gap-x-2 text-xs text-neutral-400">
          {message.ttftMs != null && (
            <span>First token in {(message.ttftMs / 1000).toFixed(1)}s</span>
          )}
          {message.costUsd != null && (
            <>
              <span aria-hidden>·</span>
              <span title="Approximate answer-generation cost (OpenAI list prices)">
                ~{formatCost(message.costUsd)}
              </span>
            </>
          )}
          {message.totalTokens != null && message.totalTokens > 0 && (
            <>
              <span aria-hidden>·</span>
              <span>{message.totalTokens.toLocaleString()} tokens</span>
            </>
          )}
        </div>
      )}

      {waiting ? (
        <div className="flex items-center gap-2 text-sm text-neutral-500">
          <Spinner /> Thinking…
        </div>
      ) : (
        <div data-answer className="markdown text-sm leading-relaxed text-foreground">
          <ReactMarkdown remarkPlugins={[remarkGfm]} components={MD}>
            {message.content}
          </ReactMarkdown>
        </div>
      )}

      {message.sources && message.sources.length > 0 && (
        <div className="space-y-1 border-l-2 border-neutral-200 pl-3 text-xs text-neutral-500 dark:border-neutral-800">
          <div className="font-semibold uppercase tracking-wide">Sources</div>
          <ol className="space-y-0.5">
            {message.sources.map((s) => (
              <li key={s.n}>
                <span className="text-neutral-400">[{s.n}]</span>{" "}
                {s.section_path.join(" › ") || s.chapter_title}
                {s.page_start != null && (
                  <span className="text-neutral-400">
                    {" "}· p.{s.page_start}
                    {s.page_end != null && s.page_end !== s.page_start ? `–${s.page_end}` : ""}
                  </span>
                )}
              </li>
            ))}
          </ol>
        </div>
      )}
    </div>
  );
}

// Format a tiny USD cost: cents for >= $0.01, otherwise two significant figures
// (e.g. $0.00072) so per-question costs stay legible.
function formatCost(usd: number): string {
  if (usd <= 0) return "$0.00";
  if (usd >= 0.01) return `$${usd.toFixed(2)}`;
  return `$${usd.toPrecision(2)}`;
}

// Markdown element styling (no typography plugin needed).
const MD = {
  p: (props: React.HTMLAttributes<HTMLParagraphElement>) => <p className="mb-3 last:mb-0" {...props} />,
  ol: (props: React.OlHTMLAttributes<HTMLOListElement>) => <ol className="mb-3 list-decimal space-y-1 pl-5" {...props} />,
  ul: (props: React.HTMLAttributes<HTMLUListElement>) => <ul className="mb-3 list-disc space-y-1 pl-5" {...props} />,
  li: (props: React.LiHTMLAttributes<HTMLLIElement>) => <li className="pl-1" {...props} />,
  strong: (props: React.HTMLAttributes<HTMLElement>) => <strong className="font-semibold" {...props} />,
  h1: (props: React.HTMLAttributes<HTMLHeadingElement>) => <h1 className="mb-2 mt-4 text-lg font-bold" {...props} />,
  h2: (props: React.HTMLAttributes<HTMLHeadingElement>) => <h2 className="mb-2 mt-4 text-base font-bold" {...props} />,
  h3: (props: React.HTMLAttributes<HTMLHeadingElement>) => <h3 className="mb-1 mt-3 font-semibold" {...props} />,
  code: (props: React.HTMLAttributes<HTMLElement>) => (
    <code className="rounded bg-neutral-200 px-1 py-0.5 font-mono text-[0.85em] dark:bg-neutral-800" {...props} />
  ),
  a: (props: React.AnchorHTMLAttributes<HTMLAnchorElement>) => <a className="underline" {...props} />,
};

function ThemeToggle({ dark, onToggle }: { dark: boolean; onToggle: () => void }) {
  return (
    <div className="flex items-center gap-2">
      <SunIcon />
      <button
        onClick={onToggle}
        role="switch"
        aria-checked={dark}
        aria-label="Toggle dark mode"
        className="inline-flex h-6 w-11 items-center rounded-full bg-neutral-300 px-0.5 transition-colors dark:bg-neutral-600"
      >
        <span
          className={`h-5 w-5 rounded-full bg-white shadow transition-transform ${
            dark ? "translate-x-5" : "translate-x-0"
          }`}
        />
      </button>
      <MoonIcon />
    </div>
  );
}

/* --- icons --- */

function SendIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m22 2-7 20-4-9-9-4Z" />
      <path d="M22 2 11 13" />
    </svg>
  );
}

function QuoteIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor" className="shrink-0 opacity-70">
      <path d="M7 7h4v6a4 4 0 0 1-4 4H6v-2h1a2 2 0 0 0 2-2v-1H7Zm8 0h4v6a4 4 0 0 1-4 4h-1v-2h1a2 2 0 0 0 2-2v-1h-2Z" />
    </svg>
  );
}

function CloseIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M18 6 6 18M6 6l12 12" />
    </svg>
  );
}

function SunIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-neutral-500">
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41" />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-neutral-500">
      <path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z" />
    </svg>
  );
}

function Spinner() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" className="animate-spin">
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2" strokeOpacity="0.25" />
      <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

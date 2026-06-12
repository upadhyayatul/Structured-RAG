"use client";

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Source } from "@/app/types";

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  ttftMs?: number;
  streaming?: boolean;
  error?: boolean;
}

export default function Home() {
  const [dark, setDark] = useState(true);
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  // One stable id per page load = one conversation = one Langfuse Session.
  const sessionId = useRef<string>(crypto.randomUUID());

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
  }, [dark]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Patch the most recent message in place (used while streaming).
  function patchLast(patch: Partial<ChatMessage>) {
    setMessages((m) => {
      const copy = [...m];
      copy[copy.length - 1] = { ...copy[copy.length - 1], ...patch };
      return copy;
    });
  }

  async function send(e: React.FormEvent) {
    e.preventDefault();
    const q = input.trim();
    if (!q || loading) return;

    setMessages((m) => [
      ...m,
      { role: "user", content: q },
      { role: "assistant", content: "", streaming: true },
    ]);
    setInput("");
    setLoading(true);

    const start = performance.now();
    let answer = "";
    let ttft: number | null = null;

    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, sessionId: sessionId.current }),
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

      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto w-full max-w-3xl px-4 pb-6">
          <header className="py-8 text-center">
            <h1 className="text-3xl font-bold">UPSC Polity Chat</h1>
            <p className="mt-2 text-sm text-neutral-500">
              Grounded RAG over M. Laxmikanth&apos;s <em>Indian Polity</em> (6th ed.) — answers with page citations.
            </p>
          </header>

          <div className="flex flex-col gap-6">
            {messages.map((m, i) =>
              m.role === "user" ? (
                <UserBubble key={i} text={m.content} />
              ) : (
                <AssistantMessage key={i} message={m} />
              ),
            )}
            <div ref={bottomRef} />
          </div>
        </div>
      </div>

      <div className="border-t border-neutral-200 dark:border-neutral-800">
        <form onSubmit={send} className="mx-auto flex w-full max-w-3xl items-center gap-2 px-4 py-4">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Send a message…"
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
  );
}

function UserBubble({ text }: { text: string }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[80%] rounded-2xl bg-neutral-900 px-4 py-2.5 text-sm text-white dark:bg-white dark:text-black">
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
      {message.ttftMs != null && (
        <div className="text-xs text-neutral-400">
          First token in {(message.ttftMs / 1000).toFixed(1)}s
        </div>
      )}

      {waiting ? (
        <div className="flex items-center gap-2 text-sm text-neutral-500">
          <Spinner /> Thinking…
        </div>
      ) : (
        <div className="markdown text-sm leading-relaxed text-foreground">
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

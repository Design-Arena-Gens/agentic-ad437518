/* eslint-disable @next/next/no-img-element */
"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  CreateMLCEngine,
  InitProgressReport,
  MLCEngineInterface,
  ModelRecord,
  prebuiltAppConfig,
} from "@mlc-ai/web-llm";

type Role = "user" | "assistant";

interface ChatMessage {
  role: Role;
  content: string;
}

const DEFAULT_RULES = `You are a loyal personal assistant that follows the user's custom rules precisely. 
Provide concise, actionable answers, always prioritizing the user's preferences. 
If a request conflicts with the user's rules, remind them of the active instructions instead of refusing outright.`;

const formatProgress = (progress?: InitProgressReport) => {
  if (!progress) return "";
  const percentage = progress.progress
    ? Math.round(progress.progress * 100)
    : undefined;
  const text = progress.text ?? "";
  if (percentage === undefined) return text;
  return `${percentage}% • ${text}`;
};

const llamaModels: ModelRecord[] = prebuiltAppConfig.model_list.filter(
  (record) => record.model_id.toLowerCase().includes("llama"),
);

const temperatureMarks: Record<number, string> = {
  0: "Focused",
  5: "Balanced",
  10: "Creative",
};

export default function Home() {
  const [selectedModelId, setSelectedModelId] = useState<string>(
    llamaModels[0]?.model_id ?? "",
  );
  const [engine, setEngine] = useState<MLCEngineInterface | null>(null);
  const [engineStatus, setEngineStatus] = useState<string>(
    "Select a model to initialize your assistant.",
  );
  const [loadProgress, setLoadProgress] = useState<InitProgressReport | null>(
    null,
  );
  const [isLoadingModel, setIsLoadingModel] = useState(false);
  const [systemRules, setSystemRules] = useState<string>(DEFAULT_RULES);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [userInput, setUserInput] = useState<string>("");
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [temperature, setTemperature] = useState<number>(6);
  const [maxTokens, setMaxTokens] = useState<number>(512);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const conversationRef = useRef<ChatMessage[]>(messages);

  useEffect(() => {
    conversationRef.current = messages;
  }, [messages]);

  useEffect(() => {
    if (!selectedModelId) return;
    let cancelled = false;
    async function init() {
      setIsLoadingModel(true);
      setEngine(null);
      setMessages([]);
      setEngineStatus(`Loading ${selectedModelId}…`);
      setLoadProgress(null);
      try {
        const record = llamaModels.find(
          (model) => model.model_id === selectedModelId,
        );
        const instance = await CreateMLCEngine(selectedModelId, {
          appConfig: record
            ? { model_list: [record], useIndexedDBCache: false }
            : undefined,
          initProgressCallback: (progress) => {
            if (cancelled) return;
            setLoadProgress(progress);
            setEngineStatus(formatProgress(progress) || "Preparing model…");
          },
        });
        if (cancelled) return;
        setEngine(instance);
        setEngineStatus(
          "Model ready. Define your custom rules and start chatting.",
        );
      } catch (error) {
        if (cancelled) return;
        setEngineStatus(
          `Failed to load model: ${
            error instanceof Error ? error.message : String(error)
          }`,
        );
      } finally {
        if (!cancelled) setIsLoadingModel(false);
      }
    }
    init();
    return () => {
      cancelled = true;
    };
  }, [selectedModelId]);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const activeSystemPrompt = useMemo(() => {
    const trimmed = systemRules.trim();
    return trimmed.length > 0 ? trimmed : DEFAULT_RULES;
  }, [systemRules]);

  const canSend =
    !!engine && !isStreaming && userInput.trim().length > 0 && !isLoadingModel;

  const handleSend = async () => {
    if (!engine) return;
    const prompt = userInput.trim();
    if (!prompt) return;
    const userMessage: ChatMessage = { role: "user", content: prompt };
    const history = [...conversationRef.current, userMessage];
    const assistantPlaceholder: ChatMessage = { role: "assistant", content: "" };
    setUserInput("");
    setMessages([...history, assistantPlaceholder]);
    setIsStreaming(true);
    try {
      const stream = await engine.chat.completions.create({
        stream: true,
        messages: [
          { role: "system", content: activeSystemPrompt },
          ...history,
        ],
        temperature: Number((temperature / 10).toFixed(2)),
        max_tokens: maxTokens,
      });

      let aggregate = "";
      for await (const chunk of stream) {
        aggregate += chunk.choices[0]?.delta.content ?? "";
        const snapshot = aggregate;
        setMessages((prev) => {
          const next = [...prev];
          const last = next[next.length - 1];
          if (last?.role === "assistant") {
            next[next.length - 1] = { role: "assistant", content: snapshot };
          }
          return next;
        });
      }

      const finalReply = await engine.getMessage();
      if (typeof finalReply === "string" && finalReply.length > 0) {
        setMessages((prev) => {
          const next = [...prev];
          const last = next[next.length - 1];
          if (last?.role === "assistant") {
            next[next.length - 1] = { role: "assistant", content: finalReply };
          }
          return next;
        });
      }
    } catch (error) {
      setMessages((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        if (last?.role === "assistant") {
          next[next.length - 1] = {
            role: "assistant",
            content: `⚠️ ${error instanceof Error ? error.message : String(error)}`,
          };
        }
        return next;
      });
    } finally {
      setIsStreaming(false);
    }
  };

  const handleReset = () => {
    setMessages([]);
    setEngineStatus(
      "Conversation cleared. You can adjust rules and start again.",
    );
  };

  return (
    <main className="relative mx-auto flex min-h-screen w-full max-w-7xl flex-col gap-6 px-4 pb-12 pt-10 sm:px-8 lg:flex-row">
      <section className="w-full space-y-6 lg:max-w-sm">
        <header className="rounded-3xl border border-white/10 bg-slate-900/60 p-6 shadow-2xl shadow-indigo-950/40 backdrop-blur">
          <div className="flex items-center gap-3">
            <div className="flex size-12 items-center justify-center rounded-2xl bg-indigo-600/20">
              <img
                src="https://cdn.jsdelivr.net/gh/tabler/tabler-icons/icons/brand-openai.svg"
                alt="Llama icon"
                className="size-7 opacity-90 invert"
              />
            </div>
            <div>
              <h1 className="text-lg font-semibold text-slate-100">
                Llama Rule Rewriter
              </h1>
              <p className="text-sm text-slate-400">
                Personalize every instruction before the model responds.
              </p>
            </div>
          </div>
        </header>

        <div className="space-y-4 rounded-3xl border border-white/10 bg-slate-900/60 p-6 backdrop-blur">
          <div>
            <label className="mb-2 block text-sm font-semibold text-slate-200">
              Llama model
            </label>
            <select
              value={selectedModelId}
              onChange={(event) => setSelectedModelId(event.target.value)}
              className="w-full rounded-xl border border-slate-700 bg-slate-800/80 px-4 py-3 text-sm text-slate-100 outline-none transition focus:border-indigo-400 focus:ring-2 focus:ring-indigo-400/40"
            >
              {llamaModels.map((model) => (
                <option key={model.model_id} value={model.model_id}>
                  {model.model_id.replaceAll("-", " ")}
                </option>
              ))}
            </select>
          </div>
          <div className="space-y-2 rounded-2xl border border-white/5 bg-slate-950/60 px-4 py-3 text-sm text-slate-300">
            <p className="font-medium text-slate-200">Model status</p>
            <p className="leading-relaxed text-slate-400">{engineStatus}</p>
            {loadProgress?.text && (
              <div className="mt-2 rounded-lg border border-slate-800 bg-slate-900/80 p-2 text-xs text-indigo-300">
                {formatProgress(loadProgress)}
              </div>
            )}
          </div>
          <div className="space-y-3 rounded-3xl border border-white/10 bg-slate-900/60 p-5">
            <div className="flex items-center justify-between text-xs uppercase tracking-wide text-slate-400">
              <span>Temperature</span>
              <span className="font-semibold text-indigo-300">
                {(temperature / 10).toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={10}
              step={1}
              value={temperature}
              onChange={(event) => setTemperature(Number(event.target.value))}
              className="w-full accent-indigo-400"
            />
            <div className="flex justify-between text-[11px] text-slate-500">
              {Object.entries(temperatureMarks).map(([value, label]) => (
                <span key={value}>{label}</span>
              ))}
            </div>
            <div className="mt-4 flex items-center justify-between text-xs uppercase tracking-wide text-slate-400">
              <span>Max Tokens</span>
              <span className="font-semibold text-indigo-300">
                {maxTokens}
              </span>
            </div>
            <input
              type="range"
              min={256}
              max={2048}
              step={64}
              value={maxTokens}
              onChange={(event) => setMaxTokens(Number(event.target.value))}
              className="w-full accent-teal-400"
            />
            <div className="flex justify-between text-[11px] text-slate-500">
              <span>Short</span>
              <span>Extended</span>
            </div>
          </div>
        </div>

        <div className="h-px w-full bg-gradient-to-r from-transparent via-white/10 to-transparent lg:hidden" />
      </section>

      <section className="flex flex-1 flex-col gap-6">
        <div className="gradient-border relative flex-1 overflow-hidden rounded-3xl">
          <div className="relative z-10 flex h-full flex-col">
            <div className="border-b border-white/5 px-8 py-6">
              <h2 className="text-base font-semibold text-slate-100">
                Custom rulebook
              </h2>
              <p className="mt-1 text-sm text-slate-400">
                Rewrite the assistant&apos;s rule set. Your instructions replace
                any baked-in guardrails.
              </p>
            </div>
            <div className="flex flex-1 flex-col gap-3 overflow-hidden p-6">
              <textarea
                value={systemRules}
                onChange={(event) => setSystemRules(event.target.value)}
                className="h-52 w-full flex-1 resize-none rounded-2xl border border-white/10 bg-slate-950/70 px-5 py-4 text-sm leading-relaxed text-slate-200 outline-none transition focus:border-indigo-400 focus:ring-2 focus:ring-indigo-400/40"
                placeholder="Describe exactly how your Llama assistant should behave…"
              />
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                <p className="text-xs text-slate-500">
                  Active rules ({activeSystemPrompt.length} characters)
                </p>
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={() => setSystemRules(DEFAULT_RULES)}
                    className="rounded-xl border border-white/10 px-4 py-2 text-sm font-medium text-slate-200 transition hover:border-white/30 hover:bg-slate-800/60"
                  >
                    Restore default rules
                  </button>
                  <button
                    type="button"
                    onClick={handleReset}
                    className="rounded-xl bg-indigo-500/80 px-4 py-2 text-sm font-semibold text-white shadow-lg shadow-indigo-950/40 transition hover:bg-indigo-500"
                  >
                    Reset conversation
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="flex min-h-[480px] flex-1 flex-col rounded-3xl border border-white/10 bg-slate-950/60 shadow-2xl shadow-black/30">
          <div className="border-b border-white/5 px-8 py-6">
            <h2 className="text-base font-semibold text-slate-100">
              Dialogue studio
            </h2>
            <p className="mt-1 text-sm text-slate-400">
              Talk with the model using your custom personality and rules.
            </p>
          </div>
          <div className="flex flex-1 flex-col gap-6 overflow-y-auto px-6 py-6">
            {messages.length === 0 ? (
              <div className="flex flex-1 items-center justify-center rounded-3xl border border-dashed border-slate-700/60 bg-slate-900/30 px-6 py-10 text-center text-sm text-slate-400">
                <p>
                  Nothing here yet. Craft your personal rulebook, then ask a
                  question to watch Llama follow your lead.
                </p>
              </div>
            ) : (
              messages.map((message, index) => (
                <article
                  key={`${message.role}-${index}`}
                  className={`max-w-2xl rounded-2xl px-5 py-4 text-sm leading-relaxed ${
                    message.role === "user"
                      ? "self-end bg-indigo-600/30 text-slate-100"
                      : "self-start border border-white/5 bg-slate-900/70 text-slate-200"
                  }`}
                >
                  <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-indigo-200/70">
                    {message.role === "user" ? "You" : "Llama"}
                  </div>
                  <div className="whitespace-pre-wrap">{message.content}</div>
                </article>
              ))
            )}
            <div ref={scrollRef} />
          </div>
          <div className="border-t border-white/5 px-6 py-5">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
              <textarea
                value={userInput}
                onChange={(event) => setUserInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    if (canSend) void handleSend();
                  }
                }}
                placeholder={
                  isLoadingModel
                    ? "Model is still loading…"
                    : "Type a request for your custom assistant…"
                }
                disabled={!engine || isStreaming || isLoadingModel}
                className="h-28 w-full flex-1 resize-none rounded-2xl border border-white/10 bg-slate-900/70 px-4 py-3 text-sm text-slate-200 outline-none transition focus:border-indigo-400 focus:ring-2 focus:ring-indigo-400/40 disabled:cursor-not-allowed disabled:opacity-50"
              />
              <button
                type="button"
                onClick={() => void handleSend()}
                disabled={!canSend}
                className="group flex w-full items-center justify-center gap-2 rounded-2xl bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-indigo-950/30 transition hover:from-indigo-400 hover:via-purple-400 hover:to-pink-400 disabled:cursor-not-allowed disabled:opacity-60 sm:w-40"
              >
                {isStreaming ? (
                  <span className="flex items-center gap-2">
                    <span className="size-2 animate-ping rounded-full bg-white" />
                    Generating…
                  </span>
                ) : (
                  "Send"
                )}
              </button>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}

import { streamText } from "ai";
import { openai } from "@ai-sdk/openai";

function normalizeMessages(messages: any[]) {
  return messages
    .map((m) => {
      if (m.role === "user") {
        return {
          role: "user",
          content: m.content ?? m.parts?.[0]?.text ?? ""
        };
      }

      if (m.role === "assistant") {
        return {
          role: "assistant",
          content: m.content ?? ""
        };
      }

      return null;
    })
    .filter(Boolean);
}

export async function POST(req: Request) {
  const { messages } = await req.json();

  const cleanMessages = normalizeMessages(messages);

  const lastUser = cleanMessages.filter((m) => m.role === "user").at(-1);

  const ragRes = await fetch("http://127.0.0.1:8000/rag", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: lastUser?.content ?? ""
    })
  });

  const rag = await ragRes.json();

  const result = streamText({
    model: openai("gpt-4o-mini"),
    messages: [
      {
        role: "system",
        content: `Use this context:\n${rag.answer}`
      },
      ...cleanMessages
    ]
  });

  return result.toUIMessageStreamResponse();
}
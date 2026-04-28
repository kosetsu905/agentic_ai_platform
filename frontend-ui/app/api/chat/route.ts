import {
  createUIMessageStream,
  createUIMessageStreamResponse,
} from "ai";

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
  return createUIMessageStreamResponse({
    status: 200,
    statusText: 'OK',
    headers: {'Custom-Header': 'value',
    },
    stream: createUIMessageStream({
      execute({ writer }) {
        writer.write({
          type: "text-start",
          id: "answer",
        });

        writer.write({
          type: "text-delta",
          id: "answer",
          delta: rag.answer,
        });

        writer.write({
          type: "text-end",
          id: "answer",
        });
      },
    }),
  });
}
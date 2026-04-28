"use client";

import { AssistantRuntimeProvider } from "@assistant-ui/react";
import { useChatRuntime, AssistantChatTransport } from "@assistant-ui/react-ai-sdk";
import { ThreadList } from "@/components/thread-list";
import { Thread } from "@/components/thread";

export default function MyApp() {
  const runtime = useChatRuntime({
    transport: new AssistantChatTransport({
      api: "/api/chat",
    }),
  });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <div className="h-screen flex">
        <ThreadList className="w-64" />
        <div className="flex-1">
          <Thread />
        </div>
      </div>
    </AssistantRuntimeProvider>
  );
}
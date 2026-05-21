"use client";

import { useState, useRef, useCallback, type FC } from "react";
import { MicIcon, SquareIcon, LoaderIcon } from "lucide-react";
import { useComposerRuntime } from "@assistant-ui/react";
import { TooltipIconButton } from "@/components/tooltip-icon-button";

type RecordingState = "idle" | "recording" | "uploading";

export const VoiceInputButton: FC = () => {
  const [state, setState] = useState<RecordingState>("idle");
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const composer = useComposerRuntime();

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "audio/webm";

      const mediaRecorder = new MediaRecorder(stream, { mimeType });
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop());

        const blob = new Blob(chunksRef.current, { type: mimeType });
        if (blob.size === 0) {
          setState("idle");
          return;
        }

        setState("uploading");

        try {
          const formData = new FormData();
          formData.append("audio", blob, "recording.webm");

          const res = await fetch("/api/transcribe", {
            method: "POST",
            body: formData,
          });

          if (!res.ok) {
            throw new Error(`Transcription failed: ${res.status}`);
          }

          const data = await res.json();
          const text = data.text?.trim();

          if (text) {
            const currentText = composer.getState().text ?? "";
            const separator = currentText ? " " : "";
            composer.setText(currentText + separator + text);
          }
        } catch (err) {
          console.error("Voice transcription error:", err);
        } finally {
          setState("idle");
        }
      };

      mediaRecorder.start();
      setState("recording");
    } catch (err) {
      console.error("Microphone access denied or not available:", err);
      setState("idle");
    }
  }, [composer]);

  const stopRecording = useCallback(() => {
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state !== "inactive") {
      recorder.stop();
    }
  }, []);

  const handleClick = useCallback(() => {
    if (state === "idle") {
      startRecording();
    } else if (state === "recording") {
      stopRecording();
    }
  }, [state, startRecording, stopRecording]);

  if (state === "uploading") {
    return (
      <div className="flex items-center gap-1.5 text-muted-foreground text-xs">
        <LoaderIcon className="size-3.5 animate-spin" />
        <span className="hidden @[20rem]:inline">识别中...</span>
      </div>
    );
  }

  const isRecording = state === "recording";

  return (
    <TooltipIconButton
      tooltip={isRecording ? "停止录音" : "语音输入"}
      side="bottom"
      variant="ghost"
      size="icon"
      className="size-8 rounded-full p-1 font-semibold text-xs hover:bg-muted-foreground/15 dark:border-muted-foreground/15 dark:hover:bg-muted-foreground/30"
      aria-label={isRecording ? "停止录音" : "语音输入"}
      type="button"
      onClick={handleClick}
    >
      {isRecording ? (
        <SquareIcon className="size-4 fill-red-500 text-red-500" />
      ) : (
        <MicIcon className="size-4 stroke-[1.5px]" />
      )}
    </TooltipIconButton>
  );
};

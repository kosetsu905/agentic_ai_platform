import React, { useState } from "react";

export default function App() {
  const [messages, setMessages] = useState([
    { role: "assistant", content: "Hello! Ask me a medical question." }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const sendMessage = () => {
    if (!input.trim() || loading) return;

    const text = input;
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setInput("");
    setLoading(true);

    setTimeout(() => {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "This is a demo response for: " + text
        }
      ]);
      setLoading(false);
    }, 1000);
  };

  return (
    <div style={styles.page}>
      <div style={styles.container}>
        <h1 style={styles.title}>Medical RAG Assistant</h1>

        <div style={styles.chatBox}>
          {messages.map((msg, index) => (
            <div
              key={index}
              style={{
                ...styles.messageRow,
                justifyContent:
                  msg.role === "user" ? "flex-end" : "flex-start"
              }}
            >
              <div
                style={{
                  ...styles.bubble,
                  backgroundColor:
                    msg.role === "user" ? "#2563eb" : "#1e293b"
                }}
              >
                {msg.content}
              </div>
            </div>
          ))}

          {loading && <p style={{ color: "#94a3b8" }}>Thinking...</p>}
        </div>

        <div style={styles.inputRow}>
          <input
            style={styles.input}
            value={input}
            placeholder="Ask a question..."
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          />
          <button style={styles.button} onClick={sendMessage}>
            Send
          </button>
        </div>
      </div>
    </div>
  );
}

const styles = {
  page: {
    background: "#0f172a",
    height: "100vh",
    color: "white",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    fontFamily: "Arial"
  },
  container: {
    width: "90%",
    maxWidth: "900px"
  },
  title: {
    marginBottom: "20px"
  },
  chatBox: {
    background: "#111827",
    height: "70vh",
    overflowY: "auto",
    padding: "20px",
    borderRadius: "16px",
    marginBottom: "15px"
  },
  messageRow: {
    display: "flex",
    marginBottom: "12px"
  },
  bubble: {
    padding: "12px 16px",
    borderRadius: "16px",
    maxWidth: "70%"
  },
  inputRow: {
    display: "flex",
    gap: "10px"
  },
  input: {
    flex: 1,
    padding: "12px",
    borderRadius: "10px",
    border: "none"
  },
  button: {
    background: "#2563eb",
    color: "white",
    border: "none",
    padding: "12px 20px",
    borderRadius: "10px",
    cursor: "pointer"
  }
};
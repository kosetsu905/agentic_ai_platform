export async function POST(req: Request) {
  const formData = await req.formData();

  const res = await fetch("http://127.0.0.1:8000/transcribe", {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    return Response.json({ error: "Transcription failed" }, { status: res.status });
  }

  return res;
}
